"""按处置策略清洗数据集，输出一份新的 LeRobot 数据集。

设计原则：
1. **原始数据集严格只读**：所有写入都落在 output_dir/cleaned_dataset 下，入口处校验
   「输出目录不得等于、也不得位于源数据集内部」。
2. **不只是删除**：按 policy 层的处置执行
   - drop  -> 剔除
   - repair -> 执行修复（边界裁帧 / 时间轴重建 / meta 索引重建 / stats 重算）
   - keep / review -> 保留
   修复优先于删除，是真实业务里「尽可能保住数据」的基本要求。
3. **优先调用 `lerobot-edit-dataset` CLI**：但它只能删不可修，因此只要存在 repair 项
   就强制走内置实现，并在结果里记录原因，不静默降级。
4. 手写实现基于 pyarrow 表操作（table.take / set_column），不做 pandas 往返，
   避免固定长度向量列（state/action）的 schema 被破坏。

子模块：types（结果结构）、cli（官方 CLI 路径）、manual（内置重写路径）、
meta（meta 目录重建）、util（共用文件/表工具）。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable

from ..loader import LeRobotDatasetView
from .cli import try_cli
from .manual import copy_untouched, manual_clean, reindex_untouched
from .meta import write_notes
from .types import CLEANED_DIRNAME, NOTES_FILENAME, CleaningResult
from .util import assert_safe_output, verify

LOG = logging.getLogger(__name__)

__all__ = ["CLEANED_DIRNAME", "NOTES_FILENAME", "CleaningResult", "clean_dataset"]


# ------------------------------------------------------------------ 主入口
def clean_dataset(
    dataset: LeRobotDatasetView,
    dispositions: dict[int, Any] | Iterable[int],
    output_dir: str | Path,
    *,
    prefer_cli: bool = True,
    overwrite: bool = True,
    dataset_issues: Iterable[Any] | None = None,
    out_root: str | Path | None = None,
    link_mode: str = "copy",
) -> CleaningResult:
    """按处置结果清洗：drop 的删除，repair 的修补，其余保留。

    Args:
        dispositions: {episode_index: Decision 或 disposition 字符串}；也接受 [ep, ...]
            表示这些 episode 全部剔除（兼容旧用法）。
        dataset_issues: 数据集级问题（validator 的 DatasetIssue 或同结构 dict）。
            其中 ``meta_index_inconsistent`` 表示全局 index 断裂，即使没有
            episode 需要删除/修复，也会按真实行数重建 index。
        out_root: 产出数据集的根目录。默认 ``output_dir/cleaned_dataset``；
            独立导出脚本可直接指定目标目录。
        link_mode: 视频整文件保留时的物化方式，``copy`` 或 ``hardlink``。
    """
    output_dir = Path(output_dir).expanduser().resolve()
    root = Path(out_root).expanduser().resolve() if out_root else output_dir / CLEANED_DIRNAME
    assert_safe_output(dataset.root, root)

    dropped, trim_plan, rebuild_ts, repair_index = _parse_dispositions(dispositions)
    all_eps = dataset.episode_indices
    kept = [e for e in all_eps if e not in set(dropped)]
    repairs = {
        "trim": sorted(trim_plan),
        "timestamps": sorted(rebuild_ts),
    }
    has_repairs = bool(trim_plan or rebuild_ts)
    index_repair = repair_index or _has_index_issue(dataset_issues)

    if not dropped and not has_repairs:
        if index_repair:
            LOG.info("检测到全局 index 不一致：按真实行数重建 index（视频与 meta 保持不变）")
            result = reindex_untouched(dataset, root, overwrite=overwrite)
            write_notes(result, root)
            return result
        LOG.info("没有需要清洗的 episode，仍产出一份结构完整的数据集副本")
        result = copy_untouched(dataset, root, overwrite=overwrite)
        write_notes(result, root)
        return result

    result: CleaningResult | None = None
    expected_frames: int | None = None
    if prefer_cli and not has_repairs:
        result = try_cli(dataset, dropped or [], root, overwrite=overwrite,
                         kept=kept, repairs=repairs)
        if result is None:
            LOG.info("lerobot-edit-dataset 不可用或执行失败，回退到内置清洗实现")
    elif prefer_cli and has_repairs:
        LOG.info("存在需修复项（%s），CLI 只能删除不能修补，改走内置实现",
                 ",".join(k for k, v in repairs.items() if v))

    if result is None:
        result, expected_frames = manual_clean(
            dataset, kept, root, overwrite=overwrite,
            trim_plan=trim_plan, rebuild_ts=rebuild_ts, repairs=repairs,
            link_mode=link_mode,
        )

    result.verification = verify(root, expected_episodes=result.kept_episodes,
                                 expected_frames=expected_frames)
    if result.verification.get("ok") is False:
        result.warnings.append(f"产出物校验未通过: {result.verification}")
    write_notes(result, root)
    return result


def _parse_dispositions(
    dispositions: dict[int, Any] | Iterable[int],
) -> tuple[list[int], dict[int, dict[str, int]], set[int], bool]:
    """归一化成 (要删的, 裁帧方案, 要重建时间轴的, 是否需要重建 index)。"""
    dropped: list[int] = []
    trim_plan: dict[int, dict[str, int]] = {}
    rebuild_ts: set[int] = set()
    repair_index = False

    if isinstance(dispositions, dict):
        for ep, item in dispositions.items():
            ep = int(ep)
            disp = getattr(item, "disposition", item)
            if disp == "drop":
                dropped.append(ep)
            elif disp == "repair":
                repair = getattr(item, "repair", None)
                trim = getattr(item, "trim", None)
                if repair == "trim" and trim:
                    trim_plan[ep] = trim
                elif repair == "timestamps":
                    rebuild_ts.add(ep)
                elif repair == "index":
                    # 没有裁帧/时间轴修复时，仍需真正重建 index（否则该处置形同 noop）
                    repair_index = True
    else:
        dropped = sorted({int(e) for e in dispositions})
    return sorted(dropped), trim_plan, rebuild_ts, repair_index


def _has_index_issue(dataset_issues: Iterable[Any] | None) -> bool:
    """数据集级 meta_index_inconsistent = 全局 index 断裂，需要重建。"""
    for issue in dataset_issues or ():
        code = issue.get("code") if isinstance(issue, dict) else getattr(issue, "code", None)
        if code == "meta_index_inconsistent":
            return True
    return False
