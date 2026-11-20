"""按处置策略清洗数据集，输出一份新的 LeRobot 数据集。

原始数据集严格只读：所有写入都落在 output_dir/cleaned_dataset 下，入口校验
输出目录不得等于或位于源数据集内部。按 policy 处置执行——drop 剔除、repair
执行修复（边界裁帧 / 时间轴重建 / meta 索引重建 / stats 重算）、keep 与 review
保留；修复优先于删除。纯删除场景优先调用 lerobot-edit-dataset CLI，存在 repair
项时走内置实现（官方 CLI 不能修），并记录原因。内置实现基于 pyarrow 表操作，
不做 pandas 往返，避免固定长度向量列的 schema 被破坏。

子模块：types（结果结构）、cli（官方 CLI 路径）、manual（内置重写路径）、
meta（meta 目录重建）、util（共用文件/表工具）。
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any, Iterable

from ..loader import LeRobotDatasetView
from .cli import try_cli
from .manual import copy_untouched, manual_clean, reindex_untouched
from .meta import write_notes
from .types import CLEANED_DIRNAME, NOTES_FILENAME, CleaningResult
from .util import assert_safe_output, publish_dir, staging_dir, verify

LOG = logging.getLogger(__name__)

__all__ = ["CLEANED_DIRNAME", "NOTES_FILENAME", "CleaningResult", "clean_dataset"]


# 主入口
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
    # 所有写入先落在同级暂存目录，校验通过后整体 rename 发布；
    # 中途失败/Ctrl+C 不会破坏最终路径上已有的上一版产物。
    stage = staging_dir(root)
    assert_safe_output(dataset.root, stage)

    (dropped, trim_plan, rebuild_ts, repair_index,
     split_plan, window_plan) = _parse_dispositions(dispositions)
    all_eps = dataset.episode_indices
    kept = [e for e in all_eps if e not in set(dropped)]
    repairs = {
        "trim": sorted(trim_plan),
        "timestamps": sorted(rebuild_ts),
        "split": sorted(split_plan),
        "window": sorted(window_plan),
    }
    has_repairs = bool(trim_plan or rebuild_ts or split_plan or window_plan)
    index_repair = repair_index or _has_index_issue(dataset_issues)

    # load_frames=False（流式模式）的公开 API 防御：
    # dataset.frames 为空 DataFrame，但 data_files 仍在——manual.py 的关键路径
    # （_source_files_in_order / _episode_row_index）都有回退到 data_files 的
    # 实现，所以纯 drop 场景是安全的。但若有 repair 项（尤其 rebuild_ts / split），
    # manual_clean 的主循环需要逐帧操作，流式模式下 SOURCE_COL 缺失不会崩溃，
    # 但会失去「源文件位置锚定」而纯粹按 parquet 顺序扫描——语义不变，
    # 但多文件场景的 traceability 降低。这里打 warning 让调用方知晓。
    if len(dataset.frames) == 0 and (has_repairs or len(dropped)):
        LOG.warning(
            "dataset.frames 为空（load_frames=False / 流式模式），clean_dataset 将回退到"
            " data_files 逐文件扫描。drop 场景安全；repair 场景逐文件扫描语义正确但"
            " 多文件 traceability 降低。主 CLI --streaming 已跳过 cleaner，此处是公共"
            " API 直调的回退路径。"
        )

    def _publish(result: CleaningResult) -> CleaningResult:
        result.output_root = str(root)
        write_notes(result, stage)
        publish_dir(stage, root)
        return result

    try:
        if not dropped and not has_repairs:
            if index_repair:
                LOG.info("检测到全局 index 不一致：按真实行数重建 index（视频与 meta 保持不变）")
                return _publish(reindex_untouched(dataset, stage, overwrite=overwrite))
            LOG.info("没有需要清洗的 episode，仍产出一份结构完整的数据集副本")
            return _publish(copy_untouched(dataset, stage, overwrite=overwrite))

        result: CleaningResult | None = None
        expected_frames: int | None = None
        if prefer_cli and not has_repairs:
            result = try_cli(dataset, dropped or [], stage, overwrite=overwrite,
                             kept=kept, repairs=repairs)
            if result is None:
                LOG.info("lerobot-edit-dataset 不可用或执行失败，回退到内置清洗实现")
        elif prefer_cli and has_repairs:
            LOG.info("存在需修复项（%s），CLI 只能删除不能修补，改走内置实现",
                     ",".join(k for k, v in repairs.items() if v))

        if result is None:
            result, expected_frames = manual_clean(
                dataset, kept, stage, overwrite=overwrite,
                trim_plan=trim_plan, rebuild_ts=rebuild_ts, repairs=repairs,
                split_plan=split_plan, window_plan=window_plan,
                link_mode=link_mode,
            )

        result.verification = verify(stage, expected_episodes=result.kept_episodes,
                                     expected_frames=expected_frames)
        if result.verification.get("ok") is False:
            result.warnings.append(f"产出物校验未通过: {result.verification}")
        # 软违规（结构细节、规则复检 findings）只告警；结构性硬失败意味着目录
        # 根本不可用，绝不能发布并覆盖上一版产物。抛错会走下面的 except 分支
        # 清掉暂存目录，最终路径保持上一版不变。
        hard = _hard_verification_failures(
            result.verification, result.kept_episodes, expected_frames
        )
        if hard:
            raise RuntimeError("产出物校验硬失败，已中止发布：" + "；".join(hard))
        return _publish(result)
    except BaseException:
        # 包括 KeyboardInterrupt：暂存目录必须清掉，最终路径保持上一版不变
        shutil.rmtree(stage, ignore_errors=True)
        raise


def _hard_verification_failures(
    verification: dict[str, Any], expected_episodes: int, expected_frames: int | None
) -> list[str]:
    """从校验结果里挑出「产出物不可用」的硬失败，区别于可告警的软违规。

    ``ok=False`` 也可能只来自时间戳抖动或规则复检的 drop 建议，那些不该阻断
    发布；但无法重新加载、episode / 帧数不符、编号不连续、产出 0 集这类问题
    说明产物本身是坏的，发布出去会直接覆盖掉上一版好数据。
    """
    failures: list[str] = []
    if verification.get("error"):
        failures.append(f"产出物无法重新加载: {verification['error']}")
    episodes = verification.get("episodes")
    if episodes != expected_episodes:
        failures.append(f"episode 数 {episodes} != 预期 {expected_episodes}")
    if verification.get("episode_indices_contiguous") is False:
        failures.append("episode_index 不连续")
    if expected_frames is not None and verification.get("frames") != expected_frames:
        failures.append(f"帧数 {verification.get('frames')} != 预期 {expected_frames}")
    if expected_episodes and episodes == 0:
        failures.append("产出 0 个 episode")
    return failures


def _parse_dispositions(
    dispositions: dict[int, Any] | Iterable[int],
) -> tuple[list[int], dict[int, dict[str, int]], set[int], bool,
           dict[int, list[dict[str, Any]]], dict[int, dict[str, dict[str, float]]]]:
    """归一化成 (要删的, 裁帧方案, 要重建时间轴的, 是否需要重建 index, 分段计划, 收窗计划)。"""
    dropped: list[int] = []
    trim_plan: dict[int, dict[str, int]] = {}
    rebuild_ts: set[int] = set()
    repair_index = False
    split_plan: dict[int, list[dict[str, Any]]] = {}
    window_plan: dict[int, dict[str, dict[str, float]]] = {}

    if isinstance(dispositions, dict):
        for ep, item in dispositions.items():
            ep = int(ep)
            # 同时支持对象（attr）和 Mapping（key）
            if isinstance(item, dict):
                disp = item.get("disposition", item)
                _g = lambda k, default=None: item.get(k, default)
            else:
                disp = getattr(item, "disposition", item)
                _g = lambda k, default=None: getattr(item, k, default)

            if disp == "drop":
                dropped.append(ep)
            elif disp == "repair":
                repair = _g("repair")
                trim = _g("trim")
                split_plan_item = _g("split_plan")
                window_plan_item = _g("window_plan")
                if repair == "trim" and trim:
                    trim_plan[ep] = dict(trim)
                elif repair == "timestamps":
                    rebuild_ts.add(ep)
                elif repair == "index":
                    # 没有裁帧/时间轴修复时，仍需真正重建 index（否则该处置形同 noop）
                    repair_index = True
                elif repair == "split" and split_plan_item:
                    split_plan[ep] = [dict(s) for s in split_plan_item]
                    rebuild_ts.add(ep)
                elif repair == "window" and window_plan_item:
                    window_plan[ep] = {k: dict(v) for k, v in window_plan_item.items()}
                elif repair is None:
                    # disp=="repair" 但没有指定 repair 子类型——说明调用方传了一个裸字符串
                    # 或 dict 缺 repair 键。默认走 index（最保守的重建动作，不会丢数据），
                    # 并打 warning 提醒调用方补全信息，否则看似修了其实只是重建 index。
                    LOG.warning(
                        "ep %s disposition=repair 但未指定 repair 子类型(trim/timestamps/index/split/window)，"
                        "默认走 index 重建；若预期 trim/split 请补全 repair 字段。", ep,
                    )
                    repair_index = True
                else:
                    LOG.warning(
                        "ep %s repair=%r 未被 _parse_dispositions 识别，该集被静默当作 keep 处理。",
                        ep, repair,
                    )
    else:
        dropped = sorted({int(e) for e in dispositions})
    return sorted(dropped), trim_plan, rebuild_ts, repair_index, split_plan, window_plan


def _has_index_issue(dataset_issues: Iterable[Any] | None) -> bool:
    """数据集级 meta_index_inconsistent = 全局 index 断裂，需要重建。"""
    for issue in dataset_issues or ():
        code = issue.get("code") if isinstance(issue, dict) else getattr(issue, "code", None)
        if code == "meta_index_inconsistent":
            return True
    return False
