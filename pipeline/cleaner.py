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
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .loader import (
    DEFAULT_DATA_PATH,
    DEFAULT_VIDEO_PATH,
    _SOURCE_COL,
    LeRobotDatasetView,
    load_dataset,
)
from .validator import probe_video_frames

LOG = logging.getLogger(__name__)

CLI_NAME = "lerobot-edit-dataset"
FFMPEG = shutil.which("ffmpeg")
CLEANED_DIRNAME = "cleaned_dataset"
NOTES_FILENAME = "quality_cleaning_notes.json"


@dataclass
class CleaningResult:
    success: bool
    method: str
    output_root: Path | None
    removed_episodes: list[int] = field(default_factory=list)
    repaired_episodes: list[int] = field(default_factory=list)
    repairs: dict[str, list[int]] = field(default_factory=dict)
    kept_episodes: int = 0
    episode_index_map: dict[int, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    cli_command: str | None = None
    cli_error: str | None = None
    verification: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "performed": bool(self.removed_episodes or self.repaired_episodes),
            "success": self.success,
            "method": self.method,
            "output_path": str(self.output_root) if self.output_root else None,
            "removed_episode_indices": self.removed_episodes,
            "removed_count": len(self.removed_episodes),
            "repaired_episode_indices": self.repaired_episodes,
            "repairs": {k: v for k, v in self.repairs.items()},
            "kept_episodes": self.kept_episodes,
            "episode_index_map": {str(k): v for k, v in sorted(self.episode_index_map.items())},
            "cli_command": self.cli_command,
            "cli_error": self.cli_error,
            "warnings": self.warnings,
            "verification": self.verification,
        }


# ------------------------------------------------------------------ 主入口
def clean_dataset(
    dataset: LeRobotDatasetView,
    dispositions: dict[int, Any] | Iterable[int],
    output_dir: str | Path,
    *,
    prefer_cli: bool = True,
    overwrite: bool = True,
) -> CleaningResult:
    """按处置结果清洗：drop 的删除，repair 的修补，其余保留。

    Args:
        dispositions: {episode_index: Decision 或 disposition 字符串}；也接受 [ep, ...]
            表示这些 episode 全部剔除（兼容旧用法）。
    """
    output_dir = Path(output_dir).expanduser().resolve()
    out_root = output_dir / CLEANED_DIRNAME
    _assert_safe_output(dataset.root, out_root)

    dropped, trim_plan, rebuild_ts = _parse_dispositions(dispositions)
    all_eps = dataset.episode_indices
    kept = [e for e in all_eps if e not in set(dropped)]
    repairs = {
        "trim": sorted(trim_plan),
        "timestamps": sorted(rebuild_ts),
    }
    has_repairs = bool(trim_plan or rebuild_ts)

    if not dropped and not has_repairs:
        LOG.info("没有需要清洗的 episode，仍产出一份结构完整的数据集副本")
        result = _copy_untouched(dataset, out_root, overwrite=overwrite)
        _write_notes(result, out_root)
        return result

    result: CleaningResult | None = None
    expected_frames: int | None = None
    if prefer_cli and not has_repairs:
        result = _try_cli(dataset, dropped or [], out_root, overwrite=overwrite,
                          kept=kept, repairs=repairs)
        if result is None:
            LOG.info("lerobot-edit-dataset 不可用或执行失败，回退到内置清洗实现")
    elif prefer_cli and has_repairs:
        LOG.info("存在需修复项（%s），CLI 只能删除不能修补，改走内置实现",
                 ",".join(k for k, v in repairs.items() if v))

    if result is None:
        result, expected_frames = _manual_clean(
            dataset, kept, out_root, overwrite=overwrite,
            trim_plan=trim_plan, rebuild_ts=rebuild_ts, repairs=repairs,
        )

    result.verification = _verify(out_root, expected_episodes=result.kept_episodes,
                                  expected_frames=expected_frames)
    if result.verification.get("ok") is False:
        result.warnings.append(f"产出物校验未通过: {result.verification}")
    _write_notes(result, out_root)
    return result


def _parse_dispositions(
    dispositions: dict[int, Any] | Iterable[int],
) -> tuple[list[int], dict[int, dict[str, int]], set[int]]:
    """归一化成 (要删的, 裁帧方案, 要重建时间轴的)。"""
    dropped: list[int] = []
    trim_plan: dict[int, dict[str, int]] = {}
    rebuild_ts: set[int] = set()

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
                    pass  # 索引重建是清洗的默认行为，无需额外参数
    else:
        dropped = sorted({int(e) for e in dispositions})
    return sorted(dropped), trim_plan, rebuild_ts


# ---------------------------------------------------------------- CLI 路径
def _try_cli(
    dataset: LeRobotDatasetView,
    bad: list[int],
    out_root: Path,
    *,
    overwrite: bool,
    kept: list[int],
    repairs: dict[str, list[int]],
) -> CleaningResult | None:
    if not bad:
        return None
    exe = os.environ.get("LEROBOT_EDIT_DATASET_CMD") or shutil.which(CLI_NAME)
    if not exe:
        LOG.debug("未找到 %s", CLI_NAME)
        return None

    try:
        help_proc = subprocess.run([exe, "--help"], capture_output=True, text=True, timeout=120, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        LOG.warning("探测 %s 失败: %s", CLI_NAME, exc)
        return None
    help_text = (help_proc.stdout or "") + (help_proc.stderr or "")
    if "delete_episodes" not in help_text:
        LOG.warning("当前 %s 不支持 delete_episodes（版本过旧），改用手写实现", CLI_NAME)
        return None

    _prepare_dir(out_root, overwrite=overwrite)
    hf_home = out_root.parent / ".hf_lerobot_home"
    cmd = [
        exe,
        "--repo-id", f"local/{dataset.root.name}",
        "--root", str(dataset.root),
        "--operation.type", "delete_episodes",
        "--operation.episode_indices", json.dumps(bad),
        "--new_repo_id", f"local/{out_root.name}",
    ]
    env = {**os.environ, "HF_LEROBOT_HOME": str(hf_home)}
    LOG.info("调用 CLI: %s", " ".join(cmd))
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=3600, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        LOG.warning("CLI 执行异常: %s", exc)
        return None
    if proc.returncode != 0:
        LOG.warning("CLI 返回码 %s，stderr: %s", proc.returncode, (proc.stderr or "")[-800:])
        return None

    produced = _find_cli_output(hf_home, out_root.name)
    if produced is None:
        LOG.warning("CLI 执行成功但未在 %s 下找到产出数据集，回退到手写实现", hf_home)
        return None

    if produced.resolve() != out_root.resolve():
        for item in produced.iterdir():
            target = out_root / item.name
            if target.is_dir():
                shutil.copytree(item, target, dirs_exist_ok=True)
            else:
                shutil.copy2(item, target)

    return CleaningResult(
        success=True,
        method="lerobot-edit-dataset",
        output_root=out_root,
        removed_episodes=sorted(bad),
        repaired_episodes=[],
        repairs=repairs,
        kept_episodes=len(kept),
        episode_index_map={old: i for i, old in enumerate(kept)},
        cli_command=" ".join(cmd),
    )


def _find_cli_output(hf_home: Path, name: str) -> Path | None:
    for cand in hf_home.rglob("meta/info.json"):
        root = cand.parent.parent
        if root.name in {name, f"{name}_tmp"} or name in root.name:
            return root
    for cand in hf_home.rglob("meta/info.json"):
        return cand.parent.parent
    return None


# ---------------------------------------------------------------- 手写路径
def _manual_clean(
    dataset: LeRobotDatasetView,
    kept: list[int],
    out_root: Path,
    *,
    overwrite: bool,
    trim_plan: dict[int, dict[str, int]],
    rebuild_ts: set[int],
    repairs: dict[str, list[int]],
) -> tuple[CleaningResult, int]:
    _prepare_dir(out_root, overwrite=overwrite)
    warnings: list[str] = []
    if FFMPEG is None:
        warnings.append("未找到 ffmpeg，视频文件将原样复制，可能残留已删除 episode 的画面（需后续重编码）。")

    # 1) 视频窗口（裁帧的 episode 需要同步收窄窗口）
    windows = _collect_video_windows(dataset, kept, trim_plan, warnings)

    # 2) 重写 data parquet（含裁帧、时间轴重建）
    ep_index_map = {old: i for i, old in enumerate(kept)}
    data_meta, cleaned_frames, kept_lengths = _rewrite_frames(
        dataset, kept, ep_index_map, out_root, trim_plan=trim_plan, rebuild_ts=rebuild_ts
    )

    # 3) 重建视频
    new_video_meta, video_warnings = _rebuild_videos(dataset, kept, out_root, windows, kept_lengths)
    warnings.extend(video_warnings)

    # 4) 重建 meta
    _rewrite_episodes_meta(dataset, kept, ep_index_map, data_meta, new_video_meta, out_root)
    _copy_tasks(dataset.root, out_root)
    total_frames = int(sum(m["length"] for m in data_meta.values()))
    _rewrite_info(dataset, kept, total_frames, out_root)
    _rewrite_stats(dataset, cleaned_frames, out_root)

    result = CleaningResult(
        success=True,
        method="manual",
        output_root=out_root,
        removed_episodes=sorted(set(dataset.episode_indices) - set(kept)),
        repaired_episodes=sorted(set(trim_plan) | rebuild_ts),
        repairs=repairs,
        kept_episodes=len(kept),
        episode_index_map=ep_index_map,
        warnings=warnings,
    )
    LOG.info("清洗完成：%d -> %d 个 episode（修复 %d 个），写入 %s",
             len(dataset.episode_indices), len(kept), len(trim_plan) + len(rebuild_ts), out_root)
    return result, total_frames


def _collect_video_windows(
    dataset: LeRobotDatasetView, kept: list[int], trim_plan: dict[int, dict[str, int]], warnings: list[str]
) -> dict[int, dict[str, dict[str, Any]]]:
    """{ep: {cam: {path, from, to}}}；裁帧时按 fps 收窄时间窗。"""
    fps = dataset.fps or 30.0
    out: dict[int, dict[str, dict[str, Any]]] = {}
    for ep in kept:
        trim = trim_plan.get(ep) or {}
        head = int(trim.get("head_frames", 0))
        tail = int(trim.get("tail_frames", 0))
        per_ep: dict[str, dict[str, Any]] = {}
        for cam in dataset.video_features:
            loc = dataset.video_location(cam, ep)
            if loc is None:
                warnings.append(f"ep {ep} 相机 {cam} 无法定位视频文件")
                continue
            path, from_ts, to_ts = loc
            if not path.is_file():
                warnings.append(f"ep {ep} 相机 {cam} 视频文件缺失: {path.name}")
                continue
            if (head or tail) and to_ts > from_ts >= 0:
                from_ts += head / fps
                to_ts -= tail / fps
            per_ep[cam] = {"path": path, "from": from_ts, "to": to_ts}
        if per_ep:
            out[ep] = per_ep
    return out


def _rewrite_frames(
    dataset: LeRobotDatasetView,
    kept: list[int],
    ep_index_map: dict[int, int],
    out_root: Path,
    *,
    trim_plan: dict[int, dict[str, int]],
    rebuild_ts: set[int],
) -> tuple[dict[int, dict[str, Any]], pd.DataFrame, dict[int, int]]:
    """按来源文件重写 parquet。返回 (episode 元信息, 清洗后长表, 每个新 episode 的帧数)。"""
    template = dataset.info.get("data_path") or DEFAULT_DATA_PATH
    fps = dataset.fps or 30.0
    running_frames: dict[int, int] = {ep_index_map[ep]: 0 for ep in kept}
    inverse_map = {v: k for k, v in ep_index_map.items()}
    rebuild_new = {ep_index_map[ep] for ep in rebuild_ts if ep in ep_index_map}
    data_meta: dict[int, dict[str, Any]] = {}
    global_index = 0
    file_index = 0
    cleaned_parts: list[pd.DataFrame] = []

    for src_file in _source_files_in_order(dataset):
        table = pq.read_table(src_file)
        if "episode_index" not in table.column_names:
            continue
        ep_values = np.asarray(table.column("episode_index").to_pylist())
        mask = np.isin(ep_values, np.asarray(kept, dtype=np.int64))
        if not mask.any():
            continue
        positions = np.where(mask)[0]
        sub = table.take(pa.array(positions, type=pa.int64()))
        old_eps = ep_values[positions].astype(np.int64)
        new_eps = np.asarray([ep_index_map[int(e)] for e in old_eps], dtype=np.int64)

        # 裁帧：丢掉每个 episode 的首 head 帧 / 尾 tail 帧
        if trim_plan:
            row_pos = np.empty(len(new_eps), dtype=np.int64)
            counter: dict[int, int] = {}
            for i, e in enumerate(new_eps):
                c = counter.get(int(e), 0)
                row_pos[i] = c
                counter[int(e)] = c + 1
            totals = counter
            keep_row = np.ones(len(new_eps), dtype=bool)
            for i, e in enumerate(new_eps):
                trim = trim_plan.get(inverse_map[int(e)]) or {}
                if not trim:
                    continue
                head = int(trim.get("head_frames", 0))
                tail = int(trim.get("tail_frames", 0))
                total = totals[int(e)]
                keep_row[i] = head <= row_pos[i] < total - tail
            if not keep_row.all():
                sub = sub.take(pa.array(np.where(keep_row)[0], type=pa.int64()))
                new_eps = new_eps[keep_row]
                LOG.debug("%s 裁帧后保留 %d/%d 行", src_file.name, sub.num_rows, len(keep_row))

        # frame_index：按新 episode 跨文件连续编号
        frame_idx = np.empty(len(new_eps), dtype=np.int64)
        for i, ep in enumerate(new_eps):
            frame_idx[i] = running_frames[int(ep)]
            running_frames[int(ep)] += 1

        idx_values = np.arange(global_index, global_index + len(new_eps), dtype=np.int64)
        global_index += len(new_eps)

        sub = _set_column(sub, "episode_index", new_eps)
        sub = _set_column(sub, "frame_index", frame_idx)
        sub = _set_column(sub, "index", idx_values)

        # 时间轴重建：timestamp = frame_index / fps（修复非单调）
        if any(int(e) in rebuild_new for e in new_eps):
            ts = (frame_idx / fps).astype(np.float64)
            sub = _set_column(sub, "timestamp", ts)
            LOG.info("%s 重建了 %d 个 episode 的时间轴", src_file.name,
                     len({int(e) for e in new_eps if int(e) in rebuild_new}))

        rel = template.format(chunk_index=0, file_index=file_index)
        out_path = out_root / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(sub, out_path)
        LOG.debug("写入 %s (%d 行)", out_path, sub.num_rows)

        for ep_new in np.unique(new_eps):
            ep_pos = np.where(new_eps == ep_new)[0]
            data_meta[inverse_map[int(ep_new)]] = {
                "length": int(running_frames[int(ep_new)]),
                "from_index": int(idx_values[ep_pos].min()),
                "to_index": int(idx_values[ep_pos].max()) + 1,
                "file_index": file_index,
                "chunk_index": 0,
            }

        cleaned_parts.append(sub.to_pandas())
        file_index += 1

    cleaned = pd.concat(cleaned_parts, ignore_index=True) if cleaned_parts else pd.DataFrame()
    if _SOURCE_COL in cleaned.columns:
        cleaned = cleaned.drop(columns=[_SOURCE_COL])
    kept_lengths = {ep: int(m["length"]) for ep, m in data_meta.items()}
    LOG.info("data parquet 重写完成：%d 个文件 / %d 行", file_index, len(cleaned))
    return data_meta, cleaned, kept_lengths


def _rewrite_episodes_meta(
    dataset: LeRobotDatasetView,
    kept: list[int],
    ep_index_map: dict[int, int],
    data_meta: dict[int, dict[str, Any]],
    video_meta: dict[int, dict[str, dict[str, Any]]],
    out_root: Path,
) -> None:
    """重建 meta/episodes/chunk-000/file-000.parquet。"""
    ep_dir = dataset.root / "meta" / "episodes"
    tables: list[pa.Table] = []
    if ep_dir.is_dir():
        for f in sorted(ep_dir.rglob("*.parquet")):
            try:
                tables.append(pq.read_table(f))
            except Exception as exc:  # noqa: BLE001
                LOG.warning("读取 %s 失败: %s", f, exc)
    if not tables:
        tables = [pa.table({
            "episode_index": pa.array([], type=pa.int64()),
            "length": pa.array([], type=pa.int64()),
            "tasks": pa.array([], type=pa.list_(pa.string())),
            "dataset_from_index": pa.array([], type=pa.int64()),
            "dataset_to_index": pa.array([], type=pa.int64()),
        })]

    combined = tables[0]
    for t in tables[1:]:
        try:
            combined = pa.concat_tables([combined, t], promote_options="permissive")
        except (TypeError, pa.lib.ArrowInvalid):
            combined = pa.concat_tables([combined, t])

    ep_all = np.asarray(combined.column("episode_index").to_pylist()).astype(np.int64)
    positions = np.where(np.isin(ep_all, np.asarray(kept, dtype=np.int64)))[0]
    sub = combined.take(pa.array(positions, type=pa.int64()))
    order = np.argsort([ep_index_map[int(e)] for e in ep_all[positions]])
    sub = sub.take(pa.array(order, type=pa.int64()))

    old_eps = [int(ep_all[positions][i]) for i in order]
    sub = _set_column(sub, "episode_index", np.asarray([ep_index_map[e] for e in old_eps], dtype=np.int64))
    sub = _set_column(sub, "length", np.asarray([data_meta[e]["length"] for e in old_eps], dtype=np.int64))
    sub = _set_column(sub, "dataset_from_index",
                      np.asarray([data_meta[e]["from_index"] for e in old_eps], dtype=np.int64))
    sub = _set_column(sub, "dataset_to_index",
                      np.asarray([data_meta[e]["to_index"] for e in old_eps], dtype=np.int64))
    sub = _set_column(sub, "data/chunk_index", np.asarray([data_meta[e]["chunk_index"] for e in old_eps], dtype=np.int64))
    sub = _set_column(sub, "data/file_index", np.asarray([data_meta[e]["file_index"] for e in old_eps], dtype=np.int64))

    for cam in dataset.video_features:
        chunk_vals, file_vals, from_vals, to_vals = [], [], [], []
        for e in old_eps:
            info = video_meta.get(e, {}).get(cam)
            if info is None:
                chunk_vals.append(0)
                file_vals.append(0)
                from_vals.append(-1.0)
                to_vals.append(-1.0)
            else:
                chunk_vals.append(int(info.get("chunk_index", 0)))
                file_vals.append(int(info.get("file_index", 0)))
                from_vals.append(float(info.get("from", 0.0)))
                to_vals.append(float(info.get("to", -1.0)))
        sub = _set_column(sub, f"videos/{cam}/chunk_index", np.asarray(chunk_vals, dtype=np.int64))
        sub = _set_column(sub, f"videos/{cam}/file_index", np.asarray(file_vals, dtype=np.int64))
        sub = _set_column(sub, f"videos/{cam}/from_timestamp", np.asarray(from_vals, dtype=np.float64))
        sub = _set_column(sub, f"videos/{cam}/to_timestamp", np.asarray(to_vals, dtype=np.float64))

    out_path = out_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(sub, out_path)
    LOG.info("meta/episodes 重建完成：%d 条", sub.num_rows)


def _rebuild_videos(
    dataset: LeRobotDatasetView,
    kept: list[int],
    out_root: Path,
    windows: dict[int, dict[str, dict[str, Any]]],
    kept_lengths: dict[int, int],
) -> tuple[dict[int, dict[str, dict[str, Any]]], list[str]]:
    """按相机重建 mp4：同一源文件内保留需要的区间并重新拼接。"""
    warnings: list[str] = []
    result: dict[int, dict[str, dict[str, Any]]] = {}
    template = dataset.info.get("video_path") or DEFAULT_VIDEO_PATH
    fps = dataset.fps or 30.0

    for cam in dataset.video_features:
        src_to_eps: dict[Path, list[int]] = {}
        for ep in dataset.episode_indices:
            loc = dataset.video_location(cam, ep)
            if loc is not None and loc[0].is_file():
                src_to_eps.setdefault(loc[0], []).append(ep)

        out_index = 0
        for src_path, all_eps in sorted(src_to_eps.items(), key=lambda kv: str(kv[0])):
            keep_here = [e for e in sorted(all_eps) if e in windows and cam in windows[e]]
            if not keep_here:
                continue

            rel = template.format(video_key=cam, chunk_index=0, file_index=out_index)
            out_path = out_root / rel
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_index += 1

            def assign(entries: list[tuple[int, int]]) -> None:
                cum = 0
                for ep, frames in entries:
                    result.setdefault(ep, {})[cam] = {
                        "chunk_index": 0,
                        "file_index": out_index - 1,
                        "from": cum / fps,
                        "to": (cum + frames) / fps,
                    }
                    cum += frames

            whole_file_kept = set(keep_here) == set(all_eps)
            window_covers_file = all(
                windows[e][cam]["from"] <= 0.001 for e in keep_here
            )
            if whole_file_kept and window_covers_file and all(
                not (windows[e][cam]["to"] > 0 and windows[e][cam]["to"] < kept_lengths.get(e, 0) / fps - 0.001)
                for e in keep_here
            ):
                shutil.copy2(src_path, out_path)
                assign([(e, kept_lengths[e]) for e in keep_here])
                continue

            segments = [(e, windows[e][cam]["from"], windows[e][cam]["to"]) for e in keep_here]
            if not FFMPEG or any(f < 0 or t <= f for _, f, t in segments):
                shutil.copy2(src_path, out_path)
                warnings.append(
                    f"相机 {cam} 的 {src_path.name} 含已删除 episode 且无法按时间戳切分，已原样复制（画面会残留脏 episode）。"
                )
                assign([(e, kept_lengths[e]) for e in keep_here])
                continue

            built = _concat_segments(src_path, segments, out_path, fps, warnings)
            assign(built)
            expected = sum(kept_lengths.get(e, 0) for e, _ in built)
            got = sum(f for _, f in built)
            if abs(got - expected) > len(built):
                warnings.append(
                    f"相机 {cam} 重建后总帧数 {got} 与 state 行数 {expected} 存在偏差，建议人工抽查该视频。"
                )
    return result, warnings


def _concat_segments(
    src_path: Path, segments: list[tuple[int, float, float]], out_path: Path, fps: float, warnings: list[str]
) -> list[tuple[int, int]]:
    """把若干 [from, to) 区间抽成独立片段，再无损拼接。返回 [(ep, 实际帧数)]。"""
    built: list[tuple[int, int]] = []
    with tempfile.TemporaryDirectory(prefix="lerobot_clean_") as tmp:
        tmp_dir = Path(tmp)
        seg_files: list[Path] = []
        for ep, from_ts, to_ts in segments:
            seg = tmp_dir / f"seg_{ep:06d}.mp4"
            dur = max(to_ts - from_ts, 1.0 / fps)
            cmd = [
                str(FFMPEG), "-y", "-v", "error",
                "-ss", f"{from_ts:.6f}", "-i", str(src_path), "-t", f"{dur:.6f}",
                "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                "-pix_fmt", "yuv420p", "-r", f"{fps:g}", str(seg),
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, check=False)
            if proc.returncode != 0 or not seg.is_file():
                warnings.append(f"切片失败 ep {ep} ({src_path.name}): {(proc.stderr or '')[-300:]}")
                continue
            frames = probe_video_frames(seg) or int(round(dur * fps))
            built.append((ep, frames))
            seg_files.append(seg)

        if not seg_files:
            shutil.copy2(src_path, out_path)
            warnings.append(f"{src_path.name} 切片全部失败，已原样复制。")
            return built

        if len(seg_files) == 1:
            shutil.copy2(seg_files[0], out_path)
            return built

        list_file = tmp_dir / "concat.txt"
        list_file.write_text("\n".join(f"file '{p.as_posix()}'" for p in seg_files), encoding="utf-8")
        cmd = [
            str(FFMPEG), "-y", "-v", "error", "-f", "concat", "-safe", "0",
            "-i", str(list_file), "-c", "copy", "-movflags", "+faststart", str(out_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, check=False)
        if proc.returncode != 0 or not out_path.is_file():
            shutil.copy2(seg_files[0], out_path)
            warnings.append(f"视频拼接失败 ({out_path.name}): {(proc.stderr or '')[-300:]}")
            return built[:1]
    return built


# ------------------------------------------------------------------ meta 更新
def _rewrite_info(dataset: LeRobotDatasetView, kept: list[int], total_frames: int, out_root: Path) -> None:
    info = dict(dataset.info)
    info["total_episodes"] = len(kept)
    info["total_frames"] = int(total_frames)
    info["total_tasks"] = len(dataset.tasks)
    info["splits"] = {"train": f"0:{len(kept)}"}
    info["data_path"] = dataset.info.get("data_path") or DEFAULT_DATA_PATH
    info["video_path"] = dataset.info.get("video_path") or DEFAULT_VIDEO_PATH
    path = out_root / "meta" / "info.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(info, fh, ensure_ascii=False, indent=2)
    LOG.info("meta/info.json 已更新：total_episodes=%d, total_frames=%d", len(kept), total_frames)


def _rewrite_stats(dataset: LeRobotDatasetView, cleaned: pd.DataFrame, out_root: Path) -> None:
    """在清洗后的数据上重算归一化统计（同时也修复了 stats_stale）。"""
    stats: dict[str, Any] = {}
    for key, spec in dataset.features.items():
        if key not in cleaned.columns:
            if key in dataset.stats:
                stats[key] = dataset.stats[key]
            continue
        try:
            arr = np.asarray(cleaned[key].to_list(), dtype=float)
        except (TypeError, ValueError):
            continue
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        arr = arr[np.isfinite(arr).all(axis=1)] if arr.size else arr
        if not arr.size:
            continue
        stats[key] = {
            "count": int(arr.shape[0]),
            "mean": _vec(np.mean(arr, axis=0)),
            "std": _vec(np.std(arr, axis=0)),
            "min": _vec(np.min(arr, axis=0)),
            "max": _vec(np.max(arr, axis=0)),
            "q01": _vec(np.quantile(arr, 0.01, axis=0)),
            "q10": _vec(np.quantile(arr, 0.10, axis=0)),
            "q50": _vec(np.quantile(arr, 0.50, axis=0)),
            "q90": _vec(np.quantile(arr, 0.90, axis=0)),
            "q99": _vec(np.quantile(arr, 0.99, axis=0)),
        }
    path = out_root / "meta" / "stats.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(stats, fh, ensure_ascii=False, indent=2)
    LOG.info("meta/stats.json 已重算（%d 个特征）", len(stats))


def _copy_tasks(src_root: Path, out_root: Path) -> None:
    for name in ("tasks.parquet", "tasks.jsonl", "subtasks.parquet"):
        src = src_root / "meta" / name
        if src.is_file():
            dst = out_root / "meta" / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


def _write_notes(result: CleaningResult, out_root: Path) -> None:
    """把清洗决策留痕在数据集内，方便回溯「哪一版删了什么、修了什么」。"""
    notes = {
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "method": result.method,
        "removed_episode_indices": result.removed_episodes,
        "repaired_episode_indices": result.repaired_episodes,
        "repairs": result.repairs,
        "kept_episodes": result.kept_episodes,
        "episode_index_map": {str(k): v for k, v in sorted(result.episode_index_map.items())},
        "cli_command": result.cli_command,
        "warnings": result.warnings,
        "verification": result.verification,
    }
    path = out_root / "meta" / NOTES_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(notes, fh, ensure_ascii=False, indent=2, default=str)


# ---------------------------------------------------------------- 无异常分支
def _copy_untouched(dataset: LeRobotDatasetView, out_root: Path, *, overwrite: bool) -> CleaningResult:
    _prepare_dir(out_root, overwrite=overwrite)
    shutil.copytree(dataset.root, out_root, dirs_exist_ok=True)
    return CleaningResult(
        success=True,
        method="noop-copy",
        output_root=out_root,
        removed_episodes=[],
        kept_episodes=len(dataset.episode_indices),
        episode_index_map={e: e for e in dataset.episode_indices},
    )


# ------------------------------------------------------------------ 工具函数
def _source_files_in_order(dataset: LeRobotDatasetView) -> list[Path]:
    if _SOURCE_COL not in dataset.frames.columns:
        return []
    return [Path(v) for v in dataset.frames[_SOURCE_COL].drop_duplicates().tolist()]


def _set_column(table: pa.Table, name: str, values: np.ndarray | list) -> pa.Table:
    """保持原 arrow 类型替换列；列不存在时原样返回。"""
    idx = table.schema.get_field_index(name)
    if idx < 0:
        return table
    field = table.schema.field(idx)
    try:
        arr = pa.array(values, type=field.type)
    except (pa.lib.ArrowInvalid, pa.lib.ArrowNotImplementedError, TypeError) as exc:
        LOG.warning("列 %s 类型转换失败(%s)，尝试无类型推断", name, exc)
        arr = pa.array([float(v) if isinstance(v, np.floating) else v for v in values])
    return table.set_column(idx, field, arr)


def _assert_safe_output(src_root: Path, out_root: Path) -> None:
    src_root = src_root.resolve()
    out_root = out_root.resolve()
    if out_root == src_root or src_root in out_root.parents:
        raise ValueError(f"输出目录 {out_root} 位于源数据集 {src_root} 内部，拒绝写入（源数据集必须保持只读）")


def _prepare_dir(out_root: Path, *, overwrite: bool) -> None:
    if out_root.exists():
        if not overwrite:
            raise FileExistsError(f"输出目录已存在: {out_root}")
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)


def _verify(out_root: Path, expected_episodes: int, expected_frames: int | None) -> dict[str, Any]:
    try:
        view = load_dataset(out_root)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"重新加载产出物失败: {exc}"}
    ok = view.num_episodes == expected_episodes and (expected_frames is None or view.num_frames == expected_frames)
    if view.num_episodes != expected_episodes:
        LOG.warning("校验：产出 %d 个 episode，预期 %d", view.num_episodes, expected_episodes)
    contiguous = view.episode_indices == list(range(view.num_episodes))
    return {
        "ok": bool(ok and contiguous),
        "episodes": view.num_episodes,
        "frames": view.num_frames,
        "episode_indices_contiguous": contiguous,
    }


def _vec(arr: np.ndarray) -> list[float]:
    return [round(float(x), 6) for x in np.asarray(arr).reshape(-1)]
