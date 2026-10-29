"""内置清洗实现：删除 + 边界裁帧 + 时间轴重建 + 视频重编码。"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ..loader import (
    DEFAULT_DATA_PATH,
    DEFAULT_VIDEO_PATH,
    SOURCE_COL,
    LeRobotDatasetView,
)
from ..validator import probe_video_frames
from .meta import copy_tasks, rewrite_episodes_meta, rewrite_info, rewrite_stats
from .types import CleaningResult
from .util import prepare_dir, set_column

LOG = logging.getLogger(__name__)

FFMPEG = shutil.which("ffmpeg")


def manual_clean(
    dataset: LeRobotDatasetView,
    kept: list[int],
    out_root: Path,
    *,
    overwrite: bool,
    trim_plan: dict[int, dict[str, int]],
    rebuild_ts: set[int],
    repairs: dict[str, list[int]],
) -> tuple[CleaningResult, int]:
    prepare_dir(out_root, overwrite=overwrite)
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
    rewrite_episodes_meta(dataset, kept, ep_index_map, data_meta, new_video_meta, out_root)
    copy_tasks(dataset.root, out_root)
    total_frames = int(sum(m["length"] for m in data_meta.values()))
    rewrite_info(dataset, kept, total_frames, out_root)
    rewrite_stats(dataset, cleaned_frames, out_root)

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


def copy_untouched(dataset: LeRobotDatasetView, out_root: Path, *, overwrite: bool) -> CleaningResult:
    prepare_dir(out_root, overwrite=overwrite)
    shutil.copytree(dataset.root, out_root, dirs_exist_ok=True)
    return CleaningResult(
        success=True,
        method="noop-copy",
        output_root=out_root,
        removed_episodes=[],
        kept_episodes=len(dataset.episode_indices),
        episode_index_map={e: e for e in dataset.episode_indices},
    )


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

        sub = set_column(sub, "episode_index", new_eps)
        sub = set_column(sub, "frame_index", frame_idx)
        sub = set_column(sub, "index", idx_values)

        # 时间轴重建：timestamp = frame_index / fps（修复非单调）
        if any(int(e) in rebuild_new for e in new_eps):
            ts = (frame_idx / fps).astype(np.float64)
            sub = set_column(sub, "timestamp", ts)
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
    if SOURCE_COL in cleaned.columns:
        cleaned = cleaned.drop(columns=[SOURCE_COL])
    kept_lengths = {ep: int(m["length"]) for ep, m in data_meta.items()}
    LOG.info("data parquet 重写完成：%d 个文件 / %d 行", file_index, len(cleaned))
    return data_meta, cleaned, kept_lengths


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


def _source_files_in_order(dataset: LeRobotDatasetView) -> list[Path]:
    if SOURCE_COL not in dataset.frames.columns:
        return []
    return [Path(v) for v in dataset.frames[SOURCE_COL].drop_duplicates().tolist()]
