"""流式（分片）质检：不把整张逐帧表读进内存。

适用场景：10–50GB 的 LeRobot 数据集在 16–24GB 内存 / 机械盘上跑质检。
做法是两遍扫描 data/*.parquet，每次只驻留一个文件：

  Pass 1（数据集级上下文）
    - 全局帧数：按文件累计每个 episode 的实际行数（与内存模式一致，meta 只做兜底）
    - action 跳变阈值：只采样前 N 个 episode 的 action（streaming.action_sample_episodes）
    - stats.json 时效性：采样前 N 个 episode 的 state 求均值
    - 全局 index 连续性：只累计 (行数, 去重个数, min, max, NaN 数)
    - 重复 episode 指纹：fp -> 首次出现的 episode，必须覆盖全量才能识别跨文件重复

  Pass 2（逐 episode 规则）
    - 直接复用内存模式的 `_validate_non_video`，保证规则与严重度完全一致
    - 同时累计 frames_by_episode 与 state/action 的 min/max/mean/NaN/Inf
    - 最后统一跑视频探针（仍有界并发）

与内存模式的差异只有一处：action 阈值来自采样而非全量。样本足够大时
（默认 200 个 episode）median/MAD 已经稳定，报告里的 resolved_thresholds 会如实记录。
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd

from ..config import cfg_get, load_config
from ..loader import (
    ACTION_KEY,
    STATE_KEY,
    TASK_INDEX_KEY,
    DatasetFormatError,
    LeRobotDatasetView,
    iter_frame_chunks,
)
from .context import (
    _as_float_matrix,
    _index_continuity_issue,
    _resolve_frame_bounds,
    _stats_staleness_issue,
    _thresholds_from_arrays,
)
from .probes import FFMPEG, FFPROBE
from .types import DatasetIssue, EpisodeResult, ValidationResult

LOG = logging.getLogger(__name__)

PASS1_COLUMNS = ("episode_index", "frame_index", "index", ACTION_KEY, STATE_KEY)
PASS2_COLUMNS = ("episode_index", "frame_index", "timestamp", ACTION_KEY, STATE_KEY, TASK_INDEX_KEY)


def validate_dataset_streaming(
    dataset: LeRobotDatasetView,
    config: dict[str, Any] | None = None,
    *,
    prior: Mapping[int, EpisodeResult] | None = None,
    only: Iterable[int] | None = None,
) -> ValidationResult:
    """与 validate_dataset 同签名的流式实现（入参语义一致）。"""
    # 避免与包 __init__ 循环导入：运行期再取共享的逐 episode 实现
    from . import _refresh_duplicate, _run_video_checks, _validate_non_video

    config = config or load_config(robot_type=dataset.info.get("robot_type"))
    columns = set(dataset.column_names)
    skipped: dict[str, str] = {}

    blank_enabled = bool(cfg_get(config, "blank_video.enabled", True)) and FFMPEG is not None
    if not blank_enabled:
        skipped["blank_video"] = "未找到 ffmpeg 或已禁用视频内容检测"
    precise = bool(cfg_get(config, "probe.precise", False))
    if not FFPROBE:
        skipped["frame_mismatch"] = "未找到 ffprobe，跳过视频帧对齐"
    if not dataset.video_features:
        skipped["frame_mismatch"] = "数据集中没有视频特征"

    stream_eps = _streamed_episodes(dataset)
    ctx, skipped_extra = _build_context(dataset, config, columns, stream_eps=stream_eps)
    skipped.update(skipped_extra)

    cameras = dataset.video_features[: int(cfg_get(config, "probe.max_cameras", 3))]
    max_probes = int(cfg_get(config, "blank_video.max_probes", 400))
    revalidate = None if only is None else {int(ep) for ep in only}

    # ------------------------------------------------------------ Pass 2
    results: dict[int, EpisodeResult] = {}
    frames_by_episode: dict[int, int] = {}
    field_stats = _FieldStats([key for key in (STATE_KEY, ACTION_KEY) if key in columns])
    pending_video: list[tuple[EpisodeResult, int, bool]] = []
    reused = 0

    pass2_columns = [c for c in PASS2_COLUMNS if c in columns]
    for chunk in iter_frame_chunks(dataset, columns=pass2_columns, episodes=stream_eps):
        if "episode_index" not in chunk.columns:
            raise DatasetFormatError("data/*.parquet 缺少 episode_index 列，无法按 episode 质检")
        field_stats.update(chunk)
        for raw_ep, ep_df in chunk.groupby("episode_index", sort=True):
            ep_index = int(raw_ep)
            if "frame_index" in ep_df.columns:
                ep_df = ep_df.sort_values("frame_index", kind="stable")
            frames_by_episode[ep_index] = frames_by_episode.get(ep_index, 0) + int(len(ep_df))

            if revalidate is not None and ep_index not in revalidate and prior and ep_index in prior:
                res = EpisodeResult.from_dict(prior[ep_index].to_dict())
                _refresh_duplicate(res, ep_df, ctx, config)
                results[ep_index] = res
                reused += 1
                continue

            res = _validate_non_video(ep_df, ep_index, dataset, ctx, config, skipped)
            results[ep_index] = res
            want_blank = blank_enabled and len(pending_video) < max_probes
            pending_video.append((res, ep_index, want_blank))

    _run_video_checks(pending_video, dataset, cameras, precise, config)
    if reused:
        LOG.info("增量质检：复用 %d 个 episode 的历史结论，重检 %d 个", reused, len(results) - reused)

    total_frames = int(sum(frames_by_episode.values()))
    result = ValidationResult(
        total_episodes=len(results),
        results=results,
        config=config,
        fps=dataset.fps,
        dataset_root=str(dataset.root),
        resolved_thresholds=ctx.resolved,
        dataset_issues=ctx.issues,
        skipped_checks=skipped,
        probe_available=bool(FFPROBE),
        aggregates={
            "total_frames": total_frames,
            "frames_by_episode": frames_by_episode,
            "field_stats": field_stats.result(),
        },
    )
    sev = result.severity_summary()
    LOG.info(
        "质检完成（流式）：%d 个 episode / %d 帧，正常 %d / 异常 %d（error %d, warning %d, info %d）",
        result.total_episodes, total_frames, result.valid_episodes, len(result.invalid_episodes),
        sev.get("error", 0), sev.get("warning", 0), sev.get("info", 0),
    )
    return result


# ------------------------------------------------------------------ Pass 1
def _build_context(
    dataset: LeRobotDatasetView,
    config: dict[str, Any],
    columns: set[str],
    stream_eps: list[int] | None = None,
) -> tuple[Any, dict[str, str]]:
    """第一遍扫描：阈值、stats 时效、index 连续性、重复指纹。"""
    from .context import DatasetContext

    skipped: dict[str, str] = {}
    sample_limit = max(1, int(cfg_get(config, "streaming.action_sample_episodes", 200)))
    action_sample = set(dataset.episode_indices[:sample_limit])

    stale_limit = (
        int(cfg_get(config, "stale_stats.sample_episodes", 200))
        if cfg_get(config, "stale_stats.enabled", True) and dataset.stats and STATE_KEY in columns
        else 0
    )
    state_sample = set(dataset.episode_indices[:stale_limit]) if stale_limit else set()

    dup_enabled = bool(cfg_get(config, "duplicate_episode.enabled", True)) and ACTION_KEY in columns
    decimals = int(cfg_get(config, "duplicate_episode.quantize_decimals", 3))

    action_parts: dict[int, list[np.ndarray]] = {}
    state_parts: dict[int, list[np.ndarray]] = {}
    fingerprints: dict[str, int] = {}
    counts: dict[int, int] = {}
    rows = 0
    nan_index = 0
    idx_min: int | None = None
    idx_max: int | None = None

    # state 只在需要判 stats 时效时读；否则少读一半列宽，50GB 级别差距明显
    pass1_columns = [
        c for c in PASS1_COLUMNS
        if c in columns and (c != STATE_KEY or state_sample)
    ]
    for chunk in iter_frame_chunks(dataset, columns=pass1_columns, episodes=stream_eps):
        if "episode_index" not in chunk.columns:
            raise DatasetFormatError("data/*.parquet 缺少 episode_index 列，无法按 episode 质检")
        rows += int(len(chunk))

        if "index" in chunk.columns:
            idx = pd.to_numeric(chunk["index"], errors="coerce").to_numpy(dtype=float)
            nan_index += int(np.isnan(idx).sum())
            finite = idx[np.isfinite(idx)]
            if finite.size:
                lo, hi = int(finite.min()), int(finite.max())
                idx_min = lo if idx_min is None else min(idx_min, lo)
                idx_max = hi if idx_max is None else max(idx_max, hi)

        for raw_ep, ep_df in chunk.groupby("episode_index", sort=True):
            ep_index = int(raw_ep)
            if "frame_index" in ep_df.columns:
                ep_df = ep_df.sort_values("frame_index", kind="stable")
            counts[ep_index] = counts.get(ep_index, 0) + int(len(ep_df))

            if ep_index in action_sample and ACTION_KEY in ep_df.columns:
                arr = _as_float_matrix(ep_df[ACTION_KEY])
                if arr is not None:
                    action_parts.setdefault(ep_index, []).append(arr)
            if ep_index in state_sample and STATE_KEY in ep_df.columns:
                arr = _as_float_matrix(ep_df[STATE_KEY])
                if arr is not None:
                    state_parts.setdefault(ep_index, []).append(arr)

            if dup_enabled:
                arr = _as_float_matrix(ep_df[ACTION_KEY])
                fingerprint = _fingerprint(arr, decimals)
                if fingerprint is not None:
                    prev = fingerprints.get(fingerprint)
                    fingerprints[fingerprint] = ep_index if prev is None else min(prev, ep_index)

    # ---- 帧数区间：优先实际行数（与内存模式一致），完全没有行时退回 meta length
    if counts:
        count_values = np.asarray([counts[ep] for ep in sorted(counts)], dtype=float)
    else:
        meta_counts = dataset.frames_by_episode()
        count_values = meta_counts.to_numpy(dtype=float) if len(meta_counts) else np.asarray([], dtype=float)
    frame_bounds = _resolve_frame_bounds(count_values, config)

    # ---- action 阈值（采样）
    threshold_arrays = [_stack(action_parts[ep]) for ep in sorted(action_parts)]
    action_thresholds, action_meta = _thresholds_from_arrays(threshold_arrays, config)
    if not action_thresholds:
        skipped["action_jump"] = "无可用 action 数据推导阈值"

    # ---- stats.json 时效性（采样）
    issues: list[DatasetIssue] = []
    stale_issue = None
    if state_sample:
        state_arrays = [_stack(state_parts[ep]) for ep in sorted(state_parts)]
        actual_mean = _finite_mean(state_arrays)
        stale_issue = _stats_staleness_issue(dataset.stats, actual_mean, config)
    if stale_issue:
        issues.append(stale_issue)

    if "index" in columns:
        continuity = _index_continuity_issue(
            rows=rows,
            count=rows - nan_index,
            min_index=idx_min,
            max_index=idx_max,
            nan_count=nan_index,
        )
        if continuity:
            issues.append(continuity)

    resolved = {
        "frame_count": frame_bounds,
        "action_jump": action_meta,
        "timestamp_step_tolerance": round(
            float(cfg_get(config, "timestamp.step_tolerance_ratio", 0.25)) / max(dataset.fps, 1e-6), 6
        ),
        "video_frame_tolerance": int(cfg_get(config, "video_frame_tolerance", 1)),
    }
    LOG.info(
        "流式第一遍完成：%d 行 / %d 个 episode，action 阈值采样 %d 个 episode",
        rows, len(counts), len(action_parts),
    )
    return DatasetContext(frame_bounds, action_thresholds, fingerprints, issues, resolved), skipped


def _fingerprint(arr: np.ndarray | None, decimals: int) -> str | None:
    if arr is None:
        return None
    quantized = np.round(np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0), decimals).astype(np.float32)
    return hashlib.sha1(quantized.tobytes()).hexdigest()[:16]


def _stack(parts: list[np.ndarray]) -> np.ndarray:
    return parts[0] if len(parts) == 1 else np.vstack(parts)


def _streamed_episodes(dataset: LeRobotDatasetView) -> list[int] | None:
    """--max_episodes 生效时只扫保留的 episode；否则扫全部（None 表示不过滤）。"""
    if dataset.truncated_to is None:
        return None
    return sorted(dataset.episode_indices)


def _finite_mean(arrays: list[np.ndarray]) -> np.ndarray | None:
    if not arrays:
        return None
    stacked = np.vstack(arrays)
    finite = stacked[np.isfinite(stacked).all(axis=1)]
    if finite.size == 0:
        return None
    return finite.mean(axis=0)


# --------------------------------------------------------------- 字段统计
class _FieldStats:
    """流式累计 state/action 的 min/max/mean/NaN/Inf，产物字段与 reporter 内存版一致。"""

    def __init__(self, keys: list[str]) -> None:
        self.keys = keys
        self._state: dict[str, dict[str, Any]] = {
            key: {"dims": 0, "nan": 0, "inf": 0, "min": None, "max": None, "sum": 0.0, "finite": 0,
                  "error": None}
            for key in keys
        }

    def update(self, chunk: pd.DataFrame) -> None:
        for key in self.keys:
            if key not in chunk.columns:
                continue
            st = self._state[key]
            try:
                arr = np.asarray(chunk[key].to_list(), dtype=float)
            except (TypeError, ValueError):
                st["error"] = "无法解析为数值矩阵"
                continue
            if arr.ndim == 0 or arr.size == 0:
                continue
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            elif arr.ndim > 2:
                arr = arr.reshape(arr.shape[0], -1)
            if arr.ndim != 2:
                continue
            st["dims"] = max(st["dims"], int(arr.shape[1]))
            st["nan"] += int(np.isnan(arr).sum())
            st["inf"] += int(np.isinf(arr).sum())
            finite = arr[np.isfinite(arr)]
            if finite.size:
                lo, hi = float(finite.min()), float(finite.max())
                st["min"] = lo if st["min"] is None else min(st["min"], lo)
                st["max"] = hi if st["max"] is None else max(st["max"], hi)
                st["sum"] += float(finite.sum())
                st["finite"] += int(finite.size)

    def result(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, st in self._state.items():
            if st["error"]:
                out[key] = {"error": st["error"]}
                continue
            if st["dims"] == 0 and not st["nan"] and not st["inf"]:
                continue
            out[key] = {
                "dims": st["dims"],
                "nan_count": int(st["nan"]),
                "inf_count": int(st["inf"]),
                "min": round(st["min"], 6) if st["min"] is not None else None,
                "max": round(st["max"], 6) if st["max"] is not None else None,
                "mean": round(st["sum"] / st["finite"], 6) if st["finite"] else None,
            }
        return out
