"""数据集级上下文：自适应阈值推导与全局一致性检查。"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..config import cfg_get
from ..loader import ACTION_KEY, STATE_KEY, LeRobotDatasetView
from .types import DatasetIssue

LOG = logging.getLogger(__name__)


@dataclass
class DatasetContext:
    frame_bounds: dict[str, Any]
    action_thresholds: list[float]
    fingerprints: dict[str, int]
    issues: list[DatasetIssue]
    resolved: dict[str, Any]

    @classmethod
    def build(
        cls, dataset: LeRobotDatasetView, config: dict[str, Any], skipped: dict[str, str]
    ) -> "DatasetContext":
        counts = dataset.frames.groupby("episode_index").size().to_numpy(dtype=float)
        frame_bounds = _resolve_frame_bounds(counts, config)
        action_thresholds, action_meta = _resolve_action_thresholds(dataset, config)
        issues: list[DatasetIssue] = []
        for issue in (_check_stats_staleness(dataset, config), _check_index_continuity(dataset)):
            if issue:
                issues.append(issue)
        if not action_thresholds:
            skipped["action_jump"] = "无可用 action 数据推导阈值"

        resolved = {
            "frame_count": frame_bounds,
            "action_jump": action_meta,
            "timestamp_step_tolerance": round(
                float(cfg_get(config, "timestamp.step_tolerance_ratio", 0.25)) / max(dataset.fps, 1e-6), 6
            ),
            "video_frame_tolerance": int(cfg_get(config, "video_frame_tolerance", 1)),
        }
        # 指纹预计算覆盖全部 episode：这样即使增量跳过某些 episode，
        # 新增 episode 与历史 episode 的重复仍能被识别。
        fingerprints = _fingerprint_actions(dataset, config)
        return cls(frame_bounds, action_thresholds, fingerprints, issues, resolved)


def _fingerprint_actions(dataset: LeRobotDatasetView, config: dict[str, Any]) -> dict[str, int]:
    """action 序列量化指纹 -> 首次出现的 episode（按 episode_index 升序）。"""
    if not cfg_get(config, "duplicate_episode.enabled", True):
        return {}
    if ACTION_KEY not in dataset.frames.columns:
        return {}
    decimals = int(cfg_get(config, "duplicate_episode.quantize_decimals", 3))
    out: dict[str, int] = {}
    for ep_index, ep_df in dataset.frames.groupby("episode_index", sort=True):
        try:
            arr = np.asarray(ep_df[ACTION_KEY].to_list(), dtype=float)
        except (TypeError, ValueError):
            continue
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        quantized = np.round(np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0), decimals).astype(np.float32)
        fingerprint = hashlib.sha1(quantized.tobytes()).hexdigest()[:16]
        out.setdefault(fingerprint, int(ep_index))
    return out


def _resolve_frame_bounds(counts: np.ndarray, config: dict[str, Any]) -> dict[str, Any]:
    """median ± k·MAD，再夹到 [floor, ceiling]。"""
    mode = str(cfg_get(config, "frame_count.mode", "auto"))
    floor = int(cfg_get(config, "frame_count.floor", 8))
    ceiling = int(cfg_get(config, "frame_count.ceiling", 20000))
    if counts.size == 0:
        return {"mode": mode, "min": floor, "max": ceiling, "median": 0.0, "mad": 0.0}
    median = float(np.median(counts))
    mad = float(np.median(np.abs(counts - median)))
    if mode == "fixed":
        return {"mode": "fixed", "min": floor, "max": ceiling, "median": median, "mad": mad}
    sigma = 1.4826 * mad if mad > 0 else 0.0
    low = max(floor, median - float(cfg_get(config, "frame_count.mad_k_low", 3.0)) * sigma)
    high = min(ceiling, median + float(cfg_get(config, "frame_count.mad_k_high", 8.0)) * sigma)
    return {
        "mode": "auto",
        "min": int(max(floor, np.floor(low))),
        "max": int(min(ceiling, np.ceil(high))),
        "median": round(median, 1),
        "mad": round(mad, 1),
    }


def _resolve_action_thresholds(
    dataset: LeRobotDatasetView, config: dict[str, Any]
) -> tuple[list[float], dict[str, Any]]:
    """逐维阈值 = max(abs_floor, median|Δ| + mad_k·1.4826·MAD|Δ|)。"""
    if ACTION_KEY not in dataset.frames.columns:
        return _thresholds_from_arrays([], config)

    arrays: list[np.ndarray] = []
    for _, ep_df in dataset.frames.groupby("episode_index", sort=False):
        arr = _as_float_matrix(ep_df[ACTION_KEY])
        if arr is not None:
            arrays.append(arr)
    return _thresholds_from_arrays(arrays, config)


def _thresholds_from_arrays(
    arrays: list[np.ndarray], config: dict[str, Any]
) -> tuple[list[float], dict[str, Any]]:
    """由若干 episode 的 action 矩阵推导逐维阈值。

    流式模式只采样部分 episode，因此把「收集 episode」与「算阈值」拆开；
    内存模式传入全量 episode，结果与原来完全一致。
    """
    abs_floor = float(cfg_get(config, "action_jump.abs_floor", 1.0))
    mad_k = float(cfg_get(config, "action_jump.mad_k", 12.0))

    deltas: list[np.ndarray] = []
    for arr in arrays:
        arr = np.asarray(arr, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.ndim != 2 or arr.shape[0] < 2:
            continue
        d = np.abs(np.diff(arr, axis=0))
        d = d[np.isfinite(d).all(axis=1)]
        if d.size:
            deltas.append(d)
    if not deltas:
        return [], {"abs_floor": abs_floor, "mad_k": mad_k, "per_dim": []}

    stacked = np.vstack(deltas)
    med = np.median(stacked, axis=0)
    mad = np.median(np.abs(stacked - med), axis=0)
    thresholds = np.maximum(abs_floor, med + mad_k * 1.4826 * np.where(mad > 0, mad, 0.0))
    return (
        [float(t) for t in thresholds],
        {
            "abs_floor": abs_floor,
            "mad_k": mad_k,
            "per_dim": [round(float(t), 4) for t in thresholds],
            "median_delta": [round(float(m), 4) for m in med],
            "mad_delta": [round(float(m), 4) for m in mad],
        },
    )


def _check_stats_staleness(dataset: LeRobotDatasetView, config: dict[str, Any]) -> DatasetIssue | None:
    """meta/stats.json 与实测数据偏差过大 → 训练归一化会全错（数据集级问题）。"""
    if not cfg_get(config, "stale_stats.enabled", True) or not dataset.stats:
        return None
    if STATE_KEY not in dataset.frames.columns:
        return None
    limit = int(cfg_get(config, "stale_stats.sample_episodes", 200))
    rows = dataset.frames[dataset.frames["episode_index"].isin(dataset.episode_indices[:limit])]
    arr = _as_float_matrix(rows[STATE_KEY])
    if arr is None:
        return None
    arr = arr[np.isfinite(arr).all(axis=1)]
    if arr.size == 0:
        return None
    return _stats_staleness_issue(dataset.stats, arr.mean(axis=0), config)


def _stats_staleness_issue(
    stats: dict[str, Any], actual_mean: np.ndarray | None, config: dict[str, Any]
) -> DatasetIssue | None:
    """stats 与实测均值的偏差判定；流式模式直接传入采样算出的均值。"""
    if actual_mean is None or not stats:
        return None
    actual_mean = np.asarray(actual_mean, dtype=float)
    try:
        recorded_arr = np.asarray((stats.get(STATE_KEY) or {}).get("mean"), dtype=float)
    except (TypeError, ValueError):
        return None
    if recorded_arr.size == 0:
        return None
    if recorded_arr.shape != actual_mean.shape:
        return DatasetIssue(
            "stats_stale", "stats.json 维度与数据不一致",
            {"recorded_shape": list(recorded_arr.shape), "actual_shape": list(actual_mean.shape)},
            repair="stats",
        )
    scale = np.maximum(np.abs(recorded_arr), 1e-6)
    deviations = np.abs(actual_mean - recorded_arr) / scale
    worst = float(np.max(deviations))
    tol = float(cfg_get(config, "stale_stats.mean_rel_tolerance", 0.05))
    if worst <= tol:
        return None
    return DatasetIssue(
        "stats_stale",
        "meta/stats.json 与实测数据偏差过大（归一化统计已过期）",
        {
            "max_relative_deviation": round(worst, 4),
            "tolerance": tol,
            "worst_dims": [int(i) for i in np.argsort(deviations)[::-1][:3]],
            "recorded_mean_first3": [round(float(x), 4) for x in recorded_arr.ravel()[:3]],
            "actual_mean_first3": [round(float(x), 4) for x in actual_mean.ravel()[:3]],
        },
        repair="stats",
    )


def _check_index_continuity(dataset: LeRobotDatasetView) -> DatasetIssue | None:
    """全局 index 必须是 0..N-1 连续；断裂会让 DataLoader 采样与 shard 划分出错。"""
    if "index" not in dataset.frames.columns:
        return None
    idx = pd.to_numeric(dataset.frames["index"], errors="coerce").to_numpy()
    unique = np.unique(idx)
    return _index_continuity_issue(
        rows=dataset.num_frames,
        count=int(unique.size),
        min_index=int(unique[0]) if unique.size else None,
        max_index=int(unique[-1]) if unique.size else None,
        nan_count=int(np.isnan(idx).sum()),
    )


def _index_continuity_issue(
    rows: int,
    count: int,
    min_index: int | None,
    max_index: int | None,
    nan_count: int = 0,
) -> DatasetIssue | None:
    """由 (行数, 去重个数, 最小/最大) 判定连续性。

    流式模式不保留全量 index，只统计这三个标量；在「行数与取值范围一致」的前提下，
    统计量等价于 np.unique（N 个互不相同的整数必然铺满 0..N-1）。
    """
    if nan_count:
        return DatasetIssue("meta_index_inconsistent", "index 列含 NaN",
                            {"nan_count": int(nan_count)}, repair="index")
    if rows == 0:
        return None
    if count == rows and min_index == 0 and max_index == rows - 1:
        return None
    return DatasetIssue(
        "meta_index_inconsistent",
        "全局 index 不连续或存在重复",
        {"rows": rows, "unique": int(count), "min": min_index, "max": max_index},
        repair="index",
    )


def _as_float_matrix(series: pd.Series) -> np.ndarray | None:
    """把向量列转成 (frames, dims) 的 float 矩阵；无法解析时返回 None。"""
    try:
        arr = np.asarray(series.to_list(), dtype=float)
    except (TypeError, ValueError):
        return None
    if arr.ndim == 0:
        return None
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)
    return arr

