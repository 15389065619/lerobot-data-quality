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
        # 全部数据文件损坏时 frames 连列都没有：counts 为空，episode 由 validator 兜底标 episode_read_error
        if "episode_index" in dataset.frames.columns:
            counts = dataset.frames.groupby("episode_index").size().to_numpy(dtype=float)
        else:
            counts = np.array([], dtype=float)
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
    """逐维阈值由数据集自身的「正常运动包络」推导，见 `_thresholds_from_arrays`。"""
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
    """由若干 episode 的 action 矩阵推导逐维跳变阈值。

    估计对象是「每集最大 |Δ|」而非逐帧 |Δ|：真实数据里 action 常是
    「保持若干帧 → 跳一步」的步进指令，逐帧分布因此双峰（大量 0 + 运动帧），
    以 median|Δ| 为中心的自适应尺度会被大量 0 拉崩，导致度制机器人（±100°）
    几乎每集都被误报。改用每集峰值后，阈值刻画的是「这个数据集里正常 episode
    的运动包络」，只有明显超出同类 episode 的才判异常。

    episode 太少（< min_episodes_for_peak）时峰值分布的 MAD 极不稳定
    （n=2 时退化成极差），退回逐帧非零 Δ 的稳健尺度，同时忽略保持帧的 0。

    流式模式只采样部分 episode，因此把「收集 episode」与「算阈值」拆开。
    """
    abs_floor = float(cfg_get(config, "action_jump.abs_floor", 1.0))
    mad_k = float(cfg_get(config, "action_jump.max_jump_mad_k", 3.0))
    min_eps = int(cfg_get(config, "action_jump.min_episodes_for_peak", 8))
    margin = float(cfg_get(config, "action_jump.min_margin_ratio", 0.05))

    def _threshold(med: float, mad: float) -> float:
        # MAD=0 时阈值正好落在典型峰值上，float32 舍入噪声就会把「正常」判成异常；
        # 留 5% 相对余量（可由 min_margin_ratio 调整）
        spread = max(mad, margin * abs(med))
        return max(abs_floor, med + mad_k * 1.4826 * spread)

    deltas: list[np.ndarray] = []
    peaks: list[np.ndarray] = []
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
            peaks.append(d.max(axis=0))
    if not deltas:
        return [], {"abs_floor": abs_floor, "max_jump_mad_k": mad_k,
                    "min_margin_ratio": margin, "per_dim": []}

    if len(peaks) >= min_eps:
        stacked = np.vstack(peaks)
        med = np.median(stacked, axis=0)
        mad = np.median(np.abs(stacked - med), axis=0)
        thresholds = np.array([
            _threshold(float(med[i]), float(mad[i])) for i in range(med.size)])
        basis, detail = "episode_max", {
            "episode_count": len(peaks),
            "episode_max_median": [round(float(m), 4) for m in med],
            "episode_max_mad": [round(float(m), 4) for m in mad],
        }
    else:
        stacked = np.vstack(deltas)
        dims = stacked.shape[1]
        thresholds = np.full(dims, abs_floor, dtype=float)
        nz_median = np.zeros(dims, dtype=float)
        for d_i in range(dims):
            col = stacked[:, d_i]
            col = col[col > 0]  # 保持帧的 0 不参与尺度估计
            if col.size == 0:
                continue
            m = np.median(col)
            nz_median[d_i] = m
            nz_mad = float(np.median(np.abs(col - m)))
            thresholds[d_i] = _threshold(float(m), nz_mad)
        basis, detail = "nonzero_delta", {
            "episode_count": len(peaks),
            "nonzero_median": [round(float(m), 4) for m in nz_median],
        }
    return (
        [float(t) for t in thresholds],
        {
            "abs_floor": abs_floor,
            "max_jump_mad_k": mad_k,
            "min_margin_ratio": margin,
            "basis": basis,
            "per_dim": [round(float(t), 4) for t in thresholds],
            **detail,
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

