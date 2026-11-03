"""逐 episode 的检查规则（数值、结构、视频内容）。"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

import numpy as np
import pandas as pd

from ..config import cfg_get
from ..loader import LeRobotDatasetView
from .context import DatasetContext
from .probes import FFMPEG, FFPROBE, probe_frame_brightness, probe_video_frames
from .types import EpisodeResult

LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------- 数值类检查
def check_finite(
    arr: np.ndarray, code: str, res: EpisodeResult, frame_col: np.ndarray | None, config: dict[str, Any]
) -> None:
    nan_mask = np.isnan(arr)
    inf_mask = np.isinf(arr)
    if not nan_mask.any() and not inf_mask.any():
        return
    bad = np.where(nan_mask.any(axis=1) | inf_mask.any(axis=1))[0]
    dims = np.where(nan_mask.any(axis=0) | inf_mask.any(axis=0))[0]
    n = arr.shape[0]

    # 边界坏帧 → 优先裁剪而不是整条丢弃（真实业务最常见的「可修」场景）
    is_bad = np.zeros(n, dtype=bool)
    is_bad[bad] = True
    head = 0
    while head < n and is_bad[head]:
        head += 1
    tail = 0
    while tail < n - head and is_bad[n - 1 - tail]:
        tail += 1
    boundary_only = int(bad.size) > 0 and (head + tail) == int(bad.size)
    max_ratio = float(cfg_get(config, "boundary_trim.max_ratio", 0.25))
    trimmable = (
        boundary_only
        and bool(cfg_get(config, "boundary_trim.enabled", True))
        and n > 0
        and max(head, tail) / n <= max_ratio
        and (head + tail) < n
    )

    res.add(
        code,
        nan_count=int(nan_mask.sum()),
        inf_count=int(inf_mask.sum()),
        affected_frames=int(bad.size),
        affected_dims=[int(d) for d in dims[:5]],
        first_bad_frames=_frame_labels(frame_col, bad, 5),
        boundary_only=bool(boundary_only),
        head_frames=int(head) if boundary_only else 0,
        tail_frames=int(tail) if boundary_only else 0,
        trimmable=bool(trimmable),
        note="坏帧集中在首尾，可裁剪保留" if trimmable else "坏帧在序列中段，无法裁剪修复",
    )


def check_action_jump(
    arr: np.ndarray, res: EpisodeResult, thresholds: list[float], config: dict[str, Any]
) -> None:
    if arr.shape[0] < 2 or not thresholds:
        return
    deltas = np.abs(np.diff(arr, axis=0))
    deltas = np.nan_to_num(deltas, nan=0.0, posinf=np.inf)
    dim = min(deltas.shape[1], len(thresholds))
    thr = np.asarray(thresholds[:dim], dtype=float)
    # 夹爪等二值/离散维的「开合」本身就是 ±2 的跳变，属正常控制指令，不能按连续维阈值判异常
    discrete_max = int(cfg_get(config, "action_jump.discrete_max_unique", 3))
    nunique = np.array([
        len(np.unique(arr[np.isfinite(arr[:, d]), d])) if np.isfinite(arr[:, d]).any() else 0
        for d in range(dim)
    ])
    discrete_dims = np.where(nunique <= discrete_max)[0]
    if discrete_dims.size:
        deltas[:, discrete_dims] = 0.0
    over = deltas[:, :dim] > thr
    if not over.any():
        return
    frames, dims = np.where(over)
    finite_deltas = np.where(np.isfinite(deltas[:, :dim]), deltas[:, :dim], 0.0)
    worst = np.unravel_index(int(np.argmax(finite_deltas)), finite_deltas.shape)
    res.add(
        "action_jump",
        jump_count=int(over.sum()),
        max_delta=round(float(np.max(deltas[:, :dim])), 4),
        thresholds_used=[round(float(t), 4) for t in thr],
        discrete_dims=[int(x) for x in discrete_dims],
        worst_frame=int(worst[0]),
        worst_dim=int(worst[1]),
        examples=[
            {"frame": int(f), "dim": int(d), "delta": round(float(deltas[f, d]), 4),
             "threshold": round(float(thr[d]), 4)}
            for f, d in list(zip(frames, dims))[:5]
        ],
    )


def check_static(arr: np.ndarray, kind: str, res: EpisodeResult, config: dict[str, Any]) -> None:
    """空转检测：整条序列几乎无变化 → 机器人没真正执行任务。"""
    if not cfg_get(config, "static_episode.enabled", True):
        return
    if arr.shape[0] < int(cfg_get(config, "static_episode.min_frames_checked", 5)):
        return
    finite = arr[np.isfinite(arr).all(axis=1)]
    if finite.shape[0] < 2:
        return
    std = float(np.mean(np.std(finite, axis=0)))
    thr = float(cfg_get(config, f"static_episode.{kind}_std_threshold", 1e-3))
    if std > thr:
        return
    detail = res.details.setdefault("static_episode", {})
    detail[f"{kind}_std"] = round(std, 6)
    detail["threshold"] = thr
    detail["note"] = "该 episode 数值几乎无变化，疑似未执行任务"
    if "static_episode" not in res.reasons:
        res.reasons.append("static_episode")


def check_joint_limits(
    arr: np.ndarray, res: EpisodeResult, config: dict[str, Any], dataset: LeRobotDatasetView
) -> None:
    if not cfg_get(config, "joint_limits.enabled", False):
        return
    lower = cfg_get(config, "joint_limits.lower")
    upper = cfg_get(config, "joint_limits.upper")
    if not lower or not upper:
        return
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    dim = min(arr.shape[1], lo.size, hi.size)
    if dim == 0:
        return
    tol = float(cfg_get(config, "joint_limits.tolerance", 1e-6))
    finite = arr[:, :dim]
    with np.errstate(invalid="ignore"):
        below = finite < (lo[:dim] - tol)
        above = finite > (hi[:dim] + tol)
    if not below.any() and not above.any():
        return
    res.add(
        "joint_limit_violation",
        violation_count=int(below.sum() + above.sum()),
        dims=[int(d) for d in np.where((below | above).any(axis=0))[0]],
        joint_names=[dataset.joint_names[d] for d in range(dim) if (below | above)[:, d].any()][:5],
    )


def check_duplicate(res: EpisodeResult, arr: np.ndarray, ctx: DatasetContext, config: dict[str, Any]) -> None:
    """同一条轨迹被重复入库会造成分布失衡。用 action 序列量化指纹识别。"""
    if not cfg_get(config, "duplicate_episode.enabled", True):
        return
    decimals = int(cfg_get(config, "duplicate_episode.quantize_decimals", 3))
    quantized = np.round(np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0), decimals).astype(np.float32)
    fingerprint = hashlib.sha1(quantized.tobytes()).hexdigest()[:16]
    first = ctx.fingerprints.get(fingerprint)
    if first is None:
        ctx.fingerprints[fingerprint] = res.episode_index
        return
    if first == res.episode_index:
        return
    res.add("duplicate_episode", duplicate_of=int(first), fingerprint=fingerprint,
            note=f"与 ep {first} 的 action 轨迹完全一致")


# ------------------------------------------------------------------ 结构类检查
def check_frame_count(res: EpisodeResult, bounds: dict[str, Any], config: dict[str, Any]) -> None:
    n = res.frame_count
    low, high = int(bounds["min"]), int(bounds["max"])
    if low <= n <= high:
        return
    res.add(
        "frame_count_abnormal",
        frame_count=n,
        min_allowed=low,
        max_allowed=high,
        direction="too_few" if n < low else "too_many",
        how_derived=f"median={bounds.get('median')} mad={bounds.get('mad')} mode={bounds.get('mode')}",
    )


def check_timestamp(ep_df: pd.DataFrame, res: EpisodeResult, config: dict[str, Any], fps: float) -> None:
    if "timestamp" not in ep_df.columns:
        res.add("timestamp_not_monotonic", reason="缺少 timestamp 列")
        return
    ts = pd.to_numeric(ep_df["timestamp"], errors="coerce").to_numpy(dtype=float)
    if np.isnan(ts).any():
        res.add("timestamp_not_monotonic", reason="timestamp 含 NaN",
                frames=[int(b) for b in np.where(np.isnan(ts))[0][:5]])
        return
    if ts.size < 2:
        return
    diffs = np.diff(ts)
    bad = np.where(diffs <= 0)[0]
    if bad.size:
        res.add(
            "timestamp_not_monotonic",
            reason="非严格递增",
            violations=int(bad.size),
            repair_hint="可用 frame_index/fps 重建时间轴",
            examples=[
                {"prev": round(float(ts[i]), 6), "next": round(float(ts[i + 1]), 6), "delta": round(float(diffs[i]), 6)}
                for i in bad[:5]
            ],
        )
        return

    # 步长偏离：掉帧/抖动，仅记录（info），不阻断
    expected = 1.0 / fps if fps else 0.0
    if expected <= 0:
        return
    ratio_tol = float(cfg_get(config, "timestamp.step_tolerance_ratio", 0.25))
    deviations = np.abs(diffs - expected) > ratio_tol * expected
    if deviations.any():
        res.add(
            "timestamp_step_deviation",
            reason="步长偏离 1/fps",
            affected_frames=int(deviations.sum()),
            expected_step=round(expected, 6),
            examples=[{"frame": int(i), "delta": round(float(diffs[i]), 6)}
                      for i in np.where(deviations)[0][:5]],
        )


def check_alignment(
    ep_df: pd.DataFrame,
    res: EpisodeResult,
    dataset: LeRobotDatasetView,
    config: dict[str, Any],
    fps: float,
) -> None:
    """多模态对齐：时间戳 ↔ 标称时间轴 ↔ 视频时间窗 ↔ frame_index 连续性。

    逐模态「各自是否合理」由 nan/timestamp/frame_mismatch 规则负责；这里只回答
    一个跨模态问题：**它们在时间上是否指向同一段物理时间**。
    """
    if not cfg_get(config, "alignment.enabled", True):
        return
    _check_timestamp_drift(ep_df, res, dataset, config, fps)
    _check_video_span(res, dataset, config, fps)
    _check_frame_index_gap(ep_df, res)


def _meta_table(dataset: LeRobotDatasetView) -> pd.DataFrame | None:
    """并行模式下 worker 拿到的是 EpisodeDatasetStub（episodes_meta），串行模式是 episodes。"""
    for attr in ("episodes", "episodes_meta"):
        table = getattr(dataset, attr, None)
        if isinstance(table, pd.DataFrame) and not table.empty:
            return table
    return None


def _video_windows(dataset: LeRobotDatasetView, episode_index: int) -> list[dict[str, Any]]:
    """从 meta 中取出该 episode 每个相机的时间窗 [from, to]。没有视频元数据时返回空列表。"""
    meta = _meta_table(dataset)
    if meta is None or "episode_index" not in meta.columns:
        return []
    rows = meta[meta["episode_index"] == episode_index]
    if not len(rows):
        return []
    windows = []
    for column in meta.columns:
        name = str(column)
        if not (name.startswith("videos/") and name.endswith("/to_timestamp")):
            continue
        camera = name[len("videos/"):-len("/to_timestamp")]
        from_col = f"videos/{camera}/from_timestamp"
        if from_col not in meta.columns:
            continue
        to_ts = rows.iloc[0].get(column)
        from_ts = rows.iloc[0].get(from_col)
        if pd.isna(to_ts) or pd.isna(from_ts):
            continue
        windows.append({"camera": camera, "from": float(from_ts), "to": float(to_ts)})
    return windows


def _check_timestamp_drift(
    ep_df: pd.DataFrame, res: EpisodeResult, dataset: LeRobotDatasetView,
    config: dict[str, Any], fps: float,
) -> None:
    """state 的首/末时间戳应落在视频时间窗内（位置对不对）。

    长度对不对由 video_span_mismatch 负责；这里只查「起点/终点偏移」，两者互不重叠。

    LeRobot v3 有两种时间戳约定，必须都支持：

    * ``timestamp`` 以 episode 起点为 0（v3 常见）：视频窗是**文件内绝对时间**，
      两者不存在可比较的绝对位置，位置检查直接跳过，时长由 ``video_span_mismatch`` 负责；
    * ``timestamp`` 是文件内全局时间：直接与 ``from/to`` 比较，偏移超容差即漂移。

    用 ``ts[0]≈0`` 区分两者；否则多 episode 拼接的 mp4 会被整体误报偏移。
    """
    if "timestamp" not in ep_df.columns or res.frame_count < 2:
        return
    ts = pd.to_numeric(ep_df["timestamp"], errors="coerce").to_numpy(dtype=float)
    if np.isnan(ts).any():
        return  # 非单调/含 NaN 已由 timestamp_not_monotonic 记录
    windows = _video_windows(dataset, res.episode_index)
    if not windows:
        return
    tol = float(cfg_get(config, "alignment.timestamp_drift_frames", 2.0)) / float(fps or 1.0)
    first, last = float(ts[0]), float(ts[-1])
    if abs(first) <= tol:
        return  # episode 相对时间轴：绝对位置不可比
    problems = []
    for win in windows:
        start_off = first - win["from"]
        end_off = last - win["to"]
        if abs(start_off) > tol or abs(end_off) > tol:
            problems.append({
                "camera": win["camera"],
                "video_from": round(win["from"], 4),
                "video_to": round(win["to"], 4),
                "state_first": round(first, 4),
                "state_last": round(last, 4),
                "start_offset_frames": round(start_off * fps, 3),
                "end_offset_frames": round(end_off * fps, 3),
            })
    if problems:
        res.add(
            "timestamp_drift",
            tolerance_frames=cfg_get(config, "alignment.timestamp_drift_frames", 2.0),
            cameras=problems,
            repair_hint="可用 frame_index/fps 重建时间轴",
        )


def _check_video_span(
    res: EpisodeResult, dataset: LeRobotDatasetView, config: dict[str, Any], fps: float
) -> None:
    """视频时间窗跨度应与 episode 时长（frame_count/fps）一致。

    帧数相等 ≠ 时间对齐：多 episode 拼接的 mp4 只要 from/to 写错，帧数校验照样通过，
    但取出来的是另一段时间的画面。
    """
    if fps <= 0 or res.frame_count <= 0:
        return
    ratio_tol = float(cfg_get(config, "alignment.video_span_tolerance_ratio", 0.05))
    expected = res.frame_count / float(fps)
    problems = []
    for win in _video_windows(dataset, res.episode_index):
        span = win["to"] - win["from"]
        if span <= 0:
            problems.append({"camera": win["camera"], "from": round(win["from"], 4),
                             "to": round(win["to"], 4), "span": round(span, 4)})
            continue
        if abs(span - expected) > ratio_tol * expected:
            problems.append({
                "camera": win["camera"],
                "from": round(win["from"], 4),
                "to": round(win["to"], 4),
                "span": round(span, 4),
                "expected_span": round(expected, 4),
                "deviation": round(span - expected, 4),
            })
    if problems:
        res.add("video_span_mismatch", expected_span=round(expected, 4),
                tolerance_ratio=ratio_tol, cameras=problems)


def _check_frame_index_gap(ep_df: pd.DataFrame, res: EpisodeResult) -> None:
    """frame_index 必须是从 0 开始的连续整数：视频按它切窗口，断裂即错位。"""
    if "frame_index" not in ep_df.columns or res.frame_count < 2:
        return
    idx = pd.to_numeric(ep_df["frame_index"], errors="coerce").to_numpy(dtype=float)
    if np.isnan(idx).any():
        return
    expected = np.arange(res.frame_count, dtype=float)
    if np.array_equal(idx, expected):
        return
    gaps = int(np.count_nonzero(idx != expected))
    res.add(
        "frame_index_gap",
        mismatched_frames=gaps,
        first_bad_frame=int(np.argmax(idx != expected)),
        head=[int(v) for v in idx[:5]],
        repair_hint="清洗时按真实行数重建全局索引",
    )


def check_meta_consistency(res: EpisodeResult, ep_index: int, dataset: LeRobotDatasetView) -> None:
    if "length" not in dataset.episodes.columns:
        return
    row = dataset.episodes.loc[dataset.episodes["episode_index"] == ep_index, "length"]
    if not len(row) or pd.isna(row.iloc[0]):
        return
    try:
        meta_length = int(row.iloc[0])
    except (TypeError, ValueError):
        return
    if meta_length != res.frame_count:
        res.add(
            "meta_index_inconsistent",
            reason="meta length 与 parquet 行数不一致",
            meta_length=meta_length,
            rows=res.frame_count,
            diff=res.frame_count - meta_length,
        )


def check_frame_mismatch(
    res: EpisodeResult, ep_index: int, dataset: LeRobotDatasetView,
    cameras: list[str], precise: bool, config: dict[str, Any],
) -> None:
    if not FFPROBE:
        return
    tol = int(cfg_get(config, "video_frame_tolerance", 1))
    problems: list[dict[str, Any]] = []

    for cam in cameras:
        loc = dataset.video_location(cam, ep_index)
        if loc is None:
            problems.append({"type": "video_unlocatable", "camera": cam})
            continue
        path, from_ts, to_ts = loc
        if not path.is_file():
            problems.append({"type": "video_missing", "camera": cam, "path": str(path)})
            continue
        total = probe_video_frames(path, precise=precise)
        if total is None:
            problems.append({"type": "video_unprobeable", "camera": cam, "path": str(path)})
            continue

        if from_ts >= 0 and to_ts > from_ts:
            start = int(round(from_ts * dataset.fps))
            end = int(round(to_ts * dataset.fps))
            seg = end - start
            if end > total + tol:
                problems.append({"type": "video_truncated", "camera": cam,
                                 "expected_end_frame": end, "actual_total_frames": total})
            elif abs(seg - res.frame_count) > tol:
                problems.append({"type": "count_mismatch", "camera": cam, "video_frames": seg,
                                 "rows": res.frame_count, "diff": seg - res.frame_count})
        elif abs(total - res.frame_count) > tol:
            problems.append({"type": "count_mismatch", "camera": cam, "video_frames": total,
                             "rows": res.frame_count, "diff": total - res.frame_count})

    if problems:
        res.add("frame_mismatch", rows=res.frame_count, tolerance=tol, problems=problems)


def check_task(res: EpisodeResult, ep_index: int, dataset: LeRobotDatasetView) -> None:
    if dataset.task_of(ep_index):
        return
    if "task_index" not in dataset.frames.columns and not dataset.tasks:
        res.add("missing_task", reason="不存在任务描述字段（task_index / tasks）")
    elif not dataset.tasks:
        res.add("missing_task", reason="meta/tasks 为空或缺失，task_index 无法解析")
    else:
        res.add("missing_task", reason="任务描述为空字符串")


# ------------------------------------------------------------------ 视频内容
def check_blank_video(
    res: EpisodeResult, ep_index: int, dataset: LeRobotDatasetView, camera: str | None, config: dict[str, Any]
) -> None:
    """黑屏/遮挡检测：一次 ffmpeg 调用抽多帧缩略图，算灰度标准差。"""
    if camera is None or not FFMPEG:
        return
    loc = dataset.video_location(camera, ep_index)
    if loc is None:
        return
    path, from_ts, to_ts = loc
    if not path.is_file():
        return
    dur = (to_ts - from_ts) if to_ts > from_ts else res.frame_count / max(dataset.fps, 1e-6)
    stats = probe_frame_brightness(path, max(from_ts, 0.0), max(dur, 0.1))
    if stats is None:
        return
    mean_lum, std_lum = stats
    if not std_lum:
        return
    thr = float(cfg_get(config, "blank_video.lum_std_threshold", 4.0))
    bad_ratio = float(np.mean(np.asarray(std_lum) < thr))
    if bad_ratio < float(cfg_get(config, "blank_video.bad_ratio_threshold", 0.6)):
        return
    res.add(
        "blank_video",
        camera=camera,
        sampled_frames=len(std_lum),
        bad_ratio=round(bad_ratio, 3),
        lum_std_threshold=thr,
        mean_luminance=round(float(np.mean(mean_lum)), 2),
        note="画面近乎纯色/全黑，疑似镜头遮挡或未推流",
    )


# -------------------------------------------------------------------- 小工具
def as_matrix(series: pd.Series, res: EpisodeResult, code: str) -> np.ndarray | None:
    try:
        arr = np.asarray(series.to_list(), dtype=float)
    except (TypeError, ValueError):
        res.add(code, error="字段无法解析为数值矩阵（dtype/长度不一致）")
        return None
    if arr.ndim == 0:
        return None
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)
    return arr


def frame_column(ep_df: pd.DataFrame) -> np.ndarray | None:
    if "frame_index" in ep_df.columns:
        return pd.to_numeric(ep_df["frame_index"], errors="coerce").to_numpy()
    return None


def _frame_labels(frame_col: np.ndarray | None, positions: np.ndarray, limit: int) -> list[int]:
    vals = frame_col[positions] if frame_col is not None else positions
    return [int(v) for v in np.asarray(vals)[:limit]]
