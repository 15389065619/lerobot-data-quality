"""点云逐帧质检规则：8 条规则的实现。"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from ..config import cfg_get
from .context import PointCloudContext
from .types import FrameResult, FramesKey, RULE_MAP
from .utils import to_matrix, xyz

LOG = logging.getLogger(__name__)


def evaluate_frame(points: np.ndarray, key: FramesKey, ctx: PointCloudContext) -> FrameResult:
    """单帧规则判定。points: (N, D)，前三列是 xyz。"""
    pts = to_matrix(points)
    ep, fi = key
    n = int(pts.shape[0])
    res = FrameResult(episode_index=ep, frame_index=fi, point_count=n, valid_count=n)
    if n == 0:
        res.add("pc_empty_or_sparse", point_count=0, valid_count=0,
                note="该帧没有任何点")
        res.metrics = {"point_count": 0, "valid_count": 0}
        return res

    finite_mask = np.isfinite(pts).all(axis=1)
    non_finite = int((~finite_mask).sum())
    finite = pts[finite_mask]
    valid_count = int(finite.shape[0])
    res.valid_count = valid_count

    coords = xyz(finite)
    centroid = np.median(coords, axis=0) if len(coords) else np.zeros(3, dtype=float)
    diag = float(np.linalg.norm(np.max(coords, axis=0) - np.min(coords, axis=0))) if len(coords) > 1 else 0.0

    metrics: dict[str, Any] = {
        "point_count": n,
        "valid_count": valid_count,
        "non_finite_ratio": round(non_finite / n, 5),
        "bbox_diag": round(diag, 5),
        "centroid": [round(float(x), 5) for x in centroid],
    }
    res.metrics = metrics

    if non_finite:
        res.add("pc_non_finite", non_finite=non_finite,
                non_finite_ratio=round(non_finite / n, 5),
                affected_points=non_finite, note="过滤后需补点，保持固定点数")

    if valid_count < ctx.min_points:
        res.add("pc_empty_or_sparse", point_count=n, valid_count=valid_count,
                min_points=ctx.min_points,
                note="有效点不足以重建几何，建议整帧丢弃/占位")
        return res

    if len(coords) >= 2 and diag <= float(cfg_get(ctx.config, "pointcloud.degenerate_diag", 1e-3)):
        res.add("pc_degenerate", bbox_diag=round(diag, 8),
                note="包围盒尺度≈0，点云坍缩为一个点或一条线")
        return res

    dup_ratio = duplicate_ratio(coords, ctx.duplicate_decimals)
    metrics["duplicate_ratio"] = round(dup_ratio, 5)
    if dup_ratio > ctx.max_duplicate_ratio:
        res.add("pc_duplicate", duplicate_ratio=round(dup_ratio, 5),
                max_allowed=ctx.max_duplicate_ratio,
                duplicate_points=int(round(dup_ratio * len(coords))))

    if ctx.outlier_radius > 1e-6:
        outlier_ratio_val = outlier_ratio(coords, ctx.outlier_radius, ctx.outlier_min_neighbors)
        metrics["outlier_ratio"] = round(outlier_ratio_val, 5)
        if outlier_ratio_val > ctx.max_outlier_ratio:
            res.add("pc_outliers", outlier_ratio=round(outlier_ratio_val, 5),
                    radius=round(ctx.outlier_radius, 5),
                    min_neighbors=ctx.outlier_min_neighbors,
                    max_allowed=ctx.max_outlier_ratio,
                    outlier_points=int(round(outlier_ratio_val * len(coords))))

    if ctx.workspace_lower is not None and ctx.workspace_upper is not None:
        outside = outside_ratio(coords, ctx.workspace_lower, ctx.workspace_upper)
        metrics["outside_ratio"] = round(outside, 5)
        if outside > ctx.max_outside_ratio:
            res.add("pc_out_of_bounds", outside_ratio=round(outside, 5),
                    max_allowed=ctx.max_outside_ratio,
                    workspace_lower=[round(float(x), 4) for x in ctx.workspace_lower],
                    workspace_upper=[round(float(x), 4) for x in ctx.workspace_upper])

    if ctx.dim >= 6:
        color_bad = color_bad_ratio(finite[:, 3:6], ctx.color_scale)
        metrics["color_bad_ratio"] = round(color_bad, 5)
        if color_bad > ctx.color_max_bad_ratio:
            res.add("pc_color_invalid", bad_ratio=round(color_bad, 5),
                    color_scale=ctx.color_scale,
                    max_allowed=ctx.color_max_bad_ratio,
                    note=f"颜色应在 [0, {ctx.color_scale:g}] 范围内")

    return res


def apply_temporal_rule(results: dict[FramesKey, FrameResult], ctx: PointCloudContext) -> None:
    """按 episode 计算质心相邻帧跳变；阈值 = max(abs_jump, median × mad_k)。"""
    if not ctx.temporal_abs_jump and not ctx.temporal_mad_k:
        return
    by_ep: dict[int, list[FrameResult]] = {}
    for res in results.values():
        by_ep.setdefault(res.episode_index, []).append(res)

    unreliable = {"pc_empty_or_sparse", "pc_degenerate", "pc_non_finite",
                  "pc_outliers", "pc_out_of_bounds"}
    for ep, frames in by_ep.items():
        frames = [r for r in frames if not (set(r.reasons) & unreliable)]
        frames.sort(key=lambda r: r.frame_index)
        if len(frames) < 2:
            continue
        centroids = np.asarray([r.metrics.get("centroid") or [np.nan, np.nan, np.nan]
                                for r in frames], dtype=float)
        jumps = np.linalg.norm(np.diff(centroids, axis=0), axis=1)
        jumps = jumps[np.isfinite(jumps)]
        if jumps.size == 0:
            continue
        median = float(np.median(jumps))
        threshold = max(ctx.temporal_abs_jump, median * ctx.temporal_mad_k)
        if median <= 0:
            threshold = max(ctx.temporal_abs_jump, 1e-6)
        for i, jump in enumerate(np.linalg.norm(np.diff(centroids, axis=0), axis=1)):
            if not np.isfinite(jump) or jump <= threshold:
                continue
            res = frames[i + 1]
            res.add("pc_temporal_jump", jump=round(float(jump), 5),
                    threshold=round(threshold, 5), median_jump=round(median, 5),
                    note="质心突变，可能是跟踪丢失/坐标系切换；需回放确认")


# ------------------------------------------------------------------ 几何工具
def neighbor_counts(coords: np.ndarray, radius: float) -> np.ndarray:
    """每个点在半径内的邻居数（不含自己），按行分块以控制内存。"""
    n = len(coords)
    counts = np.empty(n, dtype=np.int32)
    if n < 2:
        return np.zeros(n, dtype=np.int32)
    r2 = radius * radius
    block = max(1, int(4_000_000 // max(n, 1)))
    for start in range(0, n, block):
        sub = coords[start:start + block]
        d2 = ((sub[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2)
        counts[start:start + block] = (d2 <= r2).sum(axis=1) - 1
    return counts


def outlier_mask(coords: np.ndarray, radius: float, min_neighbors: int) -> np.ndarray:
    """返回布尔掩码：True 表示该点是离群点。"""
    if len(coords) < 2 or radius <= 0:
        return np.zeros(len(coords), dtype=bool)
    return neighbor_counts(coords, radius) < max(1, min_neighbors)


def duplicate_ratio(coords: np.ndarray, decimals: int) -> float:
    """计算重复点比例。"""
    if len(coords) < 2:
        return 0.0
    q = np.round(coords, decimals)
    return float(1.0 - len(np.unique(q, axis=0)) / len(q))


def outlier_ratio(coords: np.ndarray, radius: float, min_neighbors: int) -> float:
    """整帧离群点比例。"""
    if len(coords) < 2 or radius <= 0:
        return 0.0
    return float((outlier_mask(coords, radius, min_neighbors)).mean())


def outside_ratio(coords: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """超出工作空间包围盒的点比例。"""
    if len(coords) == 0:
        return 0.0
    outside = np.any((coords < lower) | (coords > upper), axis=1)
    return float(outside.mean())


def color_bad_ratio(colors: np.ndarray, scale: float) -> float:
    """颜色通道越界的点比例。"""
    if colors.size == 0:
        return 0.0
    bad = ~np.isfinite(colors) | (colors < 0) | (colors > scale)
    return float(bad.any(axis=1).mean())
