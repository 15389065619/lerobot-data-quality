"""点云阈值上下文：自适应阈值推导与采样。"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from ..config import cfg_get
from .utils import to_matrix, xyz

LOG = logging.getLogger(__name__)


@dataclass
class PointCloudContext:
    """点云质检的规则上下文（阈值、包围盒、颜色量程等）。"""
    feature_key: str
    dim: int
    expected_points: int | None
    config: dict[str, Any]
    min_points: int
    sparse_ratio: float
    duplicate_decimals: int
    max_duplicate_ratio: float
    outlier_radius: float
    outlier_min_neighbors: int
    max_outlier_ratio: float
    max_outside_ratio: float
    workspace_lower: np.ndarray | None
    workspace_upper: np.ndarray | None
    color_scale: float
    color_max_bad_ratio: float
    temporal_abs_jump: float
    temporal_mad_k: float
    max_repair_ratio: float
    jitter_ratio: float
    resolved: dict[str, Any]


def build_context(
    root: Path, files: list[Path], info: dict[str, Any], config: dict[str, Any],
    feature_key: str, expected_points: int | None, dim: int,
) -> PointCloudContext:
    """从数据集采样推导点云阈值上下文。"""
    cfg_prefix = "pointcloud"
    n = expected_points or 0
    min_points = max(1, int(cfg_get(config, f"{cfg_prefix}.min_points", 16)))
    sparse_ratio = float(cfg_get(config, f"{cfg_prefix}.sparse_ratio", 0.02))
    if n:
        min_points = max(min_points, int(np.ceil(n * sparse_ratio)))

    duplicate_decimals = int(cfg_get(config, f"{cfg_prefix}.duplicate_decimals", 4))
    max_duplicate_ratio = float(cfg_get(config, f"{cfg_prefix}.max_duplicate_ratio", 0.05))
    outlier_radius_k = float(cfg_get(config, f"{cfg_prefix}.outlier_radius_k", 2.5))
    outlier_min_neighbors = int(cfg_get(config, f"{cfg_prefix}.outlier_min_neighbors", 2))
    max_outlier_ratio = float(cfg_get(config, f"{cfg_prefix}.max_outlier_ratio", 0.10))
    max_outside_ratio = float(cfg_get(config, f"{cfg_prefix}.workspace.max_outside_ratio", 0.05))
    color_max_bad_ratio = float(cfg_get(config, f"{cfg_prefix}.color.max_bad_ratio", 0.01))
    temporal_abs_jump = float(cfg_get(config, f"{cfg_prefix}.temporal.abs_jump", 0.05))
    temporal_mad_k = float(cfg_get(config, f"{cfg_prefix}.temporal.mad_k", 6.0))
    max_repair_ratio = float(cfg_get(config, f"{cfg_prefix}.max_repair_ratio", 0.35))
    jitter_ratio = float(cfg_get(config, f"{cfg_prefix}.jitter_ratio", 0.005))

    mode = str(cfg_get(config, f"{cfg_prefix}.workspace.mode", "auto")).lower()
    lower = cfg_get(config, f"{cfg_prefix}.workspace.lower")
    upper = cfg_get(config, f"{cfg_prefix}.workspace.upper")
    sample_limit = int(cfg_get(config, f"{cfg_prefix}.sample_frames", 200))
    sample_diag = estimate_extent(files, feature_key, sample_limit)
    nn_median = median_nn_distance(files, feature_key, min(sample_limit, 60))
    if nn_median > 0:
        radius = max(1e-6, nn_median * outlier_radius_k)
    else:
        radius = max(1e-6, sample_diag * 0.04)

    if mode == "fixed" and lower and upper:
        ws_lower = np.asarray(lower[:3], dtype=float)
        ws_upper = np.asarray(upper[:3], dtype=float)
        ws_source = "fixed"
    elif sample_diag > 0:
        lo, hi = estimate_bounds(files, feature_key, sample_limit,
                                  float(cfg_get(config, f"{cfg_prefix}.workspace.margin_ratio", 0.10)))
        ws_lower, ws_upper = lo, hi
        ws_source = "auto"
    else:
        ws_lower = ws_upper = None
        ws_source = "unavailable"

    color_scale = estimate_color_scale(files, feature_key, dim, sample_limit)
    resolved = {
        "workspace_mode": ws_source,
        "workspace_lower": ws_lower.tolist() if ws_lower is not None else None,
        "workspace_upper": ws_upper.tolist() if ws_upper is not None else None,
        "outlier_radius": round(radius, 6),
        "outlier_nn_median": round(nn_median, 6),
        "outlier_radius_k": outlier_radius_k,
        "scene_extent": round(sample_diag, 6),
        "color_scale": color_scale,
        "min_points": min_points,
        "auto_workspace_note": "包围盒由样本帧 0.5%/99.5% 分位数 + 10% 边距推导",
    }
    return PointCloudContext(
        feature_key=feature_key, dim=dim, expected_points=expected_points, config=config,
        min_points=min_points, sparse_ratio=sparse_ratio,
        duplicate_decimals=duplicate_decimals, max_duplicate_ratio=max_duplicate_ratio,
        outlier_radius=radius, outlier_min_neighbors=outlier_min_neighbors,
        max_outlier_ratio=max_outlier_ratio, max_outside_ratio=max_outside_ratio,
        workspace_lower=ws_lower, workspace_upper=ws_upper,
        color_scale=color_scale, color_max_bad_ratio=color_max_bad_ratio,
        temporal_abs_jump=temporal_abs_jump, temporal_mad_k=temporal_mad_k,
        max_repair_ratio=max_repair_ratio, jitter_ratio=jitter_ratio, resolved=resolved,
    )


# ------------------------------------------------------------------ 采样函数
def sample_points(files: list[Path], feature_key: str, limit: int) -> np.ndarray:
    """取前 limit 帧的点云，拼成 (M, D) 用于推导尺度/包围盒/颜色范围。"""
    chunks: list[np.ndarray] = []
    taken = 0
    for path in files:
        try:
            pf = pq.ParquetFile(path)
            cols = [c for c in ("episode_index", "frame_index", feature_key) if c in pf.schema_arrow.names]
            if feature_key not in cols:
                continue
            for batch in pf.iter_batches(batch_size=max(1, min(limit, 256)), columns=cols):
                cells = batch.column(batch.schema.get_field_index(feature_key)).to_pylist()
                for cell in cells:
                    arr = to_matrix(cell)
                    if arr.size:
                        chunks.append(arr)
                    taken += 1
                    if taken >= limit:
                        break
                if taken >= limit:
                    break
        except Exception as exc:  # noqa: BLE001
            LOG.debug("采样 %s 失败: %s", path, exc)
        if taken >= limit:
            break
    if not chunks:
        return np.zeros((0, 0), dtype=np.float32)
    return np.concatenate(chunks, axis=0)


def estimate_extent(files: list[Path], feature_key: str, limit: int) -> float:
    """估计点云场景尺度（包围盒对角线）。"""
    pts = sample_points(files, feature_key, limit)
    coords = xyz(pts)
    coords = coords[np.isfinite(coords).all(axis=1)]
    if len(coords) < 4:
        return 0.0
    lo = np.percentile(coords, 0.5, axis=0)
    hi = np.percentile(coords, 99.5, axis=0)
    extent = float(np.linalg.norm(hi - lo))
    return extent if np.isfinite(extent) else 0.0


def median_nn_distance(files: list[Path], feature_key: str, limit: int) -> float:
    """采样若干帧，返回「每帧 1-NN 距离」的中位数，作为点云局部密度尺度。"""
    values: list[float] = []
    taken = 0
    for path in files:
        try:
            pf = pq.ParquetFile(path)
            if feature_key not in pf.schema_arrow.names:
                continue
            for batch in pf.iter_batches(batch_size=max(1, min(limit, 64)), columns=[feature_key]):
                for cell in batch.column(0).to_pylist():
                    coords = xyz(to_matrix(cell))
                    coords = coords[np.isfinite(coords).all(axis=1)]
                    if len(coords) >= 8:
                        d2 = ((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2)
                        np.fill_diagonal(d2, np.inf)
                        values.append(float(np.median(np.sqrt(d2.min(axis=1)))))
                    taken += 1
                    if taken >= limit:
                        break
                if taken >= limit:
                    break
        except Exception as exc:  # noqa: BLE001
            LOG.debug("NN 距离采样 %s 失败: %s", path, exc)
        if taken >= limit:
            break
    return float(np.median(values)) if values else 0.0


def estimate_bounds(
    files: list[Path], feature_key: str, limit: int, margin_ratio: float
) -> tuple[np.ndarray, np.ndarray]:
    """估计工作空间包围盒（分位数 + 边距）。"""
    pts = sample_points(files, feature_key, limit)
    coords = xyz(pts)
    coords = coords[np.isfinite(coords).all(axis=1)]
    if len(coords) < 4:
        return np.zeros(3), np.zeros(3)
    lo = np.percentile(coords, 0.5, axis=0)
    hi = np.percentile(coords, 99.5, axis=0)
    diag = float(np.linalg.norm(hi - lo))
    if not np.isfinite(diag):
        return np.zeros(3), np.zeros(3)
    margin = max(diag * margin_ratio, 1e-6)
    return lo - margin, hi + margin


def estimate_color_scale(files: list[Path], feature_key: str, dim: int, limit: int) -> float:
    """判断颜色量程（0~1 或 0~255）。"""
    if dim < 6:
        return 1.0
    pts = sample_points(files, feature_key, limit)
    if len(pts) == 0:
        return 1.0
    colors = pts[:, 3:6]
    finite = colors[np.isfinite(colors).all(axis=1)]
    if finite.size == 0:
        return 1.0
    return 255.0 if float(np.nanmax(finite)) > 1.5 else 1.0


def rebuild_context(validation: Any) -> PointCloudContext:
    """从 PointCloudValidation 重建规则上下文（清洗阶段使用）。"""
    from .utils import vec_or_none

    return PointCloudContext(
        feature_key=validation.feature_key, dim=validation.dim,
        expected_points=validation.expected_points, config=validation.config,
        min_points=int((validation.resolved or {}).get("min_points")
                       or cfg_get(validation.config, "pointcloud.min_points", 16)),
        sparse_ratio=float(cfg_get(validation.config, "pointcloud.sparse_ratio", 0.02)),
        duplicate_decimals=int(cfg_get(validation.config, "pointcloud.duplicate_decimals", 4)),
        max_duplicate_ratio=float(cfg_get(validation.config, "pointcloud.max_duplicate_ratio", 0.05)),
        outlier_radius=float((validation.resolved or {}).get("outlier_radius")
                             or cfg_get(validation.config, "pointcloud.outlier_radius_k", 2.5)),
        outlier_min_neighbors=int(cfg_get(validation.config, "pointcloud.outlier_min_neighbors", 2)),
        max_outlier_ratio=float(cfg_get(validation.config, "pointcloud.max_outlier_ratio", 0.10)),
        max_outside_ratio=float(cfg_get(validation.config, "pointcloud.workspace.max_outside_ratio", 0.05)),
        workspace_lower=vec_or_none((validation.resolved or {}).get("workspace_lower")),
        workspace_upper=vec_or_none((validation.resolved or {}).get("workspace_upper")),
        color_scale=float((validation.resolved or {}).get("color_scale", 1.0)),
        color_max_bad_ratio=float(cfg_get(validation.config, "pointcloud.color.max_bad_ratio", 0.01)),
        temporal_abs_jump=float(cfg_get(validation.config, "pointcloud.temporal.abs_jump", 0.05)),
        temporal_mad_k=float(cfg_get(validation.config, "pointcloud.temporal.mad_k", 6.0)),
        max_repair_ratio=float(cfg_get(validation.config, "pointcloud.max_repair_ratio", 0.35)),
        jitter_ratio=float(cfg_get(validation.config, "pointcloud.jitter_ratio", 0.005)),
        resolved=validation.resolved or {},
    )
