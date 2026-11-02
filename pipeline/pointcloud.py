"""LeRobot v3 三维点云质量校验与清洗。

业务定位：LeRobot 数据集里除 state/action/视频之外，还可能带二维点云特征
（info.json 中 shape=[N, D] 的 float 特征，例如 observation.environment_state）。
点云和关节数值的脏法完全不同：NaN、离群点、重复点、坐标系跳变、越出工作空间、
颜色通道越界，都会直接污染几何编码器/策略的输入，而且不会在 state/action 上体现。

设计原则（与数值质检保持一致）：
1. **流式读取**：512×6 的点云 × 2 万帧用 pandas 读会占用 GB 级内存，
   这里直接按 parquet batch 迭代，逐帧评估，只保留结论。
2. **能修不删**：点云是逐帧模态，删行会让 state/action/视频错位；
   因此清洗只改「点」，不改「行」。可修复的坏帧做过滤 + 抖动重采样补点；
   不可修复的帧用同一 episode 内最近的好帧占位（hold-last-good），并留痕。
3. **固定点数契约**：LeRobot 的 Array2D 特征声明了固定 [N, D]，修复后仍保持 N 点，
   不能用变长数组偷偷破坏下游 shape。
4. **阈值自适应**：离群半径按点云自身包围盒尺度缩放；工作空间包围盒默认从
   样本帧的分位数推导，避免不同机器人/场景共用一套绝对坐标。

对外入口：
    detect_pointcloud_feature(features)
    validate_pointcloud_dataset(root_or_view, config, feature_key)
    decide(validation, on_review)
    clean_pointclouds(root, validation, decisions, output_dir, feature_key)
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .config import cfg_get, load_config, severity_of, snapshot as config_snapshot
from .loader import DEFAULT_DATA_PATH, LeRobotDatasetView, heavy_feature_keys
from .manifest import config_digest

LOG = logging.getLogger(__name__)

SEVERITY_ORDER = {"info": 0, "warning": 1, "error": 2}
FramesKey = tuple[int, int]

REPORT_FILENAME = "pointcloud_report.json"
REVIEW_FILENAME = "pointcloud_review_queue.csv"
NOTES_FILENAME = "pointcloud_cleaning_notes.json"
CLEANED_DIRNAME = "pointcloud_cleaned"


# --------------------------------------------------------------------- 规则表
@dataclass(frozen=True)
class Rule:
    code: str
    label: str
    description: str
    repair: str | None = None


RULES: tuple[Rule, ...] = (
    Rule("pc_non_finite", "点云含 NaN/Inf", "深度反投影或采样器输出异常；几何计算会直接产生 NaN", "filter"),
    Rule("pc_empty_or_sparse", "有效点过少", "遮挡/掉帧/深度缺失导致点云塌缩，无法支撑几何特征", None),
    Rule("pc_degenerate", "点云几何退化", "所有点坍缩到同一点/同一直线，包围盒尺度≈0", None),
    Rule("pc_duplicate", "重复点比例过高", "采样器重复采样或去重缺失，会放大局部密度", "dedupe"),
    Rule("pc_outliers", "离群点比例过高", "深度噪声/反光/玻璃穿透；按局部密度半径的统计滤波可去除", "filter"),
    Rule("pc_out_of_bounds", "点云越出工作空间", "外参或坐标系标定漂移，几何先验失效", "bounds"),
    Rule("pc_color_invalid", "颜色通道越界", "RGB 未归一化或编码错误，视觉特征会失真", "color"),
    Rule("pc_temporal_jump", "相邻帧质心跳变", "跟踪丢失或坐标系切换，需人审确认是否真实运动", None),
)

RULE_MAP = {r.code: r for r in RULES}
RULE_ORDER = [r.code for r in RULES]

#: 可以自动修复的规则
REPAIRABLE = {r.code for r in RULES if r.repair}
#: 修复动作集合
REPAIR_ACTIONS = {"filter", "dedupe", "bounds", "color", "resample"}


# ------------------------------------------------------------------- 结果结构
@dataclass
class FrameResult:
    episode_index: int
    frame_index: int
    point_count: int
    valid_count: int
    reasons: list[str] = field(default_factory=list)
    details: dict[str, dict[str, Any]] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def valid(self) -> bool:
        return not self.reasons

    def add(self, code: str, **evidence: Any) -> None:
        if code not in self.reasons:
            self.reasons.append(code)
        self.details.setdefault(code, {}).update(evidence)

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_index": self.episode_index,
            "frame_index": self.frame_index,
            "point_count": self.point_count,
            "valid_count": self.valid_count,
            "reasons": list(self.reasons),
            "details": self.details,
            "metrics": self.metrics,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FrameResult":
        """从 to_dict 的载荷还原（点云扫描断点续传缓存的读取侧）。"""
        return cls(
            episode_index=int(data["episode_index"]),
            frame_index=int(data["frame_index"]),
            point_count=int(data.get("point_count") or 0),
            valid_count=int(data.get("valid_count") or 0),
            reasons=[str(c) for c in (data.get("reasons") or [])],
            details=dict(data.get("details") or {}),
            metrics=dict(data.get("metrics") or {}),
        )


@dataclass
class PointCloudValidation:
    feature_key: str
    dim: int
    total_frames: int
    results: dict[FramesKey, FrameResult]
    config: dict[str, Any]
    dataset_root: str
    resolved: dict[str, Any] = field(default_factory=dict)
    dataset_issues: list[dict[str, Any]] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)
    expected_points: int | None = None
    #: 读取失败被跳过的损坏数据文件（毒数据防冲击），元素为 {"file", "error", ...}
    corrupted_files: list[dict[str, Any]] = field(default_factory=list)

    @property
    def valid_frames(self) -> int:
        return sum(1 for r in self.results.values() if r.valid)

    @property
    def invalid_frames(self) -> list[FrameResult]:
        return [r for r in self.results.values() if not r.valid]

    def severity_of(self, res: FrameResult) -> str:
        levels = [severity_of(self.config, c) for c in res.reasons]
        return max(levels, key=lambda s: SEVERITY_ORDER.get(s, 0)) if levels else "ok"

    def severity_summary(self) -> dict[str, int]:
        out = {"error": 0, "warning": 0, "info": 0, "ok": 0}
        for res in self.results.values():
            sev = self.severity_of(res)
            out[sev] = out.get(sev, 0) + 1
        return out

    def reason_frequency(self) -> dict[str, int]:
        counts = {code: 0 for code in RULE_ORDER}
        for res in self.results.values():
            for code in res.reasons:
                counts[code] = counts.get(code, 0) + 1
        return counts

    def episode_summary(self) -> list[dict[str, Any]]:
        per: dict[int, dict[str, Any]] = {}
        for res in self.results.values():
            item = per.setdefault(res.episode_index, {"episode_index": res.episode_index,
                                                      "frames": 0, "invalid_frames": 0})
            item["frames"] += 1
            if not res.valid:
                item["invalid_frames"] += 1
        rows = sorted(per.values(), key=lambda r: r["episode_index"])
        for row in rows:
            row["invalid_ratio"] = round(row["invalid_frames"] / max(row["frames"], 1), 4)
        return rows

    def to_dict(self) -> dict[str, Any]:
        freq = self.reason_frequency()
        invalid = self.invalid_frames
        payload: dict[str, Any] = {
            "feature_key": self.feature_key,
            "dim": self.dim,
            "expected_points": self.expected_points,
            "total_frames": self.total_frames,
            "valid_frames": self.valid_frames,
            "invalid_frames": len(invalid),
            "severity_summary": self.severity_summary(),
            "reason_frequency": freq,
            "resolved": self.resolved,
            "dataset_issues": self.dataset_issues,
            "episodes": self.episode_summary(),
            "invalid_frame_details": [r.to_dict() for r in invalid],
        }
        if self.corrupted_files:
            payload["corrupted_files"] = self.corrupted_files
        if self.skipped:
            payload["skipped_checks"] = self.skipped
        return payload


# ------------------------------------------------------------------ 阈值上下文
@dataclass
class _Context:
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


# --------------------------------------------------------------------- 对外入口
def detect_pointcloud_feature(features: dict[str, Any]) -> str | None:
    """从 info.json 特征里挑出点云列：二维、点数≥4、维度≥3、浮点。

    优先返回名字里带 point/cloud/lidar 的列，否则取第一个符合形状的列。
    """
    candidates: list[str] = []
    for key in heavy_feature_keys(features):
        spec = features.get(key) or {}
        shape = spec.get("shape")
        if not isinstance(shape, (list, tuple)) or len(shape) != 2:
            continue
        try:
            n, d = int(shape[0]), int(shape[1])
        except (TypeError, ValueError):
            continue
        if n >= 4 and d >= 3:
            candidates.append(key)
    if not candidates:
        return None
    preferred = [k for k in candidates
                 if any(t in k.lower() for t in ("point", "cloud", "lidar", "environment_state"))]
    return (preferred or candidates)[0]


def validate_pointcloud_dataset(
    root_or_view: str | Path | LeRobotDatasetView,
    config: dict[str, Any] | None = None,
    *,
    feature_key: str | None = None,
    max_frames: int | None = None,
    workers: int = 1,
    cache_path: str | Path | None = None,
) -> PointCloudValidation:
    """流式校验点云特征，返回逐帧结论（不保留原始点云）。

    workers>1 时按 (parquet 文件, row group) 分片多进程扫描，结论在主进程归并，
    与串行结果逐位一致（跨帧的 pc_temporal_jump 始终在归并后统一执行）。
    cache_path 传入时启用断点续传：文件指纹未变的分片直接复用上次逐帧结论。
    单个文件损坏不再中断：记录进 validation.corrupted_files 并跳过。
    """
    view = root_or_view if isinstance(root_or_view, LeRobotDatasetView) else None
    root = view.root if view else Path(root_or_view).expanduser().resolve()
    info = view.info if view else _read_info(root)
    features = info.get("features") or {}
    config = config or load_config(robot_type=info.get("robot_type"))

    key = feature_key or detect_pointcloud_feature(features)
    if not key:
        raise PointCloudFormatError(
            "未在 info.json 中找到二维点云特征（需要 shape=[N, D] 且 D≥3 的浮点特征）"
        )
    spec = features.get(key) or {}
    shape = spec.get("shape") or []
    expected_points = int(shape[0]) if len(shape) == 2 else None
    dim = int(shape[1]) if len(shape) == 2 else 0

    files = _discover_data_files(root, info)
    if not files:
        raise PointCloudFormatError(f"未在 {root} 下找到 data/*.parquet")
    # 第一个分片可能就损坏：跨文件探测特征列，全部不可读/都没有该列才判定格式错误
    feature_present = False
    for path in files:
        try:
            if key in pq.ParquetFile(path).schema_arrow.names:
                feature_present = True
                break
        except Exception:  # noqa: BLE001 - 坏文件留给扫描阶段记账
            continue
    if not feature_present:
        raise PointCloudFormatError(f"数据 parquet 中不存在点云列: {key}")

    ctx = _build_context(root, files, info, config, key, expected_points, dim)
    cache = _ScanCache(Path(cache_path), config, key, ctx.resolved) if cache_path else None
    if workers > 1:
        results, skipped, corrupted = _scan_parallel(
            files, ctx, max_frames=max_frames, workers=workers, cache=cache)
    else:
        results, skipped, corrupted = _scan(files, ctx, max_frames=max_frames, cache=cache)
    if cache:
        cache.save()
    _apply_temporal_rule(results, ctx)

    validation = PointCloudValidation(
        feature_key=key,
        dim=dim,
        total_frames=len(results),
        results=results,
        config=config,
        dataset_root=str(root),
        resolved=ctx.resolved,
        skipped=skipped,
        expected_points=expected_points,
        corrupted_files=corrupted,
    )
    sev = validation.severity_summary()
    LOG.info("点云质检完成：%d 帧，正常 %d / 异常 %d（error %d, warning %d）",
             validation.total_frames, validation.valid_frames, len(validation.invalid_frames),
             sev.get("error", 0), sev.get("warning", 0))
    return validation


class PointCloudFormatError(RuntimeError):
    """数据集没有可用的点云特征。"""


# ------------------------------------------------------------------ 上下文构建
def _build_context(
    root: Path, files: list[Path], info: dict[str, Any], config: dict[str, Any],
    feature_key: str, expected_points: int | None, dim: int,
) -> _Context:
    cfg_prefix = "pointcloud"
    n = expected_points or 0
    min_points = max(1, int(cfg_get(config, f"{cfg_prefix}.min_points", 16)))
    sparse_ratio = float(cfg_get(config, f"{cfg_prefix}.sparse_ratio", 0.02))
    if n:
        min_points = max(min_points, int(np.ceil(n * sparse_ratio)))

    duplicate_decimals = int(cfg_get(config, f"{cfg_prefix}.duplicate_decimals", 4))
    max_duplicate_ratio = float(cfg_get(config, f"{cfg_prefix}.max_duplicate_ratio", 0.05))
    # 离群半径按局部密度自适应：半径 = k × 每帧 1-NN 距离中位数。
    # 用「场景对角线 × 固定比例」在稀疏点云上会把正常点全判成离群。
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
    sample_diag = _estimate_extent(files, feature_key, sample_limit)
    nn_median = _median_nn_distance(files, feature_key, min(sample_limit, 60))
    if nn_median > 0:
        radius = max(1e-6, nn_median * outlier_radius_k)
    else:
        radius = max(1e-6, sample_diag * 0.04)

    if mode == "fixed" and lower and upper:
        ws_lower = np.asarray(lower[:3], dtype=float)
        ws_upper = np.asarray(upper[:3], dtype=float)
        ws_source = "fixed"
    elif sample_diag > 0:
        lo, hi = _estimate_bounds(files, feature_key, sample_limit,
                                  float(cfg_get(config, f"{cfg_prefix}.workspace.margin_ratio", 0.10)))
        ws_lower, ws_upper = lo, hi
        ws_source = "auto"
    else:
        ws_lower = ws_upper = None
        ws_source = "unavailable"

    color_scale = _estimate_color_scale(files, feature_key, dim, sample_limit)
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
    return _Context(
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


def _sample_points(files: list[Path], feature_key: str, limit: int) -> np.ndarray:
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
                    arr = _to_matrix(cell)
                    if arr.size:
                        chunks.append(arr)
                    taken += 1
                    if taken >= limit:
                        break
                if taken >= limit:
                    break
        except Exception as exc:  # noqa: BLE001 - 采样是 best-effort，坏文件留给扫描阶段记账
            LOG.debug("采样 %s 失败: %s", path, exc)
        if taken >= limit:
            break
    if not chunks:
        return np.zeros((0, 0), dtype=np.float32)
    return np.concatenate(chunks, axis=0)


def _estimate_extent(files: list[Path], feature_key: str, limit: int) -> float:
    pts = _sample_points(files, feature_key, limit)
    xyz = _xyz(pts)
    # 只统计有限值：样本里一旦混入 NaN/Inf，nanmax/nanmin 会返回 inf，后续抖动补点全变 NaN
    xyz = xyz[np.isfinite(xyz).all(axis=1)]
    if len(xyz) < 4:
        return 0.0
    # 用分位数而不是 min/max：少量注入的远点（如 100m）会把尺度拉到几百米，
    # 后续「补点抖动」σ 按尺度缩放时就会把修复点甩出工作空间
    lo = np.percentile(xyz, 0.5, axis=0)
    hi = np.percentile(xyz, 99.5, axis=0)
    extent = float(np.linalg.norm(hi - lo))
    return extent if np.isfinite(extent) else 0.0


def _median_nn_distance(files: list[Path], feature_key: str, limit: int) -> float:
    """采样若干帧，返回「每帧 1-NN 距离」的中位数，作为点云局部密度尺度。

    离群半径如果按「场景对角线 × 固定比例」算，稀疏点云上正常点间距就会超过阈值，
    整帧被误判成离群；按 1-NN 中位数缩放才能跟点云自身采样密度走。
    """
    values: list[float] = []
    taken = 0
    for path in files:
        try:
            pf = pq.ParquetFile(path)
            if feature_key not in pf.schema_arrow.names:
                continue
            for batch in pf.iter_batches(batch_size=max(1, min(limit, 64)), columns=[feature_key]):
                for cell in batch.column(0).to_pylist():
                    xyz = _xyz(_to_matrix(cell))
                    xyz = xyz[np.isfinite(xyz).all(axis=1)]
                    if len(xyz) >= 8:
                        d2 = ((xyz[:, None, :] - xyz[None, :, :]) ** 2).sum(axis=2)
                        np.fill_diagonal(d2, np.inf)
                        values.append(float(np.median(np.sqrt(d2.min(axis=1)))))
                    taken += 1
                    if taken >= limit:
                        break
                if taken >= limit:
                    break
        except Exception as exc:  # noqa: BLE001 - 采样是 best-effort，坏文件留给扫描阶段记账
            LOG.debug("NN 距离采样 %s 失败: %s", path, exc)
        if taken >= limit:
            break
    return float(np.median(values)) if values else 0.0


def _estimate_bounds(
    files: list[Path], feature_key: str, limit: int, margin_ratio: float
) -> tuple[np.ndarray, np.ndarray]:
    pts = _sample_points(files, feature_key, limit)
    xyz = _xyz(pts)
    xyz = xyz[np.isfinite(xyz).all(axis=1)]
    if len(xyz) < 4:
        return np.zeros(3), np.zeros(3)
    lo = np.percentile(xyz, 0.5, axis=0)
    hi = np.percentile(xyz, 99.5, axis=0)
    diag = float(np.linalg.norm(hi - lo))
    if not np.isfinite(diag):
        return np.zeros(3), np.zeros(3)
    margin = max(diag * margin_ratio, 1e-6)
    return lo - margin, hi + margin


def _estimate_color_scale(files: list[Path], feature_key: str, dim: int, limit: int) -> float:
    if dim < 6:
        return 1.0
    pts = _sample_points(files, feature_key, limit)
    if len(pts) == 0:
        return 1.0
    colors = pts[:, 3:6]
    finite = colors[np.isfinite(colors).all(axis=1)]
    if finite.size == 0:
        return 1.0
    return 255.0 if float(np.nanmax(finite)) > 1.5 else 1.0


# ------------------------------------------------------------------ 逐帧校验
def _scan(
    files: list[Path], ctx: _Context, *, max_frames: int | None = None,
    cache: "_ScanCache | None" = None,
) -> tuple[dict[FramesKey, FrameResult], dict[str, str], list[dict[str, Any]]]:
    """串行流式扫描。

    毒数据防冲击：单个 parquet 损坏（截断/schema 异常/缺列读失败）记录进
    corrupted 并跳过，不再中断整个点云阶段；断点续传：cache 命中且文件
    指纹未变时直接复用上次的逐帧结论。
    """
    results: dict[FramesKey, FrameResult] = {}
    skipped = _scan_skipped(ctx)
    corrupted: list[dict[str, Any]] = []

    batch_size = max(1, int(cfg_get(ctx.config, "pointcloud.batch_size", 256)))
    for path in files:
        cached = cache.get(path) if cache else None
        if cached is not None:
            for res in cached:
                results[(res.episode_index, res.frame_index)] = res
            continue
        file_frames: list[FrameResult] = []
        try:
            pf = pq.ParquetFile(path)
            names = pf.schema_arrow.names
            cols = [c for c in ("episode_index", "frame_index", ctx.feature_key) if c in names]
            if ctx.feature_key not in cols:
                continue
            for batch in pf.iter_batches(batch_size=batch_size, columns=cols):
                data = batch.to_pydict()
                eps = data["episode_index"]
                fis = data["frame_index"]
                cells = data[ctx.feature_key]
                for ep, fi, cell in zip(eps, fis, cells):
                    key = (int(ep), int(fi))
                    res = evaluate_frame(_to_matrix(cell), key, ctx)
                    results[key] = res
                    file_frames.append(res)
                    if max_frames and len(results) >= max_frames:
                        LOG.info("--pointcloud_max_frames 生效：只校验前 %d 帧", max_frames)
                        # 截断发生在文件中途：该文件结论不完整，不入缓存
                        return results, skipped, corrupted
        except Exception as exc:  # noqa: BLE001 - 坏文件不中断流水线
            LOG.error("点云数据文件损坏，已跳过: %s -> %s", path.name, exc)
            corrupted.append({"file": str(path), "error": str(exc)})
            continue
        if cache:
            cache.put(path, file_frames)
    return results, skipped, corrupted


def _scan_skipped(ctx: _Context) -> dict[str, str]:
    """扫描前即可确定的规则跳过原因（只看全局上下文，与具体数据无关）。"""
    skipped: dict[str, str] = {}
    if ctx.dim < 6:
        skipped["pc_color_invalid"] = "点云维度 < 6，没有颜色通道"
    if ctx.workspace_lower is None:
        skipped["pc_out_of_bounds"] = "无法推导工作空间包围盒（样本点不足）"
    if ctx.outlier_radius <= 1e-6:
        skipped["pc_outliers"] = "场景尺度≈0，无法推导离群半径"
    return skipped


def _jsonable(obj: Any) -> Any:
    """把 numpy 标量/数组转成 JSON 原生类型（扫描缓存落盘用）。"""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


class _ScanCache:
    """点云逐帧扫描的断点续传缓存（写在 output_dir，不碰源数据集）。

    命中条件（全部满足才复用上次结论）：
      1. 配置指纹一致（阈值/严重度变了结论必须重算）；
      2. 同一特征列、同一解析出的自适应规则上下文（随数据分布变化）；
      3. 文件 stat 未变（路径 + 大小 + mtime_ns，廉价指纹，不读文件内容）。
    损坏修复、新增、改动的文件因指纹变化自动重扫；缓存损坏/写不进时
    静默降级为不缓存，绝不影响主流程。
    """

    VERSION = 1

    def __init__(self, path: Path, config: dict[str, Any], feature_key: str,
                 resolved: dict[str, Any]) -> None:
        self.path = path
        self._envelope = {
            "version": self.VERSION,
            "config_digest": config_digest(config),
            "feature_key": feature_key,
            "resolved": _jsonable(resolved),
        }
        self._files: dict[str, dict[str, Any]] = {}
        self._dirty = False
        if not path.is_file():
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            LOG.warning("点云扫描缓存无法解析，按首次运行处理: %s", exc)
            return
        meta = {k: payload.get(k) for k in self._envelope}
        if meta == self._envelope:
            self._files = payload.get("files") or {}
            LOG.info("断点续传：载入点云扫描缓存（%d 个文件分片）", len(self._files))
        else:
            LOG.info("配置/特征列/自适应阈值已变化，点云扫描缓存失效，将全量重扫")

    @staticmethod
    def _key(path: Path) -> str:
        try:
            st = path.stat()
            sig = {"path": str(path), "size": st.st_size, "mtime_ns": st.st_mtime_ns}
        except OSError:
            sig = {"path": str(path), "exists": False}
        return hashlib.sha1(json.dumps(sig, sort_keys=True).encode()).hexdigest()[:16]

    def get(self, path: Path) -> list[FrameResult] | None:
        entry = self._files.get(self._key(path))
        if not entry:
            return None
        return [FrameResult.from_dict(d) for d in entry.get("frames") or []]

    def put(self, path: Path, frames: list[FrameResult]) -> None:
        self._files[self._key(path)] = {
            "file": str(path),
            "frames": [_jsonable(r.to_dict()) for r in frames],
        }
        self._dirty = True

    def save(self) -> None:
        if not self._dirty and self.path.is_file():
            return
        payload = dict(self._envelope)
        payload["files"] = self._files
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(payload, ensure_ascii=False, default=str),
                                 encoding="utf-8")
            LOG.info("点云扫描缓存已保存: %s（%d 个文件分片）", self.path, len(self._files))
        except OSError as exc:
            LOG.warning("点云扫描缓存写入失败（不影响本次结果）: %s", exc)


# ------------------------------------------------------------------ 多进程扫描
# 子进程共享的只读规则上下文（由 _init_pc_worker 在子进程启动时注入一次）
_PC_CTX: "_Context | None" = None


def _init_pc_worker(ctx: _Context) -> None:
    """子进程 initializer：注入点云规则上下文（阈值 / 包围盒 / 颜色量程）。"""
    global _PC_CTX
    _PC_CTX = ctx


def _scan_row_group_worker(job: tuple[str, int]) -> dict[str, Any]:
    """worker：读 parquet 的一个 row group，逐帧跑 8 条规则。

    返回 {"file", "results", "corrupted"}：读取失败不再上抛拖垮进程池，
    而是作为 corrupted 条目交回主进程记录（毒数据防冲击）。
    纯函数契约：只读输入文件、只经返回值交回结论，不写盘、不写共享状态；
    跨帧规则（pc_temporal_jump）由主进程在结果归并后统一执行。
    """
    path, rg_index = job
    ctx = _PC_CTX
    if ctx is None:  # 防御：initializer 未跑时给出可读错误而不是 AttributeError
        raise RuntimeError("点云 worker 上下文未初始化（initializer 未执行）")
    try:
        pf = pq.ParquetFile(path)
        names = pf.schema_arrow.names
        cols = [c for c in ("episode_index", "frame_index", ctx.feature_key) if c in names]
        if ctx.feature_key not in cols:
            return {"file": path, "results": [], "corrupted": None}  # 该文件没有点云列
        table = pf.read_row_group(rg_index, columns=cols)
    except Exception as exc:  # noqa: BLE001
        return {"file": path, "results": [],
                "corrupted": {"file": path, "row_group": rg_index, "error": str(exc)}}
    data = table.to_pydict()
    out = [
        evaluate_frame(_to_matrix(cell), (int(ep), int(fi)), ctx)
        for ep, fi, cell in zip(data["episode_index"], data["frame_index"], data[ctx.feature_key])
    ]
    return {"file": path, "results": out, "corrupted": None}


def _scan_parallel(
    files: list[Path], ctx: _Context, *, max_frames: int | None, workers: int,
    cache: "_ScanCache | None" = None,
) -> tuple[dict[FramesKey, FrameResult], dict[str, str], list[dict[str, Any]]]:
    """按 (parquet 文件, row group) 分片并发扫描，主进程按 (episode, frame) 归并。

    与串行 _scan 的唯一语义差异在 --pointcloud_max_frames：串行是按文件序
    提前截断，并发是扫描全量后按 (episode_index, frame_index) 排序截取，
    两者各自确定，但截断窗口可能不同（max_frames 只是调试开关，不影响全量结果）。
    """
    from .parallel import run_pool

    skipped = _scan_skipped(ctx)
    corrupted: list[dict[str, Any]] = []
    results: dict[FramesKey, FrameResult] = {}
    per_file: dict[str, list[FrameResult]] = {}

    jobs: list[tuple[str, int]] = []
    for path in files:
        cached = cache.get(path) if cache else None
        if cached is not None:
            for res in cached:
                results[(res.episode_index, res.frame_index)] = res
            continue
        try:
            pf = pq.ParquetFile(path)
        except Exception as exc:  # noqa: BLE001 - 连文件头都打不开，主进程直接记账
            LOG.error("点云数据文件损坏，已跳过: %s -> %s", path.name, exc)
            corrupted.append({"file": str(path), "error": str(exc)})
            continue
        if ctx.feature_key not in pf.schema_arrow.names:
            continue
        per_file[str(path)] = []
        for rg in range(pf.metadata.num_row_groups):
            jobs.append((str(path), rg))

    for chunk in run_pool(
        jobs, _scan_row_group_worker, ctx, workers,
        desc="点云帧质检", unit="RG", initializer=_init_pc_worker,
    ):
        if chunk["corrupted"]:
            corrupted.append(chunk["corrupted"])
        for res in chunk["results"]:
            results[(res.episode_index, res.frame_index)] = res
            per_file.setdefault(chunk["file"], []).append(res)

    # future 完成顺序不确定：按 (episode, frame) 排序重建 dict，
    # 保证下游 invalid_frame_details / 报告的顺序与串行逐位一致
    results = {k: results[k] for k in sorted(results)}

    if cache:
        # 只缓存「所有 row group 都成功」的文件；有分片损坏的文件下次仍会重扫
        broken = {c["file"] for c in corrupted}
        for file_key, frames in per_file.items():
            if frames and file_key not in broken:
                cache.put(Path(file_key), frames)

    if max_frames and len(results) > max_frames:
        LOG.info("--pointcloud_max_frames 生效：并发模式扫描全量后按 (episode, frame) 截取前 %d 帧",
                 max_frames)
        keys = sorted(results)[:max_frames]
        results = {k: results[k] for k in keys}
    return results, skipped, corrupted


def evaluate_frame(points: np.ndarray, key: FramesKey, ctx: _Context) -> FrameResult:
    """单帧规则判定。points: (N, D)，前三列是 xyz。"""
    pts = _to_matrix(points)
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

    xyz = _xyz(finite)
    # 用中位数而不是均值：离群点会把均值拉走，导致「离群」被误判成「时序跳变」
    centroid = np.median(xyz, axis=0) if len(xyz) else np.zeros(3, dtype=float)
    diag = float(np.linalg.norm(np.max(xyz, axis=0) - np.min(xyz, axis=0))) if len(xyz) > 1 else 0.0

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

    if len(xyz) >= 2 and diag <= float(cfg_get(ctx.config, "pointcloud.degenerate_diag", 1e-3)):
        res.add("pc_degenerate", bbox_diag=round(diag, 8),
                note="包围盒尺度≈0，点云坍缩为一个点或一条线")
        return res

    dup_ratio = _duplicate_ratio(xyz, ctx.duplicate_decimals)
    metrics["duplicate_ratio"] = round(dup_ratio, 5)
    if dup_ratio > ctx.max_duplicate_ratio:
        res.add("pc_duplicate", duplicate_ratio=round(dup_ratio, 5),
                max_allowed=ctx.max_duplicate_ratio,
                duplicate_points=int(round(dup_ratio * len(xyz))))

    if ctx.outlier_radius > 1e-6:
        outlier_ratio = _outlier_ratio(xyz, ctx.outlier_radius, ctx.outlier_min_neighbors)
        metrics["outlier_ratio"] = round(outlier_ratio, 5)
        if outlier_ratio > ctx.max_outlier_ratio:
            res.add("pc_outliers", outlier_ratio=round(outlier_ratio, 5),
                    radius=round(ctx.outlier_radius, 5),
                    min_neighbors=ctx.outlier_min_neighbors,
                    max_allowed=ctx.max_outlier_ratio,
                    outlier_points=int(round(outlier_ratio * len(xyz))))

    if ctx.workspace_lower is not None and ctx.workspace_upper is not None:
        outside = _outside_ratio(xyz, ctx.workspace_lower, ctx.workspace_upper)
        metrics["outside_ratio"] = round(outside, 5)
        if outside > ctx.max_outside_ratio:
            res.add("pc_out_of_bounds", outside_ratio=round(outside, 5),
                    max_allowed=ctx.max_outside_ratio,
                    workspace_lower=[round(float(x), 4) for x in ctx.workspace_lower],
                    workspace_upper=[round(float(x), 4) for x in ctx.workspace_upper])

    if ctx.dim >= 6:
        color_bad_ratio = _color_bad_ratio(finite[:, 3:6], ctx.color_scale)
        metrics["color_bad_ratio"] = round(color_bad_ratio, 5)
        if color_bad_ratio > ctx.color_max_bad_ratio:
            res.add("pc_color_invalid", bad_ratio=round(color_bad_ratio, 5),
                    color_scale=ctx.color_scale,
                    max_allowed=ctx.color_max_bad_ratio,
                    note=f"颜色应在 [0, {ctx.color_scale:g}] 范围内")

    return res


def _apply_temporal_rule(results: dict[FramesKey, FrameResult], ctx: _Context) -> None:
    """按 episode 计算质心相邻帧跳变；阈值 = max(abs_jump, median × mad_k)。"""
    if not ctx.temporal_abs_jump and not ctx.temporal_mad_k:
        return
    by_ep: dict[int, list[FrameResult]] = {}
    for res in results.values():
        by_ep.setdefault(res.episode_index, []).append(res)

    # 稀疏/退化/含 NaN 的帧本身没有可信质心，参与比较会把「上一帧正常」也判成跳变；
    # 离群/越界帧的质心同样被坏点带偏，若不排除会连锁误报后续正常帧
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


# ------------------------------------------------------------------ 处置策略
@dataclass
class PointCloudDecision:
    key: FramesKey
    disposition: str
    severity: str
    reasons: list[str]
    repair_actions: list[str] = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_index": self.key[0],
            "frame_index": self.key[1],
            "disposition": self.disposition,
            "severity": self.severity,
            "reasons": list(self.reasons),
            "repair_actions": list(self.repair_actions),
            "note": self.note,
        }


def decide(
    validation: PointCloudValidation, *, on_review: str = "keep", overrides: dict[FramesKey, str] | None = None
) -> dict[FramesKey, PointCloudDecision]:
    """逐帧处置：keep / repair / review / drop。

    - 可修复项（NaN/离群/重复/越界/颜色）且坏点比例可控 → repair
    - 有效点过少或几何退化 → drop（不可凭空生成几何）
    - 时间跳变或无法自动修 → review（不改数据，只进复核队列）
    """
    decisions: dict[FramesKey, PointCloudDecision] = {}
    for key, res in validation.results.items():
        decisions[key] = _decide_one(key, res, validation, on_review)

    for key, override in (overrides or {}).items():
        if key not in decisions:
            continue
        disp = str(override).strip().lower()
        if disp not in {"keep", "repair", "review", "drop"}:
            continue
        d = decisions[key]
        d.disposition = disp
        d.note = "人工复核覆盖"
    counts = {d: sum(1 for x in decisions.values() if x.disposition == d)
              for d in ("keep", "repair", "review", "drop")}
    LOG.info("点云处置：%s", counts)
    return decisions


def _decide_one(key: FramesKey, res: FrameResult, validation: PointCloudValidation, on_review: str) -> PointCloudDecision:
    severity = validation.severity_of(res)
    if not res.reasons:
        return PointCloudDecision(key, "keep", severity, [])
    if "pc_empty_or_sparse" in res.reasons or "pc_degenerate" in res.reasons:
        return PointCloudDecision(key, "drop", severity, list(res.reasons),
                                  note="有效点不足/几何退化，无法修复")
    actions = [RULE_MAP[c].repair for c in res.reasons if RULE_MAP[c].repair]
    actions = [a for a in actions if a]
    non_repairable = [c for c in res.reasons if c not in REPAIRABLE]
    if non_repairable:
        disposition = "drop" if (severity == "error" and on_review == "drop") else "review"
        return PointCloudDecision(key, disposition, severity, list(res.reasons),
                                  note="需人工确认，暂不改点云")
    # 坏点比例过高时不做「大量补点」，避免用抖动点伪造几何
    ratio = max(float(res.metrics.get("non_finite_ratio", 0.0)),
                float(res.metrics.get("outlier_ratio", 0.0))
                + float(res.metrics.get("duplicate_ratio", 0.0)))
    max_repair = float(cfg_get(validation.config, "pointcloud.max_repair_ratio", 0.35))
    if ratio > max_repair:
        return PointCloudDecision(key, "drop", severity, list(res.reasons),
                                  note=f"坏点比例 {ratio:.0%} 超过 {max_repair:.0%}，无法可靠修复")
    return PointCloudDecision(key, "repair", severity, list(res.reasons),
                              repair_actions=_dedupe_actions(actions))


def _dedupe_actions(actions: Iterable[str]) -> list[str]:
    out: list[str] = []
    for action in actions:
        if action not in out:
            out.append(action)
    return out


# ------------------------------------------------------------------ 清洗实现
@dataclass
class PointCloudCleaningResult:
    success: bool
    output_root: Path | None
    method: str
    repaired_frames: list[FramesKey] = field(default_factory=list)
    held_frames: list[FramesKey] = field(default_factory=list)
    kept_frames: int = 0
    removed_points: int = 0
    added_points: int = 0
    warnings: list[str] = field(default_factory=list)
    verification: dict[str, Any] = field(default_factory=dict)
    stats_updated: bool = False
    samples: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "performed": bool(self.repaired_frames or self.held_frames),
            "success": self.success,
            "output_path": str(self.output_root) if self.output_root else None,
            "method": self.method,
            "repaired_frames": [f"{e}:{f}" for e, f in self.repaired_frames],
            "held_frames": [f"{e}:{f}" for e, f in self.held_frames],
            "repaired_count": len(self.repaired_frames),
            "held_count": len(self.held_frames),
            "kept_frames": self.kept_frames,
            "removed_points": self.removed_points,
            "added_points": self.added_points,
            "warnings": self.warnings,
            "verification": self.verification,
            "stats_updated": self.stats_updated,
            "samples": self.samples[:5],
        }


def clean_pointclouds(
    root: str | Path,
    validation: PointCloudValidation,
    decisions: dict[FramesKey, PointCloudDecision],
    output_dir: str | Path,
    *,
    feature_key: str | None = None,
    target_root: str | Path | None = None,
    overwrite: bool = True,
) -> PointCloudCleaningResult:
    """把修复结果写回数据集。

    - target_root 指向主清洗器产出的 cleaned_dataset 时，直接原地修补它的 parquet；
    - 未指定时输出到 <output_dir>/pointcloud_cleaned（复制 meta，不含视频）。
    """
    src_root = Path(root).expanduser().resolve()
    out_dir = Path(output_dir).expanduser().resolve()
    key = feature_key or validation.feature_key
    ctx = _rebuild_context(validation)
    repaired = [k for k, d in decisions.items() if d.disposition == "repair"]
    held = [k for k, d in decisions.items() if d.disposition == "drop"]

    if target_root:
        dst_root = Path(target_root).expanduser().resolve()
        same_tree = dst_root == src_root
        method = "inplace-patch" if same_tree else "patch-cleaned-dataset"
    else:
        dst_root = out_dir / CLEANED_DIRNAME
        _prepare_dir(dst_root, overwrite=overwrite)
        _copy_meta(src_root, dst_root)
        same_tree = False
        method = "standalone"

    if not repaired and not held:
        LOG.info("点云无需清洗：%d 帧全部保留", validation.total_frames)
        result = PointCloudCleaningResult(
            success=True, output_root=dst_root, method="noop" if target_root else method,
            kept_frames=validation.total_frames,
        )
        if not target_root:
            # standalone 输出需要一份可独立打开的数据副本（data 分片原样复制）
            _copy_data_parquets(src_root, dst_root)
        _attach_samples(result, validation, decisions, src_root, key, ctx, limit=3)
        _write_notes(result, validation, dst_root)
        _verify(dst_root, validation, key, result)
        return result

    files = _discover_data_files(src_root, _read_info(src_root))
    result = PointCloudCleaningResult(
        success=True, output_root=dst_root, method=method,
        repaired_frames=sorted(repaired), held_frames=sorted(held),
        kept_frames=validation.total_frames - len(repaired) - len(held),
    )
    batch_size = max(1, int(cfg_get(validation.config, "pointcloud.batch_size", 256)))
    added_points = 0
    removed_points = 0

    for src_path in files:
        try:
            pf = pq.ParquetFile(src_path)
            names = pf.schema_arrow.names
            if key not in names or "episode_index" not in names or "frame_index" not in names:
                result.warnings.append(f"{src_path.name} 缺少点云/索引列，跳过")
                continue
            dst_path = dst_root / src_path.relative_to(src_root)
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            if same_tree:
                # 同目录临时文件 + os.replace，避免边读边写覆盖源分片；不用 mkstemp 以免泄漏 fd
                tmp_path = dst_path.with_name(dst_path.name + ".tmp")
                tmp_path.unlink(missing_ok=True)
            else:
                tmp_path = dst_path

            col_index = pf.schema_arrow.get_field_index(key)
            field = pf.schema_arrow.field(col_index)
            writer = pq.ParquetWriter(tmp_path, pf.schema_arrow)
            last_good: dict[int, np.ndarray] = {}
            try:
                for batch in pf.iter_batches(batch_size=batch_size):
                    data = batch.to_pydict()
                    eps = data["episode_index"]
                    fis = data["frame_index"]
                    cells = data[key]
                    new_cells: list[Any] = []
                    for ep, fi, cell in zip(eps, fis, cells):
                        k = (int(ep), int(fi))
                        pts = _to_matrix(cell)
                        if k in decisions and decisions[k].disposition == "repair":
                            new_pts, removed, added = repair_frame(pts, decisions[k], ctx)
                            added_points += added
                            removed_points += removed
                            last_good[int(ep)] = new_pts
                            new_cells.append(_to_nested(new_pts))
                        elif k in decisions and decisions[k].disposition == "drop":
                            good = last_good.get(int(ep))
                            if good is None:
                                new_cells.append(_to_nested(pts))
                                result.warnings.append(f"ep {ep} 帧 {fi} 之前没有可参考的好帧，保留原值")
                            else:
                                new_cells.append(_to_nested(good))
                                result.warnings.append(f"ep {ep} 帧 {fi} 不可修复，使用同 episode 最近好帧占位")
                        else:
                            last_good[int(ep)] = pts if np.isfinite(pts).all() else last_good.get(int(ep), pts)
                            new_cells.append(_to_nested(pts))
                    icol = batch.schema.get_field_index(key)
                    new_col = pa.array(new_cells, type=field.type)
                    batch = batch.set_column(icol, field, new_col)
                    writer.write_batch(batch)
            finally:
                writer.close()
                # Windows 上源 parquet 仍被 ParquetFile 持有句柄时，os.replace 会报 WinError 5
                pf.close()

            if same_tree:
                os.replace(tmp_path, dst_path)
            LOG.info("点云清洗写入 %s", dst_path)
        except Exception as exc:  # noqa: BLE001 - 毒数据防冲击：坏文件原样保留，不中断清洗
            LOG.error("点云清洗：文件损坏，原样保留并继续: %s -> %s", src_path.name, exc)
            result.warnings.append(f"{src_path.name} 损坏，已原样保留: {exc}")
            if not same_tree:
                # 独立输出目录需要一份完整副本：损坏分片直接二进制复制过去
                try:
                    raw_dst = dst_root / src_path.relative_to(src_root)
                    raw_dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src_path, raw_dst)
                except OSError as copy_exc:
                    result.warnings.append(f"{src_path.name} 原样复制也失败: {copy_exc}")
            continue

    result.removed_points = removed_points
    result.added_points = added_points
    if dst_root != src_root:
        _copy_meta(src_root, dst_root)
    result.stats_updated = _rewrite_pointcloud_stats(dst_root, key, validation.config)
    _attach_samples(result, validation, decisions, src_root, key, ctx, limit=5, repaired=repaired)
    _write_notes(result, validation, dst_root)
    _verify(dst_root, validation, key, result)
    LOG.info("点云清洗完成：修复 %d 帧 / 占位 %d 帧 / 保留 %d 帧 -> %s",
             len(repaired), len(held), result.kept_frames, dst_root)
    return result


def repair_frame(points: np.ndarray, decision: PointCloudDecision, ctx: _Context) -> tuple[np.ndarray, int, int]:
    """按修复动作过滤 -> 去重 -> 裁剪颜色 -> 抖动重采样补回固定点数。"""
    pts = _to_matrix(points).astype(np.float32, copy=True)
    original_n = int(pts.shape[0])
    removed = 0
    actions = set(decision.repair_actions) | {RULE_MAP[r].repair for r in decision.reasons if RULE_MAP[r].repair}

    if "filter" in actions or "pc_non_finite" in decision.reasons:
        pts = pts[np.isfinite(pts).all(axis=1)]
    if "bounds" in actions and ctx.workspace_lower is not None:
        inside = np.all((_xyz(pts) >= ctx.workspace_lower) & (_xyz(pts) <= ctx.workspace_upper), axis=1)
        pts = pts[inside]
    if "filter" in actions and len(pts):
        outlier = _outlier_mask(_xyz(pts), ctx.outlier_radius, ctx.outlier_min_neighbors)
        pts = pts[~outlier]
    if "dedupe" in actions and len(pts):
        coords = np.round(_xyz(pts), ctx.duplicate_decimals)
        _, idx = np.unique(coords, axis=0, return_index=True)
        pts = pts[np.sort(idx)]
    if "color" in actions and ctx.dim >= 6 and len(pts):
        pts[:, 3:6] = np.clip(pts[:, 3:6], 0.0, ctx.color_scale)

    target = ctx.expected_points or original_n
    kept = int(len(pts))
    added = 0
    if target and kept != target:
        if kept == 0:
            return np.zeros((0, pts.shape[1] or ctx.dim), dtype=np.float32), original_n, 0
        if kept < target:
            rng = np.random.default_rng(abs(decision.key[0]) * 100000 + decision.key[1])
            idx = np.arange(target) % kept
            order = rng.permutation(target)
            # 抖动尺度优先用局部点距（1-NN 中位数），而不是整场景尺度：
            # 场景里有少量远点时，按场景尺度缩放会把补出来的点撒到几米外
            spacing = float(ctx.resolved.get("outlier_nn_median") or 0.0)
            scale = spacing if spacing > 0 else float(ctx.resolved.get("scene_extent") or 1.0)
            jitter = rng.normal(0.0, ctx.jitter_ratio * max(scale, 1e-6), size=(target, 3))
            picked = pts[idx[order]].copy()
            picked[:, :3] += jitter
            added = target - kept
            pts = picked
        else:
            pts = pts[:target]
    removed = max(0, original_n - kept)
    return pts, removed, added


def _neighbor_counts(xyz: np.ndarray, radius: float) -> np.ndarray:
    """每个点在半径内的邻居数（不含自己），按行分块以控制内存。"""
    n = len(xyz)
    counts = np.empty(n, dtype=np.int32)
    if n < 2:
        return np.zeros(n, dtype=np.int32)
    r2 = radius * radius
    block = max(1, int(4_000_000 // max(n, 1)))
    for start in range(0, n, block):
        sub = xyz[start:start + block]
        d2 = ((sub[:, None, :] - xyz[None, :, :]) ** 2).sum(axis=2)
        counts[start:start + block] = (d2 <= r2).sum(axis=1) - 1
    return counts


def _outlier_mask(xyz: np.ndarray, radius: float, min_neighbors: int) -> np.ndarray:
    if len(xyz) < 2 or radius <= 0:
        return np.zeros(len(xyz), dtype=bool)
    return _neighbor_counts(xyz, radius) < max(1, min_neighbors)


def _duplicate_ratio(xyz: np.ndarray, decimals: int) -> float:
    if len(xyz) < 2:
        return 0.0
    q = np.round(xyz, decimals)
    return float(1.0 - len(np.unique(q, axis=0)) / len(q))


def _outlier_ratio(xyz: np.ndarray, radius: float, min_neighbors: int) -> float:
    """整帧统计：抽稀会同时降低局部密度，导致正常点被误判，所以必须用全量点。"""
    if len(xyz) < 2 or radius <= 0:
        return 0.0
    return float((_neighbor_counts(xyz, radius) < max(1, min_neighbors)).mean())


def _outside_ratio(xyz: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    if len(xyz) == 0:
        return 0.0
    outside = np.any((xyz < lower) | (xyz > upper), axis=1)
    return float(outside.mean())


def _color_bad_ratio(colors: np.ndarray, scale: float) -> float:
    if colors.size == 0:
        return 0.0
    bad = ~np.isfinite(colors) | (colors < 0) | (colors > scale)
    return float(bad.any(axis=1).mean())


# ------------------------------------------------------------------ 报告
def build_report(
    validation: PointCloudValidation,
    decisions: dict[FramesKey, PointCloudDecision],
    cleaning: PointCloudCleaningResult | None = None,
    *,
    dataset_path: str | Path | None = None,
) -> dict[str, Any]:
    total = max(validation.total_frames, 1)
    counts = {d: sum(1 for x in decisions.values() if x.disposition == d)
              for d in ("keep", "repair", "review", "drop")}
    freq = validation.reason_frequency()
    sev = validation.severity_summary()
    invalid = len(validation.invalid_frames)
    repair_actions: dict[str, int] = {}
    for d in decisions.values():
        if d.disposition == "repair":
            for action in d.repair_actions:
                repair_actions[action] = repair_actions.get(action, 0) + 1

    report: dict[str, Any] = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "dataset": {
            "path": str(dataset_path or validation.dataset_root),
            "feature_key": validation.feature_key,
            "dim": validation.dim,
            "expected_points": validation.expected_points,
            "total_frames": validation.total_frames,
        },
        "config": config_snapshot(validation.config),
        "resolved": validation.resolved,
        "corrupted_files": validation.corrupted_files,
        "summary": {
            "total_frames": validation.total_frames,
            "valid_frames": validation.valid_frames,
            "invalid_frames": invalid,
            "valid_ratio": round(validation.valid_frames / total, 4),
            "severity": sev,
            "disposition": counts,
            "repair_actions": repair_actions,
            "health_level": _health_level(invalid / total),
            "remaining_frames": validation.total_frames - counts["drop"],
        },
        "checks": [
            {
                "code": code,
                "label": RULE_MAP[code].label,
                "description": RULE_MAP[code].description,
                "severity": severity_of(validation.config, code),
                "repair": RULE_MAP[code].repair,
                "affected_frames": freq.get(code, 0),
                "affected_ratio": round(freq.get(code, 0) / total, 4),
                "status": "skipped" if code in validation.skipped else "enabled",
            }
            for code in RULE_ORDER
        ],
        "reason_frequency": freq,
        "episodes": validation.episode_summary(),
        "invalid_frame_details": [r.to_dict() for r in validation.invalid_frames],
        "dataset_issues": validation.dataset_issues,
        "skipped_checks": validation.skipped,
        "cleaning": cleaning.to_dict() if cleaning else None,
        "artifacts": {},
        "recommendations": _recommendations(validation, decisions, cleaning),
    }
    return report


def write_report(report: dict[str, Any], output_dir: str | Path) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / REPORT_FILENAME
    with path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2, default=_json_default)
    LOG.info("pointcloud_report.json 已写入 %s", path)
    return path


def write_review_queue(
    path: str | Path, validation: PointCloudValidation, decisions: dict[FramesKey, PointCloudDecision]
) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for key in sorted(decisions):
        d = decisions[key]
        if d.disposition != "review":
            continue
        res = validation.results[key]
        rows.append({
            "episode_index": key[0],
            "frame_index": key[1],
            "severity": d.severity,
            "reasons": "|".join(d.reasons),
            "suggested_disposition": d.disposition,
            "suggested_action": d.note or "人工回放确认",
            "metrics": json.dumps(res.metrics, ensure_ascii=False, default=str),
            "reviewer_disposition": "",
            "reviewer_note": "",
        })
    headers = ["episode_index", "frame_index", "severity", "reasons", "suggested_disposition",
               "suggested_action", "metrics", "reviewer_disposition", "reviewer_note"]
    with out.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    LOG.info("点云复核队列 %d 行 -> %s", len(rows), out)
    return out


def read_overrides(path: str | Path) -> dict[FramesKey, str]:
    """读取点云复核结果：(episode_index, frame_index) -> disposition。"""
    p = Path(path).expanduser()
    if not p.is_file():
        raise FileNotFoundError(f"复核文件不存在: {p}")
    out: dict[FramesKey, str] = {}
    with p.open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            disp = (row.get("reviewer_disposition") or "").strip().lower()
            if disp not in {"keep", "repair", "review", "drop"}:
                continue
            try:
                key = (int(row["episode_index"]), int(row["frame_index"]))
            except (KeyError, TypeError, ValueError):
                continue
            out[key] = disp
    LOG.info("读取点云人工复核 %d 条 <- %s", len(out), p)
    return out


def print_summary(report: dict[str, Any], *, limit: int = 10) -> None:
    ds = report["dataset"]
    sm = report["summary"]
    line = "=" * 76
    print(line)
    print("3D 点云质检报告")
    print(line)
    print(f"数据集      : {ds['path']}")
    print(f"点云特征    : {ds['feature_key']}  ({ds.get('expected_points') or '变长'} 点 × {ds['dim']} 维)")
    print(f"帧数        : {sm['total_frames']}    正常 {sm['valid_frames']} ({sm['valid_ratio']:.1%})    "
          f"异常 {sm['invalid_frames']}    健康度: {_health_cn(sm['health_level'])}")
    print(f"严重度      : error {sm['severity'].get('error', 0)} / warning {sm['severity'].get('warning', 0)} / "
          f"info {sm['severity'].get('info', 0)}")
    print(f"处置        : 保留 {sm['disposition']['keep']} / 修复 {sm['disposition']['repair']} / "
          f"复核 {sm['disposition']['review']} / 剔除 {sm['disposition']['drop']}")
    ws = report.get("resolved") or {}
    if ws.get("workspace_lower"):
        lo, hi = ws["workspace_lower"], ws["workspace_upper"]
        print(f"工作空间    : [{lo[0]:.2f},{lo[1]:.2f},{lo[2]:.2f}] ~ [{hi[0]:.2f},{hi[1]:.2f},{hi[2]:.2f}]"
              f"（{ws.get('workspace_mode')}）")
    print(f"离群半径    : {ws.get('outlier_radius')} m（1-NN 中位数 {ws.get('outlier_nn_median')} m × "
          f"{ws.get('outlier_radius_k')}）    场景尺度: {ws.get('scene_extent')} m")
    print("-" * 76)
    print(_ljust("检查项", 30) + _rjust("级别", 8) + _rjust("帧数", 8) + _rjust("占比", 9))
    for item in report["checks"]:
        if item["status"] == "enabled" and item["affected_frames"] == 0:
            continue
        print(_ljust(item["label"][:28], 30) + _rjust(item["severity"], 8)
              + _rjust("跳过" if item["status"] == "skipped" else str(item["affected_frames"]), 8)
              + _rjust("-" if item["status"] == "skipped" else f"{item['affected_ratio']:.1%}", 9))
    details = report.get("invalid_frame_details") or []
    if details:
        print("-" * 76)
        print(f"异常帧明细（最多 {limit} 条，全量见 pointcloud_report.json）：")
        for item in details[:limit]:
            print(f"  ep {item['episode_index']:>3}  frame {item['frame_index']:>4}  "
                  f"points={item['point_count']:>4}  {','.join(item['reasons'])}")
    recs = report.get("recommendations") or []
    if recs:
        print("-" * 76)
        print("建议：")
        for i, rec in enumerate(recs, 1):
            print(f"  {i}. {rec}")
    print(line)


def _recommendations(
    validation: PointCloudValidation,
    decisions: dict[FramesKey, PointCloudDecision],
    cleaning: PointCloudCleaningResult | None,
) -> list[str]:
    freq = validation.reason_frequency()
    recs: list[str] = []
    if not validation.invalid_frames:
        recs.append("点云质量良好：未发现 NaN/离群/重复/越界问题，可直接用于训练。")
        return recs
    if freq.get("pc_non_finite"):
        recs.append(f"{freq['pc_non_finite']} 帧含 NaN/Inf：优先排查深度反投影与采样器序列化。")
    if freq.get("pc_outliers"):
        recs.append(f"{freq['pc_outliers']} 帧离群点超阈值：多为反光/透明物体穿透，已做半径滤波修复。")
    if freq.get("pc_duplicate"):
        recs.append(f"{freq['pc_duplicate']} 帧重复点超阈值：检查上游是否对点云做了重复采样。")
    if freq.get("pc_out_of_bounds"):
        recs.append(f"{freq['pc_out_of_bounds']} 帧越出工作空间：优先核对相机外参与机器人基座标系。")
    if freq.get("pc_temporal_jump"):
        recs.append(f"{freq['pc_temporal_jump']} 帧质心突变：已进入复核队列，回放相邻帧确认是真实运动还是跟踪跳变。")
    if freq.get("pc_empty_or_sparse"):
        recs.append(f"{freq['pc_empty_or_sparse']} 帧有效点过少：属不可修复采集事故，已用同 episode 最近好帧占位。")
    if cleaning and cleaning.warnings:
        recs.append(f"清洗产生 {len(cleaning.warnings)} 条告警，详见 pointcloud_cleaning_notes.json。")
    recs.append("点云修复保持固定点数契约：过滤坏点后用抖动重采样补点，不改变 parquet 行数。")
    return recs


# ------------------------------------------------------------------ 内部工具
def _to_matrix(cell: Any) -> np.ndarray:
    """把 parquet 单元格（list[list[float]] / ndarray 对象数组）统一成 (N, D) float32。"""
    if cell is None:
        return np.zeros((0, 0), dtype=np.float32)
    if isinstance(cell, np.ndarray) and cell.dtype != object:
        arr = np.asarray(cell, dtype=np.float32)
    elif isinstance(cell, np.ndarray) and cell.dtype == object:
        try:
            arr = np.asarray(cell.tolist(), dtype=np.float32)
        except (TypeError, ValueError):
            rows = [np.asarray(r, dtype=np.float32).reshape(-1) for r in cell]
            arr = np.stack(rows) if rows else np.zeros((0, 0), dtype=np.float32)
    else:
        try:
            arr = np.asarray(cell, dtype=np.float32)
        except (TypeError, ValueError):
            try:
                arr = np.asarray([np.asarray(r, dtype=np.float32).reshape(-1) for r in cell],
                                 dtype=np.float32)
            except (TypeError, ValueError):
                return np.zeros((0, 0), dtype=np.float32)
    if arr.ndim == 0:
        return np.zeros((0, 0), dtype=np.float32)
    if arr.ndim == 1:
        if arr.size % 3 == 0 and arr.size >= 3:
            arr = arr.reshape(-1, 3)
        else:
            arr = arr.reshape(-1, 1)
    if arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)
    return arr


def _xyz(points: np.ndarray) -> np.ndarray:
    if points.ndim != 2 or points.shape[1] < 3:
        return np.zeros((0, 3), dtype=np.float32)
    return points[:, :3]


def _to_nested(points: np.ndarray) -> list[list[float]]:
    return [[float(x) for x in row] for row in np.asarray(points, dtype=np.float32)]


def _rebuild_context(validation: PointCloudValidation) -> _Context:
    ctx = _Context(
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
        workspace_lower=_vec_or_none((validation.resolved or {}).get("workspace_lower")),
        workspace_upper=_vec_or_none((validation.resolved or {}).get("workspace_upper")),
        color_scale=float((validation.resolved or {}).get("color_scale", 1.0)),
        color_max_bad_ratio=float(cfg_get(validation.config, "pointcloud.color.max_bad_ratio", 0.01)),
        temporal_abs_jump=float(cfg_get(validation.config, "pointcloud.temporal.abs_jump", 0.05)),
        temporal_mad_k=float(cfg_get(validation.config, "pointcloud.temporal.mad_k", 6.0)),
        max_repair_ratio=float(cfg_get(validation.config, "pointcloud.max_repair_ratio", 0.35)),
        jitter_ratio=float(cfg_get(validation.config, "pointcloud.jitter_ratio", 0.005)),
        resolved=validation.resolved or {},
    )
    return ctx


def _vec_or_none(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=float)
    return arr if arr.size >= 3 else None


def _read_info(root: Path) -> dict[str, Any]:
    path = root / "meta" / "info.json"
    if not path.is_file():
        raise PointCloudFormatError(f"缺少 meta/info.json: {root}")
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _discover_data_files(root: Path, info: dict[str, Any]) -> list[Path]:
    from .loader import _discover_data_files as discover
    return discover(root, info)


def _copy_meta(src_root: Path, dst_root: Path) -> None:
    meta_src = src_root / "meta"
    for name in ("info.json", "stats.json", "tasks.parquet", "tasks.jsonl", "subtasks.parquet"):
        src = meta_src / name
        if src.is_file():
            dst = dst_root / "meta" / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    ep_dir = meta_src / "episodes"
    if ep_dir.is_dir():
        for path in ep_dir.rglob("*.parquet"):
            dst = dst_root / "meta" / "episodes" / path.relative_to(ep_dir)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dst)


def _copy_data_parquets(src_root: Path, dst_root: Path) -> None:
    """standalone 输出且无需改写时，原样复制 data 分片。"""
    for path in _discover_data_files(src_root, _read_info(src_root)):
        dst = dst_root / path.relative_to(src_root)
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dst)


def _prepare_dir(root: Path, *, overwrite: bool) -> None:
    if root.exists():
        if not overwrite:
            raise FileExistsError(f"输出目录已存在: {root}")
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)


def _rewrite_pointcloud_stats(dst_root: Path, feature_key: str, config: dict[str, Any]) -> bool:
    """更新 stats.json 里点云特征的均值/方差/极值（分位数沿用旧值）。"""
    stats_path = dst_root / "meta" / "stats.json"
    if not stats_path.is_file():
        return False
    try:
        with stats_path.open("r", encoding="utf-8") as fh:
            stats = json.load(fh)
    except json.JSONDecodeError:
        return False
    if feature_key not in stats:
        return False

    files = _discover_data_files(dst_root, _read_info(dst_root))
    total = 0
    s = None
    sq = None
    lo = None
    hi = None
    batch_size = max(1, int(cfg_get(config, "pointcloud.batch_size", 256)))
    for path in files:
        try:
            pf = pq.ParquetFile(path)
            if feature_key not in pf.schema_arrow.names:
                continue
            for batch in pf.iter_batches(batch_size=batch_size, columns=[feature_key]):
                cells = batch.column(0).to_pylist()
                for cell in cells:
                    arr = _to_matrix(cell)
                    arr = arr[np.isfinite(arr).all(axis=1)]
                    if not arr.size:
                        continue
                    a = arr.astype(np.float64)
                    total += len(a)
                    s = a.sum(axis=0) if s is None else s + a.sum(axis=0)
                    sq = (a ** 2).sum(axis=0) if sq is None else sq + (a ** 2).sum(axis=0)
                    lo = a.min(axis=0) if lo is None else np.minimum(lo, a.min(axis=0))
                    hi = a.max(axis=0) if hi is None else np.maximum(hi, a.max(axis=0))
        except Exception as exc:  # noqa: BLE001 - 损坏分片不参与统计，不中断
            LOG.warning("stats 重算：跳过损坏文件 %s: %s", path.name, exc)
            continue
    if total == 0 or s is None:
        return False
    mean = s / total
    std = np.sqrt(np.maximum(sq / total - mean ** 2, 0.0))
    stats[feature_key]["mean"] = [round(float(x), 6) for x in mean]
    stats[feature_key]["std"] = [round(float(x), 6) for x in std]
    stats[feature_key]["min"] = [round(float(x), 6) for x in lo]
    stats[feature_key]["max"] = [round(float(x), 6) for x in hi]
    stats[feature_key]["count"] = int(total)
    with stats_path.open("w", encoding="utf-8") as fh:
        json.dump(stats, fh, ensure_ascii=False, indent=2)
    LOG.info("meta/stats.json 点云统计已重算（%d 个点）", total)
    return True


def _attach_samples(
    result: PointCloudCleaningResult,
    validation: PointCloudValidation,
    decisions: dict[FramesKey, PointCloudDecision],
    root: Path,
    feature_key: str,
    ctx: _Context,
    *,
    limit: int,
    repaired: list[FramesKey] | None = None,
) -> None:
    """抓几帧前后对比数据给可视化用（不保留全量点云）。"""
    keys = (repaired or [])[:limit]
    if not keys:
        keys = [k for k, r in validation.results.items() if r.valid][:limit]
    if not keys:
        return
    wanted = {k: "repaired" if k in (repaired or []) else "valid" for k in keys}
    found: dict[FramesKey, np.ndarray] = {}
    for path in _discover_data_files(root, _read_info(root)):
        try:
            pf = pq.ParquetFile(path)
            if feature_key not in pf.schema_arrow.names:
                continue
            for batch in pf.iter_batches(batch_size=256, columns=["episode_index", "frame_index", feature_key]):
                data = batch.to_pydict()
                for ep, fi, cell in zip(data["episode_index"], data["frame_index"], data[feature_key]):
                    k = (int(ep), int(fi))
                    if k in wanted and k not in found:
                        found[k] = _to_matrix(cell).copy()
                if len(found) == len(keys):
                    break
        except Exception as exc:  # noqa: BLE001 - 采样是锦上添花，坏文件跳过
            LOG.debug("对比样本抓取跳过损坏文件 %s: %s", path.name, exc)
            continue
        if len(found) == len(keys):
            break
    for k in keys:
        if k not in found:
            continue
        before = found[k]
        decision = decisions.get(k)
        if decision and decision.disposition == "repair":
            after, _, _ = repair_frame(before, decision, ctx)
        else:
            after = before
        result.samples.append({
            "episode_index": k[0],
            "frame_index": k[1],
            "kind": wanted[k],
            "reasons": decision.reasons if decision else [],
            "before": _to_nested(before[:256]),
            "after": _to_nested(after[:256]),
        })


def _write_notes(result: PointCloudCleaningResult, validation: PointCloudValidation, dst_root: Path) -> None:
    notes = {
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "feature_key": validation.feature_key,
        "method": result.method,
        "repaired_frames": [f"{e}:{f}" for e, f in result.repaired_frames],
        "held_frames": [f"{e}:{f}" for e, f in result.held_frames],
        "kept_frames": result.kept_frames,
        "removed_points": result.removed_points,
        "added_points": result.added_points,
        "stats_updated": result.stats_updated,
        "warnings": result.warnings,
        "business_rules": [
            "清洗只改点、不改行：state/action/视频帧索引保持不变",
            "可修复帧：过滤 NaN/离群/越界 -> 去重 -> 颜色裁剪 -> 抖动重采样补回固定点数",
            "不可修复帧：用同 episode 最近的好帧占位（hold-last-good），并在 warnings 留痕",
            "stats.json 的点云 mean/std/min/max 按清洗后数据重算",
        ],
    }
    path = dst_root / "meta" / NOTES_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(notes, fh, ensure_ascii=False, indent=2, default=str)


def _verify(dst_root: Path, validation: PointCloudValidation, feature_key: str, result: PointCloudCleaningResult) -> None:
    try:
        pf = pq.ParquetFile(_discover_data_files(dst_root, _read_info(dst_root))[0])
        rows = sum(pq.ParquetFile(p).metadata.num_rows for p in _discover_data_files(dst_root, _read_info(dst_root)))
        result.verification = {
            "ok": bool(feature_key in pf.schema_arrow.names and rows == validation.total_frames),
            "rows": rows,
            "expected_rows": validation.total_frames,
            "feature_key": feature_key,
        }
    except Exception as exc:  # noqa: BLE001
        result.verification = {"ok": False, "error": str(exc)}


def _health_level(ratio: float) -> str:
    if ratio <= 0:
        return "clean"
    if ratio <= 0.05:
        return "minor"
    if ratio <= 0.25:
        return "moderate"
    return "severe"


def _health_cn(level: str) -> str:
    return {"clean": "干净", "minor": "轻微", "moderate": "中等", "severe": "严重"}.get(level, level)


def _json_default(obj: Any):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


# --- 终端表格对齐（中文按 2 宽） ---
def _display_width(text: str) -> int:
    import unicodedata
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in str(text))


def _ljust(text: str, width: int) -> str:
    return str(text) + " " * max(0, width - _display_width(text))


def _rjust(text: str, width: int) -> str:
    return " " * max(0, width - _display_width(text)) + str(text)
