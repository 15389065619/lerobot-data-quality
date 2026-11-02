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

import logging
from pathlib import Path
from typing import Any

from ..config import load_config
from ..loader import LeRobotDatasetView, heavy_feature_keys
from .cleaner import clean_pointclouds
from .context import PointCloudContext, build_context
from .policy import decide, read_overrides, write_review_queue
from .reporter import build_report, print_summary, write_report
from .rules import apply_temporal_rule, evaluate_frame
from .scanner import ScanCache, scan, scan_parallel
from .types import (
    CLEANED_DIRNAME,
    NOTES_FILENAME,
    REPORT_FILENAME,
    REPAIRABLE,
    REPAIR_ACTIONS,
    RULE_MAP,
    RULE_ORDER,
    RULES,
    REVIEW_FILENAME,
    SEVERITY_ORDER,
    FramesKey,
    FrameResult,
    PointCloudCleaningResult,
    PointCloudDecision,
    PointCloudFormatError,
    PointCloudValidation,
    Rule,
)
from .utils import read_info, discover_data_files, to_matrix

LOG = logging.getLogger(__name__)


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
    info = view.info if view else read_info(root)
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

    files = discover_data_files(root, info)
    if not files:
        raise PointCloudFormatError(f"未在 {root} 下找到 data/*.parquet")
    feature_present = False
    for path in files:
        try:
            import pyarrow.parquet as pq
            if key in pq.ParquetFile(path).schema_arrow.names:
                feature_present = True
                break
        except Exception:  # noqa: BLE001
            continue
    if not feature_present:
        raise PointCloudFormatError(f"数据 parquet 中不存在点云列: {key}")

    ctx = build_context(root, files, info, config, key, expected_points, dim)
    cache = ScanCache(Path(cache_path), config, key, ctx.resolved) if cache_path else None
    if workers > 1:
        results, skipped, corrupted = scan_parallel(
            files, ctx, max_frames=max_frames, workers=workers, cache=cache)
    else:
        results, skipped, corrupted = scan(files, ctx, max_frames=max_frames, cache=cache)
    if cache:
        cache.save()
    apply_temporal_rule(results, ctx)

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


__all__ = [
    "CLEANED_DIRNAME",
    "NOTES_FILENAME",
    "REPORT_FILENAME",
    "REPAIRABLE",
    "REPAIR_ACTIONS",
    "RULE_MAP",
    "RULE_ORDER",
    "RULES",
    "REVIEW_FILENAME",
    "SEVERITY_ORDER",
    "FramesKey",
    "FrameResult",
    "PointCloudCleaningResult",
    "PointCloudContext",
    "PointCloudDecision",
    "PointCloudFormatError",
    "PointCloudValidation",
    "Rule",
    "ScanCache",
    "apply_temporal_rule",
    "build_context",
    "build_report",
    "clean_pointclouds",
    "decide",
    "detect_pointcloud_feature",
    "evaluate_frame",
    "print_summary",
    "read_overrides",
    "validate_pointcloud_dataset",
    "write_report",
    "write_review_queue",
]
