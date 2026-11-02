"""点云清洗：把修复结果写回数据集。"""

from __future__ import annotations

import json
import logging
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ..config import cfg_get
from .context import PointCloudContext, rebuild_context
from .rules import duplicate_ratio, outlier_mask
from .types import (
    CLEANED_DIRNAME,
    NOTES_FILENAME,
    RULE_MAP,
    FramesKey,
    PointCloudCleaningResult,
    PointCloudDecision,
    PointCloudValidation,
)
from .utils import (
    copy_data_parquets,
    copy_meta,
    discover_data_files,
    prepare_dir,
    read_info,
    to_matrix,
    to_nested,
    xyz,
)

LOG = logging.getLogger(__name__)


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
    ctx = rebuild_context(validation)
    repaired = [k for k, d in decisions.items() if d.disposition == "repair"]
    held = [k for k, d in decisions.items() if d.disposition == "drop"]

    if target_root:
        dst_root = Path(target_root).expanduser().resolve()
        same_tree = dst_root == src_root
        method = "inplace-patch" if same_tree else "patch-cleaned-dataset"
    else:
        dst_root = out_dir / CLEANED_DIRNAME
        prepare_dir(dst_root, overwrite=overwrite)
        copy_meta(src_root, dst_root)
        same_tree = False
        method = "standalone"

    if not repaired and not held:
        LOG.info("点云无需清洗：%d 帧全部保留", validation.total_frames)
        result = PointCloudCleaningResult(
            success=True, output_root=dst_root, method="noop" if target_root else method,
            kept_frames=validation.total_frames,
        )
        if not target_root:
            copy_data_parquets(src_root, dst_root)
        _attach_samples(result, validation, decisions, src_root, key, ctx, limit=3)
        _write_notes(result, validation, dst_root)
        _verify(dst_root, validation, key, result)
        return result

    files = discover_data_files(src_root, read_info(src_root))
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
                    new_cells: list[Any] = []
                    for ep, fi, cell in zip(data["episode_index"], data["frame_index"], data[key]):
                        k = (int(ep), int(fi))
                        pts = to_matrix(cell)
                        if k in decisions and decisions[k].disposition == "repair":
                            new_pts, removed, added = repair_frame(pts, decisions[k], ctx)
                            added_points += added
                            removed_points += removed
                            last_good[int(ep)] = new_pts
                            new_cells.append(to_nested(new_pts))
                        elif k in decisions and decisions[k].disposition == "drop":
                            good = last_good.get(int(ep))
                            if good is None:
                                new_cells.append(to_nested(pts))
                                result.warnings.append(f"ep {ep} 帧 {fi} 之前没有可参考的好帧，保留原值")
                            else:
                                new_cells.append(to_nested(good))
                                result.warnings.append(f"ep {ep} 帧 {fi} 不可修复，使用同 episode 最近好帧占位")
                        else:
                            last_good[int(ep)] = pts if np.isfinite(pts).all() else last_good.get(int(ep), pts)
                            new_cells.append(to_nested(pts))
                    icol = batch.schema.get_field_index(key)
                    new_col = pa.array(new_cells, type=field.type)
                    batch = batch.set_column(icol, field, new_col)
                    writer.write_batch(batch)
            finally:
                writer.close()
                pf.close()

            if same_tree:
                os.replace(tmp_path, dst_path)
            LOG.info("点云清洗写入 %s", dst_path)
        except Exception as exc:  # noqa: BLE001
            LOG.error("点云清洗：文件损坏，原样保留并继续: %s -> %s", src_path.name, exc)
            result.warnings.append(f"{src_path.name} 损坏，已原样保留: {exc}")
            if not same_tree:
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
        copy_meta(src_root, dst_root)
    result.stats_updated = _rewrite_pointcloud_stats(dst_root, key, validation.config)
    _attach_samples(result, validation, decisions, src_root, key, ctx, limit=5, repaired=repaired)
    _write_notes(result, validation, dst_root)
    _verify(dst_root, validation, key, result)
    LOG.info("点云清洗完成：修复 %d 帧 / 占位 %d 帧 / 保留 %d 帧 -> %s",
             len(repaired), len(held), result.kept_frames, dst_root)
    return result


def repair_frame(points: np.ndarray, decision: PointCloudDecision, ctx: PointCloudContext) -> tuple[np.ndarray, int, int]:
    """按修复动作过滤 -> 去重 -> 裁剪颜色 -> 抖动重采样补回固定点数。"""
    pts = to_matrix(points).astype(np.float32, copy=True)
    original_n = int(pts.shape[0])
    removed = 0
    actions = set(decision.repair_actions) | {RULE_MAP[r].repair for r in decision.reasons if RULE_MAP[r].repair}

    if "filter" in actions or "pc_non_finite" in decision.reasons:
        pts = pts[np.isfinite(pts).all(axis=1)]
    if "bounds" in actions and ctx.workspace_lower is not None:
        inside = np.all((xyz(pts) >= ctx.workspace_lower) & (xyz(pts) <= ctx.workspace_upper), axis=1)
        pts = pts[inside]
    if "filter" in actions and len(pts):
        outlier = outlier_mask(xyz(pts), ctx.outlier_radius, ctx.outlier_min_neighbors)
        pts = pts[~outlier]
    if "dedupe" in actions and len(pts):
        coords = np.round(xyz(pts), ctx.duplicate_decimals)
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

    files = discover_data_files(dst_root, read_info(dst_root))
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
                    arr = to_matrix(cell)
                    arr = arr[np.isfinite(arr).all(axis=1)]
                    if not arr.size:
                        continue
                    a = arr.astype(np.float64)
                    total += len(a)
                    s = a.sum(axis=0) if s is None else s + a.sum(axis=0)
                    sq = (a ** 2).sum(axis=0) if sq is None else sq + (a ** 2).sum(axis=0)
                    lo = a.min(axis=0) if lo is None else np.minimum(lo, a.min(axis=0))
                    hi = a.max(axis=0) if hi is None else np.maximum(hi, a.max(axis=0))
        except Exception as exc:  # noqa: BLE001
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
    ctx: PointCloudContext,
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
    for path in discover_data_files(root, read_info(root)):
        try:
            pf = pq.ParquetFile(path)
            if feature_key not in pf.schema_arrow.names:
                continue
            for batch in pf.iter_batches(batch_size=256, columns=["episode_index", "frame_index", feature_key]):
                data = batch.to_pydict()
                for ep, fi, cell in zip(data["episode_index"], data["frame_index"], data[feature_key]):
                    k = (int(ep), int(fi))
                    if k in wanted and k not in found:
                        found[k] = to_matrix(cell).copy()
                if len(found) == len(keys):
                    break
        except Exception as exc:  # noqa: BLE001
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
            "before": to_nested(before[:256]),
            "after": to_nested(after[:256]),
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
        pf = pq.ParquetFile(discover_data_files(dst_root, read_info(dst_root))[0])
        rows = sum(pq.ParquetFile(p).metadata.num_rows for p in discover_data_files(dst_root, read_info(dst_root)))
        result.verification = {
            "ok": bool(feature_key in pf.schema_arrow.names and rows == validation.total_frames),
            "rows": rows,
            "expected_rows": validation.total_frames,
            "feature_key": feature_key,
        }
    except Exception as exc:  # noqa: BLE001
        result.verification = {"ok": False, "error": str(exc)}
