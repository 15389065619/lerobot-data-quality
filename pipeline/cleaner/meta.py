"""清洗后 meta 目录（episodes / info / stats / tasks / 留痕）的重建。"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ..loader import DEFAULT_DATA_PATH, DEFAULT_VIDEO_PATH, LeRobotDatasetView
from .stats_streaming import META_FEATURE_KEYS
from .types import CleaningResult, NOTES_FILENAME
from .util import set_column, vec

LOG = logging.getLogger(__name__)


def _load_source_episodes(dataset: LeRobotDatasetView) -> pa.Table | None:
    """读取源 episodes 元数据：优先 v3 parquet，回退 v2.1 jsonl。"""
    ep_dir = dataset.root / "meta" / "episodes"
    tables: list[pa.Table] = []
    if ep_dir.is_dir():
        for f in sorted(ep_dir.rglob("*.parquet")):
            try:
                tables.append(pq.read_table(f))
            except Exception as exc:  # noqa: BLE001
                LOG.warning("读取 %s 失败: %s", f, exc)
    if tables:
        combined = tables[0]
        for t in tables[1:]:
            try:
                combined = pa.concat_tables([combined, t], promote_options="permissive")
            except (TypeError, pa.lib.ArrowInvalid):
                combined = pa.concat_tables([combined, t])
        return combined

    jsonl = dataset.root / "meta" / "episodes.jsonl"
    if not jsonl.is_file():
        return None
    rows: list[dict[str, Any]] = []
    with jsonl.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    if not rows:
        return None
    return pa.table({
        "episode_index": pa.array(
            [int(r.get("episode_index", i)) for i, r in enumerate(rows)], type=pa.int64()
        ),
        "length": pa.array([int(r.get("length") or 0) for r in rows], type=pa.int64()),
        "tasks": pa.array([list(r.get("tasks") or []) for r in rows], type=pa.list_(pa.string())),
    })


def rewrite_episodes_meta(
    dataset: LeRobotDatasetView,
    kept: list[int],
    ep_index_map: dict[int, int],
    data_meta: dict[int, dict[str, Any]],
    video_meta: dict[int, dict[str, dict[str, Any]]],
    out_root: Path,
) -> None:
    """重建 meta/episodes/chunk-000/file-000.parquet（v3.0 输出）。"""
    combined = _load_source_episodes(dataset)
    if combined is None:
        # 源里没有任何 episodes 元数据：用已知帧数兜底，至少保证新数据集可读
        combined = pa.table({
            "episode_index": pa.array([int(e) for e in kept], type=pa.int64()),
            "length": pa.array([int(data_meta[e]["length"]) for e in kept], type=pa.int64()),
            "tasks": pa.array([[] for _ in kept], type=pa.list_(pa.string())),
        })

    ep_all = np.asarray(combined.column("episode_index").to_pylist()).astype(np.int64)
    positions = np.where(np.isin(ep_all, np.asarray(kept, dtype=np.int64)))[0]
    sub = combined.take(pa.array(positions, type=pa.int64()))
    order = np.argsort([ep_index_map[int(e)] for e in ep_all[positions]])
    sub = sub.take(pa.array(order, type=pa.int64()))

    old_eps = [int(ep_all[positions][i]) for i in order]
    sub = set_column(sub, "episode_index", np.asarray([ep_index_map[e] for e in old_eps], dtype=np.int64))
    sub = set_column(sub, "length", np.asarray([data_meta[e]["length"] for e in old_eps], dtype=np.int64))
    sub = set_column(sub, "dataset_from_index",
                     np.asarray([data_meta[e]["from_index"] for e in old_eps], dtype=np.int64))
    sub = set_column(sub, "dataset_to_index",
                     np.asarray([data_meta[e]["to_index"] for e in old_eps], dtype=np.int64))
    sub = set_column(sub, "data/chunk_index", np.asarray([data_meta[e]["chunk_index"] for e in old_eps], dtype=np.int64))
    sub = set_column(sub, "data/file_index", np.asarray([data_meta[e]["file_index"] for e in old_eps], dtype=np.int64))

    for cam in dataset.video_features:
        chunk_vals, file_vals, from_vals, to_vals = [], [], [], []
        for e in old_eps:
            info = video_meta.get(e, {}).get(cam)
            if info is None:
                chunk_vals.append(0)
                file_vals.append(0)
                from_vals.append(-1.0)
                to_vals.append(-1.0)
            else:
                chunk_vals.append(int(info.get("chunk_index", 0)))
                file_vals.append(int(info.get("file_index", 0)))
                from_vals.append(float(info.get("from", 0.0)))
                to_vals.append(float(info.get("to", -1.0)))
        sub = set_column(sub, f"videos/{cam}/chunk_index", np.asarray(chunk_vals, dtype=np.int64))
        sub = set_column(sub, f"videos/{cam}/file_index", np.asarray(file_vals, dtype=np.int64))
        sub = set_column(sub, f"videos/{cam}/from_timestamp", np.asarray(from_vals, dtype=np.float64))
        sub = set_column(sub, f"videos/{cam}/to_timestamp", np.asarray(to_vals, dtype=np.float64))

    out_path = out_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(sub, out_path)
    LOG.info("meta/episodes 重建完成：%d 条", sub.num_rows)


def rewrite_info(
    dataset: LeRobotDatasetView,
    kept: list[int],
    total_frames: int,
    out_root: Path,
    *,
    force_v3: bool = False,
) -> None:
    """刷新 info.json 的计数与 splits。

    ``force_v3=True`` 用于 manual_clean：data/videos 已按 chunk/file 重建，
    必须同步升级 codebase_version 与路径模板，否则官方 DatasetReader 会按
    v2 路径去找不存在的文件。copy/reindex 原样保留源布局，不能强制升级。
    """
    info = dict(dataset.info)
    info["total_episodes"] = len(kept)
    info["total_frames"] = int(total_frames)
    info["total_tasks"] = len(dataset.tasks)
    info["splits"] = {"train": f"0:{len(kept)}"}
    if force_v3:
        info["codebase_version"] = "v3.0"
        info["data_path"] = DEFAULT_DATA_PATH
        info["video_path"] = DEFAULT_VIDEO_PATH

    # 源数据集若缺 frame_index 声明（index 兼任了它的语义），补上声明，
    # 避免出现「parquet 有列、info.json 不声明」的漂移。
    # 只在确实有该列（或本次手动重写一定产出该列）时声明，防止反向漂移。
    has_frame_col = (
        "frame_index" in dataset.data_columns
        or "frame_index" in dataset.frames.columns
    )
    if "frame_index" not in (info.get("features") or {}) and (force_v3 or has_frame_col):
        features = dict(info.get("features") or {})
        features["frame_index"] = {"dtype": "int64", "shape": (1,), "names": None}
        info["features"] = features
    path = out_root / "meta" / "info.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(info, fh, ensure_ascii=False, indent=2)
    LOG.info("meta/info.json 已更新：total_episodes=%d, total_frames=%d", len(kept), total_frames)


def _feature_array(series: pd.Series) -> tuple[np.ndarray | None, str | None]:
    """把特征列转成数值矩阵；二维嵌套特征（如 body_pos [N,3]）逐行还原。

    pyarrow 的 ``list<fixed_size_list<float>>`` 读进 pandas 是 object 列，每行一个
    元素为 ndarray 的 ndarray；直接 ``np.asarray(series, dtype=float)`` 或
    ``np.stack([np.asarray(v, dtype=float) for v in series])`` 都会报
    “setting an array element with a sequence”，必须先用 ``tolist()`` 深转换。

    Returns:
        (数组, 失败原因)。成功时原因为 None；失败时数组为 None。
    """
    try:
        return np.asarray(series.to_list(), dtype=float), None
    except (TypeError, ValueError):
        pass
    rows: list[np.ndarray] = []
    for position, value in enumerate(series):
        if value is None:
            return None, f"第 {position} 行为空"
        try:
            raw = value.tolist() if hasattr(value, "tolist") else value
            rows.append(np.asarray(raw, dtype=float))
        except (TypeError, ValueError) as exc:
            return None, f"第 {position} 行无法转成数值（{exc}）"
    if not rows:
        return None, "列为空"
    try:
        return np.stack(rows), None
    except (TypeError, ValueError) as exc:
        return None, f"各行 shape 不一致（{exc}）"


def _stat_view(arr: np.ndarray) -> list:
    """统计量序列化：一维沿用 vec（6 位小数），二维及以上保留特征原 shape。"""
    arr = np.asarray(arr, dtype=float)
    if arr.ndim <= 1:
        return vec(arr)
    return np.round(arr, 6).tolist()


def rewrite_stats(dataset: LeRobotDatasetView, cleaned: pd.DataFrame, out_root: Path) -> list[str]:
    """在清洗后的数据上重算归一化统计（同时也修复了 stats_stale）。

    Returns:
        统计量未能生成的告警文本（同时已写入日志），供报告高亮。
    """
    stats: dict[str, Any] = {}
    warnings: list[str] = []
    for key, spec in dataset.features.items():
        numeric = str((spec or {}).get("dtype", "")).lower().startswith(
            ("float", "int", "uint", "bool")
        )
        if key not in cleaned.columns:
            if key in dataset.stats:
                stats[key] = dataset.stats[key]
            elif numeric:
                message = f"特征 {key} 统计量计算失败：清洗后数据中不存在该列，源 stats.json 也未提供"
                LOG.warning(message)
                warnings.append(message)
            continue
        arr, error = _feature_array(cleaned[key])
        if arr is None:
            # 静默丢字段会让下游归一化少一组统计量，数值型特征必须留痕
            if numeric:
                message = f"特征 {key} 统计量计算失败：{error}，stats.json 中将缺少该特征"
                LOG.warning(message)
                warnings.append(message)
            continue
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        finite = np.isfinite(arr.reshape(arr.shape[0], -1)).all(axis=1)
        arr = arr[finite]
        if not arr.shape[0]:
            if numeric:
                # 与流式路径保持一致：能回退源 stats 就回退，但必须告警（可能已过期）
                if key in dataset.stats:
                    stats[key] = dataset.stats[key]
                    message = (
                        f"特征 {key} 统计量计算失败：全部帧含 NaN/Inf，无有效样本，"
                        "已沿用源 stats.json（可能与清洗后数据不一致）"
                    )
                else:
                    message = (
                        f"特征 {key} 统计量计算失败：全部帧含 NaN/Inf，无有效样本，"
                        "stats.json 中将缺少该特征"
                    )
                LOG.warning(message)
                warnings.append(message)
            continue
        stats[key] = {
            "count": int(arr.shape[0]),
            "mean": _stat_view(np.mean(arr, axis=0)),
            "std": _stat_view(np.std(arr, axis=0)),
            "min": _stat_view(np.min(arr, axis=0)),
            "max": _stat_view(np.max(arr, axis=0)),
            "q01": _stat_view(np.quantile(arr, 0.01, axis=0)),
            "q10": _stat_view(np.quantile(arr, 0.10, axis=0)),
            "q50": _stat_view(np.quantile(arr, 0.50, axis=0)),
            "q90": _stat_view(np.quantile(arr, 0.90, axis=0)),
            "q99": _stat_view(np.quantile(arr, 0.99, axis=0)),
        }
    path = out_root / "meta" / "stats.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(stats, fh, ensure_ascii=False, indent=2)
    LOG.info("meta/stats.json 已重算（%d 个特征）", len(stats))
    return warnings


def rewrite_stats_streaming(
    dataset: LeRobotDatasetView, stats_acc: Any, out_root: Path
) -> list[str]:
    """流式版本：stats_acc 是 StreamingStats 实例，已逐文件累计完成。

    与全量 ``rewrite_stats`` 的语义保持一致：
    - 未累计成功的特征写明真实原因；
    - 回退到源 stats.json 时必须告警（沿用旧统计量可能已过期，不能沉默）；
    - 索引/时间列不参与统计，也不产生缺失告警。

    Returns:
        统计量计算失败的告警文本（同时已写入日志），供报告高亮。
    """
    stats = stats_acc.finalize(dataset.stats)
    reasons = getattr(stats_acc, "skipped", {}) or {}
    warnings: list[str] = []
    for key, spec in dataset.features.items():
        if key in stats:
            continue
        source = dataset.stats.get(key)
        if key in META_FEATURE_KEYS:
            if source is not None:
                stats[key] = source
            continue
        numeric = str((spec or {}).get("dtype", "")).lower().startswith(
            ("float", "int", "uint", "bool")
        )
        if not numeric:
            if source is not None:
                stats[key] = source
            continue
        reason = reasons.get(key, "流式模式未累计该特征")
        if source is not None:
            stats[key] = source
            message = (
                f"特征 {key} 统计量计算失败：{reason}，"
                "已沿用源 stats.json（可能与清洗后数据不一致）"
            )
        else:
            message = f"特征 {key} 统计量计算失败：{reason}，stats.json 中将缺少该特征"
        LOG.warning(message)
        warnings.append(message)
    path = out_root / "meta" / "stats.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(stats, fh, ensure_ascii=False, indent=2)
    LOG.info("meta/stats.json 已重算（流式，%d 个特征）", len(stats))
    return warnings


def copy_tasks(src_root: Path, out_root: Path) -> None:
    for name in ("tasks.parquet", "tasks.jsonl", "subtasks.parquet"):
        src = src_root / "meta" / name
        if src.is_file():
            dst = out_root / "meta" / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


def write_notes(result: CleaningResult, out_root: Path) -> None:
    """把清洗决策留痕在数据集内，方便回溯「哪一版删了什么、修了什么」。"""
    notes = {
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "method": result.method,
        "removed_episode_indices": result.removed_episodes,
        "repaired_episode_indices": result.repaired_episodes,
        "repairs": result.repairs,
        "kept_episodes": result.kept_episodes,
        "episode_index_map": {str(k): v for k, v in sorted(result.episode_index_map.items())},
        "cli_command": result.cli_command,
        "warnings": result.warnings,
        "verification": result.verification,
    }
    path = out_root / "meta" / NOTES_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(notes, fh, ensure_ascii=False, indent=2, default=str)
