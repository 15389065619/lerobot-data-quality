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
from .types import CleaningResult, NOTES_FILENAME
from .util import set_column, vec

LOG = logging.getLogger(__name__)


def rewrite_episodes_meta(
    dataset: LeRobotDatasetView,
    kept: list[int],
    ep_index_map: dict[int, int],
    data_meta: dict[int, dict[str, Any]],
    video_meta: dict[int, dict[str, dict[str, Any]]],
    out_root: Path,
) -> None:
    """重建 meta/episodes/chunk-000/file-000.parquet。"""
    ep_dir = dataset.root / "meta" / "episodes"
    tables: list[pa.Table] = []
    if ep_dir.is_dir():
        for f in sorted(ep_dir.rglob("*.parquet")):
            try:
                tables.append(pq.read_table(f))
            except Exception as exc:  # noqa: BLE001
                LOG.warning("读取 %s 失败: %s", f, exc)
    if not tables:
        tables = [pa.table({
            "episode_index": pa.array([], type=pa.int64()),
            "length": pa.array([], type=pa.int64()),
            "tasks": pa.array([], type=pa.list_(pa.string())),
            "dataset_from_index": pa.array([], type=pa.int64()),
            "dataset_to_index": pa.array([], type=pa.int64()),
        })]

    combined = tables[0]
    for t in tables[1:]:
        try:
            combined = pa.concat_tables([combined, t], promote_options="permissive")
        except (TypeError, pa.lib.ArrowInvalid):
            combined = pa.concat_tables([combined, t])

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


def rewrite_info(dataset: LeRobotDatasetView, kept: list[int], total_frames: int, out_root: Path) -> None:
    info = dict(dataset.info)
    info["total_episodes"] = len(kept)
    info["total_frames"] = int(total_frames)
    info["total_tasks"] = len(dataset.tasks)
    info["splits"] = {"train": f"0:{len(kept)}"}
    info["data_path"] = dataset.info.get("data_path") or DEFAULT_DATA_PATH
    info["video_path"] = dataset.info.get("video_path") or DEFAULT_VIDEO_PATH
    path = out_root / "meta" / "info.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(info, fh, ensure_ascii=False, indent=2)
    LOG.info("meta/info.json 已更新：total_episodes=%d, total_frames=%d", len(kept), total_frames)


def rewrite_stats(dataset: LeRobotDatasetView, cleaned: pd.DataFrame, out_root: Path) -> None:
    """在清洗后的数据上重算归一化统计（同时也修复了 stats_stale）。"""
    stats: dict[str, Any] = {}
    for key, spec in dataset.features.items():
        if key not in cleaned.columns:
            if key in dataset.stats:
                stats[key] = dataset.stats[key]
            continue
        try:
            arr = np.asarray(cleaned[key].to_list(), dtype=float)
        except (TypeError, ValueError):
            continue
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        arr = arr[np.isfinite(arr).all(axis=1)] if arr.size else arr
        if not arr.size:
            continue
        stats[key] = {
            "count": int(arr.shape[0]),
            "mean": vec(np.mean(arr, axis=0)),
            "std": vec(np.std(arr, axis=0)),
            "min": vec(np.min(arr, axis=0)),
            "max": vec(np.max(arr, axis=0)),
            "q01": vec(np.quantile(arr, 0.01, axis=0)),
            "q10": vec(np.quantile(arr, 0.10, axis=0)),
            "q50": vec(np.quantile(arr, 0.50, axis=0)),
            "q90": vec(np.quantile(arr, 0.90, axis=0)),
            "q99": vec(np.quantile(arr, 0.99, axis=0)),
        }
    path = out_root / "meta" / "stats.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(stats, fh, ensure_ascii=False, indent=2)
    LOG.info("meta/stats.json 已重算（%d 个特征）", len(stats))


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
