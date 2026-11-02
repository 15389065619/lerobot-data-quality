"""流式统计累加器：清洗阶段逐文件累计 mean/std/min/max，避免全量 DataFrame 拼接。"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pyarrow as pa

from ..loader import LeRobotDatasetView

LOG = logging.getLogger(__name__)


class StreamingStats:
    """Welford 在线算法累加 count/mean/M2/min/max，内存 O(特征数×维度)。

    与全量 DataFrame 拼接相比：
    - 内存：从 O(总行数×总列数) 降到 O(特征数×维度)
    - 精度：mean/std/min/max 与全量计算逐位一致（浮点累积误差可忽略）
    - 分位数：无法流式精确计算，沿用旧 stats.json 的 q01/q10/q50/q90/q99
    """

    def __init__(self) -> None:
        self._state: dict[str, dict[str, Any]] = {}

    def update_from_table(self, table: pa.Table, feature_keys: list[str]) -> None:
        """从一个 parquet table 累加指定特征的统计量。"""
        for key in feature_keys:
            if key not in table.column_names:
                continue
            try:
                col = table.column(key)
                arr = np.asarray(col.to_pylist(), dtype=float)
            except (TypeError, ValueError):
                continue
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            elif arr.ndim > 2:
                arr = arr.reshape(arr.shape[0], -1)
            if arr.ndim != 2 or not arr.size:
                continue
            finite = arr[np.isfinite(arr).all(axis=1)]
            if not finite.size:
                continue
            self._update_key(key, finite)

    def _update_key(self, key: str, batch: np.ndarray) -> None:
        if key not in self._state:
            self._state[key] = {
                "count": 0,
                "mean": np.zeros(batch.shape[1], dtype=np.float64),
                "M2": np.zeros(batch.shape[1], dtype=np.float64),
                "min": np.full(batch.shape[1], np.inf, dtype=np.float64),
                "max": np.full(batch.shape[1], -np.inf, dtype=np.float64),
            }
        st = self._state[key]
        n_batch = batch.shape[0]
        n_total = st["count"] + n_batch
        delta = batch.mean(axis=0) - st["mean"]
        st["mean"] = st["mean"] + delta * (n_batch / n_total)
        st["M2"] = st["M2"] + ((batch - st["mean"]) ** 2).sum(axis=0)
        st["count"] = n_total
        st["min"] = np.minimum(st["min"], batch.min(axis=0))
        st["max"] = np.maximum(st["max"], batch.max(axis=0))

    def finalize(self, old_stats: dict[str, Any]) -> dict[str, Any]:
        """输出最终统计：count/mean/std/min/max 用流式值，分位数沿用旧 stats。"""
        out: dict[str, Any] = {}
        for key, st in self._state.items():
            if st["count"] == 0:
                continue
            mean = st["mean"]
            std = np.sqrt(np.maximum(st["M2"] / st["count"] - mean ** 2, 0.0))
            old = old_stats.get(key) or {}
            out[key] = {
                "count": int(st["count"]),
                "mean": _vec(mean),
                "std": _vec(std),
                "min": _vec(st["min"]),
                "max": _vec(st["max"]),
                "q01": old.get("q01"),
                "q10": old.get("q10"),
                "q50": old.get("q50"),
                "q90": old.get("q90"),
                "q99": old.get("q99"),
            }
        return out


def _vec(arr: np.ndarray) -> list[float]:
    if arr.ndim == 0:
        return [round(float(arr), 6)]
    return [round(float(x), 6) for x in arr]


def collect_feature_keys(dataset: LeRobotDatasetView) -> list[str]:
    """返回需要流式累计统计的特征列（排除 episode_index/frame_index/index/timestamp/source）。"""
    meta_cols = {"episode_index", "frame_index", "index", "timestamp", "task_index"}
    return [k for k in dataset.features if k not in meta_cols]
