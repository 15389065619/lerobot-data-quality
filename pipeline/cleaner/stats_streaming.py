"""流式统计累加器：清洗阶段逐文件累计 mean/std/min/max，避免全量 DataFrame 拼接。"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pyarrow as pa

from ..loader import LeRobotDatasetView

LOG = logging.getLogger(__name__)

# 不属于「统计对象」的索引/时间列：不累计，也不应产生缺失告警
META_FEATURE_KEYS = {"episode_index", "frame_index", "index", "timestamp", "task_index"}


class StreamingStats:
    """Welford 在线算法累加 count/mean/M2/min/max，内存 O(特征数×维度 + 蓄水池)。

    与全量 DataFrame 拼接相比：
    - 内存：从 O(总行数×总列数) 降到 O(特征数×维度)
    - 精度：mean/std 与 numpy 全量计算一致（Welford 并行合并公式）
    - 分位数：确定性蓄水池采样（行数 ≤ 容量时精确），避免沿用旧分布

    ``feature_shapes`` 用于记录特征声明的尾部 shape（如 body_pos 的 ``(29, 3)``），
    累加时仍按展平维度计算，``finalize`` 再 reshape 回去，保证 stats.json 与
    info.json 的 shape 一致。
    """

    def __init__(
        self,
        reservoir_size: int = 100_000,
        seed: int = 0,
        feature_shapes: dict[str, tuple[int, ...]] | None = None,
    ) -> None:
        self._state: dict[str, dict[str, Any]] = {}
        self._reservoir_size = max(0, int(reservoir_size))
        self._rng = np.random.default_rng(seed)
        self._shapes = {
            key: tuple(int(v) for v in shape)
            for key, shape in (feature_shapes or {}).items()
            if shape
        }
        self._skipped: dict[str, str] = {}

    @property
    def skipped(self) -> dict[str, str]:
        """未累计成功的特征及原因（仅保留从未成功累计过的特征）。"""
        return {key: reason for key, reason in self._skipped.items() if key not in self._state}

    def update_from_table(self, table: pa.Table, feature_keys: list[str]) -> None:
        """从一个 parquet table 累加指定特征的统计量。"""
        for key in feature_keys:
            if key not in table.column_names:
                self._skipped.setdefault(key, "清洗后数据中不存在该列")
                continue
            try:
                col = table.column(key)
                arr = np.asarray(col.to_pylist(), dtype=float)
            except (TypeError, ValueError) as exc:
                self._skipped.setdefault(key, f"无法转成数值（{exc}）")
                continue
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            elif arr.ndim > 2:
                arr = arr.reshape(arr.shape[0], -1)
            if arr.ndim != 2 or not arr.size:
                self._skipped.setdefault(key, "列为空")
                continue
            finite = arr[np.isfinite(arr).all(axis=1)]
            if not finite.size:
                self._skipped.setdefault(key, "全部帧含 NaN/Inf，无有效样本")
                continue
            self._skipped.pop(key, None)
            self._update_key(key, finite)

    def _update_key(self, key: str, batch: np.ndarray) -> None:
        if key not in self._state:
            self._state[key] = {
                "count": 0,
                "mean": np.zeros(batch.shape[1], dtype=np.float64),
                "M2": np.zeros(batch.shape[1], dtype=np.float64),
                "min": np.full(batch.shape[1], np.inf, dtype=np.float64),
                "max": np.full(batch.shape[1], -np.inf, dtype=np.float64),
                "sample": None,
                "filled": 0,
                "seen": 0,
            }
        st = self._state[key]
        n_old = st["count"]
        n_batch = batch.shape[0]
        n_total = n_old + n_batch
        batch_mean = batch.mean(axis=0)
        if n_old == 0:
            st["mean"] = batch_mean
            st["M2"] = ((batch - batch_mean) ** 2).sum(axis=0)
        else:
            # Welford 并行合并：M2 += Σ(x-μ_batch)² + δ²·n_old·n_batch/n_total
            delta = batch_mean - st["mean"]
            st["mean"] = st["mean"] + delta * (n_batch / n_total)
            st["M2"] = st["M2"] + ((batch - batch_mean) ** 2).sum(axis=0) \
                + (delta ** 2) * (n_old * n_batch / n_total)
        st["count"] = n_total
        st["min"] = np.minimum(st["min"], batch.min(axis=0))
        st["max"] = np.maximum(st["max"], batch.max(axis=0))
        self._reservoir_update(st, batch)

    def _reservoir_update(self, st: dict[str, Any], batch: np.ndarray) -> None:
        """确定性蓄水池采样：保证分位数来自「清洗后的数据」而不是旧 stats。"""
        cap = self._reservoir_size
        if cap <= 0:
            return
        if st["sample"] is None:
            st["sample"] = np.empty((cap, batch.shape[1]), dtype=np.float32)
        sample = st["sample"]
        filled = st["filled"]
        seen = st["seen"]
        n = batch.shape[0]
        take = min(cap - filled, n)
        if take > 0:
            idx = self._rng.choice(n, size=take, replace=False) if n > take else np.arange(n)
            sample[filled:filled + take] = batch[idx]
            filled += take
        for i in range(take, n):
            pos = seen + i
            j = int(self._rng.integers(0, pos + 1))
            if j < cap:
                sample[j] = batch[i]
        st["filled"] = filled
        st["seen"] = seen + n

    def finalize(self, old_stats: dict[str, Any]) -> dict[str, Any]:
        """输出最终统计：count/mean/std/min/max 用流式值，分位数用蓄水池重算。"""
        out: dict[str, Any] = {}
        quantiles = (0.01, 0.10, 0.50, 0.90, 0.99)
        for key, st in self._state.items():
            if st["count"] == 0:
                continue
            shape = self._shapes.get(key, ())
            mean = st["mean"]
            # M2 已是 Σ(x-μ)²，方差 = M2/n，不能再减 mean²
            std = np.sqrt(np.maximum(st["M2"] / st["count"], 0.0))
            old = old_stats.get(key) or {}
            sample = st["sample"]
            if sample is not None and st["filled"] > 0:
                q = np.quantile(sample[:st["filled"]], quantiles, axis=0)
                q01, q10, q50, q90, q99 = (q[i] for i in range(len(quantiles)))
            else:
                q01 = old.get("q01")
                q10 = old.get("q10")
                q50 = old.get("q50")
                q90 = old.get("q90")
                q99 = old.get("q99")
            out[key] = {
                "count": int(st["count"]),
                "mean": _vec(mean, shape),
                "std": _vec(std, shape),
                "min": _vec(st["min"], shape),
                "max": _vec(st["max"], shape),
                "q01": _vec(q01, shape) if q01 is not None else None,
                "q10": _vec(q10, shape) if q10 is not None else None,
                "q50": _vec(q50, shape) if q50 is not None else None,
                "q90": _vec(q90, shape) if q90 is not None else None,
                "q99": _vec(q99, shape) if q99 is not None else None,
            }
        return out


def _vec(arr: Any, shape: tuple[int, ...] = ()) -> list:
    """序列化统计量：声明了多维 shape 时先 reshape 回去，与 info.json 对齐。"""
    a = np.asarray(arr, dtype=float)
    if len(shape) > 1 and a.size == int(np.prod(shape)):
        a = a.reshape(shape)
    if a.ndim == 0:
        return [round(float(a), 6)]
    if a.ndim == 1:
        return [round(float(x), 6) for x in a]
    return np.round(a, 6).tolist()


def collect_feature_keys(dataset: LeRobotDatasetView) -> list[str]:
    """返回需要流式累计统计的特征列（排除 episode_index/frame_index/index/timestamp/source）。"""
    return [k for k in dataset.features if k not in META_FEATURE_KEYS]
