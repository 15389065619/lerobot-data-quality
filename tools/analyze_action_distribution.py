#!/usr/bin/env python3
"""清洗前后动作分布对比：方差、跳变峰值、均值漂移。

为什么需要这个脚本
------------------
质检报告回答的是「哪些 episode 有问题」。但它不回答下一个问题：**清洗之后，
动作序列到底变成了什么样？** 没有这一步，「数据变干净了」只是主观判断。

本脚本只做一件事：把三个可复算的统计量摆到同一张表上，原始侧和清洗侧对齐。

取数与算账
----------
取数分三份，缺一不可：

1. **原始全量** —— 基线，含所有已剔除的坏 episode；
2. **原始仅保留集** —— 只取清洗后仍在的那些 episode。这一份用来把「剔除」和
   「修复」两种效应拆开：删样本必然改变方差，那是选择效应；裁剪边界帧、重建
   时间戳带来的变化才是修复效应。少了这一份，任何「方差下降」都无法归因；
3. **清洗后** —— 结果。

三个指标的口径：

- **方差**：逐维总体方差（ddof=0）跨维平均；另算离差平方和（SSE）用于归因，
  SSE 可跨 episode 相加，方差不可；
- **跳变峰值**：episode 内相邻帧 |Δ| 的 L∞ 范数（跨维取最大）的全局最大值，
  另报「每集最大跳变」的中位数/分位数，以及超过阈值的帧数——单看最大值会被
  一个尖峰主导，分位数才看得出整体分布是否收敛；
- **均值漂移**：逐维均值差的最大绝对值，并按池化标准差归一化（等价于 Cohen's d
  的分子），避免用绝对值比较不同量纲的维度。

明确不做的
----------
不训练模型，因此不声称 loss 曲线变化。这里报的是**代理指标**（归一化跳变幅度、
超阈帧数），只能说明「梯度被异常样本放大的通道变少了」，不能说明「最终精度提升
了多少」。真要那条结论，需要跑训练对照，见 docs/EVAL.md §8.5。
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

if __package__ in (None, ""):  # 允许 `python tools/analyze_action_distribution.py` 直接跑
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.config import cfg_get, load_config  # noqa: E402
from pipeline.loader import ACTION_KEY, load_dataset  # noqa: E402

#: 注入 docs/EVAL.md 的分节标记。同一份文档可能要放多组对比（默认策略 / 复核回灌后），
#: 用 section_id 区分，避免后来的覆盖前面的。
DEFAULT_SECTION_ID = "ACTION_DIST"
DEFAULT_TITLE = "## 9. 清洗前后动作分布对比（下游收益代理指标）"


def _markers(section_id: str) -> tuple[str, str]:
    return f"<!-- {section_id}:BEGIN -->", f"<!-- {section_id}:END -->"

#: 单个 float64 元素占位，用于估算分块读取的内存上限
_BYTES = 8


# --------------------------------------------------------------------------- 读
def _to_matrix(column: pa.ChunkedArray) -> np.ndarray:
    """把 action 列转成 (帧数, 维度) 的 float64 矩阵，非有限值置 NaN。

    NaN/Inf 不丢弃，而是置 NaN 后由统计量按维跳过——逐帧丢行会让「含 1 个 Inf 的
    整条 episode」从方差里消失，反而低估了原始数据的离散程度。
    """
    arr = column.combine_chunks()
    if pa.types.is_fixed_size_list(arr.type):
        dim = arr.type.list_size
        flat = np.asarray(arr.flatten(), dtype=np.float64)
        matrix = flat.reshape(len(arr), dim)
        nulls = np.asarray(arr.is_null())
        if nulls.any():
            matrix[nulls] = np.nan
    else:  # 变长 list：逐行兜底，只在大数据集出现非规范 schema 时走到
        rows = arr.to_pylist()
        dim = 0
        for row in rows:
            if row is not None:
                dim = len(row)
                break
        matrix = np.full((len(rows), dim), np.nan, dtype=np.float64)
        for i, row in enumerate(rows):
            if row:
                values = np.asarray(row, dtype=np.float64)
                matrix[i, : values.size] = values[:dim]

    bad = int((~np.isfinite(matrix)).sum())
    if bad:
        matrix = np.where(np.isfinite(matrix), matrix, np.nan)
    return matrix


@dataclass
class _EpisodeAcc:
    """单 episode 的累加器。只存 O(维度) 的中间量，不留在内存里堆原始帧。"""

    frames: int = 0
    bad: int = 0
    n: np.ndarray | None = None
    s1: np.ndarray | None = None
    s2: np.ndarray | None = None
    max_delta: float = 0.0
    max_delta_dim: int = -1
    over: int = 0
    last: np.ndarray | None = None

    def ensure(self, dim: int) -> None:
        if self.n is None:
            self.n = np.zeros(dim, dtype=np.int64)
            self.s1 = np.zeros(dim, dtype=np.float64)
            self.s2 = np.zeros(dim, dtype=np.float64)


class ActionAccumulator:
    """按 episode 累加动作统计量，支持跨 parquet 分片的 episode。

    跨文件续接靠 `_EpisodeAcc.last`：上一块的最后一行留作下一块第一条 delta 的
    起点，否则分片边界处的跳变会被漏掉（大 episode 跨文件时正是最脏的地方）。
    """

    def __init__(self, threshold: float, keep: set[int] | None = None) -> None:
        self.threshold = float(threshold)
        self.keep = keep
        self.episodes: dict[int, _EpisodeAcc] = {}
        self.frames = 0
        self.bad = 0
        self.dim = 0
        self.corrupted: list[dict[str, str]] = []

    # ------------------------------------------------------------------ 累加
    def add(self, table: pa.Table) -> None:
        names = table.column_names
        if "episode_index" not in names or ACTION_KEY not in names:
            raise KeyError(f"数据文件缺少 episode_index / {ACTION_KEY} 列: {names}")

        episodes = np.asarray(table.column("episode_index").to_numpy(zero_copy_only=False))
        matrix = _to_matrix(table.column(ACTION_KEY))
        if matrix.size == 0:
            return
        if self.dim == 0:
            self.dim = matrix.shape[1]
        elif matrix.shape[1] != self.dim:
            raise ValueError(f"action 维度不一致：先见 {self.dim}，本文件 {matrix.shape[1]}")

        order = self._row_order(table, episodes)
        if order is not None:
            episodes = episodes[order]
            matrix = matrix[order]

        self._consume(episodes, matrix)

    @staticmethod
    def _row_order(table: pa.Table, episodes: np.ndarray) -> np.ndarray | None:
        """返回重排索引；已按 (episode, frame) 有序时返回 None，省一次全量排序。"""
        if len(episodes) < 2:
            return None
        ordered = bool((np.diff(episodes) >= 0).all())
        if "frame_index" in table.column_names:
            frames = np.asarray(table.column("frame_index").to_numpy(zero_copy_only=False),
                                dtype=np.float64)
            if ordered and np.isfinite(frames).all() and (np.diff(frames) >= 0).all():
                return None
            if np.isfinite(frames).all():
                return np.lexsort((frames, episodes))
        return None if ordered else np.argsort(episodes, kind="stable")

    def _consume(self, episodes: np.ndarray, matrix: np.ndarray) -> None:
        if len(episodes) == 0:
            return
        bounds = np.flatnonzero(np.diff(episodes) != 0) + 1
        starts = np.concatenate(([0], bounds))
        ends = np.concatenate((bounds, [len(episodes)]))
        for start, end in zip(starts, ends):
            ep = int(episodes[start])
            if self.keep is not None and ep not in self.keep:
                continue
            sub = matrix[start:end]
            acc = self.episodes.get(ep)
            if acc is None:
                acc = self.episodes[ep] = _EpisodeAcc()
            acc.ensure(self.dim)
            acc.frames += len(sub)
            self.frames += len(sub)  # 只统计被纳入的 episode，keep 过滤后才能对齐

            finite = np.isfinite(sub)
            missing = int((~finite).sum())
            if missing:
                acc.bad += missing
                self.bad += missing
            acc.n += finite.sum(axis=0)
            acc.s1 += np.where(finite, sub, 0.0).sum(axis=0)
            acc.s2 += np.where(finite, sub * sub, 0.0).sum(axis=0)

            seq = sub if acc.last is None else np.vstack((acc.last, sub))
            if len(seq) > 1:
                delta = np.abs(np.diff(seq, axis=0))
                valid = np.isfinite(delta).all(axis=1)
                if valid.any():
                    kept = delta[valid]
                    row_max = kept.max(axis=1)
                    worst = int(row_max.argmax())
                    if row_max[worst] > acc.max_delta:
                        acc.max_delta = float(row_max[worst])
                        acc.max_delta_dim = int(kept[worst].argmax())
                    acc.over += int((row_max > self.threshold).sum())
            acc.last = sub[-1].copy()

    # ------------------------------------------------------------------ 汇总
    def summary(self, root: str = "") -> dict[str, Any]:
        per_episode: list[dict[str, Any]] = []
        means: list[np.ndarray] = []
        counts: list[np.ndarray] = []
        within: list[float] = []
        total_n = np.zeros(self.dim, dtype=np.int64)
        total_s1 = np.zeros(self.dim, dtype=np.float64)
        total_s2 = np.zeros(self.dim, dtype=np.float64)

        for ep in sorted(self.episodes):
            acc = self.episodes[ep]
            assert acc.n is not None and acc.s1 is not None and acc.s2 is not None
            mean = np.divide(acc.s1, acc.n, out=np.full(self.dim, np.nan), where=acc.n > 0)
            var = np.divide(
                acc.s2, acc.n, out=np.full(self.dim, np.nan), where=acc.n > 0
            ) - mean**2
            sse_within = float(np.nansum(acc.s2 - acc.s1**2 / np.maximum(acc.n, 1)))
            per_episode.append(
                {
                    "episode_index": ep,
                    "frames": acc.frames,
                    "var": float(np.nanmean(var)) if np.isfinite(var).any() else float("nan"),
                    "max_delta": acc.max_delta,
                    "over_threshold": acc.over,
                    "sse_within": sse_within,
                }
            )
            means.append(mean)
            counts.append(acc.n)
            within.append(sse_within)
            total_n += acc.n
            total_s1 += acc.s1
            total_s2 += acc.s2

        mean = np.divide(total_s1, total_n, out=np.full(self.dim, np.nan), where=total_n > 0)
        var = np.divide(
            total_s2, total_n, out=np.full(self.dim, np.nan), where=total_n > 0
        ) - mean**2
        var = np.maximum(var, 0.0)  # 抵消浮点抵消误差带来的 -1e-18
        sse = float(np.nansum(total_s2 - total_s1**2 / np.maximum(total_n, 1)))

        # 单集对总 SSE 的**可加**贡献 = 组间项 n·(均值差)² + 组内 SSE。
        # 只用组内 SSE 会让各集之和小于总量（差额是集间差异），报「占比」时会自相矛盾。
        for row, ep_mean, ep_n, sse_within in zip(per_episode, means, counts, within):
            between = float(np.nansum(ep_n * (ep_mean - mean) ** 2))
            row["sse"] = between + sse_within

        deltas = np.array([row["max_delta"] for row in per_episode], dtype=float)
        over_rows = [row for row in per_episode if row["over_threshold"] > 0]
        worst = max(per_episode, key=lambda row: row["max_delta"]) if per_episode else None
        var_value = float(np.nanmean(var)) if np.isfinite(var).any() else float("nan")
        std_value = math.sqrt(var_value) if math.isfinite(var_value) else float("nan")
        per_dim_std = float(np.nanmean(np.sqrt(var))) if np.isfinite(var).any() else float("nan")

        return {
            "root": root,
            "episodes": len(per_episode),
            "frames": self.frames,
            "dim": self.dim,
            "var": var_value,
            "std": std_value,
            #: 逐维 std 的平均，仅用于和 stats.json 互校（stats 存的是逐维 std）
            "std_per_dim_mean": per_dim_std,
            "mean": float(np.nanmean(mean)) if np.isfinite(mean).any() else float("nan"),
            "per_dim_mean": [float(v) for v in mean],
            "per_dim_var": [float(v) for v in var],
            "sse": sse,
            "max_delta": float(worst["max_delta"]) if worst else float("nan"),
            "max_delta_episode": int(worst["episode_index"]) if worst else -1,
            "delta_median": float(np.median(deltas)) if deltas.size else float("nan"),
            "delta_p90": float(np.percentile(deltas, 90)) if deltas.size else float("nan"),
            "over_threshold": int(sum(row["over_threshold"] for row in per_episode)),
            "over_episodes": len(over_rows),
            "threshold": self.threshold,
            "bad_values": self.bad,
            "per_episode": per_episode,
            "corrupted_files": self.corrupted,
        }


def analyze_dataset(
    root: str | Path,
    *,
    keep: set[int] | None = None,
    threshold: float | None = None,
    max_files: int | None = None,
) -> dict[str, Any]:
    """扫描一个数据集的 action 列，返回统计摘要。

    Args:
        root: 数据集根目录。
        keep: 只统计这些 episode（用于「原始仅保留集」基线）。
        threshold: 跳变阈值；None 时取配置 `action_jump.abs_floor` 并按 fps 折算。
        max_files: 只读前 N 个数据文件（自测/试跑用，正常不要设）。
    """
    view = load_dataset(root, load_frames=False)
    config = load_config(robot_type=view.info.get("robot_type"))
    if threshold is None:
        abs_floor = float(cfg_get(config, "action_jump.abs_floor", 1.0))
        ref_fps = float(cfg_get(config, "action_jump.ref_fps", 30.0))
        fps = float(view.fps or ref_fps)
        threshold = abs_floor * ref_fps / fps if fps > 0 else abs_floor

    acc = ActionAccumulator(threshold, keep=keep)
    files = view.data_files[:max_files] if max_files else view.data_files
    for path in files:
        try:
            schema = pq.read_schema(path)
            columns = [c for c in ("episode_index", "frame_index", ACTION_KEY) if c in schema.names]
            if ACTION_KEY not in columns or "episode_index" not in columns:
                acc.corrupted.append({"file": str(path), "error": f"缺少必要列: {schema.names}"})
                continue
            acc.add(pq.read_table(path, columns=columns))
        except Exception as exc:  # noqa: BLE001 - 损坏文件不应中断对比
            acc.corrupted.append({"file": str(path), "error": str(exc)})

    summary = acc.summary(root=str(root))
    summary["fps"] = float(view.fps or 0.0)
    summary["codebase_version"] = view.codebase_version
    summary["stats_action_std"] = _stats_std(view.stats)
    return summary


def _stats_std(stats: dict[str, Any]) -> float | None:
    """数据集自带 stats.json 里 action 各维 std 的均值，用于互校本脚本的读数。"""
    node = (stats or {}).get(ACTION_KEY) or {}
    values = node.get("std")
    if not isinstance(values, list) or not values:
        return None
    try:
        return float(np.mean([float(v) for v in values]))
    except (TypeError, ValueError):
        return None


# ------------------------------------------------------------------- episode 映射
def resolve_episode_map(cleaned_root: Path, report: dict[str, Any] | None) -> tuple[dict[int, int], str]:
    """返回 ({清洗后编号: 原始编号}, 来源)。

    导出脚本会把 episode_index 从 0 连续重排，因此两侧编号不能直接对齐。首选
    清洗说明里的 `episode_index_map`（导出时生成，最可信；方向是「旧 -> 新」，
    这里取反）；没有就按报告推导，再没有只能返回空——那种情况下「仅保留集」
    基线不可信，调用方应拒绝归因，而不是假设同名。
    """
    notes = cleaned_root / "meta" / "quality_cleaning_notes.json"
    if notes.is_file():
        try:
            payload = json.loads(notes.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            payload = {}
        mapping = payload.get("episode_index_map")
        if isinstance(mapping, dict) and mapping:
            return {int(new): int(old) for old, new in mapping.items()}, str(notes.name)

    if report is not None:
        dropped = {
            int(item["episode_index"])
            for item in report.get("invalid_episode_details") or []
            if str(item.get("disposition") or "").lower() == "drop"
            and str(item.get("episode_index", "")).lstrip("-").isdigit()
        }
        source = (report.get("dataset") or {}).get("path") or (report.get("dataset") or {}).get("root")
        if dropped and source and Path(source).is_dir():
            kept = [ep for ep in _episode_ids(Path(source)) if ep not in dropped]
            return {i: ep for i, ep in enumerate(sorted(kept))}, "报告推导"
    return {}, "无法确定"


def _episode_ids(root: Path) -> list[int]:
    """从数据文件里取 episode 编号（meta 可能与 data 不一致，取 data 侧）。"""
    view = load_dataset(root, load_frames=False)
    ids: set[int] = set()
    for path in view.data_files:
        try:
            column = pq.read_table(path, columns=["episode_index"]).column("episode_index")
        except Exception:  # noqa: BLE001
            continue
        ids.update(int(v) for v in np.unique(column.to_numpy(zero_copy_only=False)))
    return sorted(ids)


# ---------------------------------------------------------------------- 对比
def compare(
    raw_path: str | Path,
    cleaned_path: str | Path,
    *,
    report_path: str | Path | None = None,
    threshold: float | None = None,
) -> dict[str, Any]:
    """算三份统计量并给出归因；返回可直接渲染或序列化的字典。"""
    raw_path, cleaned_path = Path(raw_path), Path(cleaned_path)
    report: dict[str, Any] | None = None
    if report_path:
        report = json.loads(Path(report_path).read_text(encoding="utf-8"))

    mapping, map_source = resolve_episode_map(cleaned_path, report)

    raw_all = analyze_dataset(raw_path, threshold=threshold)
    if not mapping:
        raise SystemExit(
            "无法确定清洗后的 episode 与原始编号的对应关系："
            "缺少 meta/quality_cleaning_notes.json，且 --report 无法推导。"
            "没有这个映射就无法区分「剔除效应」和「修复效应」，因此不输出归因。"
        )
    kept_raw = set(mapping.values())
    if len(mapping) != raw_all["episodes"] - _dropped_count(report):
        # 不是硬错误：报告可能与清洗产物不是同一批次，但要显式告警
        print(
            f"[警告] 映射表 {len(mapping)} 项与「原始 {raw_all['episodes']} - 剔除 "
            f"{_dropped_count(report)}」不一致，请确认清洗产物与报告同源",
            file=sys.stderr,
        )

    raw_kept = analyze_dataset(raw_path, keep=kept_raw, threshold=threshold)
    cleaned = analyze_dataset(cleaned_path, threshold=threshold)

    if cleaned["dim"] != raw_all["dim"]:
        raise SystemExit(f"动作维度不一致：原始 {raw_all['dim']} 维，清洗后 {cleaned['dim']} 维")
    if cleaned["episodes"] != len(mapping):
        raise SystemExit(
            f"清洗后 {cleaned['episodes']} 个 episode，映射表 {len(mapping)} 项，无法对齐"
        )

    dropped_ids = sorted(set(_episode_ids(raw_path)) - kept_raw)
    dropped_rows = _dropped_rows(raw_all, dropped_ids, report)
    return {
        "raw_path": str(raw_path),
        "cleaned_path": str(cleaned_path),
        "map_source": map_source,
        "threshold": cleaned["threshold"],
        "raw_all": raw_all,
        "raw_kept": raw_kept,
        "cleaned": cleaned,
        "dropped": dropped_rows,
        "dropped_ids": dropped_ids,
        "metrics": _metrics(raw_all, raw_kept, cleaned),
    }


def _dropped_count(report: dict[str, Any] | None) -> int:
    if not report:
        return 0
    return sum(
        1
        for item in report.get("invalid_episode_details") or []
        if str(item.get("disposition") or "").lower() == "drop"
    )


def _dropped_rows(
    raw_all: dict[str, Any],
    dropped_ids: list[int],
    report: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """被剔除 episode 的尾部贡献：方差、SSE 占比、剔除原因。"""
    by_ep = {row["episode_index"]: row for row in raw_all["per_episode"]}
    reasons: dict[int, str] = {}
    if report:
        for item in report.get("invalid_episode_details") or []:
            try:
                ep = int(item.get("episode_index"))
            except (TypeError, ValueError):
                continue
            if str(item.get("disposition") or "").lower() == "drop":
                reasons[ep] = " / ".join(item.get("reasons") or []) or "-"

    total_sse = raw_all["sse"] or float("nan")
    total_frames = raw_all["frames"] or 1
    rows = []
    for ep in dropped_ids:
        stat = by_ep.get(ep)
        if stat is None:
            rows.append({"episode_index": ep, "frames": 0, "var": float("nan"),
                         "max_delta": float("nan"), "sse_share": float("nan"),
                         "frame_share": float("nan"), "reason": reasons.get(ep, "数据未读到")})
            continue
        rows.append(
            {
                "episode_index": ep,
                "frames": stat["frames"],
                "var": stat["var"],
                "max_delta": stat["max_delta"],
                "sse_share": stat["sse"] / total_sse * 100 if total_sse else float("nan"),
                "frame_share": stat["frames"] / total_frames * 100,
                "reason": reasons.get(ep, "-"),
            }
        )
    rows.sort(key=lambda row: (-(row["sse_share"] if math.isfinite(row["sse_share"]) else 0)))
    return rows


def _pct(after: float, before: float) -> float:
    if not math.isfinite(after) or not math.isfinite(before) or before == 0:
        return float("nan")
    return (after - before) / before * 100


def _metrics(raw_all: dict[str, Any], raw_kept: dict[str, Any], cleaned: dict[str, Any]) -> dict[str, Any]:
    """总变化 = 选择效应（剔除）+ 修复效应（裁剪/重建），按 SSE 可加性拆分。"""
    d_var = _pct(cleaned["var"], raw_all["var"])
    repair_var = _pct(cleaned["var"], raw_kept["var"])
    mean_drift = _max_abs_diff(cleaned["per_dim_mean"], raw_all["per_dim_mean"])
    repair_drift = _max_abs_diff(cleaned["per_dim_mean"], raw_kept["per_dim_mean"])
    return {
        "var_change_pct": d_var,
        "var_repair_pct": repair_var,
        "var_selection_pct": d_var - repair_var if math.isfinite(d_var) and math.isfinite(repair_var) else float("nan"),
        "mean_drift": mean_drift,
        "mean_drift_repair": repair_drift,
        "mean_drift_normalized": mean_drift / raw_all["std"] if raw_all["std"] else float("nan"),
        "jump_ratio_before": raw_all["max_delta"] / raw_all["std"] if raw_all["std"] else float("nan"),
        "jump_ratio_after": cleaned["max_delta"] / cleaned["std"] if cleaned["std"] else float("nan"),
        "frames_dropped": raw_all["frames"] - raw_kept["frames"],
        "frames_trimmed": raw_kept["frames"] - cleaned["frames"],
        "sse_dropped_pct": (1 - raw_kept["sse"] / raw_all["sse"]) * 100 if raw_all["sse"] else float("nan"),
    }


def _dropped_sse_share(cmp: dict[str, Any]) -> float:
    """被剔除集带走的离差平方和占比。

    直接累加 §8.3 表里的逐集贡献，保证与明细表同源；另算一个「1 - 保留/总量」
    会因组间项口径不同差出 0.05 个百分点，两个数并排出现会被当成矛盾。
    """
    values = [row["sse_share"] for row in cmp["dropped"] if math.isfinite(row["sse_share"])]
    return float(sum(values)) if values else float("nan")


def _max_abs_diff(a: list[float], b: list[float]) -> float:
    pairs = [(abs(x - y), x, y) for x, y in zip(a, b) if math.isfinite(x) and math.isfinite(y)]
    return max(pairs)[0] if pairs else float("nan")


# ---------------------------------------------------------------------- 渲染
def _num(value: float | int | None, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, int):
        return f"{value:,}"
    if not math.isfinite(value):
        return "—"
    if value == 0:
        return "0"
    magnitude = abs(value)
    if magnitude >= 1000:
        return f"{value:,.0f}"
    if magnitude >= 1:
        return f"{value:.{digits}f}"
    if magnitude >= 0.001:
        return f"{value:.4f}"
    return f"{value:.2e}"


def _pct_text(value: float, digits: int = 1) -> str:
    return "—" if not math.isfinite(value) else f"{value:+.{digits}f}%"



def _section_args(section_id: str, title: str) -> str:
    """非默认分节时把参数写进「复现」行，否则照抄命令会覆盖上一组结果。"""
    if section_id == DEFAULT_SECTION_ID and title == DEFAULT_TITLE:
        return ""
    safe_title = title.replace('"', '\\"')
    return f' --section-id {section_id} --section-title "{safe_title}"'


def render_section(
    cmp: dict[str, Any],
    *,
    section_id: str = DEFAULT_SECTION_ID,
    title: str = DEFAULT_TITLE,
) -> str:
    """把对比结果渲染成一段 Markdown（可整体注入 docs/EVAL.md）。

    Args:
        section_id: 分节标记名；同一文档多组对比时用不同 id，互不覆盖。
        title: 一级标题（含章节号），由调用方决定放在文档哪一节。
    """
    raw_all, raw_kept, cleaned = cmp["raw_all"], cmp["raw_kept"], cmp["cleaned"]
    metrics = cmp["metrics"]
    raw_name = Path(cmp["raw_path"]).name
    cleaned_name = Path(cmp["cleaned_path"]).name
    begin, end = _markers(section_id)
    match = re.search(r"##\s*(\d+)\.", title)
    sec = match.group(1) if match else "9"

    lines = [
        begin,
        title,
        "",
        f"> 原始：`{raw_name}`（{raw_all['episodes']} ep / {_num(raw_all['frames'])} 帧）"
        f" → 清洗：`{cleaned_name}`（{cleaned['episodes']} ep / {_num(cleaned['frames'])} 帧）",
        "> 中间基线：同一份原始数据，只取清洗后保留的 episode —— 用来把"
        "「剔除」和「修复」两种效应拆开。",
        f"> 跳变阈值：`|Δ|∞ > {_num(cmp['threshold'])}`（`action_jump.abs_floor` 按 fps 折算）；"
        "口径：逐维总体方差（ddof=0）跨维平均。",
        f"> 复现：`python tools/analyze_action_distribution.py --raw {raw_name} "
        f"--cleaned <cleaned_dataset> --report <quality_report.json>"
        f" --update-eval docs/EVAL.md{_section_args(section_id, title)}`",
        "",
        f"### {sec}.1 总量指标",
        "",
        "| 指标 | 原始（全部） | 原始（仅保留集） | 清洗后 | 总变化 |",
        "|---|---|---|---|---|",
        f"| episode 数 | {raw_all['episodes']} | {raw_kept['episodes']} | {cleaned['episodes']} "
        f"| {cleaned['episodes'] - raw_all['episodes']:+d} |",
        f"| 帧数 | {_num(raw_all['frames'])} | {_num(raw_kept['frames'])} | {_num(cleaned['frames'])} "
        f"| {_pct_text(_pct(cleaned['frames'], raw_all['frames']))} |",
        f"| 动作维度 | {raw_all['dim']} | {raw_kept['dim']} | {cleaned['dim']} | — |",
        f"| **逐维方差均值** | {_num(raw_all['var'])} | {_num(raw_kept['var'])} | {_num(cleaned['var'])} "
        f"| **{_pct_text(metrics['var_change_pct'])}** |",
        f"| 动作标准差 | {_num(raw_all['std'])} | {_num(raw_kept['std'])} | {_num(cleaned['std'])} "
        f"| {_pct_text(_pct(cleaned['std'], raw_all['std']))} |",
        f"| 离差平方和（SSE） | {_num(raw_all['sse'])} | {_num(raw_kept['sse'])} | {_num(cleaned['sse'])} "
        f"| {_pct_text(_pct(cleaned['sse'], raw_all['sse']))} |",
        f"| 最大跳变 max\\|Δ\\|∞ | {_num(raw_all['max_delta'])} (ep{raw_all['max_delta_episode']}) "
        f"| {_num(raw_kept['max_delta'])} | {_num(cleaned['max_delta'])} "
        f"| {_pct_text(_pct(cleaned['max_delta'], raw_all['max_delta']))} |",
        f"| 每集最大跳变中位数 | {_num(raw_all['delta_median'])} | {_num(raw_kept['delta_median'])} "
        f"| {_num(cleaned['delta_median'])} | {_pct_text(_pct(cleaned['delta_median'], raw_all['delta_median']))} |",
        f"| 每集最大跳变 P90 | {_num(raw_all['delta_p90'])} | {_num(raw_kept['delta_p90'])} "
        f"| {_num(cleaned['delta_p90'])} | {_pct_text(_pct(cleaned['delta_p90'], raw_all['delta_p90']))} |",
        f"| 超阈帧数 | {_num(raw_all['over_threshold'])} | {_num(raw_kept['over_threshold'])} "
        f"| {_num(cleaned['over_threshold'])} | {_num(cleaned['over_threshold'] - raw_all['over_threshold'])} |",
        f"| 均值（跨维平均） | {_num(raw_all['mean'], 4)} | {_num(raw_kept['mean'], 4)} "
        f"| {_num(cleaned['mean'], 4)} | {_num(cleaned['mean'] - raw_all['mean'], 4)} |",
        f"| 归一化跳变 max\\|Δ\\|/std | {_num(metrics['jump_ratio_before'])} "
        f"| {_num(cleaned['max_delta'] / raw_kept['std'] if raw_kept['std'] else float('nan'))} "
        f"| {_num(metrics['jump_ratio_after'])} "
        f"| {_pct_text(_pct(metrics['jump_ratio_after'], metrics['jump_ratio_before']))} |",
        "",
        f"### {sec}.2 变化归因：剔除 vs 修复",
        "",
        "删样本一定会改变方差，所以「方差下降 X%」必须先拆开，否则等于把删数据说成提纯。"
        "SSE 可跨 episode 相加，是这里唯一能做加法分解的量：",
        "",
        "| 效应 | 定义 | 方差变化 | 说明 |",
        "|---|---|---|---|",
        f"| 选择（剔除） | 原始全部 → 原始仅保留集 | {_pct_text(metrics['var_selection_pct'])} | "
        f"剔除 {len(cmp['dropped'])} 个 episode、{_num(metrics['frames_dropped'])} 帧"
        f"（占帧数 {_num(metrics['frames_dropped'] / raw_all['frames'] * 100 if raw_all['frames'] else float('nan'))}%），"
        f"带走 {_num(_dropped_sse_share(cmp))}% 的离差平方和 |",
        f"| 修复 | 原始仅保留集 → 清洗后 | {_pct_text(metrics['var_repair_pct'])} | "
        f"边界裁帧 {_num(metrics['frames_trimmed'])} 帧、时间戳重建；不改动作幅值 |",
        f"| **合计** | 原始全部 → 清洗后 | **{_pct_text(metrics['var_change_pct'])}** | "
        "两项之和（方差对样本构成非线性，故为近似） |",
        "",
        f"均值漂移：逐维均值差的最大值 {_num(metrics['mean_drift'], 5)}"
        f"（按原始 std 归一化 {_num(metrics['mean_drift_normalized'], 4)}），"
        f"其中由裁剪/重建贡献 {_num(metrics['mean_drift_repair'], 5)}。",
        "",
        f"### {sec}.3 被剔除 episode 的尾部贡献",
        "",
        "| episode | 帧数 | 帧数占比 | 方差 | max\\|Δ\\|∞ | SSE 占比 | 剔除原因 |",
        "|---|---|---|---|---|---|---|",
    ]

    for row in cmp["dropped"]:
        lines.append(
            f"| {row['episode_index']} | {_num(row['frames'])} | {_num(row['frame_share'])}% "
            f"| {_num(row['var'])} | {_num(row['max_delta'])} | {_num(row['sse_share'])}% "
            f"| {row['reason']} |"
        )

    frames_share = sum(r["frame_share"] for r in cmp["dropped"] if math.isfinite(r["frame_share"]))
    sse_share = sum(r["sse_share"] for r in cmp["dropped"] if math.isfinite(r["sse_share"]))
    lines += [
        f"| **合计** | {_num(metrics['frames_dropped'])} | {_num(frames_share)}% | — | — "
        f"| **{_num(sse_share)}%** | — |",
        "",
        f"### {sec}.4 结论",
        "",
    ]
    lines += _conclusion(cmp, sec)
    lines += [
        "",
        f"### {sec}.5 边界（这段不能删）",
        "",
        "- **没有训练模型。** 本节给的是代理指标：异常跳变帧减少、归一化跳变幅度收窄、"
        "对应梯度被放大的通道变少。这不等于「精度提升」，也不等于「loss 曲线更平滑」；"
        "要那条结论必须做清洗前后的训练对照（同 seed、同 step、同评测集）。",
        "- **方差下降不等于数据变好。** 剔除高方差 episode 必然降低整体方差——"
        "若这些 episode 是真实的高动态任务（而非碰撞/断连），降方差反而是信息损失。"
        f"判断依据只能是 §{sec}.3 的剔除原因，而不是方差这个数字本身。",
        "- 逐维方差对量纲敏感；跨机器人/跨任务混采的数据集不能直接和本表比较。",
        "- 跳变阈值取配置 `abs_floor`（按 fps 折算），不做量纲自适应：像素动作"
        "（如 pusht 的 0–512）上该阈值没有物理意义，重跑时用 `--jump-threshold` 显式指定。",
        "- 修复阶段只裁剪边界帧与重建时间戳，不动动作幅值，因此「修复效应」理论上应接近 0；"
        "若它明显非 0，说明清洗改写了动作内容，需要单独解释。",
    ]
    if cleaned.get("stats_action_std") is not None:
        lines += [
            "",
            f"> 互校：`stats.json` 里 action 各维 std 的均值为 {_num(cleaned['stats_action_std'])}，"
            f"本脚本逐维 std 实算均值 {_num(cleaned['std_per_dim_mean'])}"
            f"（差 {_pct_text(_pct(cleaned['std_per_dim_mean'], cleaned['stats_action_std']))}）。"
            "两者应一致；不一致说明 stats 未随清洗重算，训练侧的归一化会用错分布。",
        ]
    lines += ["", end]
    return "\n".join(lines)


def _conclusion(cmp: dict[str, Any], sec: str) -> list[str]:
    """数据驱动地生成结论：只陈述量出来的事实，下降/上升都照写。"""
    raw_all, cleaned = cmp["raw_all"], cmp["cleaned"]
    metrics = cmp["metrics"]
    out: list[str] = []

    var_pct = metrics["var_change_pct"]
    if math.isfinite(var_pct) and var_pct < 0:
        out.append(
            f"- 剔除 {len(cmp['dropped'])} 个 episode（{_num(metrics['frames_dropped'])} 帧）后，"
            f"整体动作方差由 {_num(raw_all['var'])} 降至 {_num(cleaned['var'])}，**{_pct_text(var_pct)}**；"
            f"其中选择效应 {_pct_text(metrics['var_selection_pct'])}、"
            f"修复效应 {_pct_text(metrics['var_repair_pct'])}"
            f"（两项之和与合计略有出入，方差对样本构成非线性，见 §{sec}.2）。"
        )
    elif math.isfinite(var_pct):
        out.append(
            f"- 整体动作方差为 **{_pct_text(var_pct)}**（{_num(raw_all['var'])} → {_num(cleaned['var'])}）："
            "被剔除的 episode 并不是高方差来源，说明本轮清洗的目标是「去掉不可用样本」，"
            "而不是「压缩动作动态范围」。这与剔除原因（缺失模态 / 重复 / NaN）一致。"
        )

    before, after = metrics["jump_ratio_before"], metrics["jump_ratio_after"]
    if math.isfinite(before) and math.isfinite(after):
        change = _pct(after, before)
        trend = f"收窄到 {_num(after)}（{_pct_text(change)}）" if change < 0 \
            else f"变为 {_num(after)}（{_pct_text(change)}，未收窄）"
        if cleaned["max_delta"] > 2 * cmp["threshold"]:
            # 只降了一点点但仍在阈值之上：极值所在的集没被剔除，说「降低」是误导
            tail = (f"清洗后单帧最大位移仍为 {_num(cleaned['max_delta'])}，"
                    f"超过阈值 {_num(cleaned['max_delta'] / cmp['threshold'])} 倍 —— "
                    "极值所在的 episode 不在本次剔除集合内（多为 review 未裁决，仍留在产物里）。")
        elif cleaned["max_delta"] < raw_all["max_delta"] - 1e-9:
            tail = (f"单帧最大位移从 {_num(raw_all['max_delta'])} 降到 "
                    f"{_num(cleaned['max_delta'])}，已回到阈值以内。")
        else:
            tail = f"单帧最大位移仍为 {_num(cleaned['max_delta'])}。"
        out.append(f"- 归一化跳变（max|Δ|/std）由 {_num(before)} {trend}；{tail}")

    over_before, over_after = raw_all["over_threshold"], cleaned["over_threshold"]
    out.append(
        f"- 超过 `|Δ|∞ > {_num(cmp['threshold'])}` 的帧：{_num(over_before)} → {_num(over_after)}"
        f"（涉及 {raw_all['over_episodes']} → {cleaned['over_episodes']} 个 episode）。"
    )

    drift = metrics["mean_drift_normalized"]
    if math.isfinite(drift):
        # Cohen's d 口径：<0.1 可忽略，<0.3 轻微，再大才值得提醒重算归一化统计
        if drift < 0.1:
            verdict = "可忽略，动作中心未被移动"
        elif drift < 0.3:
            verdict = "轻微，仅影响归一化常数，建议重算 stats.json 后再训练"
        else:
            verdict = "显著，训练侧归一化必须重算后才可用"
        out.append(
            f"- 均值漂移 {_num(metrics['mean_drift'], 5)}"
            f"（{_num(drift, 4)} 个 std，{verdict}）。"
        )
    return out


# ---------------------------------------------------------------------- 注入
def inject_into_eval(eval_path: Path, section: str, section_id: str = DEFAULT_SECTION_ID) -> None:
    """把章节写进 docs/EVAL.md：有标记就替换，没有就追加。可重复执行。"""
    begin, end = _markers(section_id)
    text = eval_path.read_text(encoding="utf-8") if eval_path.is_file() else ""
    if begin in text and end in text:
        head, rest = text.split(begin, 1)
        _, tail = rest.split(end, 1)
        new_text = head + section + tail
    else:
        body = text.rstrip()
        new_text = (body + "\n\n" if body else "") + section + "\n"
    eval_path.write_text(new_text, encoding="utf-8")


# ----------------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="对比清洗前后的动作方差 / 跳变峰值 / 均值漂移",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例：\n"
            "  python tools/analyze_action_distribution.py \\\n"
            "      --raw demo_dataset --cleaned outputs/cleaned_dataset \\\n"
            "      --report outputs/quality_report.json --update-eval docs/EVAL.md\n"
        ),
    )
    p.add_argument("--raw", required=True, help="原始数据集根目录")
    p.add_argument("--cleaned", required=True, help="清洗后数据集根目录")
    p.add_argument("--report", default=None, help="质检报告，用于取剔除原因与编号映射")
    p.add_argument("--jump-threshold", type=float, default=None,
                   help="跳变阈值（|Δ|∞ 超过即计数）；默认按配置 abs_floor 与 fps 折算")
    p.add_argument("--json-out", default=None, help="同时写出机器可读的 JSON 结果")
    p.add_argument("--update-eval", default=None, help="把章节写进该 Markdown（如 docs/EVAL.md）")
    p.add_argument("--section-id", default=DEFAULT_SECTION_ID,
                   help=f"注入用的分节标记名（默认 {DEFAULT_SECTION_ID}）；同文档多组对比时改成不同值")
    p.add_argument("--section-title", default=DEFAULT_TITLE,
                   help="一级标题（含章节号），如 '## 9. 复核回灌后的再测量'")
    p.add_argument("--quiet", action="store_true", help="不打印章节正文（只写文件）")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cmp = compare(
            args.raw,
            args.cleaned,
            report_path=args.report,
            threshold=args.jump_threshold,
        )
    except SystemExit as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return 3

    section = render_section(cmp, section_id=args.section_id, title=args.section_title)
    if not args.quiet:
        print(section)
    if args.update_eval:
        inject_into_eval(Path(args.update_eval), section, args.section_id)
        print(f"\n[完成] 已写入 {args.update_eval}", file=sys.stderr)
    if args.json_out:
        payload = {k: v for k, v in cmp.items() if k not in ("raw_all", "raw_kept", "cleaned")}
        payload["summary"] = {name: {k: v for k, v in cmp[name].items() if k != "per_episode"}
                             for name in ("raw_all", "raw_kept", "cleaned")}
        Path(args.json_out).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
