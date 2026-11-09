"""LeRobot v3 磁盘布局的 Schema 契约审计。

只读元数据与两列极值，不加载整表，因此对几十 GB 的数据集也是秒级。

存在的原因：很多"数据集能跑起来"并不等于"Schema 合规"。像 `index` 列退化成
episode 内帧号这种问题，全量读取不报错、吞吐正常，只在按 episode 过滤或开
`delta_timestamps` 时静默取错帧。这类问题必须在报告里指名到列，而不是笼统
说"数据质量差"。
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

#: LeRobot v3.0 每帧必备的标量列
REQUIRED_FRAME_COLUMNS = ("index", "episode_index", "frame_index", "task_index", "timestamp")


def audit_schema(
    *,
    features: dict[str, Any],
    splits: dict[str, Any] | None,
    total_episodes: int | None,
    data_columns: set[str] | frozenset[str] | None,
    data_summary: pd.DataFrame | None,
) -> list[dict[str, Any]]:
    """返回 Schema 违约项列表；每项都可定位到具体列，并给出后果。

    每项结构::

        {"column": "index", "rule": "...", "expected": "...", "actual": "...",
         "impact": "...", "severity": "error"|"warning"}
    """
    out: list[dict[str, Any]] = []
    cols = set(data_columns or ())
    has = lambda name: (name in cols) or (name in features)  # noqa: E731

    summary = data_summary if data_summary is not None and len(data_summary) else None
    per_ep_index = summary is not None and "idx_min" in summary.columns

    # ---------------------------------------------------------------- 1. frame_index
    if not has("frame_index"):
        detail = {
            "column": "frame_index",
            "rule": f"v3.0 每帧必备列 {list(REQUIRED_FRAME_COLUMNS)} 中的 frame_index 缺失",
            "expected": "parquet 与 meta/info.json.features 均声明 frame_index（episode 内帧号，每集从 0 起）",
            "actual": "parquet 无该列，info.json.features 也未声明",
            "impact": "episode 内帧号只能从 index 反推；任何依赖 frame_index 的切窗/拼接逻辑需自行构造",
            "severity": "warning",
        }
        if per_ep_index and bool((summary["idx_min"] == 0).all()):
            detail["actual"] += "；而 index 在每集都从 0 重新开始，说明 index 承担了 frame_index 的语义"
        out.append(detail)

    # ---------------------------------------------------------------- 2. index 全局唯一
    if per_ep_index and len(summary) > 1:
        rows = summary["rows"].to_numpy(dtype=np.int64)
        expect_from = np.concatenate([[0], np.cumsum(rows)[:-1]])
        actual_from = summary["idx_min"].to_numpy(dtype=np.int64)
        restart = bool((actual_from == 0).all())
        if restart or not np.array_equal(actual_from, expect_from):
            total = int(rows.sum())
            span = int(summary["idx_max"].max()) + 1
            out.append(
                {
                    "column": "index",
                    "rule": "index 必须是全数据集唯一的绝对帧号（0..total_frames-1，各出现一次）",
                    "expected": f"每集 index 起点 = 前序 episode 累计行数，即 {expect_from[:3].tolist()} … {int(expect_from[-1])}",
                    "actual": (
                        f"{len(summary)} 个 episode 的 index 起点全为 0，取值范围仅 0..{span - 1}，"
                        f"却承载 {total} 行" if restart else
                        f"index 起点与前序累计行数不符：实际 {actual_from[:3].tolist()}，应为 {expect_from[:3].tolist()}"
                    ),
                    "impact": (
                        "官方 DatasetReader 把 index 当绝对帧号做两件事："
                        "(1) _build_index_mapping 用 dict(zip(index, 位置)) 建映射，"
                        f"重复键会让 {total} 行塌成 {span} 项；(2) delta_timestamps 的窗口用 "
                        "delta_window(index, ±k, dataset_from_index, dataset_to_index) 判越界，"
                        "index 远小于区间起点会让整个窗口被判越界并压成同一帧。"
                        "两种错误都不抛异常，表现为吞吐正常但训练样本错位。"
                    ),
                    "severity": "error",
                }
            )

    # ---------------------------------------------------------------- 3. splits
    if isinstance(splits, dict) and total_episodes:
        for name, spec in splits.items():
            try:
                start, end = (int(x) for x in str(spec).split(":"))
            except (ValueError, AttributeError):
                continue
            if end - start != int(total_episodes):
                out.append(
                    {
                        "column": "meta/info.json:splits",
                        "rule": "split 区间长度必须等于 total_episodes",
                        "expected": f"{name}: 0:{total_episodes}",
                        "actual": f"{name}: {spec}（长度 {end - start}，total_episodes={total_episodes}）",
                        "impact": "按 split 加载会以为 episode 数就是区间长度，越界访问或空集",
                        "severity": "warning",
                    }
                )

    return out
