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


def _episode_nunique(summary: pd.DataFrame | None) -> np.ndarray | None:
    """每集 index 的唯一值个数；列缺失或跨文件无法合并（NaN）时该集不判唯一性。"""
    if summary is None or "idx_nunique" not in summary.columns:
        return None
    return pd.to_numeric(summary["idx_nunique"], errors="coerce").to_numpy(dtype=float)


def audit_schema(
    *,
    features: dict[str, Any],
    splits: dict[str, Any] | None,
    total_episodes: int | None,
    data_columns: set[str] | frozenset[str] | None,
    data_summary: pd.DataFrame | None,
    info_total_episodes: int | None = None,
    info_total_frames: int | None = None,
    meta_episode_rows: int | None = None,
    data_rows: int | None = None,
    codebase_version: str | None = None,
    truncated: bool = False,
) -> list[dict[str, Any]]:
    """返回 Schema 违约项列表；每项都可定位到具体列，并给出后果。

    每项结构::

        {"column": "index", "rule": "...", "expected": "...", "actual": "...",
         "impact": "...", "severity": "error"|"warning"}

    ``codebase_version`` 决定 index 的契约：v2.x 每个 episode 一个 parquet，
    index 是 episode 内帧号（每集从 0 起属规范）；v3.0 才是全局绝对帧号。
    ``truncated`` 为真时（--max_episodes）跳过 info 计数对账，子集不可比。
    """
    out: list[dict[str, Any]] = []
    cols = set(data_columns or ())
    has = lambda name: (name in cols) or (name in features)  # noqa: E731

    summary = data_summary if data_summary is not None and len(data_summary) else None
    per_ep_index = summary is not None and "idx_min" in summary.columns
    is_v2 = str(codebase_version or "").startswith("v2")

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

    # ---------------------------------------------------------------- 2. index 语义
    # v2.1：每集一个 parquet，index 就是 episode 内帧号（0..rows-1）。
    #       用 v3 的「全局唯一」判据会把完全合规的 v2.1 数据判成 error 并置顶阻断。
    if per_ep_index and len(summary) > 1 and is_v2:
        rows = summary["rows"].to_numpy(dtype=np.int64)
        idx_min = summary["idx_min"].to_numpy(dtype=np.int64)
        idx_max = summary["idx_max"].to_numpy(dtype=np.int64)
        # 只要 episode 内连续、不重复即可：起点允许是 0（原生 v2.1 每集归零），
        # 也允许是从 v3 沿用的全局偏移（kuavo4 实测：ep1 是 131..471）。
        # 连续性用「取值范围长度 == 行数」判；但重复与断号会互相抵消（[0,1,1,3] 4 行、
        # 跨度 4 也能通过），所以再用唯一值个数兜一层，与 context.py 的判据保持一致。
        span_bad = (idx_max - idx_min + 1) != rows
        nunique = _episode_nunique(summary)
        dup_bad = (
            np.isfinite(nunique) & (nunique != rows)
            if nunique is not None
            else np.zeros(rows.shape, dtype=bool)
        )
        bad = np.flatnonzero(span_bad | dup_bad)
        if bad.size:
            sample = summary.index[bad[:5]].tolist()
            uniq_note = (
                f"，唯一值个数={nunique[bad[:5]].tolist()}" if nunique is not None else ""
            )
            out.append(
                {
                    "column": "index",
                    "rule": "v2.x 每集独立 parquet，index 必须在 episode 内连续且不重复",
                    "expected": "每个 episode 的 idx_max - idx_min + 1 == rows 且唯一值个数 == rows（起点可为 0 或全局偏移）",
                    "actual": (
                        f"{bad.size}/{len(summary)} 个 episode 不符（示例 episode "
                        f"{sample}，idx_min={idx_min[bad[:5]].tolist()}，"
                        f"idx_max={idx_max[bad[:5]].tolist()}，rows={rows[bad[:5]].tolist()}"
                        f"{uniq_note}）"
                    ),
                    "impact": "episode 内帧号断裂、重复或串位，按 frame_index/index 切窗会取错帧",
                    "severity": "error",
                }
            )
    # v3.0：index 是全局绝对帧号，必须 0..total_frames-1 各出现一次
    elif per_ep_index and len(summary) > 1:
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
        else:
            # 起点序列正确但仍可能有集内重复/断号互相抵消：唯一值个数 != 行数即为证据。
            nunique = _episode_nunique(summary)
            if nunique is not None:
                bad = np.flatnonzero(np.isfinite(nunique) & (nunique != rows))
                if bad.size:
                    sample = summary.index[bad[:5]].tolist()
                    out.append(
                        {
                            "column": "index",
                            "rule": "index 必须是全数据集唯一的绝对帧号（episode 内不得重复）",
                            "expected": "每个 episode 内 index 唯一值个数 == 行数",
                            "actual": (
                                f"{bad.size}/{len(summary)} 个 episode 内 index 有重复（示例 episode "
                                f"{sample}，rows={rows[bad[:5]].tolist()}，"
                                f"唯一值个数={nunique[bad[:5]].tolist()}）"
                            ),
                            "impact": "重复键会让官方 _build_index_mapping 塌缩，训练样本静默错位",
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

    # ------------------------------------------------- 4. info 与 meta/episodes 对账
    # 数据集可能被裁剪/截断过而 info.json 没同步。这类缺口不会让任何单条
    # episode 规则命中（缺失的 episode 根本没有数据可查），只能靠计数对账发现。
    if not truncated and info_total_episodes is not None and meta_episode_rows is not None:
        declared, actual = int(info_total_episodes), int(meta_episode_rows)
        if declared != actual:
            missing = declared - actual
            out.append(
                {
                    "column": "meta/info.json:total_episodes",
                    "rule": "info.total_episodes 必须等于 meta/episodes 的实际行数",
                    "expected": f"{declared}（info.json 声明）",
                    "actual": f"meta/episodes 实际 {actual} 行，差 {missing:+d}",
                    "impact": (
                        f"有 {abs(missing)} 个 episode 只存在于声明里：按 info 分配 episode "
                        "会越界或得到空集；本次质检也无法覆盖它们，报告里的 episode 总数"
                        "小于数据集自称的数量"
                    ),
                    "severity": "error" if abs(missing) / max(declared, 1) > 0.01 else "warning",
                }
            )

    # ------------------------------------------------- 5. info 与 data 实际行数对账
    if not truncated and info_total_frames is not None and data_rows is not None:
        declared, actual = int(info_total_frames), int(data_rows)
        if declared != actual:
            missing = declared - actual
            out.append(
                {
                    "column": "meta/info.json:total_frames",
                    "rule": "info.total_frames 必须等于 data/*.parquet 的实际行数",
                    "expected": f"{declared}（info.json 声明）",
                    "actual": f"data 实际 {actual} 行，差 {missing:+d}",
                    "impact": (
                        "下游按 total_frames 分配全局 index 或切分训练集时会在末尾越界；"
                        "stats.json 也多半是按声明值算的，与实测分布不对应"
                    ),
                    "severity": "error" if abs(missing) / max(declared, 1) > 0.01 else "warning",
                }
            )

    return out
