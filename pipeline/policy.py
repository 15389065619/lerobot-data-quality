# Copyright (c) 2026 wuliangzhong
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""处置策略层：把「质检结论」翻译成「业务动作」。

真实数据团队不会「有异常就删整个 episode」——那会白白丢掉大量可修复数据，
而且删数据是不可逆操作，必须有分级策略 + 人工复核：

    keep    数据可用，原样保留
    repair  可自动修复（时间轴重建 / 边界裁帧 / meta 索引重建 / stats 重算）
    review  需人工确认（如 action 跳变：可能是真实动作突变，也可能是标注错误）
    drop    不可用，剔除

复核结果可以通过 overrides CSV 回灌，覆盖自动判定——质检因此是流程的一环，而不是一次性脚本。
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .baseline import fingerprint
from .loader import LeRobotDatasetView
from .validator import RULE_MAP, EpisodeResult, ValidationResult

LOG = logging.getLogger(__name__)

DISPOSITIONS = ("keep", "repair", "review", "drop")

#: 可自动修复的检查项 -> 修复动作
REPAIR_ACTIONS = {
    "timestamp_not_monotonic": "timestamps",
    "meta_index_inconsistent": "index",
}

#: 硬损坏：只能整集剔除，修不了也不该为几帧冒险
HARD_DROP_CODES = {
    "nan_state", "nan_action", "blank_video", "video_missing", "episode_read_error",
    "static_episode", "duplicate_episode", "joint_limit_violation", "frame_count_abnormal",
}

#: 软规则只提示复核，永不自动剔除（即使被配置成 error）
SOFT_NEVER_DROP_CODES = {"state_jump", "state_wrap", "timestamp_step_deviation"}

#: 单条规则命中占比达到该阈值时视为「系统性疑似误判」，禁止批量剔除
SYSTEMIC_HIT_RATIO = 0.95

#: 整体剔除比例达到该阈值（且 episode 数足够）时视为系统性误判，全部降级复核
MASS_DROP_RATIO = 0.80
MASS_DROP_MIN_EPISODES = 10

REVIEW_QUEUE_FILENAME = "review_queue.csv"

_HEADERS = [
    "episode_index", "severity", "suggested_disposition", "reasons", "fingerprints",
    "recommended_action", "frame_count", "detail", "video", "window",
    "reviewer_disposition", "reviewer_note",
]


@dataclass
class Decision:
    episode_index: int
    disposition: str
    severity: str
    reasons: list[str]
    recommended_action: str
    repair: str | None = None
    trim: dict[str, int] | None = None
    #: repair="split" 时的分段计划：每段 length + 每相机 (chunk/file/from/to)
    split_plan: list[dict[str, Any]] | None = None
    #: repair="window" 时每个相机收窄后的 to_timestamp
    window_plan: dict[str, dict[str, float]] | None = None
    source: str = "rule"
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_index": self.episode_index,
            "disposition": self.disposition,
            "severity": self.severity,
            "reasons": self.reasons,
            "recommended_action": self.recommended_action,
            "repair": self.repair,
            "trim": self.trim,
            "split_plan": self.split_plan,
            "window_plan": self.window_plan,
            "source": self.source,
            "note": self.note,
        }


def decide(
    validation: ValidationResult,
    *,
    on_review: str = "keep",
    overrides: dict[int, dict[str, Any]] | None = None,
    dataset: LeRobotDatasetView | None = None,
) -> dict[int, Decision]:
    """按严重度 + 可修性给出处置建议，再套用人工复核结果。

    dataset 用于读取 meta 原始多行（同集多段 take）与真实帧号分布；
    不传时退化为逐集判定，无法生成 split/window 修复计划。
    """
    if on_review not in {"keep", "drop"}:
        raise ValueError(f"on_review 只能是 keep/drop，收到 {on_review}")
    decisions: dict[int, Decision] = {}
    for ep, res in validation.results.items():
        decisions[ep] = _decide_one(res, validation, on_review, dataset)

    # 熔断：一条规则命中近乎全部 episode 时，绝大多数情况是元数据/定位问题而非数据
    # 真的全坏（behavior1k 实测：meta 用全局 episode ID，导致视频与任务定位全部落空，
    # 表面是 200/200 error）。这种情况删数据不可逆，必须先降级为人工复核。
    _apply_systemic_guard(decisions)
    _apply_mass_drop_guard(decisions)

    for ep, override in (overrides or {}).items():
        if ep not in decisions:
            LOG.warning("overrides 中的 ep %s 不在本次处理范围内，已忽略", ep)
            continue
        disp = str(override.get("disposition") or "").strip().lower()
        if disp not in DISPOSITIONS:
            LOG.warning("ep %s 的 override disposition=%r 非法，已忽略", ep, disp)
            continue
        d = decisions[ep]
        d.disposition = disp
        d.source = "override"
        d.note = str(override.get("note") or "").strip()
        if disp != "repair":
            d.repair = None
            d.trim = None
            d.split_plan = None
            d.window_plan = None
        elif d.repair is None:
            d.repair = "timestamps" if "timestamp_not_monotonic" in d.reasons else "index"
        d.recommended_action = _action_text(d)

    counts = {disp: sum(1 for d in decisions.values() if d.disposition == disp) for disp in DISPOSITIONS}
    LOG.info("处置决策：%s", counts)
    return decisions


def _nan_all_trimmable(res: EpisodeResult) -> bool:
    """trim 分支的前置校验：所有 nan 坏帧都必须落在边界（可裁剪）。

    trim_plan() 只累计 trimmable=True 项的 head/tail，不检查其余项——若中段
    还混着裁不掉的 NaN 帧而放行 trim，cleaner 按首尾切片后中段 NaN 帧会原样
    进入清洗产物（训练 loss 直接 NaN）。这种情况按硬损坏处理，宁可不救。
    """
    for code in ("nan_state", "nan_action"):
        if code in res.reasons and not (res.details.get(code) or {}).get("trimmable"):
            return False
    return True


def _decide_one(
    res: EpisodeResult,
    validation: ValidationResult,
    on_review: str,
    dataset: LeRobotDatasetView | None = None,
) -> Decision:
    severity = validation.severity_of(res)
    if not res.reasons:
        return Decision(res.episode_index, "keep", severity, [], "保留")

    # 1) 边界坏帧优先裁剪（最常见、最值得救的情况）
    trim = res.trim_plan()
    if trim and set(res.reasons) <= {"nan_state", "nan_action"} and _nan_all_trimmable(res):
        d = Decision(res.episode_index, "repair", severity, list(res.reasons),
                     f"裁掉首{trim['head_frames']}帧/尾{trim['tail_frames']}帧后保留",
                     repair="trim", trim=trim)
        return d

    # 2) 硬损坏：不可修复，只能剔除。warning 级的（如帧数过短）仍按人工复核处理，
    #    否则 --frame_count min 这类统计阈值会把合法短任务批量删掉。
    hard = sorted(set(res.reasons) & HARD_DROP_CODES)
    if hard and severity == "error":
        return Decision(
            res.episode_index, "drop", severity, list(res.reasons),
            f"剔除（硬损坏：{','.join(hard)}）",
            note="硬损坏不可修复",
        )

    # 3) 纯结构损坏：优先按 meta 分段拆分，其次收窄视频窗口，最后重建时间轴/索引。
    #    这几条都不丢数据，绝不能和硬损坏一样整集剔除。
    if dataset is not None:
        split = _split_plan(res, dataset)
        if split:
            return Decision(
                res.episode_index, "repair", severity, list(res.reasons),
                f"按 meta 拆分为 {len(split)} 段后保留",
                repair="split", split_plan=split,
            )
        window = _window_plan(res, dataset)
        if window:
            return Decision(
                res.episode_index, "repair", severity, list(res.reasons),
                "按真实帧数收窄视频窗口后保留",
                repair="window", window_plan=window,
            )

    if all(code in REPAIR_ACTIONS for code in res.reasons):
        repair = REPAIR_ACTIONS[res.reasons[0]]
        action = "用 frame_index/fps 重建时间轴" if repair == "timestamps" else "按真实行数重建 meta 索引"
        return Decision(res.episode_index, "repair", severity, list(res.reasons), action, repair=repair)

    # 4) info 级：仅记录，不影响可用性
    if severity == "info":
        return Decision(res.episode_index, "keep", severity, list(res.reasons), "保留（仅记录）")

    # 5) 软规则永远只交人工复核，即使被配置成 error 也不自动删
    soft = [c for c in res.reasons if c in SOFT_NEVER_DROP_CODES
            or (RULE_MAP.get(c) and RULE_MAP[c].tier == "soft")]
    if soft and all(c in SOFT_NEVER_DROP_CODES or (RULE_MAP.get(c) and RULE_MAP[c].tier == "soft")
                    for c in res.reasons):
        return Decision(
            res.episode_index, "review", severity, list(res.reasons),
            "人工复核后决定（软规则不自动剔除）",
            note=f"软规则 {','.join(soft)} 不参与自动剔除",
        )

    # 6) error 级：不可用
    if severity == "error":
        if soft:
            return Decision(
                res.episode_index, "review", severity, list(res.reasons),
                "人工复核后决定（软规则不自动剔除）",
                note=f"软规则 {','.join(soft)} 不参与自动剔除",
            )
        return Decision(res.episode_index, "drop", severity, list(res.reasons), "剔除")

    # 7) warning 级但没有自动修复方案 → 交人工复核
    disposition = "drop" if on_review == "drop" else "review"
    action = "剔除（--on-review drop）" if on_review == "drop" else "人工复核后决定"
    return Decision(res.episode_index, disposition, severity, list(res.reasons), action)


def _row_window(row: pd.Series, camera: str) -> tuple[float, float] | None:
    """取 meta 行里某相机的 (from, to)；字段缺失或非法时返回 None。"""
    from_ts = row.get(f"videos/{camera}/from_timestamp")
    to_ts = row.get(f"videos/{camera}/to_timestamp")
    if pd.isna(from_ts) or pd.isna(to_ts):
        return None
    try:
        return float(from_ts), float(to_ts)
    except (TypeError, ValueError):
        return None


def _split_plan(
    res: EpisodeResult, dataset: LeRobotDatasetView
) -> list[dict[str, Any]] | None:
    """meta 多行且各段 length 之和 == data 行数、帧号回退点吻合时，给出分段计划。

    分段顺序以 data 的行序为准（frame_index 回退点），不能假定与 meta 行序一致：
    v3 数据集里同一集的两段可能落在不同 parquet 文件，行序由文件名决定，
    而 meta 行的顺序可能相反。因此这里只按 length 把物理段与 meta 行配对。
    """
    try:
        segments = dataset.episode_segments(res.episode_index)
    except Exception:  # noqa: BLE001 - 定位失败就不拆，交给其它分支
        return None
    if segments is None or len(segments) < 2:
        return None
    lengths: list[int] = []
    for value in segments["length"]:
        try:
            length = int(round(float(value)))
        except (TypeError, ValueError):
            return None
        if length <= 0:
            return None
        lengths.append(length)
    if sum(lengths) != res.frame_count:
        return None
    physical = _physical_segment_lengths(dataset, res.episode_index)
    if physical is None or sorted(physical) != sorted(lengths):
        return None

    # 每个物理段按长度匹配一个尚未使用的 meta 行；同长度多行时按出现顺序消耗。
    unused = list(range(len(lengths)))
    fps = float(dataset.fps or 30.0)
    plan: list[dict[str, Any]] = []
    for length in physical:
        match = next((i for i in unused if lengths[i] == length), None)
        if match is None:
            return None
        unused.remove(match)
        row = segments.iloc[match]
        videos: dict[str, dict[str, Any]] = {}
        for cam in dataset.video_features:
            chunk = row.get(f"videos/{cam}/chunk_index")
            file_index = row.get(f"videos/{cam}/file_index")
            window = _row_window(row, cam)
            if pd.isna(chunk) or pd.isna(file_index) or window is None:
                return None
            from_ts, to_ts = window
            # 该段只有 length 帧；个别段的 meta 窗口比自身长（串到了相邻段），
            # 直接按 length 收窄，保证窗口跨度与数据严格一致。
            max_to = from_ts + length / fps
            if to_ts > max_to:
                to_ts = max_to
            videos[cam] = {
                "chunk_index": int(chunk),
                "file_index": int(file_index),
                "from": from_ts,
                "to": to_ts,
            }
        plan.append({"length": length, "videos": videos})
    return plan


def _physical_segment_lengths(
    dataset: LeRobotDatasetView, episode_index: int
) -> list[int] | None:
    """按 data 行序里的 frame_index 回退点切出物理段长度；每段必须是 0..n-1。

    必须用未排序的原始行序：多段 episode 的 frame_index 在每段都从 0 开始，
    排序会把两段交错，切出来的段就不再连续。
    """
    getter = getattr(dataset, "episode_frame_raw", None) or dataset.episode_frame
    try:
        ep = getter(episode_index)
    except Exception:  # noqa: BLE001 - 读不到就不拆
        return None
    if ep.empty or "frame_index" not in ep.columns:
        return None
    values = pd.to_numeric(ep["frame_index"], errors="coerce").to_numpy(dtype=float)
    if np.isnan(values).any():
        return None
    breaks = np.where(np.diff(values) <= 0)[0]
    bounds = [0] + [int(b) + 1 for b in breaks] + [len(values)]
    if len(bounds) < 3:
        return None
    lengths: list[int] = []
    for start, end in zip(bounds[:-1], bounds[1:]):
        seg = values[start:end].astype(np.int64)
        if not np.array_equal(seg, np.arange(len(seg))):
            return None
        lengths.append(int(len(seg)))
    return lengths if len(lengths) >= 2 else None


def _window_plan(
    res: EpisodeResult, dataset: LeRobotDatasetView
) -> dict[str, dict[str, float]] | None:
    """单段且视频窗口跨度大于数据行数时，把 to_timestamp 收窄到 from + length/fps。"""
    if not ({"video_span_mismatch", "frame_mismatch"} & set(res.reasons)):
        return None
    try:
        segments = dataset.episode_segments(res.episode_index)
    except Exception:  # noqa: BLE001
        return None
    if segments is None or len(segments) != 1:
        return None
    row = segments.iloc[0]
    fps = float(dataset.fps or 30.0)
    length = int(res.frame_count)
    tolerance = 1.0 / fps
    plan: dict[str, dict[str, float]] = {}
    for cam in dataset.video_features:
        window = _row_window(row, cam)
        if window is None:
            return None
        from_ts, to_ts = window
        if from_ts < 0 or to_ts <= from_ts:
            return None
        span = to_ts - from_ts
        if span <= length / fps + tolerance:
            continue
        plan[cam] = {"from": from_ts, "to": from_ts + length / fps}
    return plan or None


def _apply_systemic_guard(decisions: dict[int, Decision]) -> list[str]:
    """单规则近乎全量命中时，把「剔除」降级为「人工复核」。

    数据治理里 100% 命中几乎不可能是数据问题：更常见的是编号空间、定位方式、
    单位约定这类元数据层面的系统性错配。删数据不可逆，所以这里选择阻断而不是执行。
    """
    total = len(decisions)
    if total < 2:
        return []
    hits: dict[str, int] = {}
    for d in decisions.values():
        if d.disposition != "drop":
            continue
        for code in d.reasons:
            hits[code] = hits.get(code, 0) + 1

    flagged: list[str] = []
    for code, count in hits.items():
        share = count / total
        if share < SYSTEMIC_HIT_RATIO:
            continue
        flagged.append(code)
        for d in decisions.values():
            if d.disposition == "drop" and code in d.reasons:
                d.disposition = "review"
                d.source = "systemic-guard"
                d.note = (
                    f"规则 {code} 命中 {share:.0%} 的 episode，疑似系统性元数据/定位问题，"
                    "已阻止批量剔除，需人工确认后再处置"
                )
                d.recommended_action = "人工复核（疑似系统性误判，已阻止批量剔除）"
        LOG.warning(
            "熔断：规则 %s 命中 %.1f%% 的 episode（%d/%d），批量剔除已降级为人工复核",
            code, share * 100, count, total,
        )
    return flagged


def _apply_mass_drop_guard(decisions: dict[int, Decision]) -> list[int]:
    """整体剔除比例过高时，把全部 drop 降级为 review。

    单规则熔断只看「某一规则」，但多条规则各命中一部分时仍可能整体删掉 80%+ 的数据。
    对不可逆的删除操作，这种量级更可能是阈值/元数据系统性偏严，先阻断再让人确认。
    """
    total = len(decisions)
    if total < MASS_DROP_MIN_EPISODES:
        return []
    drops = [d for d in decisions.values() if d.disposition == "drop"]
    share = len(drops) / total
    if share < MASS_DROP_RATIO:
        return []
    for d in drops:
        d.disposition = "review"
        d.source = "mass-drop-guard"
        d.note = (
            f"本次将剔除 {len(drops)}/{total} 个 episode（{share:.0%}），"
            "疑似阈值系统性偏严，已全部降级为人工复核"
        )
        d.recommended_action = "人工复核（整体剔除比例过高，已阻止批量删除）"
    LOG.warning(
        "熔断：整体剔除比例 %.1f%%（%d/%d）达到阈值，已全部降级为人工复核",
        share * 100, len(drops), total,
    )
    return [d.episode_index for d in drops]


def systemic_decisions(decisions: dict[int, Decision]) -> list[int]:
    """被熔断保护下来的 episode 编号，供报告如实标注。"""
    return sorted(
        ep for ep, d in decisions.items() if d.source in {"systemic-guard", "mass-drop-guard"}
    )


def _action_text(d: Decision) -> str:
    if d.disposition == "keep":
        return "保留"
    if d.disposition == "drop":
        return "剔除"
    if d.disposition == "review":
        return "人工复核后决定"
    if d.repair == "trim" and d.trim:
        return f"裁掉首{d.trim.get('head_frames', 0)}帧/尾{d.trim.get('tail_frames', 0)}帧后保留"
    if d.repair == "split":
        return f"按 meta 拆分为 {len(d.split_plan or [])} 段后保留"
    if d.repair == "window":
        return "按真实帧数收窄视频窗口后保留"
    return "自动修复后保留"


# 人工复核
def write_review_queue(
    path: str | Path,
    validation: ValidationResult,
    decisions: dict[int, Decision],
    dataset: LeRobotDatasetView,
) -> Path:
    """导出复核队列 CSV：默认只列 review/drop，供数据同学填写 reviewer_disposition。"""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    dataset_name = dataset.root.name
    rows = []
    for ep in sorted(decisions):
        d = decisions[ep]
        if d.disposition == "keep":
            continue
        res = validation.results[ep]
        rows.append(
            {
                "episode_index": ep,
                "severity": d.severity,
                "suggested_disposition": d.disposition,
                "reasons": "|".join(d.reasons),
                # 确认误报后可复制到 `--accept-baseline` 或配置 baseline_whitelist
                "fingerprints": "|".join(
                    fingerprint(dataset_name, ep, code) for code in d.reasons
                ),
                "recommended_action": d.recommended_action,
                "frame_count": res.frame_count,
                "detail": _summarize_details(res),
                "video": _video_ref(dataset, ep),
                "window": _window_ref(dataset, ep),
                "reviewer_disposition": "",
                "reviewer_note": "",
            }
        )
    with out.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_HEADERS)
        writer.writeheader()
        writer.writerows(rows)
    LOG.info("复核队列已导出 %d 行 -> %s", len(rows), out)
    return out


def read_overrides(path: str | Path) -> dict[int, dict[str, Any]]:
    """读取人工复核结果。支持 reviewer_disposition / disposition 两种列名。"""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"overrides 文件不存在: {p}")
    out: dict[int, dict[str, Any]] = {}
    with p.open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            raw_ep = (row.get("episode_index") or "").strip()
            if not raw_ep:
                continue
            try:
                ep = int(float(raw_ep))
            except ValueError:
                LOG.warning("overrides 中 episode_index=%r 无法解析，跳过", raw_ep)
                continue
            disp = (row.get("reviewer_disposition") or row.get("disposition") or "").strip().lower()
            if not disp:
                continue
            out[ep] = {"disposition": disp, "note": (row.get("reviewer_note") or row.get("note") or "").strip()}
    LOG.info("读取 overrides %d 条 <- %s", len(out), p)
    return out


def summarize(decisions: dict[int, Decision]) -> dict[str, Any]:
    """报告用的处置统计。"""
    counts = {disp: 0 for disp in DISPOSITIONS}
    by_repair: dict[str, int] = {}
    for d in decisions.values():
        counts[d.disposition] = counts.get(d.disposition, 0) + 1
        if d.repair:
            by_repair[d.repair] = by_repair.get(d.repair, 0) + 1
    overridden = sum(1 for d in decisions.values() if d.source == "override")
    return {
        "counts": counts,
        "repair_breakdown": by_repair,
        "overridden": overridden,
        "kept_after_cleaning": counts["keep"] + counts["repair"] + counts["review"],
    }


# 小工具
def _summarize_details(res: EpisodeResult) -> str:
    parts: list[str] = []
    for code, detail in res.details.items():
        keys = ("nan_count", "inf_count", "jump_count", "max_delta", "direction",
                "duplicate_of", "bad_ratio", "meta_length", "violations")
        picked = {k: detail[k] for k in keys if k in detail}
        if picked:
            parts.append(f"{code}:" + json.dumps(picked, ensure_ascii=False, default=str))
        elif detail.get("note"):
            parts.append(f"{code}:{detail['note']}")
    return " | ".join(parts)[:500]


def _video_ref(dataset: LeRobotDatasetView, ep: int) -> str:
    cams = dataset.video_features
    if not cams:
        return ""
    loc = dataset.video_location(cams[0], ep)
    return str(loc[0]) if loc else ""


def _window_ref(dataset: LeRobotDatasetView, ep: int) -> str:
    cams = dataset.video_features
    if not cams:
        return ""
    loc = dataset.video_location(cams[0], ep)
    if not loc:
        return ""
    _, start, end = loc
    return f"{start:.2f}-{end:.2f}s" if end > start else ""
