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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .baseline import fingerprint
from .config import severity_of
from .loader import LeRobotDatasetView
from .validator import RULE_MAP, EpisodeResult, ValidationResult

LOG = logging.getLogger(__name__)

DISPOSITIONS = ("keep", "repair", "review", "drop")

#: 可自动修复的检查项 -> 修复动作
REPAIR_ACTIONS = {
    "timestamp_not_monotonic": "timestamps",
    "meta_index_inconsistent": "index",
}

#: 单条规则命中占比达到该阈值时视为「系统性疑似误判」，禁止批量剔除
SYSTEMIC_HIT_RATIO = 0.95

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
            "source": self.source,
            "note": self.note,
        }


def decide(
    validation: ValidationResult,
    *,
    on_review: str = "keep",
    overrides: dict[int, dict[str, Any]] | None = None,
) -> dict[int, Decision]:
    """按严重度 + 可修性给出处置建议，再套用人工复核结果。"""
    if on_review not in {"keep", "drop"}:
        raise ValueError(f"on_review 只能是 keep/drop，收到 {on_review}")
    decisions: dict[int, Decision] = {}
    for ep, res in validation.results.items():
        decisions[ep] = _decide_one(res, validation, on_review)

    # 熔断：一条规则命中近乎全部 episode 时，绝大多数情况是元数据/定位问题而非数据
    # 真的全坏（behavior1k 实测：meta 用全局 episode ID，导致视频与任务定位全部落空，
    # 表面是 200/200 error）。这种情况删数据不可逆，必须先降级为人工复核。
    systemic = _apply_systemic_guard(decisions)

    for ep, override in (overrides or {}).items():
        target = overrides.get(ep)
        if target is None:
            continue
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
        elif d.repair is None:
            d.repair = "timestamps" if "timestamp_not_monotonic" in d.reasons else "index"
        d.recommended_action = _action_text(d)

    counts = {disp: sum(1 for d in decisions.values() if d.disposition == disp) for disp in DISPOSITIONS}
    LOG.info("处置决策：%s", counts)
    return decisions


def _decide_one(res: EpisodeResult, validation: ValidationResult, on_review: str) -> Decision:
    severity = validation.severity_of(res)
    if not res.reasons:
        return Decision(res.episode_index, "keep", severity, [], "保留")

    # 1) 边界坏帧优先裁剪（最常见、最值得救的情况）
    trim = res.trim_plan()
    if trim and set(res.reasons) <= {"nan_state", "nan_action"}:
        d = Decision(res.episode_index, "repair", severity, list(res.reasons),
                     f"裁掉首{trim['head_frames']}帧/尾{trim['tail_frames']}帧后保留",
                     repair="trim", trim=trim)
        return d

    # 2) 纯可修复项
    if all(code in REPAIR_ACTIONS for code in res.reasons):
        repair = REPAIR_ACTIONS[res.reasons[0]]
        action = "用 frame_index/fps 重建时间轴" if repair == "timestamps" else "按真实行数重建 meta 索引"
        return Decision(res.episode_index, "repair", severity, list(res.reasons), action, repair=repair)

    # 3) error 级：不可用
    if severity == "error":
        return Decision(res.episode_index, "drop", severity, list(res.reasons), "剔除")

    # 4) info 级：仅记录，不影响可用性
    if severity == "info":
        return Decision(res.episode_index, "keep", severity, list(res.reasons), "保留（仅记录）")

    # 5) warning 级但没有自动修复方案 → 交人工复核
    disposition = "drop" if on_review == "drop" else "review"
    action = "剔除（--on-review drop）" if on_review == "drop" else "人工复核后决定"
    return Decision(res.episode_index, disposition, severity, list(res.reasons), action)


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


def systemic_decisions(decisions: dict[int, Decision]) -> list[int]:
    """被熔断保护下来的 episode 编号，供报告如实标注。"""
    return sorted(ep for ep, d in decisions.items() if d.source == "systemic-guard")


def _action_text(d: Decision) -> str:
    if d.disposition == "keep":
        return "保留"
    if d.disposition == "drop":
        return "剔除"
    if d.disposition == "review":
        return "人工复核后决定"
    if d.repair == "trim" and d.trim:
        return f"裁掉首{d.trim.get('head_frames', 0)}帧/尾{d.trim.get('tail_frames', 0)}帧后保留"
    return "自动修复后保留"


# ------------------------------------------------------------------ 人工复核
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


# --------------------------------------------------------------------- 小工具
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


def rule_label(code: str) -> str:
    rule = RULE_MAP.get(code)
    return rule.label if rule else code
