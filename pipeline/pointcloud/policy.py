"""点云处置策略：decide / write_review_queue / read_overrides。"""

from __future__ import annotations

import csv
import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import cfg_get, severity_of
from .types import (
    RULE_MAP,
    REPAIRABLE,
    RULE_ORDER,
    FramesKey,
    FrameResult,
    PointCloudDecision,
    PointCloudValidation,
)

LOG = logging.getLogger(__name__)


def decide(
    validation: PointCloudValidation, *, on_review: str = "keep", overrides: dict[FramesKey, str] | None = None
) -> dict[FramesKey, PointCloudDecision]:
    """逐帧处置：keep / repair / review / drop。

    - 可修复项（NaN/离群/重复/越界/颜色）且坏点比例可控 → repair
    - 有效点过少或几何退化 → drop（不可凭空生成几何）
    - 时间跳变或无法自动修 → review（不改数据，只进复核队列）
    """
    decisions: dict[FramesKey, PointCloudDecision] = {}
    for key, res in validation.results.items():
        decisions[key] = _decide_one(key, res, validation, on_review)

    for key, override in (overrides or {}).items():
        if key not in decisions:
            continue
        disp = str(override).strip().lower()
        if disp not in {"keep", "repair", "review", "drop"}:
            continue
        d = decisions[key]
        d.disposition = disp
        d.note = "人工复核覆盖"
    counts = {d: sum(1 for x in decisions.values() if x.disposition == d)
              for d in ("keep", "repair", "review", "drop")}
    LOG.info("点云处置：%s", counts)
    return decisions


def _decide_one(key: FramesKey, res: FrameResult, validation: PointCloudValidation, on_review: str) -> PointCloudDecision:
    severity = validation.severity_of(res)
    if not res.reasons:
        return PointCloudDecision(key, "keep", severity, [])
    if "pc_empty_or_sparse" in res.reasons or "pc_degenerate" in res.reasons:
        return PointCloudDecision(key, "drop", severity, list(res.reasons),
                                  note="有效点不足/几何退化，无法修复")
    actions = [RULE_MAP[c].repair for c in res.reasons if RULE_MAP[c].repair]
    actions = [a for a in actions if a]
    non_repairable = [c for c in res.reasons if c not in REPAIRABLE]
    if non_repairable:
        disposition = "drop" if (severity == "error" and on_review == "drop") else "review"
        return PointCloudDecision(key, disposition, severity, list(res.reasons),
                                  note="需人工确认，暂不改点云")
    ratio = max(float(res.metrics.get("non_finite_ratio", 0.0)),
                float(res.metrics.get("outlier_ratio", 0.0))
                + float(res.metrics.get("duplicate_ratio", 0.0)))
    max_repair = float(cfg_get(validation.config, "pointcloud.max_repair_ratio", 0.35))
    if ratio > max_repair:
        return PointCloudDecision(key, "drop", severity, list(res.reasons),
                                  note=f"坏点比例 {ratio:.0%} 超过 {max_repair:.0%}，无法可靠修复")
    return PointCloudDecision(key, "repair", severity, list(res.reasons),
                              repair_actions=_dedupe_actions(actions))


def _dedupe_actions(actions: Iterable[str]) -> list[str]:
    out: list[str] = []
    for action in actions:
        if action not in out:
            out.append(action)
    return out


def write_review_queue(
    path: str | Path, validation: PointCloudValidation, decisions: dict[FramesKey, PointCloudDecision]
) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for key in sorted(decisions):
        d = decisions[key]
        if d.disposition != "review":
            continue
        res = validation.results[key]
        rows.append({
            "episode_index": key[0],
            "frame_index": key[1],
            "severity": d.severity,
            "reasons": "|".join(d.reasons),
            "suggested_disposition": d.disposition,
            "suggested_action": d.note or "人工回放确认",
            "metrics": json.dumps(res.metrics, ensure_ascii=False, default=str),
            "reviewer_disposition": "",
            "reviewer_note": "",
        })
    headers = ["episode_index", "frame_index", "severity", "reasons", "suggested_disposition",
               "suggested_action", "metrics", "reviewer_disposition", "reviewer_note"]
    with out.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    LOG.info("点云复核队列 %d 行 -> %s", len(rows), out)
    return out


def read_overrides(path: str | Path) -> dict[FramesKey, str]:
    """读取点云复核结果：(episode_index, frame_index) -> disposition。"""
    p = Path(path).expanduser()
    if not p.is_file():
        raise FileNotFoundError(f"复核文件不存在: {p}")
    out: dict[FramesKey, str] = {}
    with p.open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            disp = (row.get("reviewer_disposition") or "").strip().lower()
            if disp not in {"keep", "repair", "review", "drop"}:
                continue
            try:
                key = (int(row["episode_index"]), int(row["frame_index"]))
            except (KeyError, TypeError, ValueError):
                continue
            out[key] = disp
    LOG.info("读取点云人工复核 %d 条 <- %s", len(out), p)
    return out
