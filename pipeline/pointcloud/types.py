"""点云模块的数据类型定义：规则表、帧结论、验证结果、处置决策、清洗结果。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..config import severity_of

FramesKey = tuple[int, int]

REPORT_FILENAME = "pointcloud_report.json"
REVIEW_FILENAME = "pointcloud_review_queue.csv"
NOTES_FILENAME = "pointcloud_cleaning_notes.json"
CLEANED_DIRNAME = "pointcloud_cleaned"

SEVERITY_ORDER = {"info": 0, "warning": 1, "error": 2}


# --------------------------------------------------------------------- 规则表
@dataclass(frozen=True)
class Rule:
    code: str
    label: str
    description: str
    repair: str | None = None


RULES: tuple[Rule, ...] = (
    Rule("pc_non_finite", "点云含 NaN/Inf", "深度反投影或采样器输出异常；几何计算会直接产生 NaN", "filter"),
    Rule("pc_empty_or_sparse", "有效点过少", "遮挡/掉帧/深度缺失导致点云塌缩，无法支撑几何特征", None),
    Rule("pc_degenerate", "点云几何退化", "所有点坍缩到同一点/同一直线，包围盒尺度≈0", None),
    Rule("pc_duplicate", "重复点比例过高", "采样器重复采样或去重缺失，会放大局部密度", "dedupe"),
    Rule("pc_outliers", "离群点比例过高", "深度噪声/反光/玻璃穿透；按局部密度半径的统计滤波可去除", "filter"),
    Rule("pc_out_of_bounds", "点云越出工作空间", "外参或坐标系标定漂移，几何先验失效", "bounds"),
    Rule("pc_color_invalid", "颜色通道越界", "RGB 未归一化或编码错误，视觉特征会失真", "color"),
    Rule("pc_temporal_jump", "相邻帧质心跳变", "跟踪丢失或坐标系切换，需人审确认是否真实运动", None),
)

RULE_MAP = {r.code: r for r in RULES}
RULE_ORDER = [r.code for r in RULES]
REPAIRABLE = {r.code for r in RULES if r.repair}
REPAIR_ACTIONS = {"filter", "dedupe", "bounds", "color", "resample"}


# ------------------------------------------------------------------- 结果结构
@dataclass
class FrameResult:
    episode_index: int
    frame_index: int
    point_count: int
    valid_count: int
    reasons: list[str] = field(default_factory=list)
    details: dict[str, dict[str, Any]] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def valid(self) -> bool:
        return not self.reasons

    def add(self, code: str, **evidence: Any) -> None:
        if code not in self.reasons:
            self.reasons.append(code)
        self.details.setdefault(code, {}).update(evidence)

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_index": self.episode_index,
            "frame_index": self.frame_index,
            "point_count": self.point_count,
            "valid_count": self.valid_count,
            "reasons": list(self.reasons),
            "details": self.details,
            "metrics": self.metrics,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FrameResult":
        return cls(
            episode_index=int(data["episode_index"]),
            frame_index=int(data["frame_index"]),
            point_count=int(data.get("point_count") or 0),
            valid_count=int(data.get("valid_count") or 0),
            reasons=[str(c) for c in (data.get("reasons") or [])],
            details=dict(data.get("details") or {}),
            metrics=dict(data.get("metrics") or {}),
        )


@dataclass
class PointCloudValidation:
    feature_key: str
    dim: int
    total_frames: int
    results: dict[FramesKey, FrameResult]
    config: dict[str, Any]
    dataset_root: str
    resolved: dict[str, Any] = field(default_factory=dict)
    dataset_issues: list[dict[str, Any]] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)
    expected_points: int | None = None
    corrupted_files: list[dict[str, Any]] = field(default_factory=list)
    frame_alignment: dict[str, Any] | None = None

    @property
    def valid_frames(self) -> int:
        return sum(1 for r in self.results.values() if r.valid)

    @property
    def invalid_frames(self) -> list[FrameResult]:
        return [r for r in self.results.values() if not r.valid]

    def severity_of(self, res: FrameResult) -> str:
        levels = [severity_of(self.config, c) for c in res.reasons]
        return max(levels, key=lambda s: SEVERITY_ORDER.get(s, 0)) if levels else "ok"

    def severity_summary(self) -> dict[str, int]:
        out = {"error": 0, "warning": 0, "info": 0, "ok": 0}
        for res in self.results.values():
            sev = self.severity_of(res)
            out[sev] = out.get(sev, 0) + 1
        return out

    def reason_frequency(self) -> dict[str, int]:
        counts = {code: 0 for code in RULE_ORDER}
        for res in self.results.values():
            for code in res.reasons:
                counts[code] = counts.get(code, 0) + 1
        return counts

    def episode_summary(self) -> list[dict[str, Any]]:
        per: dict[int, dict[str, Any]] = {}
        for res in self.results.values():
            item = per.setdefault(res.episode_index, {"episode_index": res.episode_index,
                                                      "frames": 0, "invalid_frames": 0})
            item["frames"] += 1
            if not res.valid:
                item["invalid_frames"] += 1
        rows = sorted(per.values(), key=lambda r: r["episode_index"])
        for row in rows:
            row["invalid_ratio"] = round(row["invalid_frames"] / max(row["frames"], 1), 4)
        return rows

    def to_dict(self) -> dict[str, Any]:
        freq = self.reason_frequency()
        invalid = self.invalid_frames
        payload: dict[str, Any] = {
            "feature_key": self.feature_key,
            "dim": self.dim,
            "expected_points": self.expected_points,
            "total_frames": self.total_frames,
            "valid_frames": self.valid_frames,
            "invalid_frames": len(invalid),
            "severity_summary": self.severity_summary(),
            "reason_frequency": freq,
            "resolved": self.resolved,
            "dataset_issues": self.dataset_issues,
            "episodes": self.episode_summary(),
            "invalid_frame_details": [r.to_dict() for r in invalid],
        }
        if self.corrupted_files:
            payload["corrupted_files"] = self.corrupted_files
        if self.frame_alignment:
            payload["frame_alignment"] = self.frame_alignment
        if self.skipped:
            payload["skipped_checks"] = self.skipped
        return payload


# ------------------------------------------------------------------ 处置决策
@dataclass
class PointCloudDecision:
    key: FramesKey
    disposition: str
    severity: str
    reasons: list[str]
    repair_actions: list[str] = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_index": self.key[0],
            "frame_index": self.key[1],
            "disposition": self.disposition,
            "severity": self.severity,
            "reasons": list(self.reasons),
            "repair_actions": list(self.repair_actions),
            "note": self.note,
        }


# ------------------------------------------------------------------ 清洗结果
@dataclass
class PointCloudCleaningResult:
    success: bool
    output_root: Any | None
    method: str
    repaired_frames: list[FramesKey] = field(default_factory=list)
    held_frames: list[FramesKey] = field(default_factory=list)
    kept_frames: int = 0
    removed_points: int = 0
    added_points: int = 0
    warnings: list[str] = field(default_factory=list)
    verification: dict[str, Any] = field(default_factory=dict)
    stats_updated: bool = False
    samples: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "performed": bool(self.repaired_frames or self.held_frames),
            "success": self.success,
            "output_path": str(self.output_root) if self.output_root else None,
            "method": self.method,
            "repaired_frames": [f"{e}:{f}" for e, f in self.repaired_frames],
            "held_frames": [f"{e}:{f}" for e, f in self.held_frames],
            "repaired_count": len(self.repaired_frames),
            "held_count": len(self.held_frames),
            "kept_frames": self.kept_frames,
            "removed_points": self.removed_points,
            "added_points": self.added_points,
            "warnings": self.warnings,
            "verification": self.verification,
            "stats_updated": self.stats_updated,
            "samples": self.samples[:5],
        }


class PointCloudFormatError(RuntimeError):
    """数据集没有可用的点云特征。"""
