"""质检结果的数据结构：规则表、单条 episode 结论、数据集级问题、整体结果。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..config import severity_of

SEVERITY_ORDER = {"info": 0, "warning": 1, "error": 2}


# --------------------------------------------------------------------- 规则表
@dataclass(frozen=True)
class Rule:
    code: str
    label: str
    description: str
    #: 可执行的修复动作，None 表示只能保留/丢弃
    repair: str | None = None


RULES: tuple[Rule, ...] = (
    Rule("nan_state", "observation.state 存在 NaN/Inf", "传感器掉线、序列化失败；无法插值修复", None),
    Rule("nan_action", "action 存在 NaN/Inf", "控制指令未写入；训练时 loss 直接 NaN", None),
    Rule("action_jump", "action 相邻帧跳变异常", "遥操作接管/碰撞/控制周期抖动；需人审确认是否真实动作", None),
    Rule("frame_count_abnormal", "episode 帧数超出分布区间", "过短无法构成时序窗口，过长多为未切分", None),
    Rule("timestamp_not_monotonic", "timestamp 非严格递增", "时间轴损坏；可用 frame_index/fps 重建", "timestamps"),
    Rule("timestamp_step_deviation", "timestamp 步长偏离 1/fps", "掉帧或时间戳抖动；仅记录，不阻断", None),
    Rule("frame_mismatch", "视频帧数与 state 行数不一致", "采集中断导致 mp4 未收尾，按帧对齐会错位", None),
    Rule("meta_index_inconsistent", "meta/episodes 与数据不一致", "length/索引漂移；清洗时会按真实行数重建", "index"),
    Rule("missing_task", "任务描述缺失或为空", "语言条件策略会直接报错；需补标", None),
    Rule("static_episode", "空转 episode（动作几乎无变化）", "机器人未真正执行任务，属无效采集", None),
    Rule("blank_video", "视频黑屏/纯色遮挡", "镜头被挡或相机未推流；模态信息完全丢失", None),
    Rule("duplicate_episode", "重复采样 episode", "同一条 action 轨迹被重复入库，会造成分布失衡", None),
    Rule("joint_limit_violation", "state 超出关节限位", "标定错误或越界动作；需检查 URDF 限位", None),
)

RULE_MAP = {r.code: r for r in RULES}
RULE_ORDER = [r.code for r in RULES]


# ------------------------------------------------------------------- 结果结构
@dataclass
class EpisodeResult:
    episode_index: int
    frame_count: int
    reasons: list[str] = field(default_factory=list)
    details: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def valid(self) -> bool:
        return not self.reasons

    def add(self, code: str, **evidence: Any) -> None:
        if code not in self.reasons:
            self.reasons.append(code)
        self.details.setdefault(code, {}).update(evidence)

    def trim_plan(self) -> dict[str, int] | None:
        """若问题是「边界坏帧」，返回裁剪方案（供 policy → cleaner 使用）。"""
        head = tail = 0
        for code in ("nan_state", "nan_action"):
            detail = self.details.get(code) or {}
            if detail.get("trimmable"):
                head = max(head, int(detail.get("head_frames", 0)))
                tail = max(tail, int(detail.get("tail_frames", 0)))
        return {"head_frames": head, "tail_frames": tail} if (head or tail) else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_index": self.episode_index,
            "frame_count": self.frame_count,
            "reasons": list(self.reasons),
            "details": self.details,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EpisodeResult":
        """从 manifest 反序列化；缺字段时给出安全的默认值。"""
        return cls(
            episode_index=int(data.get("episode_index", -1)),
            frame_count=int(data.get("frame_count", 0)),
            reasons=list(data.get("reasons") or []),
            details=dict(data.get("details") or {}),
        )


@dataclass
class DatasetIssue:
    """数据集级（非单个 episode）的问题，例如 stats.json 过期、全局 index 断裂。"""

    code: str
    label: str
    detail: dict[str, Any]
    repair: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "label": self.label, "detail": self.detail, "repair": self.repair}


@dataclass
class ValidationResult:
    total_episodes: int
    results: dict[int, EpisodeResult]
    config: dict[str, Any]
    fps: float
    dataset_root: str
    resolved_thresholds: dict[str, Any] = field(default_factory=dict)
    dataset_issues: list[DatasetIssue] = field(default_factory=list)
    skipped_checks: dict[str, str] = field(default_factory=dict)
    probe_available: bool = True
    #: 流式模式下的聚合结果（total_frames / frames_by_episode / field_stats）；
    #: 内存模式为 None，报告仍直接读 frames。
    aggregates: dict[str, Any] | None = None

    @property
    def valid_episodes(self) -> int:
        return sum(1 for r in self.results.values() if r.valid)

    @property
    def invalid_episodes(self) -> list[EpisodeResult]:
        return [r for r in self.results.values() if not r.valid]

    @property
    def invalid_indices(self) -> list[int]:
        return sorted(r.episode_index for r in self.invalid_episodes)

    def severity_of(self, res: EpisodeResult) -> str:
        levels = [severity_of(self.config, code) for code in res.reasons]
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

    def to_dict(self) -> dict[str, Any]:
        """保留约定的三字段契约，同时补上分级与数据集级问题。"""
        payload: dict[str, Any] = {
            "total_episodes": self.total_episodes,
            "valid_episodes": self.valid_episodes,
            "invalid_episodes": [
                {
                    "episode_index": r.episode_index,
                    "reasons": list(r.reasons),
                    "frame_count": r.frame_count,
                    "severity": self.severity_of(r),
                    "details": r.details,
                }
                for r in self.invalid_episodes
            ],
            "severity_summary": self.severity_summary(),
            "resolved_thresholds": self.resolved_thresholds,
            "dataset_issues": [i.to_dict() for i in self.dataset_issues],
        }
        if self.skipped_checks:
            payload["skipped_checks"] = self.skipped_checks
        return payload
