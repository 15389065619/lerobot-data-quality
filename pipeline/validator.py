"""逐 episode 数据质检。

设计取向（对齐真实数据工程）：
1. **阈值自适应**：帧数区间、action 跳变阈值都由数据集自身分布（median + k·MAD）推导，
   配置里的 abs_floor / floor 只是安全下限。不同机器人的关节单位不同，绝对阈值必然误报。
2. **严重度分级**：error（不可用）/ warning（需人审或可修）/ info（仅记录），
   而不是所有异常一律删除——处置动作由 policy 层决定。
3. **可修性判定**：时间戳错乱、NaN 仅出现在首尾、meta 索引漂移都属于「可修复」，
   在细节里给出修复所需参数（如 head_frames / tail_frames），交由 cleaner 执行。
4. **性能**：视频探针优先读容器元数据（nb_frames / duration），仅在必要时全解码；
   黑屏检测一次 ffmpeg 调用抽多帧，避免逐帧 subprocess。

规则清单见 RULES。
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import cfg_get, load_config, severity_of
from .loader import ACTION_KEY, STATE_KEY, LeRobotDatasetView

LOG = logging.getLogger(__name__)

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

_FFPROBE = shutil.which("ffprobe")
_FFMPEG = shutil.which("ffmpeg")
_FRAME_CACHE: dict[str, int | None] = {}
_LUMA_CACHE: dict[str, Any] = {}


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


# ------------------------------------------------------------------ 主入口
def validate_dataset(dataset: LeRobotDatasetView, config: dict[str, Any] | None = None) -> ValidationResult:
    config = config or load_config(robot_type=dataset.info.get("robot_type"))
    skipped: dict[str, str] = {}
    ctx = _DatasetContext.build(dataset, config, skipped)

    results: dict[int, EpisodeResult] = {}
    probes = 0
    blank_enabled = bool(cfg_get(config, "blank_video.enabled", True)) and _FFMPEG is not None
    if not blank_enabled:
        skipped["blank_video"] = "未找到 ffmpeg 或已禁用视频内容检测"
    precise = bool(cfg_get(config, "probe.precise", False))
    if not _FFPROBE:
        skipped["frame_mismatch"] = "未找到 ffprobe，跳过视频帧对齐"
    if not dataset.video_features:
        skipped["frame_mismatch"] = "数据集中没有视频特征"

    cameras = dataset.video_features[: int(cfg_get(config, "probe.max_cameras", 3))]
    max_probes = int(cfg_get(config, "blank_video.max_probes", 400))

    for ep_index, ep_df in dataset.frames.groupby("episode_index", sort=True):
        ep_index = int(ep_index)
        res = EpisodeResult(episode_index=ep_index, frame_count=int(len(ep_df)))
        frame_col = _frame_column(ep_df)

        if STATE_KEY in ep_df.columns:
            arr_s = _as_matrix(ep_df[STATE_KEY], res, "nan_state")
            if arr_s is not None:
                _check_finite(arr_s, "nan_state", res, frame_col, config)
                _check_static(arr_s, "state", res, config)
                _check_joint_limits(arr_s, res, config, dataset)
        else:
            skipped.setdefault("nan_state", f"不存在 {STATE_KEY} 列")

        if ACTION_KEY in ep_df.columns:
            arr_a = _as_matrix(ep_df[ACTION_KEY], res, "nan_action")
            if arr_a is not None:
                _check_finite(arr_a, "nan_action", res, frame_col, config)
                _check_action_jump(arr_a, res, ctx.action_thresholds, config)
                _check_static(arr_a, "action", res, config)
                _check_duplicate(res, arr_a, ctx, config)
        else:
            skipped.setdefault("action_jump", f"不存在 {ACTION_KEY} 列")

        _check_frame_count(res, ctx.frame_bounds, config)
        _check_timestamp(ep_df, res, config, dataset.fps)
        _check_meta_consistency(res, ep_index, dataset)
        _check_frame_mismatch(res, ep_index, dataset, cameras, precise, config)

        if blank_enabled and probes < max_probes:
            probes += 1
            _check_blank_video(res, ep_index, dataset, cameras[0] if cameras else None, config)

        _check_task(res, ep_index, dataset)
        results[ep_index] = res

    result = ValidationResult(
        total_episodes=len(results),
        results=results,
        config=config,
        fps=dataset.fps,
        dataset_root=str(dataset.root),
        resolved_thresholds=ctx.resolved,
        dataset_issues=ctx.issues,
        skipped_checks=skipped,
        probe_available=bool(_FFPROBE),
    )
    sev = result.severity_summary()
    LOG.info(
        "质检完成：%d 个 episode，正常 %d / 异常 %d（error %d, warning %d, info %d）",
        result.total_episodes, result.valid_episodes, len(result.invalid_episodes),
        sev.get("error", 0), sev.get("warning", 0), sev.get("info", 0),
    )
    return result


# --------------------------------------------- 数据集级上下文（自适应阈值/全局检查）
@dataclass
class _DatasetContext:
    frame_bounds: dict[str, Any]
    action_thresholds: list[float]
    fingerprints: dict[str, int]
    issues: list[DatasetIssue]
    resolved: dict[str, Any]

    @classmethod
    def build(
        cls, dataset: LeRobotDatasetView, config: dict[str, Any], skipped: dict[str, str]
    ) -> "_DatasetContext":
        counts = dataset.frames.groupby("episode_index").size().to_numpy(dtype=float)
        frame_bounds = _resolve_frame_bounds(counts, config)
        action_thresholds, action_meta = _resolve_action_thresholds(dataset, config)
        issues: list[DatasetIssue] = []
        for issue in (_check_stats_staleness(dataset, config), _check_index_continuity(dataset)):
            if issue:
                issues.append(issue)
        if not action_thresholds:
            skipped["action_jump"] = "无可用 action 数据推导阈值"

        resolved = {
            "frame_count": frame_bounds,
            "action_jump": action_meta,
            "timestamp_step_tolerance": round(
                float(cfg_get(config, "timestamp.step_tolerance_ratio", 0.25)) / max(dataset.fps, 1e-6), 6
            ),
            "video_frame_tolerance": int(cfg_get(config, "video_frame_tolerance", 1)),
        }
        return cls(frame_bounds, action_thresholds, {}, issues, resolved)


def _resolve_frame_bounds(counts: np.ndarray, config: dict[str, Any]) -> dict[str, Any]:
    """median ± k·MAD，再夹到 [floor, ceiling]。"""
    mode = str(cfg_get(config, "frame_count.mode", "auto"))
    floor = int(cfg_get(config, "frame_count.floor", 8))
    ceiling = int(cfg_get(config, "frame_count.ceiling", 20000))
    if counts.size == 0:
        return {"mode": mode, "min": floor, "max": ceiling, "median": 0.0, "mad": 0.0}
    median = float(np.median(counts))
    mad = float(np.median(np.abs(counts - median)))
    if mode == "fixed":
        return {"mode": "fixed", "min": floor, "max": ceiling, "median": median, "mad": mad}
    sigma = 1.4826 * mad if mad > 0 else 0.0
    low = max(floor, median - float(cfg_get(config, "frame_count.mad_k_low", 3.0)) * sigma)
    high = min(ceiling, median + float(cfg_get(config, "frame_count.mad_k_high", 8.0)) * sigma)
    return {
        "mode": "auto",
        "min": int(max(floor, np.floor(low))),
        "max": int(min(ceiling, np.ceil(high))),
        "median": round(median, 1),
        "mad": round(mad, 1),
    }


def _resolve_action_thresholds(
    dataset: LeRobotDatasetView, config: dict[str, Any]
) -> tuple[list[float], dict[str, Any]]:
    """逐维阈值 = max(abs_floor, median|Δ| + mad_k·1.4826·MAD|Δ|)。"""
    abs_floor = float(cfg_get(config, "action_jump.abs_floor", 1.0))
    mad_k = float(cfg_get(config, "action_jump.mad_k", 12.0))
    if ACTION_KEY not in dataset.frames.columns:
        return [], {"abs_floor": abs_floor, "mad_k": mad_k, "per_dim": []}

    deltas: list[np.ndarray] = []
    for _, ep_df in dataset.frames.groupby("episode_index", sort=False):
        try:
            arr = np.asarray(ep_df[ACTION_KEY].to_list(), dtype=float)
        except (TypeError, ValueError):
            continue
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.shape[0] < 2:
            continue
        d = np.abs(np.diff(arr, axis=0))
        d = d[np.isfinite(d).all(axis=1)]
        if d.size:
            deltas.append(d)
    if not deltas:
        return [], {"abs_floor": abs_floor, "mad_k": mad_k, "per_dim": []}

    stacked = np.vstack(deltas)
    med = np.median(stacked, axis=0)
    mad = np.median(np.abs(stacked - med), axis=0)
    thresholds = np.maximum(abs_floor, med + mad_k * 1.4826 * np.where(mad > 0, mad, 0.0))
    return (
        [float(t) for t in thresholds],
        {
            "abs_floor": abs_floor,
            "mad_k": mad_k,
            "per_dim": [round(float(t), 4) for t in thresholds],
            "median_delta": [round(float(m), 4) for m in med],
            "mad_delta": [round(float(m), 4) for m in mad],
        },
    )


def _check_stats_staleness(dataset: LeRobotDatasetView, config: dict[str, Any]) -> DatasetIssue | None:
    """meta/stats.json 与实测数据偏差过大 → 训练归一化会全错（数据集级问题）。"""
    if not cfg_get(config, "stale_stats.enabled", True) or not dataset.stats:
        return None
    if STATE_KEY not in dataset.frames.columns:
        return None
    limit = int(cfg_get(config, "stale_stats.sample_episodes", 200))
    rows = dataset.frames[dataset.frames["episode_index"].isin(dataset.episode_indices[:limit])]
    try:
        arr = np.asarray(rows[STATE_KEY].to_list(), dtype=float)
    except (TypeError, ValueError):
        return None
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    arr = arr[np.isfinite(arr).all(axis=1)]
    if arr.size == 0:
        return None
    actual_mean = arr.mean(axis=0)
    try:
        recorded_arr = np.asarray((dataset.stats.get(STATE_KEY) or {}).get("mean"), dtype=float)
    except (TypeError, ValueError):
        return None
    if recorded_arr.size == 0:
        return None
    if recorded_arr.shape != actual_mean.shape:
        return DatasetIssue(
            "stats_stale", "stats.json 维度与数据不一致",
            {"recorded_shape": list(recorded_arr.shape), "actual_shape": list(actual_mean.shape)},
            repair="stats",
        )
    scale = np.maximum(np.abs(recorded_arr), 1e-6)
    deviations = np.abs(actual_mean - recorded_arr) / scale
    worst = float(np.max(deviations))
    tol = float(cfg_get(config, "stale_stats.mean_rel_tolerance", 0.05))
    if worst <= tol:
        return None
    return DatasetIssue(
        "stats_stale",
        "meta/stats.json 与实测数据偏差过大（归一化统计已过期）",
        {
            "max_relative_deviation": round(worst, 4),
            "tolerance": tol,
            "worst_dims": [int(i) for i in np.argsort(deviations)[::-1][:3]],
            "recorded_mean_first3": [round(float(x), 4) for x in recorded_arr.ravel()[:3]],
            "actual_mean_first3": [round(float(x), 4) for x in actual_mean.ravel()[:3]],
        },
        repair="stats",
    )


def _check_index_continuity(dataset: LeRobotDatasetView) -> DatasetIssue | None:
    """全局 index 必须是 0..N-1 连续；断裂会让 DataLoader 采样与 shard 划分出错。"""
    if "index" not in dataset.frames.columns:
        return None
    idx = pd.to_numeric(dataset.frames["index"], errors="coerce").to_numpy()
    if np.isnan(idx).any():
        return DatasetIssue("meta_index_inconsistent", "index 列含 NaN",
                            {"nan_count": int(np.isnan(idx).sum())}, repair="index")
    unique = np.unique(idx)
    if unique.size == dataset.num_frames and unique[0] == 0 and unique[-1] == dataset.num_frames - 1:
        return None
    return DatasetIssue(
        "meta_index_inconsistent",
        "全局 index 不连续或存在重复",
        {"rows": dataset.num_frames, "unique": int(unique.size),
         "min": int(unique[0]), "max": int(unique[-1])},
        repair="index",
    )


# ---------------------------------------------------------------- 数值类检查
def _check_finite(
    arr: np.ndarray, code: str, res: EpisodeResult, frame_col: np.ndarray | None, config: dict[str, Any]
) -> None:
    nan_mask = np.isnan(arr)
    inf_mask = np.isinf(arr)
    if not nan_mask.any() and not inf_mask.any():
        return
    bad = np.where(nan_mask.any(axis=1) | inf_mask.any(axis=1))[0]
    dims = np.where(nan_mask.any(axis=0) | inf_mask.any(axis=0))[0]
    n = arr.shape[0]

    # 边界坏帧 → 优先裁剪而不是整条丢弃（真实业务最常见的「可修」场景）
    is_bad = np.zeros(n, dtype=bool)
    is_bad[bad] = True
    head = 0
    while head < n and is_bad[head]:
        head += 1
    tail = 0
    while tail < n - head and is_bad[n - 1 - tail]:
        tail += 1
    boundary_only = int(bad.size) > 0 and (head + tail) == int(bad.size)
    max_ratio = float(cfg_get(config, "boundary_trim.max_ratio", 0.25))
    trimmable = (
        boundary_only
        and bool(cfg_get(config, "boundary_trim.enabled", True))
        and n > 0
        and max(head, tail) / n <= max_ratio
        and (head + tail) < n
    )

    res.add(
        code,
        nan_count=int(nan_mask.sum()),
        inf_count=int(inf_mask.sum()),
        affected_frames=int(bad.size),
        affected_dims=[int(d) for d in dims[:5]],
        first_bad_frames=_frame_labels(frame_col, bad, 5),
        boundary_only=bool(boundary_only),
        head_frames=int(head) if boundary_only else 0,
        tail_frames=int(tail) if boundary_only else 0,
        trimmable=bool(trimmable),
        note="坏帧集中在首尾，可裁剪保留" if trimmable else "坏帧在序列中段，无法裁剪修复",
    )


def _check_action_jump(
    arr: np.ndarray, res: EpisodeResult, thresholds: list[float], config: dict[str, Any]
) -> None:
    if arr.shape[0] < 2 or not thresholds:
        return
    deltas = np.abs(np.diff(arr, axis=0))
    deltas = np.nan_to_num(deltas, nan=0.0, posinf=np.inf)
    dim = min(deltas.shape[1], len(thresholds))
    thr = np.asarray(thresholds[:dim], dtype=float)
    over = deltas[:, :dim] > thr
    if not over.any():
        return
    frames, dims = np.where(over)
    finite_deltas = np.where(np.isfinite(deltas[:, :dim]), deltas[:, :dim], 0.0)
    worst = np.unravel_index(int(np.argmax(finite_deltas)), finite_deltas.shape)
    res.add(
        "action_jump",
        jump_count=int(over.sum()),
        max_delta=round(float(np.max(deltas[:, :dim])), 4),
        thresholds_used=[round(float(t), 4) for t in thr],
        worst_frame=int(worst[0]),
        worst_dim=int(worst[1]),
        examples=[
            {"frame": int(f), "dim": int(d), "delta": round(float(deltas[f, d]), 4),
             "threshold": round(float(thr[d]), 4)}
            for f, d in list(zip(frames, dims))[:5]
        ],
    )


def _check_static(arr: np.ndarray, kind: str, res: EpisodeResult, config: dict[str, Any]) -> None:
    """空转检测：整条序列几乎无变化 → 机器人没真正执行任务。"""
    if not cfg_get(config, "static_episode.enabled", True):
        return
    if arr.shape[0] < int(cfg_get(config, "static_episode.min_frames_checked", 5)):
        return
    finite = arr[np.isfinite(arr).all(axis=1)]
    if finite.shape[0] < 2:
        return
    std = float(np.mean(np.std(finite, axis=0)))
    thr = float(cfg_get(config, f"static_episode.{kind}_std_threshold", 1e-3))
    if std > thr:
        return
    detail = res.details.setdefault("static_episode", {})
    detail[f"{kind}_std"] = round(std, 6)
    detail["threshold"] = thr
    detail["note"] = "该 episode 数值几乎无变化，疑似未执行任务"
    if "static_episode" not in res.reasons:
        res.reasons.append("static_episode")


def _check_joint_limits(
    arr: np.ndarray, res: EpisodeResult, config: dict[str, Any], dataset: LeRobotDatasetView
) -> None:
    if not cfg_get(config, "joint_limits.enabled", False):
        return
    lower = cfg_get(config, "joint_limits.lower")
    upper = cfg_get(config, "joint_limits.upper")
    if not lower or not upper:
        return
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    dim = min(arr.shape[1], lo.size, hi.size)
    if dim == 0:
        return
    tol = float(cfg_get(config, "joint_limits.tolerance", 1e-6))
    finite = arr[:, :dim]
    with np.errstate(invalid="ignore"):
        below = finite < (lo[:dim] - tol)
        above = finite > (hi[:dim] + tol)
    if not below.any() and not above.any():
        return
    res.add(
        "joint_limit_violation",
        violation_count=int(below.sum() + above.sum()),
        dims=[int(d) for d in np.where((below | above).any(axis=0))[0]],
        joint_names=[dataset.joint_names[d] for d in range(dim) if (below | above)[:, d].any()][:5],
    )


def _check_duplicate(res: EpisodeResult, arr: np.ndarray, ctx: _DatasetContext, config: dict[str, Any]) -> None:
    """同一条轨迹被重复入库会造成分布失衡。用 action 序列量化指纹识别。"""
    if not cfg_get(config, "duplicate_episode.enabled", True):
        return
    decimals = int(cfg_get(config, "duplicate_episode.quantize_decimals", 3))
    quantized = np.round(np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0), decimals).astype(np.float32)
    fingerprint = hashlib.sha1(quantized.tobytes()).hexdigest()[:16]
    first = ctx.fingerprints.get(fingerprint)
    if first is None:
        ctx.fingerprints[fingerprint] = res.episode_index
        return
    if first == res.episode_index:
        return
    res.add("duplicate_episode", duplicate_of=int(first), fingerprint=fingerprint,
            note=f"与 ep {first} 的 action 轨迹完全一致")


# ------------------------------------------------------------------ 结构类检查
def _check_frame_count(res: EpisodeResult, bounds: dict[str, Any], config: dict[str, Any]) -> None:
    n = res.frame_count
    low, high = int(bounds["min"]), int(bounds["max"])
    if low <= n <= high:
        return
    res.add(
        "frame_count_abnormal",
        frame_count=n,
        min_allowed=low,
        max_allowed=high,
        direction="too_few" if n < low else "too_many",
        how_derived=f"median={bounds.get('median')} mad={bounds.get('mad')} mode={bounds.get('mode')}",
    )


def _check_timestamp(ep_df: pd.DataFrame, res: EpisodeResult, config: dict[str, Any], fps: float) -> None:
    if "timestamp" not in ep_df.columns:
        res.add("timestamp_not_monotonic", reason="缺少 timestamp 列")
        return
    ts = pd.to_numeric(ep_df["timestamp"], errors="coerce").to_numpy(dtype=float)
    if np.isnan(ts).any():
        res.add("timestamp_not_monotonic", reason="timestamp 含 NaN",
                frames=[int(b) for b in np.where(np.isnan(ts))[0][:5]])
        return
    if ts.size < 2:
        return
    diffs = np.diff(ts)
    bad = np.where(diffs <= 0)[0]
    if bad.size:
        res.add(
            "timestamp_not_monotonic",
            reason="非严格递增",
            violations=int(bad.size),
            repair_hint="可用 frame_index/fps 重建时间轴",
            examples=[
                {"prev": round(float(ts[i]), 6), "next": round(float(ts[i + 1]), 6), "delta": round(float(diffs[i]), 6)}
                for i in bad[:5]
            ],
        )
        return

    # 步长偏离：掉帧/抖动，仅记录（info），不阻断
    expected = 1.0 / fps if fps else 0.0
    if expected <= 0:
        return
    ratio_tol = float(cfg_get(config, "timestamp.step_tolerance_ratio", 0.25))
    deviations = np.abs(diffs - expected) > ratio_tol * expected
    if deviations.any():
        res.add(
            "timestamp_step_deviation",
            reason="步长偏离 1/fps",
            affected_frames=int(deviations.sum()),
            expected_step=round(expected, 6),
            examples=[{"frame": int(i), "delta": round(float(diffs[i]), 6)}
                      for i in np.where(deviations)[0][:5]],
        )


def _check_meta_consistency(res: EpisodeResult, ep_index: int, dataset: LeRobotDatasetView) -> None:
    if "length" not in dataset.episodes.columns:
        return
    row = dataset.episodes.loc[dataset.episodes["episode_index"] == ep_index, "length"]
    if not len(row) or pd.isna(row.iloc[0]):
        return
    try:
        meta_length = int(row.iloc[0])
    except (TypeError, ValueError):
        return
    if meta_length != res.frame_count:
        res.add(
            "meta_index_inconsistent",
            reason="meta length 与 parquet 行数不一致",
            meta_length=meta_length,
            rows=res.frame_count,
            diff=res.frame_count - meta_length,
        )


def _check_frame_mismatch(
    res: EpisodeResult, ep_index: int, dataset: LeRobotDatasetView,
    cameras: list[str], precise: bool, config: dict[str, Any],
) -> None:
    if not _FFPROBE:
        return
    tol = int(cfg_get(config, "video_frame_tolerance", 1))
    problems: list[dict[str, Any]] = []

    for cam in cameras:
        loc = dataset.video_location(cam, ep_index)
        if loc is None:
            problems.append({"type": "video_unlocatable", "camera": cam})
            continue
        path, from_ts, to_ts = loc
        if not path.is_file():
            problems.append({"type": "video_missing", "camera": cam, "path": str(path)})
            continue
        total = probe_video_frames(path, precise=precise)
        if total is None:
            problems.append({"type": "video_unprobeable", "camera": cam, "path": str(path)})
            continue

        if from_ts >= 0 and to_ts > from_ts:
            start = int(round(from_ts * dataset.fps))
            end = int(round(to_ts * dataset.fps))
            seg = end - start
            if end > total + tol:
                problems.append({"type": "video_truncated", "camera": cam,
                                 "expected_end_frame": end, "actual_total_frames": total})
            elif abs(seg - res.frame_count) > tol:
                problems.append({"type": "count_mismatch", "camera": cam, "video_frames": seg,
                                 "rows": res.frame_count, "diff": seg - res.frame_count})
        elif abs(total - res.frame_count) > tol:
            problems.append({"type": "count_mismatch", "camera": cam, "video_frames": total,
                             "rows": res.frame_count, "diff": total - res.frame_count})

    if problems:
        res.add("frame_mismatch", rows=res.frame_count, tolerance=tol, problems=problems)


def _check_task(res: EpisodeResult, ep_index: int, dataset: LeRobotDatasetView) -> None:
    if dataset.task_of(ep_index):
        return
    if "task_index" not in dataset.frames.columns and not dataset.tasks:
        res.add("missing_task", reason="不存在任务描述字段（task_index / tasks）")
    elif not dataset.tasks:
        res.add("missing_task", reason="meta/tasks 为空或缺失，task_index 无法解析")
    else:
        res.add("missing_task", reason="任务描述为空字符串")


# ------------------------------------------------------------------ 视频内容
def _check_blank_video(
    res: EpisodeResult, ep_index: int, dataset: LeRobotDatasetView, camera: str | None, config: dict[str, Any]
) -> None:
    """黑屏/遮挡检测：一次 ffmpeg 调用抽多帧缩略图，算灰度标准差。"""
    if camera is None or not _FFMPEG:
        return
    loc = dataset.video_location(camera, ep_index)
    if loc is None:
        return
    path, from_ts, to_ts = loc
    if not path.is_file():
        return
    dur = (to_ts - from_ts) if to_ts > from_ts else res.frame_count / max(dataset.fps, 1e-6)
    stats = probe_frame_brightness(path, max(from_ts, 0.0), max(dur, 0.1))
    if stats is None:
        return
    mean_lum, std_lum = stats
    if not std_lum:
        return
    thr = float(cfg_get(config, "blank_video.lum_std_threshold", 4.0))
    bad_ratio = float(np.mean(np.asarray(std_lum) < thr))
    if bad_ratio < float(cfg_get(config, "blank_video.bad_ratio_threshold", 0.6)):
        return
    res.add(
        "blank_video",
        camera=camera,
        sampled_frames=len(std_lum),
        bad_ratio=round(bad_ratio, 3),
        lum_std_threshold=thr,
        mean_luminance=round(float(np.mean(mean_lum)), 2),
        note="画面近乎纯色/全黑，疑似镜头遮挡或未推流",
    )


def probe_frame_brightness(path: str | Path, start_ts: float, duration: float) -> tuple[list[float], list[float]] | None:
    """抽样若干帧，返回 (每帧平均亮度, 每帧亮度标准差)。"""
    key = f"{path}@{start_ts:.2f}+{duration:.2f}"
    if key in _LUMA_CACHE:
        return _LUMA_CACHE[key]
    if not _FFMPEG:
        return None
    w, h = 64, 48
    cmd = [
        str(_FFMPEG), "-v", "error", "-ss", f"{start_ts:.3f}", "-t", f"{duration:.3f}",
        "-i", str(path), "-vf", f"fps=2,scale={w}:{h}", "-f", "rawvideo", "-pix_fmt", "gray", "-",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=120, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        LOG.debug("亮度抽样失败 %s: %s", path, exc)
        _LUMA_CACHE[key] = None
        return None
    buf = np.frombuffer(proc.stdout, dtype=np.uint8)
    frame_size = w * h
    if proc.returncode != 0 or buf.size < frame_size:
        _LUMA_CACHE[key] = None
        return None
    n = buf.size // frame_size
    frames = buf[: n * frame_size].reshape(n, frame_size).astype(np.float32)
    result = (frames.mean(axis=1).tolist(), frames.std(axis=1).tolist())
    _LUMA_CACHE[key] = result
    return result


# ---------------------------------------------------------------- 视频帧探针
def probe_video_frames(path: str | Path, precise: bool = False) -> int | None:
    """统计视频帧数。

    真实数据集上全解码代价极高（几十 GB），因此优先读容器元数据：
    1. stream.nb_frames（最快，h264/mp4 通常有）
    2. duration × fps（无 nb_frames 时估算）
    3. -count_frames（精确但需全解码，仅在 precise=True 时使用）
    """
    key = f"{path}|{precise}"
    if key in _FRAME_CACHE:
        return _FRAME_CACHE[key]
    if not _FFPROBE:
        _FRAME_CACHE[key] = None
        return None

    count = _ffprobe_int(path, ["-show_entries", "stream=nb_frames"])
    if count is None:
        count = _ffprobe_estimate(path)
    if count is None and precise:
        count = _ffprobe_int(path, ["-count_frames", "-show_entries", "stream=nb_read_frames"])

    _FRAME_CACHE[key] = count
    if count is None:
        LOG.debug("无法获取视频帧数: %s", path)
    return count


def _ffprobe_int(path: str | Path, extra: list[str]) -> int | None:
    cmd = [str(_FFPROBE), "-v", "error", "-select_streams", "v:0", *extra,
           "-of", "default=nw=1:nk=1", str(path)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    for line in (proc.stdout or "").splitlines():
        line = line.strip()
        if not line or line.upper() == "N/A":
            continue
        try:
            value = float(line)
        except ValueError:
            continue
        if value > 0:
            return int(round(value))
    return None


def _ffprobe_estimate(path: str | Path) -> int | None:
    cmd = [str(_FFPROBE), "-v", "error", "-select_streams", "v:0",
           "-show_entries", "stream=avg_frame_rate,duration", "-of", "json", str(path)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    try:
        stream = (json.loads(proc.stdout or "{}").get("streams") or [{}])[0]
        num, _, den = str(stream.get("avg_frame_rate") or "0/0").partition("/")
        fps = float(num) / float(den) if float(den or 0) else 0.0
        duration = float(stream.get("duration") or 0.0)
    except (ValueError, ZeroDivisionError, TypeError, json.JSONDecodeError):
        return None
    if fps <= 0 or duration <= 0:
        return None
    return int(round(fps * duration))


# -------------------------------------------------------------------- 小工具
def _as_matrix(series: pd.Series, res: EpisodeResult, code: str) -> np.ndarray | None:
    try:
        arr = np.asarray(series.to_list(), dtype=float)
    except (TypeError, ValueError):
        res.add(code, error="字段无法解析为数值矩阵（dtype/长度不一致）")
        return None
    if arr.ndim == 0:
        return None
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)
    return arr


def _frame_column(ep_df: pd.DataFrame) -> np.ndarray | None:
    if "frame_index" in ep_df.columns:
        return pd.to_numeric(ep_df["frame_index"], errors="coerce").to_numpy()
    return None


def _frame_labels(frame_col: np.ndarray | None, positions: np.ndarray, limit: int) -> list[int]:
    vals = frame_col[positions] if frame_col is not None else positions
    return [int(v) for v in np.asarray(vals)[:limit]]
