"""数据集级上下文：自适应阈值推导与全局一致性检查。"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..config import cfg_get
from ..loader import ACTION_KEY, STATE_KEY, LeRobotDatasetView
from .types import DatasetIssue

LOG = logging.getLogger(__name__)


@dataclass
class DatasetContext:
    frame_bounds: dict[str, Any]
    action_thresholds: list[float]
    fingerprints: dict[str, int]
    issues: list[DatasetIssue]
    resolved: dict[str, Any]
    #: state 跳变阈值（与 action 同一套「每集峰值」推导），空列表表示不可评估
    state_thresholds: list[float] = field(default_factory=list)
    #: 全局逐维 median(|Δ|)：spike 核验「剩余邻居 <2」时的回退参照（空 = 不可用）
    action_diff_ref: list[float] = field(default_factory=list)
    state_diff_ref: list[float] = field(default_factory=list)
    #: 特征维度名，用于按四元数分组识别 q≡-q（元数据缺失时为 None）
    action_dim_names: list[str] | None = None
    state_dim_names: list[str] | None = None

    @classmethod
    def build(
        cls, dataset: LeRobotDatasetView, config: dict[str, Any], skipped: dict[str, str]
    ) -> "DatasetContext":
        # 全部数据文件损坏时 frames 连列都没有：counts 为空，episode 由 validator 兜底标 episode_read_error
        if "episode_index" in dataset.frames.columns:
            counts = dataset.frames.groupby("episode_index").size().to_numpy(dtype=float)
        else:
            counts = np.array([], dtype=float)
        frame_bounds = _resolve_frame_bounds(counts, config, dataset.fps)
        action_thresholds, action_meta = _resolve_action_thresholds(dataset, config)
        state_thresholds, state_meta = _resolve_action_thresholds(
            dataset, config, key=STATE_KEY, section="state_jump"
        )
        issues: list[DatasetIssue] = []
        for issue in (
            _check_stats_staleness(dataset, config, skipped),
            _check_episode_alignment(dataset),
            _check_index_continuity(dataset),
            _check_meta_duplicates(dataset),
            _check_feature_declarations(dataset),
        ):
            if issue:
                issues.append(issue)
        if not action_thresholds:
            skipped["action_jump"] = "无可用 action 数据推导阈值"
        if not state_thresholds:
            skipped["state_jump"] = "无可用 state 数据推导阈值"

        resolved = {
            "frame_count": frame_bounds,
            "action_jump": action_meta,
            "state_jump": state_meta,
            "timestamp_step_tolerance": round(
                float(cfg_get(config, "timestamp.step_tolerance_ratio", 0.25)) / max(dataset.fps, 1e-6), 6
            ),
            "video_frame_tolerance": scale_frames(
                int(cfg_get(config, "video_frame_tolerance", 1)), config, dataset.fps
            ),
        }
        # 指纹预计算覆盖全部 episode：这样即使增量跳过某些 episode，
        # 新增 episode 与历史 episode 的重复仍能被识别。
        fingerprints = _fingerprint_actions(dataset, config)
        return cls(
            frame_bounds, action_thresholds, fingerprints, issues, resolved,
            state_thresholds=state_thresholds,
            action_diff_ref=action_meta.get("diff_ref") or [],
            state_diff_ref=state_meta.get("diff_ref") or [],
            action_dim_names=dataset.feature_names(ACTION_KEY) or None,
            state_dim_names=dataset.feature_names(STATE_KEY) or None,
        )


def _fingerprint_actions(dataset: LeRobotDatasetView, config: dict[str, Any]) -> dict[str, int]:
    """action 序列量化指纹 -> 首次出现的 episode（按 episode_index 升序）。"""
    if not cfg_get(config, "duplicate_episode.enabled", True):
        return {}
    if ACTION_KEY not in dataset.frames.columns:
        return {}
    if "episode_index" not in dataset.frames.columns:
        # 非标准字段名：没有 episode 标识就无法判断「首次出现」，重复检测降级为空
        return {}
    decimals = int(cfg_get(config, "duplicate_episode.quantize_decimals", 3))
    out: dict[str, int] = {}
    for ep_index, ep_df in dataset.frames.groupby("episode_index", sort=True):
        try:
            arr = np.asarray(ep_df[ACTION_KEY].to_list(), dtype=float)
        except (TypeError, ValueError):
            continue
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        quantized = np.round(np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0), decimals).astype(np.float32)
        fingerprint = hashlib.sha1(quantized.tobytes()).hexdigest()[:16]
        out.setdefault(fingerprint, int(ep_index))
    return out


def _ref_fps(config: dict[str, Any]) -> float:
    """配置里帧数类阈值按 ref_fps 标定（默认 30Hz）。"""
    value = float(cfg_get(config, "action_jump.ref_fps", 30.0))
    return value if value > 0 else 30.0


def scale_frames(value: float, config: dict[str, Any], fps: float | None) -> int:
    """把按 ref_fps 标定的帧数折算到实际帧率。

    120Hz 数据 1 秒有 120 帧，按 30Hz 标定的 window/tolerance/下限直接套用会
    把正常数据判成异常（或反之）。fps 未知时按 ref_fps 处理，保持原行为。
    """
    ref = _ref_fps(config)
    eff = float(fps) if fps and fps > 0 else ref
    return max(1, int(round(float(value) * eff / ref)))


def _resolve_frame_bounds(
    counts: np.ndarray, config: dict[str, Any], fps: float | None = None
) -> dict[str, Any]:
    """解析帧数上下限：业务硬约束（manual）优先，否则按 median ± k·MAD 推导。

    配置 `frame_count.min` / `frame_count.max` 任一给定时进入 manual 模式：
    显式的一侧直接用绝对值（不再参与统计学推导），未指定的一侧仍按数据集分布推导，
    并在 `source` 里标注两侧各自来源，便于在报告里追溯判定依据。

    `frame_count.floor` / `ceiling` 是按 30Hz 标定的默认边界，需按实际 fps 折算，
    否则 120Hz 的长 episode 会因帧数多 4 倍被判 `too_many` 后误删。
    用户显式给出的 min/max 视为业务绝对值，不做折算。
    """
    mode = str(cfg_get(config, "frame_count.mode", "auto"))
    ref = _ref_fps(config)
    eff = float(fps) if fps and fps > 0 else ref
    scale = eff / ref
    floor = max(1, int(round(int(cfg_get(config, "frame_count.floor", 8)) * scale)))
    ceiling = max(floor, int(round(int(cfg_get(config, "frame_count.ceiling", 20000)) * scale)))
    manual_min = cfg_get(config, "frame_count.min")
    manual_max = cfg_get(config, "frame_count.max")

    median = float(np.median(counts)) if counts.size else 0.0
    mad = float(np.median(np.abs(counts - median))) if counts.size else 0.0
    # MAD=0 且存在离散（如录制工艺稳定、帧数只有 ±1~2 帧抖动的流水线数据）：
    # 旧实现塌缩成单点，tokyo 实测 237/238/239 三种长度把 24/50 集全体判越界。
    # 改为最小宽度区间 median ± max(1, ceil(ratio×median))，ratio 可配。
    # 刻意不用观测 [min,max]——那等于用被检样本定义边界，49×239+1×100 的
    # 上传截断集从此永久合规（零检测）。真正唯一的离群集仍落在界外照常告警。
    # 注意这是对旧「刻意塌缩」策略的策略变更：塌缩抓离群的价值由
    # mad>0 路径与全等长 margin 路径保留，此处只处理「高一致性 + 微抖动」场景。
    # 只改 auto 路径的区间值，不提前返回——manual/fixed 的优先级更高。
    mad0_half: int | None = None
    if mad > 0:
        sigma = 1.4826 * mad
    elif counts.size and np.ptp(counts) == 0:
        # 全部等长时 MAD=0，区间会塌成单点，±1 帧就判 error 并整条 drop；
        # 只有「没有任何离散」这一种情况需要补相对余量，存在离群值时应保持塌缩判定。
        margin = float(cfg_get(config, "frame_count.min_margin_ratio", 0.05))
        sigma = 1.4826 * margin * abs(median)
    elif counts.size:
        mad0_half = max(1, int(np.ceil(
            float(cfg_get(config, "frame_count.mad0_min_width_ratio", 0.01)) * abs(median)
        )))
        sigma = 0.0
    else:
        sigma = 0.0
    if mad0_half is not None:
        auto_low = median - mad0_half
        auto_high = median + mad0_half
    else:
        auto_low = max(floor, median - float(cfg_get(config, "frame_count.mad_k_low", 3.0)) * sigma)
        auto_high = min(ceiling, median + float(cfg_get(config, "frame_count.mad_k_high", 8.0)) * sigma)
    auto_min = int(max(floor, np.floor(auto_low)))
    auto_max = int(min(ceiling, np.ceil(auto_high)))

    if manual_min is not None or manual_max is not None:
        if manual_min is not None and manual_max is not None:
            low, high = int(manual_min), int(manual_max)
            if low > high:
                raise ValueError(
                    f"frame_count.min={low} 不能大于 frame_count.max={high}，请检查配置"
                )
        elif manual_min is not None:
            # 只给下限：若数据集整体低于业务下限，auto 上限（=分布上界）已低于下限，
            # 失去约束意义，直接放开到 ceiling，避免把「仅下限」误当成固定区间。
            low = int(manual_min)
            high = auto_max if auto_max >= low else ceiling
        else:
            # 只给上限：对称处理，分布下界高于业务上限时放开到 floor。
            high = int(manual_max)
            low = auto_min if auto_min <= high else floor
        return {
            "mode": "manual",
            "min": low,
            "max": high,
            "source": {
                "min": "manual" if manual_min is not None else "auto",
                "max": "manual" if manual_max is not None else "auto",
            },
            "configured": {"min": manual_min, "max": manual_max},
            "median": round(median, 1),
            "mad": round(mad, 1),
        }

    if counts.size == 0:
        return {"mode": mode, "min": floor, "max": ceiling, "median": 0.0, "mad": 0.0}
    if mode == "fixed":
        return {
            "mode": "fixed", "min": floor, "max": ceiling,
            "median": median, "mad": mad, "source": {"min": "fixed", "max": "fixed"},
        }
    return {
        "mode": "auto",
        "min": auto_min,
        "max": auto_max,
        "median": round(median, 1),
        "mad": round(mad, 1),
        "source": {
            "min": "auto:mad0-min-width" if mad0_half is not None else "auto",
            "max": "auto:mad0-min-width" if mad0_half is not None else "auto",
        },
    }


def _resolve_action_thresholds(
    dataset: LeRobotDatasetView,
    config: dict[str, Any],
    *,
    key: str = ACTION_KEY,
    section: str = "action_jump",
) -> tuple[list[float], dict[str, Any]]:
    """逐维阈值由数据集自身的「正常运动包络」推导，见 `_thresholds_from_arrays`。"""
    if key not in dataset.frames.columns:
        return _thresholds_from_arrays([], config, fps=dataset.fps, section=section)

    arrays: list[np.ndarray] = []
    if "episode_index" in dataset.frames.columns:
        groups = (ep_df for _, ep_df in dataset.frames.groupby("episode_index", sort=False))
    else:
        # 非标准字段名：缺 episode 标识时整表当作一条序列推导，避免 groupby KeyError
        groups = iter([dataset.frames])
    for ep_df in groups:
        arr = _as_float_matrix(ep_df[key])
        if arr is not None:
            arrays.append(arr)
    return _thresholds_from_arrays(arrays, config, fps=dataset.fps, section=section)


def _align_dims(matrices: list[np.ndarray], label: str) -> tuple[list[np.ndarray], int]:
    """按多数维度保留矩阵，返回 (保留列表, 被排除数量)。

    np.vstack 要求除拼接轴外维度一致；跨 episode / 跨分片的列宽不一致会抛
    ValueError 让整次质检中断，因此统一按出现次数最多的维度对齐，
    被排除的矩阵由调用方决定是否降级为告警。
    """
    if not matrices:
        return [], 0
    dims = np.array([m.shape[1] for m in matrices])
    values, freq = np.unique(dims, return_counts=True)
    common_dim = int(values[int(np.argmax(freq))])
    kept = [m for m in matrices if m.shape[1] == common_dim]
    dropped = len(matrices) - len(kept)
    if dropped:
        LOG.warning("%d 个 %s 数组的宽度与多数（%d 维）不一致，已排除",
                    dropped, label, common_dim)
    return kept, dropped


def _thresholds_from_arrays(
    arrays: list[np.ndarray],
    config: dict[str, Any],
    fps: float | None = None,
    section: str = "action_jump",
) -> tuple[list[float], dict[str, Any]]:
    """由若干 episode 的 action 矩阵推导逐维跳变阈值。

    估计对象是「每集最大 |Δ|」而非逐帧 |Δ|：步进保持型 action 的逐帧分布双峰
    （大量 0 + 运动帧），median|Δ| 会被 0 拉崩，改用每集峰值刻画正常 episode
    的运动包络。episode 太少（< min_episodes_for_peak）时峰值分布的 MAD 不稳，
    退回逐帧非零 Δ 的稳健尺度。流式模式只采样部分 episode，因此把收集与计算拆开。
    """
    abs_floor = float(cfg_get(config, f"{section}.abs_floor", 1.0))
    mad_k = float(cfg_get(config, f"{section}.max_jump_mad_k", 3.0))
    min_eps = int(cfg_get(config, f"{section}.min_episodes_for_peak", 8))
    margin = float(cfg_get(config, f"{section}.min_margin_ratio", 0.05))
    # abs_floor 按 30Hz 标定，按数据集 fps 折算成「每秒位移」，与帧率解耦
    ref_fps = float(cfg_get(config, "action_jump.ref_fps", 30.0))
    eff_fps = float(fps) if fps and fps > 0 else ref_fps
    floor = abs_floor * ref_fps / eff_fps

    def _threshold(med: float, mad: float) -> float:
        # MAD=0 时阈值正好落在典型峰值上，留 5% 相对余量防 float32 舍入噪声误报
        spread = max(mad, margin * abs(med))
        return max(floor, med + mad_k * 1.4826 * spread)

    matrices: list[np.ndarray] = []
    for arr in arrays:
        arr = np.asarray(arr, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.ndim != 2 or arr.shape[0] < 2:
            continue
        matrices.append(arr)

    # 各 episode 的 action 宽度可能不一致（混合机器人配置、修复后残留的列等）。
    # 异常集不参与阈值推导，由 episode 级规则单独裁决。
    matrices, dim_mismatch = _align_dims(matrices, "episode action")

    deltas: list[np.ndarray] = []
    peaks: list[np.ndarray] = []
    for arr in matrices:
        d = np.abs(np.diff(arr, axis=0))
        d = d[np.isfinite(d).all(axis=1)]
        if d.size:
            deltas.append(d)
            peaks.append(d.max(axis=0))
    if not deltas:
        return [], {"abs_floor": abs_floor, "max_jump_mad_k": mad_k,
                    "min_margin_ratio": margin,
                    "fps": eff_fps,
                    "abs_floor_effective": round(floor, 6),
                    "spike_ratio": float(cfg_get(config, f"{section}.spike_ratio", 3.0)),
                    "diff_ref": [], "diff_ref_basis": "unavailable",
                    "dim_mismatch_episodes": dim_mismatch, "per_dim": []}

    # 全局逐维 median(|Δ|)：spike 核验中「剩余邻居 <2」时的回退参照。
    # 成对尖峰互为对方的邻域，剔除后邻居不足 2 个，只能退到全局口径；
    # 与阈值同源（同一批 Δ），流式模式下同样是采样口径，basis 如实记录。
    # 必须剔除 0：step-hold 维（大量零位移）直接取 median 会得 0，而 ref<=0 在
    # 回退路径上等价于「任何超阈都算尖峰」，会把快速运动整片放进来。
    stacked_all = np.vstack(deltas)
    diff_ref = []
    for d_i in range(stacked_all.shape[1]):
        col = stacked_all[:, d_i]
        col = col[col > 0]
        diff_ref.append(round(float(np.median(col)) if col.size else 0.0, 6))

    if len(peaks) >= min_eps:
        stacked = np.vstack(peaks)
        med = np.median(stacked, axis=0)
        mad = np.median(np.abs(stacked - med), axis=0)
        thresholds = np.array([
            _threshold(float(med[i]), float(mad[i])) for i in range(med.size)])
        basis, detail = "episode_max", {
            "episode_count": len(peaks),
            "dim_mismatch_episodes": dim_mismatch,
            "episode_max_median": [round(float(m), 4) for m in med],
            "episode_max_mad": [round(float(m), 4) for m in mad],
        }
    else:
        stacked = np.vstack(deltas)
        dims = stacked.shape[1]
        thresholds = np.full(dims, floor, dtype=float)
        nz_median = np.zeros(dims, dtype=float)
        for d_i in range(dims):
            col = stacked[:, d_i]
            col = col[col > 0]  # 保持帧的 0 不参与尺度估计
            if col.size == 0:
                continue
            m = np.median(col)
            nz_median[d_i] = m
            nz_mad = float(np.median(np.abs(col - m)))
            thresholds[d_i] = _threshold(float(m), nz_mad)
        basis, detail = "nonzero_delta", {
            "episode_count": len(peaks),
            "dim_mismatch_episodes": dim_mismatch,
            "nonzero_median": [round(float(m), 4) for m in nz_median],
        }
    return (
        [float(t) for t in thresholds],
        {
            "abs_floor": abs_floor,
            "abs_floor_effective": round(floor, 6),
            "fps": eff_fps,
            "max_jump_mad_k": mad_k,
            "min_margin_ratio": margin,
            "spike_ratio": float(cfg_get(config, f"{section}.spike_ratio", 3.0)),
            "basis": basis,
            "diff_ref": diff_ref,
            "diff_ref_basis": basis,
            "per_dim": [round(float(t), 4) for t in thresholds],
            **detail,
        },
    )


def _check_stats_staleness(
    dataset: LeRobotDatasetView, config: dict[str, Any], skipped: dict[str, str] | None = None
) -> DatasetIssue | None:
    """meta/stats.json 与实测数据偏差过大 → 训练归一化会全错（数据集级问题）。"""
    if not cfg_get(config, "stale_stats.enabled", True) or not dataset.stats:
        return None
    if STATE_KEY not in dataset.frames.columns:
        return None
    # 子集均值与全量 stats 不可比：--max_episodes 下必然出现偏差，
    # 报成「stats 过期」是假阳性，如实标注为跳过更诚实。
    if getattr(dataset, "truncated_to", None):
        if skipped is not None:
            skipped["stats_stale"] = (
                f"只扫描了前 {dataset.truncated_to} 个 episode，子集均值与全量 stats.json 不可比，"
                "需全量运行才能判定 stats 是否过期"
            )
        return None
    limit = int(cfg_get(config, "stale_stats.sample_episodes", 200))
    if "episode_index" in dataset.frames.columns:
        rows = dataset.frames[dataset.frames["episode_index"].isin(dataset.episode_indices[:limit])]
    else:
        # 非标准字段名：没有 episode 标识时用整表均值，仍能做 stats 时效性判定
        rows = dataset.frames
    arr = _as_float_matrix(rows[STATE_KEY])
    if arr is None:
        return None
    arr = arr[np.isfinite(arr).all(axis=1)]
    if arr.size == 0:
        return None
    return _stats_staleness_issue(dataset.stats, arr.mean(axis=0), config)


def _stats_staleness_issue(
    stats: dict[str, Any], actual_mean: np.ndarray | None, config: dict[str, Any]
) -> DatasetIssue | None:
    """stats 与实测均值的偏差判定；流式模式直接传入采样算出的均值。"""
    if actual_mean is None or not stats:
        return None
    actual_mean = np.asarray(actual_mean, dtype=float)
    try:
        recorded_arr = np.asarray((stats.get(STATE_KEY) or {}).get("mean"), dtype=float)
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
    # 尺度基准：只用 |mean| 会让「均值本就接近 0」的维度（关节零位、对称信号）
    # 把微小漂移放大成上百倍，产生看似惊人的假数字。改用该维自身的量级
    # max(|mean|, std) 作为分母：既有意义，也不会因近零维度爆表。
    recorded_std = None
    try:
        recorded_std = np.asarray((stats.get(STATE_KEY) or {}).get("std"), dtype=float)
    except (TypeError, ValueError):
        recorded_std = None
    if recorded_std is None or recorded_std.shape != recorded_arr.shape:
        recorded_std = np.zeros_like(recorded_arr)
    scale = np.maximum.reduce([np.abs(recorded_arr), np.abs(recorded_std), np.full_like(recorded_arr, 1e-6)])
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
    """校验 index 语义，判据按 codebase_version 区分。

    v3.0：index 是全局绝对帧号，必须 0..N-1 各出现一次。
    v2.x：每个 episode 一个 parquet，index 是 episode 内帧号，**每集从 0 起是规范**。
          用 v3 的「全表唯一」判据会把完全合规的 v2.1 数据判成 index 断裂——
          这正是 humanoid_everyday_h1（1418 集、每集归零）被误报的原因。
    """
    if "index" not in dataset.frames.columns:
        return None
    idx = pd.to_numeric(dataset.frames["index"], errors="coerce").to_numpy()

    if str(dataset.codebase_version or "").startswith("v2"):
        if "episode_index" not in dataset.frames.columns:
            return None
        if int(np.isnan(idx).sum()):
            return DatasetIssue("meta_index_inconsistent", "index 列含 NaN",
                                {"nan_count": int(np.isnan(idx).sum())}, repair="index")
        bad: list[int] = []
        for ep, sub in dataset.frames.groupby("episode_index", sort=True):
            vals = pd.to_numeric(sub["index"], errors="coerce").to_numpy()
            vals = vals[~np.isnan(vals)]
            n = int(vals.size)
            if n == 0:
                continue
            # v2.x 只要求 episode 内连续且不重复：起点为 0（原生每集归零）或
            # 全局偏移（从 v3 沿用，kuavo4 实测 ep1 为 131..471）都属合法。
            contiguous = (vals.max() - vals.min() + 1) == n
            if not contiguous or np.unique(vals).size != n:
                bad.append(int(ep))
        if not bad:
            return None
        return DatasetIssue(
            "meta_index_inconsistent",
            "v2.x 每个 episode 的 index 应在 episode 内连续且不重复",
            {"bad_episodes": len(bad), "sample": bad[:5]},
            repair="index",
        )

    # NaN/Inf 不能参与 min/max（int(nan) 会抛 ValueError）；先剔除再统计，
    # 非有限值数量单独传给 _index_continuity_issue 判定。
    non_finite = ~np.isfinite(idx)
    unique = np.unique(idx[~non_finite])
    return _index_continuity_issue(
        rows=dataset.num_frames,
        count=int(unique.size),
        min_index=int(unique[0]) if unique.size else None,
        max_index=int(unique[-1]) if unique.size else None,
        nan_count=int(non_finite.sum()),
    )


def _check_episode_alignment(dataset: LeRobotDatasetView) -> DatasetIssue | None:
    """meta/episodes 的 episode_index 必须与 data 中出现的 episode 一致。

    behavior1k 实测：meta 使用全局 ID（420010, 420020, …）而 data 用 0..N-1，
    直接 join 会全部落空，导致 task/视频定位整体失效。
    """
    if dataset.frames.empty or "episode_index" not in dataset.frames.columns:
        return None  # 流式模式在 streaming._build_context 中另行检测
    if dataset.episodes.empty or "episode_index" not in dataset.episodes.columns:
        return None
    data_values = pd.to_numeric(dataset.frames["episode_index"], errors="coerce").dropna()
    meta_values = pd.to_numeric(dataset.episodes["episode_index"], errors="coerce").dropna()
    data_ids = {int(x) for x in data_values}
    meta_ids = {int(x) for x in meta_values}
    issue = _episode_alignment_issue(data_ids, meta_ids)
    return _annotate_reconciliation(issue, dataset)


def _check_meta_duplicates(dataset: LeRobotDatasetView) -> DatasetIssue | None:
    """meta/episodes 同一 episode 多行且 length/视频窗冲突。

    典型来源是多次导出叠加：同一集 meta 有两行、数据又把两次录制拼成一行集，
    静默保留首行会让后续只看到「帧数不符」，看不到真正的重复导出。
    """
    conflicts = getattr(dataset, "meta_duplicate_conflicts", None) or {}
    episodes = conflicts.get("episodes") or []
    if not episodes:
        return None
    return DatasetIssue(
        "meta_episode_duplicate",
        f"meta/episodes 同一 episode 多行且内容冲突（{len(episodes)} 集）",
        {
            "episodes": episodes,
            "duplicate_rows": conflicts.get("duplicate_rows", 0),
            "rows": conflicts.get("rows"),
            "unique_episodes": conflicts.get("unique_episodes"),
        },
    )


def _check_feature_declarations(dataset: LeRobotDatasetView) -> DatasetIssue | None:
    """info.json 的 features 必须覆盖磁盘与 meta 中出现的每一路相机。

    Pico 上传链路丢过相机特征：``features`` 里删掉 wrist，`videos/` 目录与
    `meta/episodes` 的 ``videos/*`` 时间窗仍在。读侧只遍历 ``features``，
    该相机因此静默消失，所有帧级规则照常通过——报告必须显式指出。
    """
    declared = set((dataset.features or {}).keys())
    # v2.x 的部分数据集用短名存目录（features 为 observation.images.egocentric，
    # 目录却是 videos/chunk-000/egocentric/），不能因此误判为「缺声明」。
    declared |= {key.rsplit(".", 1)[-1] for key in declared}
    cameras: set[str] = set()
    evidence: set[str] = set()

    video_root = dataset.root / "videos"
    if video_root.is_dir():
        # v3.0：videos/<camera>/chunk-xxx/file-xxx.mp4
        # v2.x：videos/chunk-xxx/<camera>/episode_xxx.mp4
        for entry in video_root.iterdir():
            if not entry.is_dir():
                continue
            if entry.name.startswith("chunk-"):
                cameras.update(c.name for c in entry.iterdir() if c.is_dir())
            else:
                cameras.add(entry.name)
            evidence.add("videos/")

    episodes = getattr(dataset, "episodes", None)
    if episodes is not None and len(episodes.columns):
        for column in episodes.columns:
            parts = str(column).split("/")
            if len(parts) >= 3 and parts[0] == "videos":
                cameras.add(parts[1])
                evidence.add("meta/episodes")

    missing = sorted(cameras - declared)
    if not missing:
        return None
    return DatasetIssue(
        "feature_missing_declaration",
        f"info.json 缺少 {len(missing)} 路相机的特征声明",
        {"cameras": missing, "evidence": sorted(evidence)},
    )


def _annotate_reconciliation(issue: DatasetIssue | None, dataset: LeRobotDatasetView) -> DatasetIssue | None:
    """编号不一致已在加载器完成自动对齐时，在问题条目上标注「已解决」。

    否则报告读者会把它当成未处理的阻断项，与实际结论（本次质检基于正确关联）冲突。
    """
    if issue is None:
        return None
    recon = getattr(dataset, "id_reconciliation", None) or {}
    mode = str(recon.get("mode", "native"))
    if mode == "remapped":
        issue.detail["auto_reconciled"] = True
        issue.detail["reconciliation"] = {
            "meta_id_range": recon.get("meta_id_range"),
            "data_id_range": recon.get("data_id_range"),
            "length_agreement": recon.get("length_agreement"),
            "video_agreement": recon.get("video_agreement"),
        }
        issue.detail["impact"] = "已按顺序映射并通过校验，本次结论基于正确关联；建议上游统一编号"
    elif mode == "ambiguous":
        issue.detail["auto_reconciled"] = False
        issue.detail["impact"] = "无法可靠映射，依赖 meta 的检查不可评估，禁止据此批量剔除"
    return issue


def _episode_alignment_issue(data_ids: set[int], meta_ids: set[int]) -> DatasetIssue | None:
    """data/meta 两侧 episode_index 集合不一致时返回数据集级问题。"""
    if not data_ids or not meta_ids or data_ids == meta_ids:
        return None
    return DatasetIssue(
        "meta_episode_mismatch",
        "meta/episodes 的 episode_index 与数据不一致",
        {
            "data_episodes": len(data_ids),
            "meta_episodes": len(meta_ids),
            "overlap": len(data_ids & meta_ids),
            "data_only_sample": sorted(data_ids - meta_ids)[:5],
            "meta_only_sample": sorted(meta_ids - data_ids)[:5],
        },
    )


def _index_continuity_issue(
    rows: int,
    count: int | None,
    min_index: int | None,
    max_index: int | None,
    nan_count: int = 0,
) -> DatasetIssue | None:
    """由 (行数, 去重个数, 最小/最大) 判定连续性。

    流式模式不保留全量 index，只统计 (rows, min, max, NaN 数)，此时 count 传
    None：整数全部落在 [min, max] 且恰好 rows 个，当且仅当 min==0 且
    max==rows-1 时等价于 0..N-1 连续（鸽巢原理，无需知道去重个数）。
    """
    if nan_count:
        return DatasetIssue("meta_index_inconsistent", "index 列含 NaN",
                            {"nan_count": int(nan_count)}, repair="index")
    if rows == 0:
        return None
    if count is None:
        if min_index == 0 and max_index is not None and max_index == rows - 1:
            return None
        return DatasetIssue(
            "meta_index_inconsistent",
            "全局 index 不连续或存在重复",
            {"rows": rows, "min": min_index, "max": max_index},
            repair="index",
        )
    if count == rows and min_index == 0 and max_index == rows - 1:
        return None
    return DatasetIssue(
        "meta_index_inconsistent",
        "全局 index 不连续或存在重复",
        {"rows": rows, "unique": int(count), "min": min_index, "max": max_index},
        repair="index",
    )


def _as_float_matrix(series: pd.Series) -> np.ndarray | None:
    """把向量列转成 (frames, dims) 的 float 矩阵；无法解析时返回 None。"""
    try:
        arr = np.asarray(series.to_list(), dtype=float)
    except (TypeError, ValueError):
        return None
    if arr.ndim == 0:
        return None
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)
    return arr

