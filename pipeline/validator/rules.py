# Copyright (c) 2026 wu
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""逐 episode 的检查规则（数值、结构、视频内容）。"""

from __future__ import annotations

import hashlib
import logging
import math
from typing import Any, NamedTuple

import numpy as np
import pandas as pd

from ..config import cfg_get
from ..loader import LeRobotDatasetView
from .context import DatasetContext, scale_frames
from .probes import FFMPEG, FFPROBE, probe_frame_brightness, probe_video_frames
from .types import EpisodeResult

LOG = logging.getLogger(__name__)


# 数值类检查
def check_finite(
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


def _quaternion_dim_mask(dim_names: list[str] | None, dim: int) -> np.ndarray | None:
    """按特征维度名识别四元数维度。

    返回长度 dim 的布尔掩码；维度名缺失（元数据不可用）时返回 None，
    调用方据此回退到数值启发式。名字存在但不含四元数时返回全 False，
    表示「这些维度不是四元数，q≡-q 不适用」。
    """
    if not dim_names or len(dim_names) < dim:
        return None
    names = [str(n).strip().lower() for n in dim_names[:dim]]
    mask = np.zeros(dim, dtype=bool)
    for i, name in enumerate(names):
        if "quat" in name or "orient" in name:
            mask[i] = True
    # 同前缀的 x/y/z/w 四连也视为一个四元数组（如 pose.x / pose.y / pose.z / pose.w）
    for i in range(dim - 3):
        parts = [
            n.rsplit(".", 1) if "." in n else (n.rsplit("_", 1) if "_" in n else [n, n])
            for n in names[i:i + 4]
        ]
        if len({p[0] for p in parts}) == 1 and {p[-1] for p in parts} == {"x", "y", "z", "w"}:
            mask[i:i + 4] = True
    return mask


class _SpikeScan(NamedTuple):
    """spike 核验的公共中间结果；两条规则各自决定证据字段与严重度。"""

    deltas: np.ndarray
    dim: int
    thr: np.ndarray
    discrete_dims: np.ndarray
    spike_ratio: float
    window: int
    kept: list
    sign_flips: int
    smooth_ignored: int
    #: 连续 ≥3 个超阈 Δ 的包络段数（整段跳过，不参与尖峰核验）
    envelope_runs: int
    #: 长度恰为 2 且「异号往返」的脉冲段中出现命中的个数
    paired_spike: int
    #: 命中的脉冲物理帧（同一脉冲的 up/down 两个 diff 去重为一个帧）
    spike_frames: list
    #: 按维拆分的脉冲物理帧（供 stats_pollution 精确剔除）
    spike_frames_by_dim: dict[int, list[int]]
    #: 每个命中的 ref 来源（local/global），供调用方按需汇总
    hit_sources: dict[tuple[int, int], str]
    global_hits: int
    local_hits: int


#: 「往返」判据：两段有符号 Δ 的净位移不超过单段幅值的 20%
_PULSE_NET_RATIO = 0.2


def _pulse_shaped(run: np.ndarray, arr: np.ndarray, d: int) -> bool:
    """长度 2 的 run 是否为脉冲（出去再回来）而非快速斜坡。

    同号 2-diff（如 pusht 的连续加速帧）净位移约为单段幅值之和，是正常运动，
    不能借全局 median（运动尺度而非噪声底）兜底判成尖峰。
    """
    if run.size != 2:
        return False
    f0, f1 = int(run[0]), int(run[1])
    if f1 != f0 + 1 or f0 + 2 >= arr.shape[0]:
        return False
    d0 = float(arr[f0 + 1, d] - arr[f0, d])
    d1 = float(arr[f0 + 2, d] - arr[f0 + 1, d])
    if not (np.isfinite(d0) and np.isfinite(d1)) or d0 == 0.0 or d1 == 0.0:
        return False
    if np.sign(d0) == np.sign(d1):
        return False
    return abs(d0 + d1) <= _PULSE_NET_RATIO * max(abs(d0), abs(d1))


def _scan_spikes(
    arr: np.ndarray,
    thresholds: list[float],
    config: dict[str, Any],
    *,
    prefix: str,
    min_frames: int,
    fps: float | None = None,
    dim_names: list[str] | None = None,
    diff_ref: list[float] | None = None,
) -> _SpikeScan | None:
    """check_action_jump / check_state_jump 的公共核验（提取自两份重复实现）。

    判据分四步：超阈值（必要条件）→ 离散维豁免 / 四元数取反豁免 →
    连续 run 分组（≥3 视为运动包络整体跳过）→ 相对邻域突跳核验。
    邻域 ref 计算时剔除其他超阈 Δ：成对尖峰（脉冲的 up/down 两个 diff）互为
    对方的邻域，会把中位数抬到 ~脉冲幅值的一半，spike_ratio=3 判不过而双双漏检
    （tokyo fps=5 实测 4 个 ±2π 脉冲全部漏报）。剔除后剩余邻居 <2 时，只有
    异号往返的脉冲段才回退全局逐维 median(|Δ|)（ctx 提供，来源口径见
    resolved_thresholds.diff_ref）；同号 2-diff 是快速斜坡，不借全局回退兜底。
    flip 参数刻意沿用 action_jump.*（state_jump 历史上就复用同一组标定）。
    min_frames / 配置前缀由调用方决定，本函数不做任何严重度判断。
    """
    if arr.shape[0] < min_frames or not thresholds:
        return None
    deltas = np.abs(np.diff(arr, axis=0))
    deltas = np.nan_to_num(deltas, nan=0.0, posinf=np.inf)
    dim = min(deltas.shape[1], len(thresholds))
    thr = np.asarray(thresholds[:dim], dtype=float)
    # 夹爪等二值/离散维的「开合」本身就是 ±2 的跳变，属正常控制指令，不能按连续维阈值判异常
    discrete_max = int(cfg_get(config, f"{prefix}.discrete_max_unique", 3))
    nunique = np.array([
        len(np.unique(arr[np.isfinite(arr[:, d]), d])) if np.isfinite(arr[:, d]).any() else 0
        for d in range(dim)
    ])
    discrete_dims = np.where(nunique <= discrete_max)[0]
    if discrete_dims.size:
        deltas[:, discrete_dims] = 0.0
    over = deltas[:, :dim] > thr
    if not over.any():
        return None
    frames, dims = np.where(over)
    spike_ratio = float(cfg_get(config, f"{prefix}.spike_ratio", 3.0))
    # spike_window 按 30Hz 标定（默认 3 帧≈0.1s），高帧率下需按 fps 折算，
    # 否则 120Hz 的邻域过短，正常的连续加减速会被误判成单帧尖峰。
    window = scale_frames(int(cfg_get(config, f"{prefix}.spike_window", 3)), config, fps)
    # 四元数 q 与 -q 表示同一姿态：整体取反会让 4 个分量在同一帧同时出现 ~2|x| 的跳变。
    # 这是表示等价而非数据缺陷（120Hz 光捕数据里很常见），需要逐帧成组识别后排除；
    # 只翻转单个维度的仍按真实跳变处理。
    flip_tol = float(cfg_get(config, "action_jump.sign_flip_tol", 0.25))
    flip_min_dims = int(cfg_get(config, "action_jump.sign_flip_min_dims", 2))
    head, tail = arr[:-1, :dim], arr[1:, :dim]
    finite = np.isfinite(head) & np.isfinite(tail)
    with np.errstate(invalid="ignore"):
        step = np.abs(tail - head)
        residual = np.abs(head + tail)
        # 跳变量主要由「正负号反了」贡献：残差 |x_t + x_{t-1}| 远小于步长 |x_t - x_{t-1}|
        mirror = (residual <= flip_tol * step) & (step > 0)
    mirror &= finite
    # 只在四元数维度内识别整体取反：元数据可用时非四元数维度的符号变化
    # 属于真实跳变，不能借「多维同时翻转」蒙混过关；元数据缺失才退回数值启发式。
    quat_mask = _quaternion_dim_mask(dim_names, dim)
    if quat_mask is not None:
        mirror &= quat_mask[None, :]
    # 只有同一帧上「多个维同时镜像翻转」才算四元数整体取反，单维翻转仍按真实跳变处理
    flip_mask = mirror & (mirror.sum(axis=1, keepdims=True) >= flip_min_dims)

    # 四元数取反先行剔除（保号：flip 的记账口径与提取前一致），剩下的超阈 Δ
    # 按维分组为连续 run——run≥3 是连续加速/减速的运动包络，整体跳过；
    # run≤2 才是尖峰候选（成对脉冲即长度 2 的 run）。
    sign_flips = int(flip_mask[frames, dims].sum())
    candidate_set = {(int(f), int(d)) for f, d in zip(frames.tolist(), dims.tolist())
                     if not flip_mask[f, d]}
    envelope_runs = 0
    for d in range(dim):
        fs = sorted(f for f, dd in candidate_set if dd == d)
        if len(fs) < 3:
            continue
        splits = np.where(np.diff(fs) != 1)[0] + 1
        for run in np.split(np.asarray(fs), splits):
            if run.size >= 3:
                envelope_runs += 1
                for f in run:
                    candidate_set.discard((int(f), int(d)))

    kept: list = []
    global_hits = 0
    local_hits = 0
    paired_spike = 0
    spike_frames: list = []
    spike_frames_by_dim: dict[int, list[int]] = {}
    hit_sources: dict[tuple[int, int], str] = {}
    # 按维重走候选（保持 run 与命中的对应关系）
    for d in range(dim):
        fs = sorted(f for f, dd in candidate_set if dd == d)
        if not fs:
            continue
        splits = np.where(np.diff(fs) != 1)[0] + 1
        for run in np.split(np.asarray(fs), splits):
            pulse_run = _pulse_shaped(run, arr, d)
            hits_in_run = 0
            for f in run.tolist():
                lo, hi = max(0, f - window), min(deltas.shape[0], f + window + 1)
                neighbours = np.delete(deltas[lo:hi, d], f - lo)
                neighbours = neighbours[np.isfinite(neighbours)]
                # 剔除邻域里的其他超阈 Δ：成对尖峰会互相把对方抬成「邻域中位数」
                if thr.size:
                    neighbours = neighbours[neighbours <= thr[d]]
                ref_source = "local"
                if neighbours.size >= 2:
                    ref = float(np.median(neighbours))
                elif run.size == 2 and not pulse_run:
                    # 同号 2-diff 是快速斜坡（出去不回来），剔除同伴后已无干净邻域，
                    # 而此时回退全局 median（运动尺度）会把正常加速判成尖峰 → 不命中
                    continue
                else:
                    # 剩余邻居不足（fps 低时窗口塌缩，成对脉冲剔除后只剩 0~1 个）：
                    # 只有异号往返的脉冲才允许回退全局逐维 median(|Δ|)；
                    # 孤立单帧尖峰（run 长度 1）维持旧语义，无全局统计时 ref=0。
                    g = float(diff_ref[d]) if diff_ref and d < len(diff_ref) else 0.0
                    ref = float(g)
                    ref_source = "global"
                if ref <= 0 or deltas[f, d] >= spike_ratio * ref:
                    kept.append((f, d))
                    hit_sources[(f, d)] = ref_source
                    hits_in_run += 1
                    if ref_source == "global":
                        global_hits += 1
                    else:
                        local_hits += 1
            if hits_in_run:
                if pulse_run:
                    paired_spike += 1
                # 脉冲物理帧：run 覆盖 diff i..j → 候选帧 i+1..j+1，取 |值| 最大者
                lo_f, hi_f = int(run[0]) + 1, min(int(run[-1]) + 2, arr.shape[0])
                vals = np.abs(arr[lo_f:hi_f, d])
                if vals.size:
                    frame = int(lo_f + int(np.argmax(vals)))
                    spike_frames.append(frame)
                    spike_frames_by_dim.setdefault(int(d), []).append(frame)
    smooth_ignored = int(frames.size) - len(kept) - sign_flips
    return _SpikeScan(
        deltas=deltas, dim=dim, thr=thr, discrete_dims=discrete_dims,
        spike_ratio=spike_ratio, window=window, kept=kept,
        sign_flips=sign_flips, smooth_ignored=smooth_ignored,
        envelope_runs=envelope_runs, paired_spike=paired_spike,
        spike_frames=sorted(set(spike_frames)),
        spike_frames_by_dim={k: sorted(set(v)) for k, v in spike_frames_by_dim.items()},
        hit_sources=hit_sources,
        global_hits=global_hits, local_hits=local_hits,
    )


def check_action_jump(
    arr: np.ndarray,
    res: EpisodeResult,
    thresholds: list[float],
    config: dict[str, Any],
    fps: float | None = None,
    dim_names: list[str] | None = None,
    diff_ref: list[float] | None = None,
) -> None:
    scan = _scan_spikes(arr, thresholds, config, prefix="action_jump", min_frames=2,
                        fps=fps, dim_names=dim_names, diff_ref=diff_ref)
    if scan is None or not scan.kept:
        return
    kept = scan.kept
    kept_deltas = np.array([scan.deltas[f, d] for f, d in kept], dtype=float)
    worst_pos = int(np.argmax(kept_deltas))
    # 超阈幅度分级：这个阈值是「每集峰值」的 3σ 上界，本身就在分布尾部，
    # 所以 1.0~1.5 倍的命中几乎都是「快速但连续」的正常动作，报成 warning 只会制造
    # 稳定误报（实测 pusht 17/206）。真正的跳变通常远超阈值，按倍数区分。
    worst_dim = kept[worst_pos][1]
    exceed_ratio = float(kept_deltas[worst_pos] / max(float(scan.thr[worst_dim]), 1e-9))
    downgrade_ratio = float(cfg_get(config, "action_jump.downgrade_ratio", 1.5))
    escalate_ratio = cfg_get(config, "action_jump.escalate_ratio")
    severity: str | None = None
    if escalate_ratio is not None and exceed_ratio >= float(escalate_ratio):
        severity = "error"
    elif exceed_ratio < downgrade_ratio:
        severity = "info"
    # 成对尖峰不做幅度降级：downgrade_ratio 针对「快速但连续」的正常运动，
    # 而成对 ±2π 脉冲是语义伪影（如 ±π 卷绕的忠实 delta），keep 会把它放回训练数据。
    # 下限 warning（进复核）；escalate_ratio 仍可升到 error。
    if scan.paired_spike > 0 and severity != "error":
        severity = "warning"
    if scan.global_hits and scan.local_hits:
        ref_source = "mixed"
    elif scan.global_hits:
        ref_source = "global"
    else:
        ref_source = "local"
    # 归一化污染证据（G5c）：成对脉冲会把该维的 std 抬高数个量级
    # （tokyo action dim5：4 个 ±2π 脉冲使 std 0.003 → 0.115），
    # 训练侧按 recorded stats 归一化会在这些帧上炸出离群值。
    # 只报证据不改 stats——修复与否由下游决定；reporter 聚合并对照 stats.json。
    stats_pollution: list[dict[str, Any]] = []
    if scan.paired_spike > 0:
        for d in sorted({d for _, d in kept}):
            pulse_frames = scan.spike_frames_by_dim.get(int(d), [])
            if not pulse_frames:
                continue
            col = arr[:, d]
            finite = col[np.isfinite(col)]
            if finite.size < 2:
                continue
            mask = np.ones(col.size, dtype=bool)
            for fr in pulse_frames:
                if 0 <= fr < mask.size:
                    mask[fr] = False
            trimmed = col[mask]
            trimmed = trimmed[np.isfinite(trimmed)]
            if trimmed.size < 2:
                continue
            stats_pollution.append({
                "dim": int(d),
                "spike_count": len(pulse_frames),
                "observed_std": round(float(np.std(finite)), 6),
                "trimmed_std": round(float(np.std(trimmed)), 6),
                "n_frames": int(col.size),
            })
    res.add(
        "action_jump",
        severity=severity,
        exceed_ratio=round(exceed_ratio, 2),
        jump_count=len(kept),
        max_delta=round(float(kept_deltas[worst_pos]), 4),
        thresholds_used=[round(float(t), 4) for t in scan.thr],
        discrete_dims=[int(x) for x in scan.discrete_dims],
        smooth_peaks_ignored=scan.smooth_ignored,
        sign_flips_ignored=scan.sign_flips,
        envelope_runs_ignored=scan.envelope_runs,
        spike_ratio=scan.spike_ratio,
        paired_spike=scan.paired_spike,
        spike_frames=scan.spike_frames[:20],
        ref_source=ref_source,
        **({"stats_pollution": stats_pollution} if stats_pollution else {}),
        worst_frame=kept[worst_pos][0],
        worst_dim=worst_dim,
        examples=[
            {"frame": f, "dim": d, "delta": round(float(scan.deltas[f, d]), 4),
             "threshold": round(float(scan.thr[d]), 4)}
            for f, d in kept[:5]
        ],
    )


def _is_pi_wrap(prev: float, nxt: float, delta: float, tol: float) -> bool:
    """±π 表示卷绕判据：两侧端点分别贴住 −π/+π（容差内），且 |Δ| ≈ 2π。

    刻意不用「整维历史值域 ⊆ [−π,π]」——合法摆动维（tokyo dim3 实测）的值域
    也在该区间内，会被误豁免；端点判据只认真正跨边界的那个 transition。
    """
    two_pi = 2.0 * math.pi
    up = abs(prev - (-math.pi)) <= tol and abs(nxt - math.pi) <= tol
    down = abs(prev - math.pi) <= tol and abs(nxt - (-math.pi)) <= tol
    return (up or down) and abs(delta - two_pi) <= tol


def check_state_jump(
    arr: np.ndarray,
    res: EpisodeResult,
    thresholds: list[float],
    config: dict[str, Any],
    fps: float | None = None,
    dim_names: list[str] | None = None,
    diff_ref: list[float] | None = None,
) -> None:
    """state 单帧突跳（追踪丢失、设备重置、瞬移）的软规则。

    与 action_jump 的判据一致（超阈值 + 相对邻域尖峰），但只报 warning：
    state 突跳可能来自真实快速运动，也可能只是采集中断，必须人工确认；
    RULE_MAP 中该规则 tier=soft，策略层永远不会据此自动剔除。

    命中里满足 ±π 卷绕判据（端点分别贴 ±π、|Δ|≈2π）的部分是**表示层等价**
    而非数据缺陷：转入独立的 info 级 `state_wrap` 留痕（数据自洽、不修复，
    下游按连续量使用需自行 unwrap），避免把合法表示当成突跳送复核。
    """
    if not cfg_get(config, "state_jump.enabled", True):
        return
    scan = _scan_spikes(arr, thresholds, config, prefix="state_jump", min_frames=3,
                        fps=fps, dim_names=dim_names, diff_ref=diff_ref)
    if scan is None or not scan.kept:
        return
    wrap_tol = float(cfg_get(config, "state_jump.wrap_endpoint_tol", 0.05))
    wraps: list[dict[str, Any]] = []
    real_hits: list = []
    for f, d in scan.kept:
        prev_v = float(arr[f, d])
        next_v = float(arr[f + 1, d])
        if _is_pi_wrap(prev_v, next_v, float(scan.deltas[f, d]), wrap_tol):
            wraps.append({
                "frame": int(f + 1),
                "prev": round(prev_v, 4),
                "next": round(next_v, 4),
                "delta": round(float(scan.deltas[f, d]), 4),
                "dim": int(d),
            })
        else:
            real_hits.append((f, d))
    if wraps:
        res.add(
            "state_wrap",
            wrap_ignored=len(wraps),
            examples=wraps[:5],
        )
    if not real_hits:
        return
    kept = real_hits
    kept_deltas = np.array([scan.deltas[f, d] for f, d in kept], dtype=float)
    worst_pos = int(np.argmax(kept_deltas))
    # 按真实命中（剔除卷绕后）汇总 ref 来源，避免把只出现在 wrap 上的 global 算进来
    sources = {scan.hit_sources.get((f, d), "local") for f, d in kept}
    ref_source = "mixed" if len(sources) > 1 else next(iter(sources))
    res.add(
        "state_jump",
        severity="warning",  # 软规则上限：即使配置成 error 也只提示复核
        jump_count=len(kept),
        max_delta=round(float(kept_deltas[worst_pos]), 4),
        thresholds_used=[round(float(t), 4) for t in scan.thr],
        discrete_dims=[int(x) for x in scan.discrete_dims],
        paired_spike=scan.paired_spike,
        ref_source=ref_source,
        worst_frame=kept[worst_pos][0],
        worst_dim=kept[worst_pos][1],
        note="state 单帧突跳，疑似追踪丢失/重置/瞬移，需人工确认",
        examples=[
            {"frame": f, "dim": d, "delta": round(float(scan.deltas[f, d]), 4),
             "threshold": round(float(scan.thr[d]), 4)}
            for f, d in kept[:5]
        ],
        **({"wrap_ignored": len(wraps)} if wraps else {}),
    )


def check_static(
    arr: np.ndarray,
    kind: str,
    res: EpisodeResult,
    config: dict[str, Any],
    fps: float | None = None,
) -> None:
    """空转检测：整条序列几乎无变化 → 机器人没真正执行任务。"""
    if not cfg_get(config, "static_episode.enabled", True):
        return
    min_frames = scale_frames(int(cfg_get(config, "static_episode.min_frames_checked", 5)), config, fps)
    if arr.shape[0] < min_frames:
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


def check_joint_limits(
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
    # 配置限位长度可以大于 info 声明的关节名数量（数据列比 info 宽是野外常见情况），
    # 越界维度按 dim_N 兜底，不能让报错路径自己 IndexError。
    names = list(dataset.joint_names)
    res.add(
        "joint_limit_violation",
        violation_count=int(below.sum() + above.sum()),
        dims=[int(d) for d in np.where((below | above).any(axis=0))[0]],
        joint_names=[
            names[d] if d < len(names) else f"dim_{d}"
            for d in range(dim) if (below | above)[:, d].any()
        ][:5],
    )


def check_duplicate(res: EpisodeResult, arr: np.ndarray, ctx: DatasetContext, config: dict[str, Any]) -> None:
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


# 结构类检查
def check_frame_count(res: EpisodeResult, bounds: dict[str, Any], config: dict[str, Any]) -> None:
    """帧数区间检查。

    方向决定严重度：过短降为 warning 交人工复核——任务步数本就少的合法数据
    （如 LIBERO）容易被自适应下限误杀；过长通常意味着拼接失败或未正确切分，
    保持 error 并剔除。
    """
    n = res.frame_count
    low, high = int(bounds["min"]), int(bounds["max"])
    if low <= n <= high:
        return
    too_short = n < low
    res.add(
        "frame_count_abnormal",
        severity="warning" if too_short else None,
        frame_count=n,
        min_allowed=low,
        max_allowed=high,
        direction="too_few" if too_short else "too_many",
        direction_label="过短" if too_short else "过长",
        how_derived=f"median={bounds.get('median')} mad={bounds.get('mad')} mode={bounds.get('mode')}",
    )


def check_timestamp(ep_df: pd.DataFrame, res: EpisodeResult, config: dict[str, Any], fps: float) -> None:
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


def check_alignment(
    ep_df: pd.DataFrame,
    res: EpisodeResult,
    dataset: LeRobotDatasetView,
    config: dict[str, Any],
    fps: float,
) -> None:
    """多模态对齐：时间戳 ↔ 标称时间轴 ↔ 视频时间窗 ↔ frame_index 连续性。

    逐模态「各自是否合理」由 nan/timestamp/frame_mismatch 规则负责；这里只回答
    一个跨模态问题：它们在时间上是否指向同一段物理时间。
    """
    if not cfg_get(config, "alignment.enabled", True):
        return
    _check_timestamp_drift(ep_df, res, dataset, config, fps)
    _check_video_span(res, dataset, config, fps)
    _check_frame_index_gap(ep_df, res)


def _meta_table(dataset: LeRobotDatasetView) -> pd.DataFrame | None:
    """并行模式下 worker 拿到的是 EpisodeDatasetStub（episodes_meta），串行模式是 episodes。"""
    for attr in ("episodes", "episodes_meta"):
        table = getattr(dataset, attr, None)
        if isinstance(table, pd.DataFrame) and not table.empty:
            return table
    return None


def _video_windows(dataset: LeRobotDatasetView, episode_index: int) -> list[dict[str, Any]]:
    """该 episode 全部 meta 段的相机时间窗 [from, to]（按源文件顺序）。

    多段/双 take 数据集（同一 episode_index 多行 meta，每行自带 length 与窗口）
    必须按段取窗：旧口径只取第一行窗口、拿全集行数去比，会把正常双 take
    系统性误报成 video_span_mismatch / frame_mismatch（29 个 drop 集假阳性
    的根因，复盘见 kaggle_ab/results.md）。没有视频元数据时返回空列表。
    """
    if hasattr(dataset, "episode_segments"):
        segments = dataset.episode_segments(episode_index)
    else:  # 极简替身（测试里的 SimpleNamespace 等只带 episodes 表）
        meta = _meta_table(dataset)
        if meta is None or "episode_index" not in meta.columns:
            return []
        segments = meta[meta["episode_index"] == episode_index]
    if segments.empty or "episode_index" not in segments.columns:
        return []
    windows = []
    for column in segments.columns:
        name = str(column)
        if not (name.startswith("videos/") and name.endswith("/to_timestamp")):
            continue
        camera = name[len("videos/"):-len("/to_timestamp")]
        from_col = f"videos/{camera}/from_timestamp"
        if from_col not in segments.columns:
            continue
        for _, row in segments.iterrows():
            to_ts = row.get(column)
            from_ts = row.get(from_col)
            if pd.isna(to_ts) or pd.isna(from_ts):
                continue
            windows.append({"camera": camera, "from": float(from_ts), "to": float(to_ts)})
    return windows


def _check_timestamp_drift(
    ep_df: pd.DataFrame, res: EpisodeResult, dataset: LeRobotDatasetView,
    config: dict[str, Any], fps: float,
) -> None:
    """state 的首/末时间戳应落在视频时间窗内（位置对不对）。

    长度对不对由 video_span_mismatch 负责；这里只查「起点/终点偏移」，两者互不重叠。
    v3 有两种时间戳约定：timestamp 以 episode 起点为 0（视频窗是文件内绝对时间，无
    可比较位置，直接跳过）；或 timestamp 为文件内全局时间（直接与 from/to 比较）。
    用 ts[0]≈0 区分，否则多 episode 拼接的 mp4 会被整体误报偏移。
    """
    if "timestamp" not in ep_df.columns or res.frame_count < 2:
        return
    ts = pd.to_numeric(ep_df["timestamp"], errors="coerce").to_numpy(dtype=float)
    if np.isnan(ts).any():
        return  # 非单调/含 NaN 已由 timestamp_not_monotonic 记录
    windows = _video_windows(dataset, res.episode_index)
    if not windows:
        return
    tol = float(cfg_get(config, "alignment.timestamp_drift_frames", 2.0)) / float(fps or 1.0)
    first, last = float(ts[0]), float(ts[-1])
    if abs(first) <= tol:
        return  # episode 相对时间轴：绝对位置不可比
    problems = []
    by_camera: dict[str, list[dict[str, Any]]] = {}
    for win in windows:
        by_camera.setdefault(win["camera"], []).append(win)
    for camera, wins in sorted(by_camera.items()):
        wins = sorted(wins, key=lambda w: (w["from"], w["to"]))
        # 多段/双 take：state 的首/末时间戳应分别落在首段起点与末段终点附近；
        # 段与段交界处的对齐由 span/帧数规则按段计量，这里只查两端。
        first_win, last_win = wins[0], wins[-1]
        start_off = first - first_win["from"]
        end_off = last - last_win["to"]
        if abs(start_off) > tol or abs(end_off) > tol:
            problems.append({
                "camera": camera,
                "video_from": round(first_win["from"], 4),
                "video_to": round(last_win["to"], 4),
                "state_first": round(first, 4),
                "state_last": round(last, 4),
                "start_offset_frames": round(start_off * fps, 3),
                "end_offset_frames": round(end_off * fps, 3),
                "segments": len(wins),
            })
    if problems:
        res.add(
            "timestamp_drift",
            tolerance_frames=cfg_get(config, "alignment.timestamp_drift_frames", 2.0),
            cameras=problems,
            repair_hint="可用 frame_index/fps 重建时间轴",
        )


def _check_video_span(
    res: EpisodeResult, dataset: LeRobotDatasetView, config: dict[str, Any], fps: float
) -> None:
    """视频时间窗跨度应与 episode 时长（frame_count/fps）一致。

    多段/双 take episode 按段计量：Σ 各段窗口跨度 ≈ frame_count/fps（旧口径
    拿单段窗口比全集时长，见 _video_windows 的误判复盘）。段内的对不齐仍会被
    本规则按段抓出（每段跨度都进证据）。
    """
    if fps <= 0 or res.frame_count <= 0:
        return
    ratio_tol = float(cfg_get(config, "alignment.video_span_tolerance_ratio", 0.05))
    expected = res.frame_count / float(fps)
    by_camera: dict[str, list[dict[str, Any]]] = {}
    for win in _video_windows(dataset, res.episode_index):
        by_camera.setdefault(win["camera"], []).append(win)
    problems = []
    for camera, wins in sorted(by_camera.items()):
        wins = sorted(wins, key=lambda w: (w["from"], w["to"]))
        if len(wins) == 1:
            win = wins[0]
            span = win["to"] - win["from"]
            if span <= 0:
                problems.append({"camera": camera, "from": round(win["from"], 4),
                                 "to": round(win["to"], 4), "span": round(span, 4)})
                continue
            if abs(span - expected) > ratio_tol * expected:
                problems.append({
                    "camera": camera,
                    "from": round(win["from"], 4),
                    "to": round(win["to"], 4),
                    "span": round(span, 4),
                    "expected_span": round(expected, 4),
                    "deviation": round(span - expected, 4),
                })
            continue
        # 多段：Σ 各段跨度 vs 全集时长（按 take 量计量）
        spans = [w["to"] - w["from"] for w in wins]
        total = sum(spans)
        if total <= 0:
            problems.append({"camera": camera, "segments": len(wins),
                             "span_total": round(total, 4)})
            continue
        if abs(total - expected) > ratio_tol * expected:
            problems.append({
                "camera": camera,
                "segments": len(wins),
                "span_total": round(total, 4),
                "expected_span": round(expected, 4),
                "deviation": round(total - expected, 4),
                "per_segment": [{"from": round(w["from"], 4), "to": round(w["to"], 4),
                                 "span": round(s, 4)} for w, s in zip(wins, spans)],
            })
    if problems:
        res.add("video_span_mismatch", expected_span=round(expected, 4),
                tolerance_ratio=ratio_tol, cameras=problems)


def _check_frame_index_gap(ep_df: pd.DataFrame, res: EpisodeResult) -> None:
    """frame_index 必须是从 0 开始的连续整数：视频按它切窗口，断裂即错位。"""
    if "frame_index" not in ep_df.columns or res.frame_count < 2:
        return
    idx = pd.to_numeric(ep_df["frame_index"], errors="coerce").to_numpy(dtype=float)
    if np.isnan(idx).any():
        return
    expected = np.arange(res.frame_count, dtype=float)
    if np.array_equal(idx, expected):
        return
    gaps = int(np.count_nonzero(idx != expected))
    res.add(
        "frame_index_gap",
        mismatched_frames=gaps,
        first_bad_frame=int(np.argmax(idx != expected)),
        head=[int(v) for v in idx[:5]],
        repair_hint="清洗时按真实行数重建全局索引",
    )


def check_meta_consistency(res: EpisodeResult, ep_index: int, dataset: LeRobotDatasetView) -> None:
    if "length" not in dataset.episodes.columns:
        return
    # 走 view.meta_row 而不是直接比较：两侧编号空间不同时由视图负责换算
    meta_row = getattr(dataset, "meta_row", None)
    if callable(meta_row):
        row_obj = meta_row(ep_index)
        if row_obj is None or pd.isna(row_obj.get("length")):
            return
        try:
            meta_length = int(row_obj["length"])
        except (TypeError, ValueError):
            return
    else:
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


def check_frame_mismatch(
    res: EpisodeResult, ep_index: int, dataset: LeRobotDatasetView,
    cameras: list[str], precise: bool, config: dict[str, Any],
) -> None:
    if not FFPROBE:
        return
    # 编号空间对不上时，meta 里根本没有该 episode 的视频窗口 → 定位必然失败。
    # 这属于「不可评估」，不是数据异常；报成 error 会让整批数据被误删。
    if not dataset.locator_available:
        return
    # 视频帧容差按 30Hz 标定，高帧率采样下应同步放大，否则几帧的正常编码偏差会被误报
    tol = scale_frames(int(cfg_get(config, "video_frame_tolerance", 1)), config, dataset.fps)
    problems: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []

    for cam in cameras:
        # require_exists=False：先拿到候选路径，才能把「文件不存在」（数据缺陷）
        # 与「meta 无定位信息」（不可评估）分开报。多段/双 take 每段一项。
        if hasattr(dataset, "video_locations"):
            locs = dataset.video_locations(cam, ep_index, require_exists=False)
        else:  # 极简替身（测试桩）只带单段 video_location
            loc = dataset.video_location(cam, ep_index, require_exists=False)
            locs = [loc] if loc is not None else []
        if not locs:
            problems.append({"type": "video_unlocatable", "camera": cam})
            continue

        # 探测按文件去重：双 take 的各段可能落在同一文件里
        totals: dict[Path, int | None] = {}
        segs: list[dict[str, Any]] = []
        for path, from_ts, to_ts in locs:
            if not path.is_file():
                missing.append({"camera": cam, "path": str(path)})
                continue
            if path not in totals:
                totals[path] = probe_video_frames(path, precise=precise)
            total = totals[path]
            if total is None:
                problems.append({"type": "video_unprobeable", "camera": cam, "path": str(path)})
                continue
            # 脏元数据可能给出 NaN/Inf 时间戳：先判有限性，避免 int(round(inf)) 抛
            # OverflowError；这类窗口无法用于帧对齐，退化为按整段视频比对总帧数。
            segs.append({
                "path": path, "total": total,
                "window_valid": (
                    math.isfinite(from_ts) and math.isfinite(to_ts)
                    and from_ts >= 0 and to_ts > from_ts
                ),
                "from_ts": from_ts, "to_ts": to_ts,
            })
        if not segs:
            continue

        if all(s["window_valid"] for s in segs):
            # 按 take 量计量：Σ 各段窗口帧数 vs 全集行数。单段 locs 长度 1，
            # 行为与旧口径一致。
            seg_total = 0
            truncated = False
            for s in segs:
                start = int(round(s["from_ts"] * dataset.fps))
                end = int(round(s["to_ts"] * dataset.fps))
                seg_total += end - start
                if end > s["total"] + tol:
                    problems.append({"type": "video_truncated", "camera": cam,
                                     "expected_end_frame": end,
                                     "actual_total_frames": s["total"]})
                    truncated = True
            if not truncated and abs(seg_total - res.frame_count) > tol:
                problems.append({"type": "count_mismatch", "camera": cam,
                                 "video_frames": seg_total, "rows": res.frame_count,
                                 "diff": seg_total - res.frame_count,
                                 "segments": len(segs)})
        else:
            file_total = sum({s["path"]: s["total"] for s in segs}.values())
            if abs(file_total - res.frame_count) > tol:
                problems.append({"type": "count_mismatch", "camera": cam,
                                 "video_frames": file_total, "rows": res.frame_count,
                                 "diff": file_total - res.frame_count})

    if missing:
        # 文件整缺 = 模态丢失，和「帧数差几帧」是两件事，必须单独成规则，
        # 否则报告会写「视频帧数与 state 行数不一致」，把人引去查帧率和编码。
        res.add(
            "video_missing",
            cameras=[m["camera"] for m in missing],
            missing_count=len(missing),
            expected_cameras=len(cameras),
            sample_paths=[m["path"] for m in missing[:3]],
            note="声明的视频文件在磁盘上不存在（或该相机目录为空）",
        )

    if problems:
        # 全是 unlocatable 说明 meta 给不出路径（编号/模板问题），属不可评估而非数据异常，
        # 降为 info：既不误删数据，又保留在报告里可追溯。
        only_unlocatable = all(p["type"] == "video_unlocatable" for p in problems)
        res.add(
            "frame_mismatch",
            severity="info" if only_unlocatable else None,
            rows=res.frame_count,
            tolerance=tol,
            problems=problems,
        )


def check_task(res: EpisodeResult, ep_index: int, dataset: LeRobotDatasetView) -> None:
    if not dataset.locator_available:
        return  # 同上：任务信息定位不到 ≠ 任务缺失
    if dataset.task_of(ep_index):
        return
    if "task_index" not in dataset.frames.columns and not dataset.tasks:
        res.add("missing_task", reason="不存在任务描述字段（task_index / tasks）")
    elif not dataset.tasks:
        res.add("missing_task", reason="meta/tasks 为空或缺失，task_index 无法解析")
    else:
        res.add("missing_task", reason="任务描述为空字符串")


# 视频内容
def check_blank_video(
    res: EpisodeResult, ep_index: int, dataset: LeRobotDatasetView, camera: str | None, config: dict[str, Any]
) -> None:
    """黑屏/遮挡检测：一次 ffmpeg 调用抽多帧缩略图，算灰度标准差。"""
    if camera is None or not FFMPEG:
        return
    loc = dataset.video_location(camera, ep_index)
    if loc is None:
        return
    path, from_ts, to_ts = loc
    if not path.is_file():
        return
    if from_ts < 0:
        # meta 给不出该集的视频时间窗（v3 缺列时为 (-1,-1)）：从文件 0 点抽帧
        # 实际判的是同文件里别的 episode 画面，属于不可评估而非数据异常。
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


# 小工具
def as_matrix(series: pd.Series, res: EpisodeResult, code: str) -> np.ndarray | None:
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


def frame_column(ep_df: pd.DataFrame) -> np.ndarray | None:
    if "frame_index" in ep_df.columns:
        return pd.to_numeric(ep_df["frame_index"], errors="coerce").to_numpy()
    return None


def _frame_labels(frame_col: np.ndarray | None, positions: np.ndarray, limit: int) -> list[int]:
    vals = frame_col[positions] if frame_col is not None else positions
    finite = np.asarray(vals, dtype=float)
    finite = finite[np.isfinite(finite)]
    return [int(v) for v in finite[:limit]]
