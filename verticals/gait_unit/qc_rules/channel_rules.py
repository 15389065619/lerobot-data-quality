"""Tier 1 通道级 QC 规则：只依赖 MotionClip（通道名 + T×D 数值），零骨骼知识。

规则清单（每条返回 findings 列表，dict 留痕，无副作用）：
  rule_mad_jump          每通道稳健 Z 跳变（median/MAD 范式，与 pipeline/ 一致、代码独立）
  rule_nan_runs          NaN 缺测段（连续长度分档留痕）
  rule_constant_channel  常值/死通道（方差为零，跳变规则对它天然失明，单列）
  rule_range_check       通道值域检查（limits 缺省 None=跳过；Tier 2 接 G1 URDF 限位表）

约定：
  - NaN 不参与 MAD 统计（用 nanmedian/nan-abs），但会被 nan 规则单独留痕；
  - findings 的 frame 下标一律 [start, end) 左闭右开；
  - 规则无处置权：findings 只进报告/复核队列， disposition 由 policy 层决定。
"""

from __future__ import annotations

import fnmatch

import numpy as np

from verticals.gait_unit.motion import MotionClip

__all__ = ["rule_mad_jump", "rule_nan_runs", "rule_constant_channel", "rule_range_check"]

_MAD_SCALE = 1.4826  # 使 MAD 在正态假设下无偏于标准差


def _runs(flags: np.ndarray) -> list[tuple[int, int]]:
    """布尔序列 → 连续 True 段 [(start, end)，左闭右开]。"""
    if not flags.any():
        return []
    idx = np.flatnonzero(flags)
    breaks = np.where(np.diff(idx) > 1)[0]
    starts = np.concatenate(([idx[0]], idx[breaks + 1]))
    ends = np.concatenate((idx[breaks], [idx[-1]])) + 1
    return [(int(s), int(e)) for s, e in zip(starts, ends)]


def rule_mad_jump(clip: MotionClip, k: float = 6.0,
                  skip_channel_patterns: list[str] | None = None) -> list[dict]:
    """每通道稳健 Z 跳变：|x−median|/MAD 超过 k 倍判跳变（NaN 不参与统计）。

    skip_channel_patterns：fnmatch 模式，匹配的通道整条跳过（不动统计）。
    用于动作类别分档——高动态类里 root_pos_* 的"跳变"被动作本身支配
    （校准：Standing_High_Jump k=6 时 root_pos_z 独占 1055/4740 帧），
    详见 profiles.py。

    MAD=0 的通道（常值/半常值）跳过——由 rule_constant_channel 单独留痕，
    否则稳健 Z 的分母为零会产生假警报。
    """
    patterns = skip_channel_patterns or []
    findings = []
    for d, name in enumerate(clip.channel_names):
        if any(fnmatch.fnmatch(name, pat) for pat in patterns):
            continue
        x = clip.channels[:, d]
        valid = np.isfinite(x)
        n_valid = int(valid.sum())
        if n_valid == 0:
            continue
        med = float(np.median(x[valid]))
        mad = float(np.median(np.abs(x[valid] - med)))
        if mad <= 0:
            continue
        z = np.zeros(clip.n_frames)
        # 稳健 Z 的标准定义：|x − median| / (1.4826·MAD)
        z[valid] = np.abs(x[valid] - med) / (_MAD_SCALE * mad)
        flags = z > k
        runs = _runs(flags)
        if runs:
            findings.append({
                "rule": "mad_jump", "channel": name, "k": k,
                "median": med, "mad": mad,
                "runs": runs, "n_flagged": int(flags.sum()),
                "max_z": round(float(z.max()), 2),
            })
    return findings


def rule_nan_runs(clip: MotionClip) -> list[dict]:
    """NaN 缺测段：逐通道列出连续缺测段与占比。"""
    findings = []
    for d, name in enumerate(clip.channel_names):
        nan_flags = ~np.isfinite(clip.channels[:, d])
        n_nan = int(nan_flags.sum())
        if n_nan == 0:
            continue
        runs = _runs(nan_flags)
        findings.append({
            "rule": "nan_runs", "channel": name,
            "n_nan": n_nan, "ratio": round(n_nan / clip.n_frames, 4),
            "runs": runs,
            "longest_run": max((e - s for s, e in runs), default=0),
        })
    return findings


def rule_constant_channel(clip: MotionClip, tol: float = 1e-12) -> list[dict]:
    """常值/死通道：max−min ≤ tol 的通道（传感器冻结/写死）。"""
    findings = []
    for d, name in enumerate(clip.channel_names):
        x = clip.channels[:, d]
        valid = x[np.isfinite(x)]
        if len(valid) < 2:
            continue
        span = float(valid.max() - valid.min())
        if span <= tol:
            findings.append({"rule": "constant_channel", "channel": name,
                             "span": span, "value": float(np.nanmedian(valid))})
    return findings


def rule_range_check(clip: MotionClip,
                     limits: dict[str, tuple[float, float]] | None = None) -> list[dict]:
    """通道值域检查：limits = {通道名: (下限, 上限)}；未列出的通道跳过。

    Tier 2 接线点：把 G1 URDF 的关节限位表填进来后，本规则升级为
    生理/机械限位检查。现在接口先固化。
    """
    if not limits:
        return []
    findings = []
    for d, name in enumerate(clip.channel_names):
        if name not in limits:
            continue
        lo, hi = limits[name]
        x = clip.channels[:, d]
        bad = np.isfinite(x) & ((x < lo) | (x > hi))
        runs = _runs(bad)
        if runs:
            findings.append({"rule": "range_check", "channel": name,
                             "lo": lo, "hi": hi, "runs": runs,
                             "n_violations": int(bad.sum())})
    return findings
