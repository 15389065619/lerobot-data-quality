"""FZMotion marker 点级规则：作用于点级 GaitClip（parents/rotations = None）。

  rule_marker_dropout  单 marker 缺测占比超阈（遮挡丢点）
  rule_marker_jump     单 marker 位移跳变（误关联/互换——MAD 稳健 Z）

与 pipeline/ 的关系：MAD 范式一致、代码独立（gait_unit 零耦合原则）。
"""

from __future__ import annotations

import numpy as np

from lerobot_data_quality.gait_unit.ir import GaitClip

__all__ = ["rule_marker_dropout", "rule_marker_jump"]

_MAD_SCALE = 1.4826


def _runs(flags: np.ndarray) -> list[tuple[int, int]]:
    if not flags.any():
        return []
    idx = np.flatnonzero(flags)
    breaks = np.where(np.diff(idx) > 1)[0]
    starts = np.concatenate(([idx[0]], idx[breaks + 1]))
    ends = np.concatenate((idx[breaks], [idx[-1]])) + 1
    return [(int(s), int(e)) for s, e in zip(starts, ends)]


def rule_marker_dropout(clip: GaitClip, max_ratio: float = 0.05) -> list[dict]:
    """单 marker 缺测占比超阈：NaN 比例 > max_ratio 的 marker 留痕。"""
    findings = []
    T = clip.n_frames
    for m, name in enumerate(clip.joint_names):
        x = clip.positions[:, m, :]
        nan_flags = ~np.isfinite(x).all(axis=1)
        n_nan = int(nan_flags.sum())
        ratio = n_nan / T
        if ratio <= max_ratio:
            continue
        runs = _runs(nan_flags)
        findings.append({"rule": "marker_dropout", "marker": name,
                         "ratio": round(ratio, 4), "n_nan": n_nan,
                         "longest_run": max((e - s for s, e in runs), default=0),
                         "runs": runs[:10]})
    return findings


def rule_marker_jump(clip: GaitClip, k: float = 8.0) -> list[dict]:
    """单 marker 位移跳变：逐帧位移幅度的稳健 Z 超阈（marker 误关联/互换）。

    位移 = |p[f] − p[f−1]|；任一端 NaN 的步长不参与统计（那是 dropout 的
    事，不在这里制造双重报警）。
    """
    findings = []
    T = clip.n_frames
    if T < 3:
        return findings
    for m, name in enumerate(clip.joint_names):
        p = clip.positions[:, m, :]
        step = np.linalg.norm(p[1:] - p[:-1], axis=1)
        valid = np.isfinite(step)
        if valid.sum() < 3:
            continue
        med = float(np.median(step[valid]))
        mad = float(np.median(np.abs(step[valid] - med)))
        if mad <= 0:
            continue
        z = np.zeros(T - 1)
        z[valid] = np.abs(step[valid] - med) / (_MAD_SCALE * mad)
        flags = np.zeros(T - 1, dtype=bool)
        flags[valid] = z > k
        runs = _runs(flags)
        if runs:
            findings.append({"rule": "marker_jump", "marker": name, "k": k,
                             "median_step": round(med, 5),
                             "runs": [(s + 1, e + 1) for s, e in runs],
                             "max_z": round(float(z.max()), 2)})
    return findings
