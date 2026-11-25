# Copyright (c) 2026 wu
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""骨骼级规则：作用于带层级/旋转的 GaitClip（BVH/Pico 测量型骨架）。

规则清单（对应 G1 六条中的四条可即时报错的子集）：
  rule_timestamp_gaps       时间戳缺口（丢帧——采集丢包）
  rule_timestamp_duplicates 盖章重复（采集粗化，同刻多帧）
  rule_joint_rotation_limit 关节局部旋转角超限（扭曲/反关节的粗探针）
  rule_bone_length_drift    骨长漂移（局部骨架被拖走——刚体被破坏的证据）

诚实边界：
  - 骨长规则只对**测量型骨架**有意义（Pico/BVH 解析）；FK 产物骨长由刚体
    模型数学保证恒定，跑了也恒为零差异（kinematics.py 已文档化）；
  - 全局漂移（整条骨架被拖走）骨长不变，抓不到——那是 G1.5 慢漂移
    趋势检验的靶子（本地漂移/局部漂移才能被骨长规则抓住）；
  - 旋转限位是**粗探针**：局部旋转角（相对静止姿态的总旋转量）超阈，
    不做逐关节解剖学限位表——那个需要 URDF/人体模型，Tier 2 接线。
"""

from __future__ import annotations

import math

import numpy as np

from verticals.gait_unit.ir import GaitClip, bone_lengths, quat_angle

__all__ = [
    "rule_timestamp_gaps",
    "rule_timestamp_duplicates",
    "rule_joint_rotation_limit",
    "rule_bone_length_drift",
]


def _runs(flags: np.ndarray) -> list[tuple[int, int]]:
    if not flags.any():
        return []
    idx = np.flatnonzero(flags)
    breaks = np.where(np.diff(idx) > 1)[0]
    starts = np.concatenate(([idx[0]], idx[breaks + 1]))
    ends = np.concatenate((idx[breaks], [idx[-1]])) + 1
    return [(int(s), int(e)) for s, e in zip(starts, ends)]


def rule_timestamp_gaps(clip: GaitClip, k: float = 3.0) -> list[dict]:
    """时间戳缺口：相邻帧间隔 > k × 中位间隔（丢帧——采集丢包）。"""
    findings = []
    if clip.n_frames < 3:
        return findings
    diffs = np.diff(clip.timestamps)
    med = float(np.median(diffs[diffs > 0]))
    if med <= 0:
        return findings
    gap_flags = diffs > k * med
    for s, e in _runs(gap_flags):
        findings.append({
            "rule": "timestamp_gaps", "frames": (int(s), int(e + 1)),
            "gap_s": round(float(diffs[s]), 4),
            "expected_s": round(med, 4),
            "n_missing_approx": max(0, int(round(diffs[s] / med)) - 1),
        })
    return findings


def rule_timestamp_duplicates(clip: GaitClip) -> list[dict]:
    """盖章重复：间隔为 0 的相邻帧段（采集粗化，同刻多帧）。"""
    findings = []
    if clip.n_frames < 2:
        return findings
    dup_flags = np.diff(clip.timestamps) == 0
    for s, e in _runs(dup_flags):
        findings.append({
            "rule": "timestamp_duplicates", "frames": (int(s), int(e + 2)),
            "n_duplicate_frames": int(e - s + 1),
        })
    return findings


def rule_joint_rotation_limit(clip: GaitClip, max_deg: float = 170.0,
                              joints: tuple[str, ...] | None = None) -> list[dict]:
    """关节局部旋转角超限（粗探针）：局部旋转相对单位四元数的总旋转角
    超过 max_deg 的帧段。

    局部旋转角 = |角度(quat)|，即相对静止姿态的总偏转——不做逐关节
    解剖学限位表（那个需要人体模型，Tier 2 接线）。默认 170°：正常
    动作极少有单关节相对父系偏转超过它。
    """
    if clip.rotations is None:
        return [{"rule": "joint_rotation_limit", "skipped": True,
                 "reason": "clip 无旋转数据（点级 IR），规则不适用"}]
    target = joints or clip.joint_names
    limit_rad = math.radians(max_deg)
    findings = []
    for name in target:
        if name not in clip.joint_names:
            continue
        j = clip.joint_names.index(name)
        angles = np.array([quat_angle(tuple(clip.rotations[f, j]))
                           for f in range(clip.n_frames)])
        flags = angles > limit_rad
        runs = _runs(flags)
        if runs:
            findings.append({
                "rule": "joint_rotation_limit", "joint": name,
                "max_deg": max_deg,
                "max_observed_deg": round(float(angles.max()) * 180 / math.pi, 1),
                "runs": runs, "n_frames": int(flags.sum()),
            })
    return findings


def rule_bone_length_drift(clip: GaitClip, tol_rel: float = 0.02) -> list[dict]:
    """骨长漂移：任一骨骼的长度偏离自身中位数超过 tol_rel（相对）的帧段。

    局部漂移（骨架一部分被拖走）会拉长/压短跨区域的骨长——刚体被破坏
    的直接证据。注意：全局漂移（整体平移）骨长不变，此规则抓不到。
    需要 clip.parents（层级），缺失时如实跳过。
    """
    if clip.parents is None:
        return [{"rule": "bone_length_drift", "skipped": True,
                 "reason": "clip 无层级（parents=None），骨长规则不适用"}]
    findings = []
    for bone, seq in bone_lengths(clip).items():
        valid = np.isfinite(seq)
        if valid.sum() < 3:
            continue
        med = float(np.median(seq[valid]))
        if med <= 0:
            continue
        dev = np.abs(seq - med) / med
        flags = valid & (dev > tol_rel)
        runs = _runs(flags)
        if runs:
            findings.append({
                "rule": "bone_length_drift", "bone": f"{bone[0]}→{bone[1]}",
                "tol_rel": tol_rel, "median_len": round(med, 4),
                "max_dev": round(float(dev.max()), 4),
                "runs": runs, "n_frames": int(flags.sum()),
            })
    return findings
