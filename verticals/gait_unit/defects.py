# Copyright (c) 2026 wuliangzhong
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""G0 缺陷注入器 + 干净基线 clip 生成器。

职责与哲学（与 examples/fzmotion_sim 一脉相承）：
  规则先有靶子再有实现——每一条未来 G1 规则的验收标准 = 对应注入器注入的
  缺陷**必须被检测到**。注入器返回 (clip, manifest)，manifest 是注入缺陷的
  地面真值（类型/帧区间/参数），留痕原则同样适用于测试。

四类注入对应真实采集最常见的病（Pico 漂移/跳帧/盖章粗化/超限扭曲）：
  inject_drift           整条骨架被拖走（全局低频漂移，骨长保持不变——
                         这是有意为之：说明骨长规则抓不到这类漂移，
                         它是 G1.5 慢漂移趋势检验的靶子）
  inject_frame_drop      连续帧整体缺失（采集丢包）
  inject_duplicate_stamp 盖章粗化（同一时刻盖两行，Pico 15Hz 盖章/30Hz 数据）
  inject_joint_limit     关节角超出生理极限（扭曲/反关节）
"""

from __future__ import annotations

import math

import numpy as np

from verticals.gait_unit.ir import (
    GaitClip,
    quat_angle,
    quat_from_axis_angle,
    quat_mul,
    quat_rotate,
)

_JOINTS = ("Hips", "ThighL", "ShankL", "FootL", "ThighR", "ShankR", "FootR")
_PARENTS = (-1, 0, 1, 2, 0, 4, 5)
_OFFSETS = {
    "ThighL": (0.10, -0.02, 0.0), "ShankL": (0.0, -0.40, 0.0), "FootL": (0.0, -0.42, 0.0),
    "ThighR": (-0.10, -0.02, 0.0), "ShankR": (0.0, -0.40, 0.0), "FootR": (0.0, -0.42, 0.0),
}


def make_clean_clip(fps: float = 30.0, seconds: float = 10.0,
                    stride_hz: float = 1.4) -> GaitClip:
    """生成干净的双腿步行基线 clip（合成数据，无缺陷，骨长恒定）。

    运动 = 髋部正弦摆动（双相位）+ 膝伸展 + 根部前进/起伏；
    所有角度控制在生理极限内——它是所有注入器的干净底片。
    """
    T = int(round(fps * seconds))
    t = np.arange(T, dtype=np.float64) / fps
    phase = 2.0 * math.pi * stride_hz * t

    swing_l = 0.45 * np.sin(phase)                 # 左腿摆动（弧度）
    swing_r = 0.45 * np.sin(phase + math.pi)       # 右腿反相
    knee_l = 0.30 * np.maximum(0.0, np.sin(phase + 0.6))
    knee_r = 0.30 * np.maximum(0.0, np.sin(phase + 0.6 + math.pi))

    local = np.zeros((T, 7, 4))
    local[:, 0] = (1.0, 0.0, 0.0, 0.0)             # Hips：朝前走，无附加旋转
    local[:, 1] = [quat_from_axis_angle((1, 0, 0), a) for a in swing_l]
    local[:, 2] = [quat_from_axis_angle((1, 0, 0), a) for a in knee_l]
    local[:, 3] = (1.0, 0.0, 0.0, 0.0)
    local[:, 4] = [quat_from_axis_angle((1, 0, 0), a) for a in swing_r]
    local[:, 5] = [quat_from_axis_angle((1, 0, 0), a) for a in knee_r]
    local[:, 6] = (1.0, 0.0, 0.0, 0.0)

    positions = np.zeros((T, 7, 3))
    for f in range(T):
        hips = np.array([0.0, 0.95 - 0.02 * abs(math.sin(phase[f])), 1.2 * t[f]])
        positions[f, 0] = hips
        wq = [None] * 7
        for j in range(1, 7):
            p = _PARENTS[j]
            # Hips 的世界旋转 = 单位四元数（根无附加旋转），直接显式表达
            parent_wq = (1.0, 0.0, 0.0, 0.0) if p == 0 else wq[p]
            wq[j] = quat_mul(parent_wq, tuple(local[f, j]))
            positions[f, j] = positions[f, p] + \
                quat_rotate(parent_wq, _OFFSETS[_JOINTS[j]])

    return GaitClip(
        fps=fps,
        timestamps=t.copy(),
        joint_names=_JOINTS,
        positions=positions,
        parents=_PARENTS,
        rotations=local,
        source={"source": "synthetic_clean", "stride_hz": stride_hz,
                "note": "G0 干净基线：所有注入器的底片"},
    )


def inject_drift(clip: GaitClip, rate: float = 0.01, axis: str = "x",
                 joints: tuple[str, ...] | None = None,
                 start: int = 0) -> tuple[GaitClip, dict]:
    """骨架被拖走：给指定关节（缺省=全部）加随时间线性增长的偏移。

    joints=None → 全部关节（刚体平移，骨长不变——G1.5 慢漂移的靶子）；
    joints=子树 → 只有子树内关节漂移，跨子树的骨长被拉长——
    G1.3 骨长漂移规则的靶子。start>0 时缺陷从该帧开始（前段干净）。
    """
    out = clip.copy()
    t = out.timestamps - out.timestamps[0]
    t = np.maximum(t, 0.0)
    axis_i = "xyz".index(axis)
    if joints is None:
        targets = list(range(out.n_joints))
    else:
        targets = [out.joint_names.index(n) for n in joints]
    drift = rate * t[:, None]
    drift[:start] = 0.0
    out.positions[:, targets, axis_i] += drift
    return out, {"type": "drift", "rate_per_s": rate, "axis": axis,
                 "joints": [out.joint_names[j] for j in targets],
                 "start_frame": start,
                 "total_offset": rate * float(max(out.timestamps[-1] - out.timestamps[0], 0.0))}


def inject_frame_drop(clip: GaitClip, start: int = 40, n: int = 3) -> tuple[GaitClip, dict]:
    """连续 n 帧整体缺失（采集丢包），时间戳出现缺口。

    manifest["gap_s"] = 删完后跨越缺口的间隔（原 ts[start+n] − ts[start-1]）；
    缺口贴边（start=0 或删到末尾）时没有"跨越"可言，记 None。
    """
    out = clip.copy()
    keep = np.ones(out.n_frames, dtype=bool)
    keep[start:start + n] = False
    out.timestamps = out.timestamps[keep]
    out.positions = out.positions[keep]
    if out.rotations is not None:
        out.rotations = out.rotations[keep]
    if start > 0 and start + n < clip.n_frames:
        gap_s = float(clip.timestamps[start + n] - clip.timestamps[start - 1])
    else:
        gap_s = None
    return out, {"type": "frame_drop", "start": start, "n": n, "gap_s": gap_s}


def inject_duplicate_stamp(clip: GaitClip, start: int | None = None,
                           n: int | None = None) -> tuple[GaitClip, dict]:
    """盖章粗化：奇数帧的时间戳复制自前一帧（同一时刻盖两行）。

    start/n 给定时只在 [start, start+n) 段内粗化（基准随机化用）；
    缺省 = 全片交替粗化（Pico 15Hz 盖章/30Hz 数据的形态）。
    """
    out = clip.copy()
    if start is None or n is None:
        out.timestamps[1::2] = out.timestamps[0::2][:len(out.timestamps[1::2])]
        n_repeat = int((np.diff(out.timestamps) == 0).sum())
        return out, {"type": "duplicate_stamp", "repeat_count": n_repeat,
                     "frames": (0, out.n_frames)}
    s, e = max(0, start), min(out.n_frames, start + n)
    for f in range(s + 1, e):
        out.timestamps[f] = out.timestamps[f - 1]
    n_repeat = int((np.diff(out.timestamps) == 0).sum())
    return out, {"type": "duplicate_stamp", "repeat_count": n_repeat,
                 "frames": (s, e)}


def inject_joint_limit(clip: GaitClip, joint: str = "ShankL",
                       axis: str = "X", extra_deg: float = 80.0
                       ) -> tuple[GaitClip, dict]:
    """关节角超出生理极限：从第 0 帧起给目标关节附加一段超限旋转。"""
    out = clip.copy()
    j = out.joint_names.index(joint)
    if out.rotations is None:
        raise ValueError("注入关节超限需要 clip.rotations（点级 IR 不适用）")
    extra = quat_from_axis_angle({"X": (1, 0, 0), "Y": (0, 1, 0), "Z": (0, 0, 1)}[axis],
                                 math.radians(extra_deg))
    before = [quat_angle(tuple(out.rotations[f, j])) for f in range(out.n_frames)]
    for f in range(out.n_frames):
        out.rotations[f, j] = quat_mul(tuple(out.rotations[f, j]), extra)
    after = [quat_angle(tuple(out.rotations[f, j])) for f in range(out.n_frames)]
    return out, {"type": "joint_limit", "joint": joint, "axis": axis,
                 "extra_deg": extra_deg,
                 "angle_before_max_deg": round(max(before) * 180 / math.pi, 2),
                 "angle_after_max_deg": round(max(after) * 180 / math.pi, 2)}
