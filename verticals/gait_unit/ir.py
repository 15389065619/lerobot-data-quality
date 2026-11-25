# Copyright (c) 2026 wuliangzhong
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""步态单元的中间格式（IR）：标准骨骼时序帧表。

为什么是 dataclass 而不是 DataFrame（与 loaders/ 的 LeRobot 帧表不同构）：
  骨骼时序是 (T, J, ·) 的三维张量——每个时间帧、每个关节、多个分量（位置/四元数），
  DataFrame 的二维结构装不下这种层级，硬拍平会丢掉关节名与层级关系。
  loaders/ 面向 VLA 帧表（episode_index + state + action），与本 IR 是两个数据域，
  各自独立演化，互不依赖。

IR 契约（所有 adapter 的输出、所有检测规则的输入都必须是 GaitClip）：
  - timestamps   (T,)  秒，单调**不减**（重复盖章是质检对象，不是结构错误）
  - joint_names  (J,)  关节/标记点名，来源内排序确定
  - parents      (J,) | None  父关节索引（-1=根）；源数据不带层级时为 None，
                 需要 parents 的规则（G1.3 骨长一致性）自动标记不适用
  - positions    (T, J, 3)  全局位置（米；BVH 等源单位未验证时在 source 注明）
  - rotations    (T, J, 4) | None  局部旋转四元数 (w, x, y, z)，相对父关节；
                 源数据没有旋转时为 None
  - source       dict  溯源（source 类型 / 文件 / 导出器 / 单位声明）

设计约束：
  - 缺陷不在这里裁决：NaN、重复盖章、跳变都原样保留进 IR，交给 G1 规则；
    validate_clip 只做**结构层**校验（形状/有限值/时间戳倒退），不做内容判断。
  - 四元数工具仅实现本项目用到的最小集合（乘法/旋转/轴角/旋角），不做完整
    旋转库；约定 wxyz 顺序（与 torch/lerobot 一致）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

__all__ = [
    "GaitClip", "validate_clip", "bone_lengths",
    "quat_from_axis_angle", "quat_mul", "quat_rotate", "quat_angle",
]


# --------------------------------------------------------------------------
# 四元数最小工具集（wxyz 约定）
# --------------------------------------------------------------------------

def quat_from_axis_angle(axis: tuple[float, float, float], angle_rad: float) -> tuple[float, float, float, float]:
    """轴角 → 四元数 (w, x, y, z)。axis 会被归一化。"""
    n = math.sqrt(axis[0] ** 2 + axis[1] ** 2 + axis[2] ** 2)
    if n == 0:
        raise ValueError("旋转轴不能为零向量")
    ux, uy, uz = axis[0] / n, axis[1] / n, axis[2] / n
    half = angle_rad / 2.0
    s = math.sin(half)
    return (math.cos(half), ux * s, uy * s, uz * s)


def quat_mul(a, b):
    """四元数乘法 (w,x,y,z)。内旋组合时按声明顺序左乘：q = q1 ⊗ q2 ⊗ ..."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def quat_rotate(q, v):
    """用四元数旋转三维向量 v（列向量约定），返回 np.ndarray (3,)。"""
    qv = (0.0, float(v[0]), float(v[1]), float(v[2]))
    conj = (q[0], -q[1], -q[2], -q[3])
    r = quat_mul(quat_mul(q, qv), conj)
    return np.array([r[1], r[2], r[3]])


def quat_angle(q) -> float:
    """四元数表示的旋转角（弧度）＝ 2·acos(w)。用于关节超限检测。"""
    return 2.0 * math.acos(max(-1.0, min(1.0, float(q[0]))))


# --------------------------------------------------------------------------
# IR 主结构
# --------------------------------------------------------------------------

@dataclass
class GaitClip:
    """标准骨骼时序帧表（步态单元唯一的中间交换格式）。"""

    fps: float
    timestamps: np.ndarray          # (T,) 秒，单调不减
    joint_names: tuple[str, ...]    # (J,)
    positions: np.ndarray           # (T, J, 3) 全局位置
    parents: tuple[int, ...] | None = None   # (J,) 父关节索引，-1=根
    rotations: np.ndarray | None = None      # (T, J, 4) 局部旋转 wxyz
    source: dict = field(default_factory=dict)  # 溯源信息

    def __post_init__(self):
        self.timestamps = np.asarray(self.timestamps, dtype=np.float64)
        self.positions = np.asarray(self.positions, dtype=np.float64)
        if self.rotations is not None:
            self.rotations = np.asarray(self.rotations, dtype=np.float64)

    @property
    def n_frames(self) -> int:
        return len(self.timestamps)

    @property
    def n_joints(self) -> int:
        return len(self.joint_names)

    def copy(self) -> "GaitClip":
        """深拷贝（注入器返回新对象，原 clip 不被改动——留痕原则）。"""
        return GaitClip(
            fps=self.fps,
            timestamps=self.timestamps.copy(),
            joint_names=self.joint_names,
            positions=self.positions.copy(),
            parents=self.parents,
            rotations=None if self.rotations is None else self.rotations.copy(),
            source=dict(self.source),
        )


def validate_clip(clip: GaitClip) -> list[str]:
    """结构层校验：只查形状/有限值/时间戳倒退，不做内容判断。

    返回问题列表（空 = 结构合法）。注意：
      - NaN **不算结构错误**（遮挡丢点是真实采集现象，原样保留给 G1 规则）；
      - 重复时间戳也不算（盖章粗化是质检对象）；
      - 只有**时间戳倒退**（后帧时间早于前帧）属于结构错误，说明源文件损坏。
    """
    issues: list[str] = []
    T, J, _ = clip.positions.shape
    if len(clip.timestamps) != T:
        issues.append(f"timestamps 长度 {len(clip.timestamps)} != positions 帧数 {T}")
    if len(clip.joint_names) != J:
        issues.append(f"joint_names 数量 {len(clip.joint_names)} != positions 关节数 {J}")
    if not np.isfinite(clip.timestamps).all():
        issues.append("timestamps 含 NaN/Inf")
    if not np.isfinite(clip.positions).all():
        # positions 的 NaN 允许存在（marker 遮挡），只查 Inf
        if np.isinf(clip.positions).any():
            issues.append("positions 含 Inf")
    if clip.rotations is not None and clip.rotations.shape[:2] != (T, J):
        issues.append(f"rotations 形状 {clip.rotations.shape[:2]} != (T, J) = ({T}, {J})")
    if T >= 2:
        back = np.where(np.diff(clip.timestamps) < 0)[0]
        if len(back):
            issues.append(f"时间戳倒退 {len(back)} 处（首个在帧 {back[0]}→{back[0] + 1}）")
    if clip.fps <= 0:
        issues.append(f"fps={clip.fps} 非法")
    return issues


def bone_lengths(clip: GaitClip) -> dict[tuple[str, str], np.ndarray]:
    """逐帧计算每根骨骼的长度 {(关节, 父关节): (T,) 长度序列}。

    骨骼是刚体，理论上长度恒定；随时间漂移 = 骨骼被拖走/缩放的证据，
    是 G1.3（骨骼段长度一致性）与慢漂移检测的数据基础。
    需要 clip.parents，缺失时抛错。
    """
    if clip.parents is None:
        raise ValueError("clip.parents 为 None（来源不含层级），无法计算骨骼长度")
    out: dict[tuple[str, str], np.ndarray] = {}
    for j, parent in enumerate(clip.parents):
        if parent < 0:
            continue
        seg = clip.positions[:, j, :] - clip.positions[:, parent, :]
        out[(clip.joint_names[parent], clip.joint_names[j])] = np.linalg.norm(seg, axis=1)
    return out
