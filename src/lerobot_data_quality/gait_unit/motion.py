"""光捕/重定向轨迹的通道级 IR（MotionClip）。

为什么是第二个 IR 而不是复用 GaitClip：
  CHINGMU MotionDecode 导出的是**重定向后的 1-DOF 关节角轨迹**（29 个
  dof_* 通道 + 根位置/四元数，无显式时间戳，帧序即时间）。它没有逐关节
  位置，也没有关节局部四元数——塞进 GaitClip 需要伪造字段。
  通道级 IR 承认"我只有 T×D 数值通道 + 通道名"这一事实，让 QC 分层：
    Tier 1 通道级规则（MAD 跳变/NaN 段/常值通道）——只依赖本 IR，立即可用
    Tier 2 关节级规则（限位/角速度）——需要各通道的限位表（G1 URDF 公开可得）
    Tier 3 空间级规则（骨长/接触/质心）——需要 URDF 运动学做 FK，之后升级为
           GaitClip，规则与 vision/gait 共用

设计约束与 GaitClip 一致：缺陷（NaN、跳变）原样保留不裁决；
validate 只做结构层（形状/Inf/常值通道算不算错误交给规则）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["MotionClip", "validate_motion"]


@dataclass
class MotionClip:
    """通道级轨迹 IR：T 帧 × D 通道，通道名自带语义（如 dof_left_knee_joint(rad)）。"""

    fps: float
    channel_names: tuple[str, ...]   # (D,) 源列名原样保留（含单位后缀）
    channels: np.ndarray             # (T, D) 数值；NaN = 缺测/丢帧，保留给规则
    source: dict = field(default_factory=dict)  # 溯源

    def __post_init__(self):
        self.channels = np.asarray(self.channels, dtype=np.float64)

    @property
    def n_frames(self) -> int:
        return self.channels.shape[0]

    @property
    def n_channels(self) -> int:
        return self.channels.shape[1]

    def copy(self) -> "MotionClip":
        return MotionClip(
            fps=self.fps,
            channel_names=self.channel_names,
            channels=self.channels.copy(),
            source=dict(self.source),
        )


def validate_motion(clip: MotionClip) -> list[str]:
    """结构层校验：形状 / Inf / fps。NaN 是质检对象（丢帧缺测），不算结构错误。"""
    issues: list[str] = []
    T, D = clip.channels.shape
    if len(clip.channel_names) != D:
        issues.append(f"channel_names 数量 {len(clip.channel_names)} != 通道数 {D}")
    if np.isinf(clip.channels).any():
        issues.append("channels 含 Inf")
    if clip.fps <= 0:
        issues.append(f"fps={clip.fps} 非法")
    return issues
