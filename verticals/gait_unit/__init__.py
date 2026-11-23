"""步态数据质检单元（gait_unit）——独立于 vision/VLA 主线的人体步态时序数据治理。

与 pipeline/（LeRobot 视觉数据质检）的关系：
  - 零代码依赖、零数据格式依赖，只共享设计范式（MAD 自适应阈值、
    缺陷注入验收、manifest 留痕、规则先有靶子再写实现）；
  - 数据域不同：vision 面向 LeRobot 帧表（RGB+state+action），
    gait 面向骨骼时序（GaitClip IR，见 ir.py）。

分层（按 2026-11-20 设计方案，资产门控见 README/设计文档）：
  G0 数据契约层：ir.py + adapters/（Pico/BVH/FZMotion）+ defects.py 注入器
  G1 骨骼时序质检引擎：qc_rules/（通道/骨架/标记点/空间规则，Tier 1-3 已落地）
  G2 修复单元：repair.py（规划中）
  G3 重定向：retarget/（门控——T800 URDF/MJCF + 验证闭环到位才开工，当前只有设计文档）
"""

from verticals.gait_unit.ir import (
    GaitClip,
    bone_lengths,
    quat_angle,
    validate_clip,
)

__all__ = [
    "GaitClip", "validate_clip", "bone_lengths", "quat_angle",
]
