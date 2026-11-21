"""步态单元的 QC 规则层。

按语义可得性分层（见 motion.py 的 Tier 说明）：
  Tier 1 channel_rules：通道级，只依赖 MotionClip（MAD 跳变/NaN 段/常值/值域）
  Tier 2 关节级：限位——接线点在 rule_range_check 的 limits 参数，
        限位表来源 = Unitree G1 公开 URDF
  Tier 3 空间级：骨长/接触/质心——需要 URDF 运动学 FK 后升级为 GaitClip
"""

from lerobot_data_quality.gait_unit.qc_rules.channel_rules import (
    rule_constant_channel,
    rule_mad_jump,
    rule_nan_runs,
    rule_range_check,
)

__all__ = ["rule_mad_jump", "rule_nan_runs", "rule_constant_channel", "rule_range_check"]
