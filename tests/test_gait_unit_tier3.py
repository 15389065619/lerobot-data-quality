"""Tier 3 验收：URDF FK 已知答案 + 空间规则（接触/滑步/穿透/腾空）+ 真实样本。"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from lerobot_data_quality.gait_unit.adapters import load_motiondecode_csv
from lerobot_data_quality.gait_unit.ir import GaitClip, bone_lengths, validate_clip
from lerobot_data_quality.gait_unit.kinematics import fk_motion_clip, load_skeleton
from lerobot_data_quality.gait_unit.qc_rules.spatial_rules import (
    compute_foot_contact,
    rule_flight_phase,
    rule_foot_skating,
    rule_ground_penetration,
)

REAL_SAMPLE = (Path("data/chingmu_motiondecode/samples")
               / "1.1.Basic_Movement_Category/1.1.1.High_Dynamic_Movement"
               / "1.1.1.1.Standing_High_Jump/BM_Standing_High_Jump_00001.csv")
REAL_URDF = Path("data/external/g1_description/g1_29dof.urdf")

_TINY_URDF = """<?xml version="1.0"?>
<robot name="tiny">
  <link name="pelvis"/>
  <link name="l1"/>
  <link name="l2"/>
  <joint name="j1" type="revolute">
    <parent link="pelvis"/><child link="l1"/>
    <origin xyz="0 0 0" rpy="0 0 0"/><axis xyz="0 0 1"/>
    <limit lower="-3" upper="3" effort="1"/>
  </joint>
  <joint name="j2" type="revolute">
    <parent link="l1"/><child link="l2"/>
    <origin xyz="0 10 0" rpy="0 0 0"/><axis xyz="0 0 1"/>
    <limit lower="-3" upper="3" effort="1"/>
  </joint>
</robot>
"""


def _tiny_motion(j1_deg: float, root=(0.0, 0.0, 0.0)) -> "MotionClip":
    from lerobot_data_quality.gait_unit.motion import MotionClip
    ch = {"root_pos_x(m)": root[0], "root_pos_y(m)": root[1], "root_pos_z(m)": root[2],
          "root_rot_w": 1.0, "root_rot_x": 0.0, "root_rot_y": 0.0, "root_rot_z": 0.0,
          "dof_j1(rad)": math.radians(j1_deg), "dof_j2(rad)": 0.0}
    return MotionClip(fps=100.0, channel_names=tuple(ch),
                      channels=np.array([[v] for v in ch.values()]).T)


def test_tiny_urdf_fk_known_answer(tmp_path):
    urdf = tmp_path / "tiny.urdf"
    urdf.write_text(_TINY_URDF, encoding="utf-8")
    spec = load_skeleton(urdf)
    assert spec.root_link == "pelvis"
    assert spec.actuated_names == ("j1", "j2")
    assert spec.anchor_names == ("pelvis(root)", "j1", "j2")

    gait = fk_motion_clip(_tiny_motion(90.0), spec)
    # j1 锚点在根原点；j2 锚点 = 根 + Rz90·(0,10,0) = (-10, 0, 0)
    assert np.allclose(gait.positions[0, 0], (0, 0, 0))
    assert np.allclose(gait.positions[0, 1], (0, 0, 0))
    assert np.allclose(gait.positions[0, 2], (-10, 0, 0), atol=1e-9)
    # 骨长恒定（FK 的数学性质——测试把它写死，防止以后有人改动 FK 破坏刚体性）
    for bone, seq in bone_lengths(gait).items():
        assert seq.max() - seq.min() < 1e-9


def test_tiny_urdf_fk_missing_channel_raises(tmp_path):
    urdf = tmp_path / "tiny.urdf"
    urdf.write_text(_TINY_URDF, encoding="utf-8")
    spec = load_skeleton(urdf)
    clip = _tiny_motion(0.0)
    from lerobot_data_quality.gait_unit.motion import MotionClip
    broken = MotionClip(fps=clip.fps, channel_names=clip.channel_names[:6],
                        channels=clip.channels[:, :6])
    with pytest.raises(ValueError, match="缺少通道"):
        fk_motion_clip(broken, spec)


# --------------------------------------------------------------------------
# 空间规则：手工构造的接触/腾空/滑步/穿透
# --------------------------------------------------------------------------

def _contact_fixture() -> "GaitClip":
    """双脚 clip：0-49 双脚触地（左脚 10-20 横滑=打滑），50-79 双脚腾空
    且持续下落中 80-99 触地；最后 5 帧左脚穿地（z=-0.03）。"""
    T = 135
    pos = np.zeros((T, 2, 3))
    pos[:, :, 2] = 0.0                          # 地面 z=0（世界系）
    pos[:50, :, 0] = np.linspace(0, 0.1, 50)[:, None]      # 触地缓行
    pos[10:21, 0, 0] = np.linspace(0, 0.6, 11)             # 左脚触地期横滑 0.6m
    pos[50:80, :, 2] = 0.4                      # 腾空
    pos[50:80, :, 0] += np.linspace(0, 0.5, 30)[:, None]
    pos[80:, :, 0] += 0.5
    pos[130:, 0, 2] = -0.03                     # 左脚穿地
    return GaitClip(
        fps=30.0,
        timestamps=np.arange(T, dtype=np.float64) / 30.0,
        joint_names=("left_ankle_roll_joint", "right_ankle_roll_joint"),
        positions=pos,
        parents=(-1, 0),
        rotations=None,
        source={"source": "synthetic_contact"},
    )


def test_contact_and_flight_and_skate():
    clip = _contact_fixture()
    contact = compute_foot_contact(clip)
    # 腾空段两脚都离地
    assert not contact["contact"]["left_ankle_roll_joint"][60]
    assert not contact["contact"]["right_ankle_roll_joint"][60]
    assert contact["contact"]["left_ankle_roll_joint"][0]

    skate = rule_foot_skating(clip, speed_max=0.4)
    left_skate = [f for f in skate if f["foot"] == "left_ankle_roll_joint"]
    assert left_skate, "触地期横滑 0.6m 必须被标记为打滑"
    assert left_skate[0]["max_speed"] > 0.4

    flight = rule_flight_phase(clip, min_frames=6)
    assert flight and (50, 80) in [tuple(r) for r in flight[0]["runs"]]


def test_penetration_needs_explicit_ground():
    clip = _contact_fixture()
    # 不给 ground_z：规则如实退化为 no-op（自动地面=观测最小值，穿透不可能）
    skipped = rule_ground_penetration(clip)
    assert len(skipped) == 1 and skipped[0]["skipped"] is True
    assert "未提供世界系地面高度 ground_z" in skipped[0]["reason"]
    # 显式 ground_z=0：左脚 z=-0.03 的帧必须被抓到
    findings = rule_ground_penetration(clip, ground_z=0.0)
    left = [f for f in findings if f["foot"] == "left_ankle_roll_joint"]
    assert left and left[0]["n_frames"] == 5


# --------------------------------------------------------------------------
# 真实样本：G1 全身 FK + 跳高的接触/腾空
# --------------------------------------------------------------------------

@pytest.mark.skipif(not (REAL_SAMPLE.exists() and REAL_URDF.exists()),
                    reason="真实样本或 URDF 不在本机")
def test_g1_fk_and_jump_contact_real():
    clip = load_motiondecode_csv(REAL_SAMPLE)
    spec = load_skeleton(REAL_URDF)
    assert spec.root_link == "pelvis"
    assert len(spec.actuated_names) == 29

    gait = fk_motion_clip(clip, spec)
    assert validate_clip(gait) == []
    pelvis_idx = gait.joint_names.index("pelvis(root)")
    pelvis_z = gait.positions[:, pelvis_idx, 2]
    assert float(pelvis_z.max() - pelvis_z.min()) > 0.15, "跳高样本骨盆应有明显起伏"

    # 跳高必须有腾空段；起始应有接触（站立预备）
    flight = rule_flight_phase(gait, min_frames=12)
    assert flight and flight[0]["n_runs"] >= 1
    contact = compute_foot_contact(gait)
    for name, c in contact["contact"].items():
        # 跳高 = 预备(触地) + 腾空 + 落地(触地)：触地占多数，但不要求连续
        assert c.mean() > 0.3, f"{name} 触地占比应过半（跳高以站立预备/落地为主）"
