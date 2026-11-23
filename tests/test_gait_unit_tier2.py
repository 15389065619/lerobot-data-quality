"""Tier 2 验收：G1 限位表接线 + 按动作类别的规则参数分档。

  1. 限位表加载：29/29 dof 全覆盖、通道映射契约、坏表拒绝
  2. 真实跳高样本 × 真实限位 → 0 越界（重定向质量的正面信号）
  3. 档位解析：High_Dynamic 路径 → high_dynamic 档
  4. 分档效果：跳高样本 default 档的 root_pos/knee findings 在
     high_dynamic 档消失（校准自 n=1，扩样本后须重新校准）
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from verticals.gait_unit.adapters import load_motiondecode_csv
from verticals.gait_unit.limits import load_g1_limits
from verticals.gait_unit.profiles import profile_params, resolve_profile
from verticals.gait_unit.qc_rules import rule_mad_jump, rule_range_check

REAL_SAMPLE = (Path("data/chingmu_motiondecode/samples")
               / "1.1.Basic_Movement_Category/1.1.1.High_Dynamic_Movement"
               / "1.1.1.1.Standing_High_Jump/BM_Standing_High_Jump_00001.csv")


def test_g1_limits_coverage():
    limits = load_g1_limits()
    assert len(limits) >= 29
    # 通道映射契约：dof_<urdf 名>(rad)
    assert "dof_left_hip_pitch_joint(rad)" in limits
    assert "dof_right_wrist_yaw_joint(rad)" in limits
    lo, hi = limits["dof_left_knee_joint(rad)"]
    assert lo < hi and lo > -0.5 and hi < 3.0   # 膝不可反向、不超 3 rad


def test_g1_limits_rejects_bad_tables(tmp_path):
    truncated = {"joints": {"left_hip_pitch_joint": {"lower": -1.0, "upper": 1.0}}}
    p = tmp_path / "truncated.json"
    p.write_text(json.dumps(truncated), encoding="utf-8")
    with pytest.raises(ValueError, match="29"):
        load_g1_limits(p)

    inverted = {"joints": {f"j{i:02d}": {"lower": 1.0, "upper": -1.0} for i in range(29)}}
    p2 = tmp_path / "inverted.json"
    p2.write_text(json.dumps(inverted), encoding="utf-8")
    with pytest.raises(ValueError, match="倒挂"):
        load_g1_limits(p2)

    with pytest.raises(FileNotFoundError):
        load_g1_limits(tmp_path / "nope.json")


@pytest.mark.skipif(not REAL_SAMPLE.exists(), reason="真实样本不在本机")
def test_real_jump_sample_within_g1_limits():
    clip = load_motiondecode_csv(REAL_SAMPLE)
    limits = load_g1_limits()
    findings = rule_range_check(clip, limits)
    assert findings == [], "厂商重定向输出不应超出 G1 关节限位"


def test_profile_resolution():
    assert resolve_profile(REAL_SAMPLE) == "high_dynamic"
    # 类别关键词分档（LAFAN1 校准后新增）：路径里的动作类别自动进对应档
    assert resolve_profile("data/other/BM_Walk_00001.csv") == "walk"
    assert resolve_profile("data/lafan1/extracted/bvh/walk3_subject3.bvh") == "walk"
    assert resolve_profile("data/lafan1/extracted/bvh/run2_subject1.bvh") == "run"
    assert resolve_profile("data/other/unknown_action.csv") == "default"
    # 未知档位名回落 default（宁松勿紧，不抛错）
    params = profile_params("nonexistent_profile")
    assert params["mad_jump_k"] == 6.0
    assert profile_params("high_dynamic")["mad_jump_k"] == 10.0


def test_profile_params_carry_calibrated_baseline():
    """LAFAN1 校准表存在时，类别档位应附带本底报警率（k=6）。"""
    params = profile_params("walk")
    assert params.get("baseline_rate_per_1k") == pytest.approx(21.32)
    # 未校准的类别不带 baseline 字段
    assert "baseline_rate_per_1k" not in profile_params("high_dynamic")


@pytest.mark.skipif(not REAL_SAMPLE.exists(), reason="真实样本不在本机")
def test_high_dynamic_profile_suppresses_action_dominated_findings():
    clip = load_motiondecode_csv(REAL_SAMPLE)

    default_fs = rule_mad_jump(clip, k=6.0)
    default_ch = {f["channel"] for f in default_fs}
    assert "root_pos_z(m)" in default_ch, "default 档（k=6）应报 root_pos_z——这正是要分档的原因"

    params = profile_params(resolve_profile(REAL_SAMPLE))
    hd_fs = rule_mad_jump(clip, k=params["mad_jump_k"],
                          skip_channel_patterns=params["mad_jump_skip_channel_patterns"])
    hd_ch = {f["channel"] for f in hd_fs}
    assert not any(c.startswith("root_pos") for c in hd_ch), \
        "high_dynamic 档跳过 root_pos_*：跳高动作本身支配根位置"
    # 校准事实：膝角在 k=6 的 342/72 帧 = 快速屈膝（动作），k=10 下消失
    assert "dof_left_knee_joint(rad)" not in hd_ch
    assert "dof_right_knee_joint(rad)" not in hd_ch
