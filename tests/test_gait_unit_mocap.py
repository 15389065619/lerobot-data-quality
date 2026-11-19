"""光捕通道级 QC（Tier 1）验收：MotionClip IR + MotionDecode adapter + 四条规则。

  1. IR 结构校验（形状/Inf/NaN 不裁决）
  2. MotionDecode adapter：真实样本冒烟（存在才跑）+ 缺列报错 + 脏列丢弃
  3. 规则地面真值：注入的跳变/NaN 段/常值通道/越界，必须被对应规则找到
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lerobot_data_quality.gait_unit.adapters import load_motiondecode_csv
from lerobot_data_quality.gait_unit.motion import MotionClip, validate_motion
from lerobot_data_quality.gait_unit.qc_rules import (
    rule_constant_channel,
    rule_mad_jump,
    rule_nan_runs,
    rule_range_check,
)

# 真实样本（本仓库 data/ 下；缺失则冒烟测试自动跳过）
REAL_SAMPLE = (Path("data/chingmu_motiondecode/samples")
               / "1.1.Basic_Movement_Category/1.1.1.High_Dynamic_Movement"
               / "1.1.1.1.Standing_High_Jump/BM_Standing_High_Jump_00001.csv")


def _fixture_clip() -> MotionClip:
    """四通道合成 clip，每条规则埋一个已知缺陷。"""
    T = 600
    t = np.arange(T)
    rng = np.random.default_rng(11)
    a = np.sin(t * 0.05) + 0.02 * rng.standard_normal(T)   # 干净 + 注入尖峰
    a[300] += 20.0
    b = np.full(T, 1.25)                                    # 死通道
    c = np.sin(t * 0.03) + 0.01 * rng.standard_normal(T)
    c[100:105] = np.nan                                     # 连续 5 帧缺测
    c[420] = np.nan                                         # 单帧缺测
    d = 0.6 * np.sin(t * 0.04) + 0.02 * rng.standard_normal(T)
    d[200:206] = 2.5                                        # 越界段
    return MotionClip(
        fps=120.0,
        channel_names=("a", "b", "c", "dof_left_knee_joint(rad)"),
        channels=np.stack([a, b, c, d], axis=1),
        source={"source": "synthetic"},
    )


# --------------------------------------------------------------------------
# 1. IR 结构校验
# --------------------------------------------------------------------------

def test_motion_clip_validate():
    clip = _fixture_clip()
    assert validate_motion(clip) == []
    bad = MotionClip(fps=-1, channel_names=("a", "b"), channels=clip.channels[:, :2])
    issues = validate_motion(bad)
    assert any("fps" in s for s in issues)
    wrong_names = MotionClip(fps=120, channel_names=("a",),
                             channels=clip.channels[:, :2])
    assert any("channel_names" in s for s in validate_motion(wrong_names))


def test_motion_clip_nan_is_qc_object_not_structural():
    clip = _fixture_clip()
    issues = validate_motion(clip)
    assert not any("NaN" in s for s in issues), "NaN 是质检对象，不属于结构错误"


# --------------------------------------------------------------------------
# 2. MotionDecode adapter
# --------------------------------------------------------------------------

@pytest.mark.skipif(not REAL_SAMPLE.exists(), reason="真实样本不在本机")
def test_motiondecode_real_sample_smoke():
    clip = load_motiondecode_csv(REAL_SAMPLE)
    assert clip.n_channels == 36
    assert clip.channel_names[0] == "root_pos_x(m)"
    assert clip.channel_names[7] == "dof_left_hip_pitch_joint(rad)"
    assert clip.n_frames >= 1200         # 跳高片段，39.5 s @ 120fps ≈ 4740 帧
    assert validate_motion(clip) == []
    # 规则跑通即可（真实数据的 findings 数量不预注册，只要求不崩且留痕完整）
    for rule in (rule_mad_jump, rule_nan_runs, rule_constant_channel):
        for f in rule(clip):
            assert f["channel"] in clip.channel_names


def test_motiondecode_missing_column_fails(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("foo,bar\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="缺少必需列"):
        load_motiondecode_csv(bad)


def test_motiondecode_dirty_column_dropped(tmp_path):
    csv = tmp_path / "with_label.csv"
    csv.write_text(
        "root_pos_x(m),root_pos_y(m),root_pos_z(m),dof_left_knee_joint(rad),label\n"
        "0.1,0.2,0.3,0.05,walk\n"
        "0.1,0.2,0.3,0.06,walk\n",
        encoding="utf-8",
    )
    clip = load_motiondecode_csv(csv)
    assert clip.n_channels == 4
    assert "label" in clip.source["dropped_columns"]


# --------------------------------------------------------------------------
# 3. 规则地面真值
# --------------------------------------------------------------------------

def test_rule_mad_jump_finds_injected_spike():
    clip = _fixture_clip()
    findings = rule_mad_jump(clip, k=6.0)
    by_ch = {f["channel"]: f for f in findings}
    assert "a" in by_ch, "注入的尖峰必须被抓到"
    f = by_ch["a"]
    assert (300, 301) in [tuple(r) for r in f["runs"]]
    assert f["max_z"] > 6.0
    assert "dof_left_knee_joint(rad)" not in by_ch, "干净通道不应误报"


def test_rule_nan_runs_finds_injected_runs():
    clip = _fixture_clip()
    findings = rule_nan_runs(clip)
    by_ch = {f["channel"]: f for f in findings}
    assert by_ch["c"]["longest_run"] == 5
    assert by_ch["c"]["n_nan"] == 6          # 5 连续 + 1 单帧
    assert [tuple(r) for r in by_ch["c"]["runs"]] == [(100, 105), (420, 421)]


def test_rule_constant_channel_finds_dead():
    clip = _fixture_clip()
    findings = rule_constant_channel(clip)
    assert len(findings) == 1 and findings[0]["channel"] == "b"
    assert abs(findings[0]["value"] - 1.25) < 1e-9


def test_rule_range_check_uses_limits_and_skips_rest():
    clip = _fixture_clip()
    limits = {"dof_left_knee_joint(rad)": (-1.0, 1.0)}
    findings = rule_range_check(clip, limits)
    assert len(findings) == 1
    f = findings[0]
    assert f["channel"] == "dof_left_knee_joint(rad)"
    assert f["n_violations"] == 6            # d 通道 200:206 = 2.5 > 1.0
    assert (200, 206) in [tuple(r) for r in f["runs"]]
    # 未配 limits 的通道（如 b/c/a）必须全部跳过
    assert all(x["channel"] in limits for x in findings)


def test_rule_range_check_empty_limits_is_noop():
    clip = _fixture_clip()
    assert rule_range_check(clip, None) == []
