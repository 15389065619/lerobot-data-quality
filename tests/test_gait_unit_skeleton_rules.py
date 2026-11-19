"""骨架级四规则验收：时间戳缺口/盖章重复/旋转超限/骨长漂移——注入地面真值断言。"""

from __future__ import annotations

import math

import numpy as np

from lerobot_data_quality.gait_unit.defects import (
    inject_drift,
    inject_duplicate_stamp,
    inject_frame_drop,
    inject_joint_limit,
    make_clean_clip,
)
from lerobot_data_quality.gait_unit.qc_rules.skeleton_rules import (
    rule_bone_length_drift,
    rule_joint_rotation_limit,
    rule_timestamp_duplicates,
    rule_timestamp_gaps,
)


def test_timestamp_gaps_finds_dropped_frames():
    clip = make_clean_clip(seconds=10.0)
    injected, manifest = inject_frame_drop(clip, start=100, n=5)
    findings = rule_timestamp_gaps(injected, k=3.0)
    assert len(findings) == 1
    f = findings[0]
    s, e = f["frames"]
    assert s <= 100 < e, f"缺口帧区间应覆盖注入位置: {f}"
    assert f["n_missing_approx"] == 5


def test_timestamp_duplicates_finds_coarse_stamps():
    clip = make_clean_clip(seconds=10.0)
    injected, manifest = inject_duplicate_stamp(clip, start=100, n=60)
    findings = rule_timestamp_duplicates(injected)
    assert len(findings) == 1
    s, e = findings[0]["frames"]
    assert s <= 100 < e
    assert findings[0]["n_duplicate_frames"] >= 30


def test_joint_rotation_limit_finds_injected_overrotation():
    clip = make_clean_clip(seconds=10.0)
    injected, manifest = inject_joint_limit(clip, joint="ShankL",
                                            axis="X", extra_deg=200.0)
    findings = rule_joint_rotation_limit(injected, max_deg=170.0)
    shank = [f for f in findings if f["joint"] == "ShankL"]
    assert shank, "基线 ~17° + 附加 200° = 217°，应超 170° 阈"
    assert shank[0]["max_observed_deg"] > 170.0
    # 干净 clip 无超限
    assert rule_joint_rotation_limit(clip, max_deg=170.0) == []


def test_bone_length_drift_finds_subtree_drift():
    clip = make_clean_clip(seconds=10.0)
    # 左腿链漂移：跨子树边界的那根骨（pelvis→ThighL 之外……实际是
    # Hips→ThighL 段被拉长，因为 ThighL 及其子树整体平移而 Hips 不动）
    chain = ("ThighL", "ShankL", "FootL")
    injected, manifest = inject_drift(clip, rate=0.03, axis="x",
                                      joints=chain, start=150)
    findings = rule_bone_length_drift(injected, tol_rel=0.02)
    assert findings, "子树漂移必须被骨长规则抓到"
    drifted_bones = {f["bone"] for f in findings}
    assert any("ThighL" in b for b in drifted_bones), \
        f"跨子树边界的骨（Hips→ThighL）应漂移: {drifted_bones}"
    # 干净 clip 零 findings
    assert rule_bone_length_drift(clip, tol_rel=0.02) == []


def test_skeleton_rules_clean_clip_all_quiet():
    clip = make_clean_clip(seconds=10.0)
    assert rule_timestamp_gaps(clip) == []
    assert rule_timestamp_duplicates(clip) == []
    assert rule_joint_rotation_limit(clip, max_deg=170.0) == []
    assert rule_bone_length_drift(clip, tol_rel=0.02) == []
