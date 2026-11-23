"""基准 harness 验收：注入确定性 / 匹配逻辑 / 合成底片上的规则全查全。"""

from __future__ import annotations

import numpy as np

from verticals.gait_unit.benchmark import (
    _match,
    _normalize,
    benchmark_on_clip,
    inject_random,
)
from verticals.gait_unit.defects import make_clean_clip
from verticals.gait_unit.qc_rules.skeleton_rules import (
    rule_bone_length_drift,
    rule_joint_rotation_limit,
    rule_timestamp_duplicates,
    rule_timestamp_gaps,
)


def test_inject_random_deterministic():
    clip = make_clean_clip(seconds=30.0)
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    out1, truth1 = inject_random(clip, rng1)
    out2, truth2 = inject_random(clip, rng2)
    assert truth1 == truth2
    assert np.array_equal(out1.positions, out2.positions)
    assert np.array_equal(out1.timestamps, out2.timestamps)


def test_inject_random_covers_all_types():
    clip = make_clean_clip(seconds=60.0)
    seen = set()
    rng = np.random.default_rng(7)
    for _ in range(40):
        _, truth = inject_random(clip, rng)
        seen.add(truth[0]["type"])
    assert seen == {"frame_drop", "duplicate_stamp", "joint_limit", "drift"}


def test_match_target_and_frame_overlap():
    truth = [{"type": "joint_limit", "target": "j1", "joints": {"j1"},
              "frames": (100, 150)}]
    norm = [
        {"rule": "joint_rotation_limit", "target": "j1", "runs": [(120, 130)]},   # 命中
        {"rule": "joint_rotation_limit", "target": "j2", "runs": [(120, 130)]},   # 目标不同
        {"rule": "joint_rotation_limit", "target": "j1", "runs": [(10, 20)]},     # 帧不交叠
    ]
    hit_truth, hit_findings = _match(truth, norm)
    assert hit_truth == {0}
    assert hit_findings == {0}


def test_match_clip_level_truth():
    truth = [{"type": "frame_drop", "target": None, "joints": None,
              "frames": (40, 45)}]
    norm = [{"rule": "timestamp_gaps", "target": None, "runs": [(39, 41)]}]
    hit_truth, hit_findings = _match(truth, norm)
    assert hit_truth == {0} and hit_findings == {0}
    # 没有规则命中时
    hit_truth, hit_findings = _match(truth, [])
    assert hit_truth == set() and hit_findings == set()


def test_benchmark_on_clip_rules_catch_all_types():
    """合成干净底片上，四类注入缺陷必须全部被对应规则查全（查全率 100%）；
    误报 = 未命中真值 **且** 底片基线解释不了的 finding——必须为零。"""
    from verticals.gait_unit.benchmark import _matches_baseline
    from verticals.gait_unit.qc_rules.channel_rules import rule_mad_jump

    clip = make_clean_clip(seconds=60.0)
    # 底片基线：干净 clip 上全规则的 findings（步态拐点等自有动态）
    baseline = [
        rule_timestamp_gaps(clip),
        rule_timestamp_duplicates(clip),
        rule_joint_rotation_limit(clip, max_deg=170.0),
        rule_bone_length_drift(clip, tol_rel=0.02),
    ]
    from verticals.gait_unit.benchmark import _angle_motion_clip
    baseline += [rule_mad_jump(_angle_motion_clip(clip), k=6.0)]
    baseline_norm = _normalize(clip, baseline)

    rng = np.random.default_rng(123)
    per_type: dict[str, list[bool]] = {}
    n_fp = 0
    for _ in range(12):
        ep = benchmark_on_clip(clip, rng)
        tp_idx = set(ep["tp_findings_indices"])
        for ti, inj in enumerate(ep["injected"]):
            per_type.setdefault(inj["type"], []).append(ti in ep["detected"])
        fp = [f for fi, f in enumerate(ep["findings_detail"])
              if fi not in tp_idx and not _matches_baseline(f, baseline_norm)]
        n_fp += len(fp)
    assert all(all(v) for v in per_type.values()), \
        f"存在漏检: { {k: sum(v) for k, v in per_type.items()} }"
    assert n_fp == 0, f"基线解释后仍有 {n_fp} 个误报 findings"
