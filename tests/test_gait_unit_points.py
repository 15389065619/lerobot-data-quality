"""marker 点级规则验收：dropout / jump 的地面真值断言。"""

from __future__ import annotations

import numpy as np
import pytest

from verticals.gait_unit.ir import GaitClip
from verticals.gait_unit.qc_rules.point_rules import (
    rule_marker_dropout,
    rule_marker_jump,
)


def _fixture() -> GaitClip:
    """3 marker × 200 帧：m0 干净；m1 有 20 帧连续缺测（10%>5% 阈）；
    m2 在第 100 帧有一个位移尖峰（误关联）。注意 m2 轨迹带微噪声——
    完美均匀的步长会让 MAD=0，跳变规则按设计跳过（这是文档化行为）。"""
    T = 200
    rng = np.random.default_rng(5)
    pos = np.zeros((T, 3, 3))
    pos[:, 0, :] = np.stack([np.linspace(0, 1, T), np.zeros(T), np.zeros(T)], axis=1)
    pos[:, 1, :] = np.stack([np.zeros(T), np.linspace(0, 1, T), np.zeros(T)], axis=1)
    pos[:, 2, 0] = np.linspace(1, 2, T) + rng.normal(0, 0.002, T)
    pos[:, 2, 1:] = 0.0
    pos[40:60, 1, :] = np.nan                     # m1 缺测段
    pos[100, 2, 0] += 0.5                         # m2 尖峰（正常步长 ~0.005）
    return GaitClip(fps=30.0, timestamps=np.arange(T, dtype=np.float64) / 30.0,
                    joint_names=("marker_000", "marker_001", "marker_002"),
                    positions=pos, parents=None, rotations=None,
                    source={"source": "synthetic_points"})


def test_marker_dropout_finds_long_nan_run():
    clip = _fixture()
    findings = rule_marker_dropout(clip, max_ratio=0.05)
    assert len(findings) == 1
    assert findings[0]["marker"] == "marker_001"
    assert findings[0]["ratio"] == pytest.approx(0.10)
    assert findings[0]["longest_run"] == 20
    # m0/m2 干净，不得误报
    assert all(f["marker"] != "marker_000" for f in findings)


def test_marker_jump_finds_injected_spike():
    clip = _fixture()
    findings = rule_marker_jump(clip, k=8.0)
    m2 = [f for f in findings if f["marker"] == "marker_002"]
    assert m2, "位移尖峰必须被抓到"
    # 尖峰(99→100)与回落(100→101)构成一个连续异常段 [100, 102)
    runs = [tuple(r) for r in m2[0]["runs"]]
    assert any(s <= 100 and e >= 101 for s, e in runs), f"尖峰及回落应被覆盖: {runs}"
    assert m2[0]["max_z"] > 8.0
    # m0 匀速直线：无跳变
    assert all(f["marker"] != "marker_000" for f in findings)
