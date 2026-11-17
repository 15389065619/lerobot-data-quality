"""成对尖峰漏检修复的验证（G5a）。

背景：fps=5 时 spike_window 折算后塌缩为 1 帧，成对（连续两帧）超阈 Δ 互为
对方的邻域，中位数被抬到 ~π，spike_ratio=3 判不过 → 整对漏检（tokyo 实测
action dim5 的 4 个 ±2π 脉冲全部漏报）。修复三件套：
1. 邻域 ref 剔除超阈 Δ（成对尖峰不再互相抬轿）；
2. 剩余邻居 <2 时回退全局逐维 median(|Δ|)；
3. 连续 ≥3 个超阈 Δ 视为运动包络整体跳过（反向护栏：修复不得把斜坡误报）。
"""

from __future__ import annotations

import types

import numpy as np
import pytest

from pipeline.config import load_config
from pipeline.loader import load_dataset
from pipeline.reporter import _pollution_issue
from pipeline.validator import validate_dataset
from pipeline.validator.rules import _scan_spikes

from conftest import build_dataset


def _validate(path, **overrides):
    dataset = load_dataset(path)
    return dataset, validate_dataset(
        dataset, load_config(robot_type="test_robot", overrides=overrides or None)
    )


def _noise_eps(n: int, seed: int, dim: int = 3, sigma: float = 0.003) -> dict:
    rng = np.random.default_rng(seed)
    vals = rng.normal(0.0, sigma, size=(n, dim))
    return {"state": vals.copy(), "action": vals.copy()}


class TestPairedSpikeHit:
    def test_fps5_paired_2pi_pulse_is_hit(self, tmp_path):
        """T1：fps=5 下单帧 ±2π 脉冲（成对 diff）必须命中，severity=warning。

        旧逻辑：window 塌缩为 1，成对尖峰互抬邻域中位数 → 整对漏检。
        """
        episodes = {i: _noise_eps(240, seed=i) for i in range(11)}
        pulse = _noise_eps(240, seed=100)
        pulse["action"][100, 2] = 6.28308  # 单帧脉冲：diff 99 与 diff 100 成对超阈
        episodes[7] = pulse
        path = build_dataset(tmp_path / "ds", episodes, fps=5)
        _, result = _validate(path)

        res = result.results.get(7)
        assert res is not None, "episode 7 无结论"
        assert "action_jump" in res.reasons, f"成对尖峰仍漏检：reasons={res.reasons}"
        detail = res.details["action_jump"]
        assert detail["jump_count"] == 2
        assert detail["paired_spike"] == 1
        # severity 不落在 details 里（等于配置默认时被剥除）；episode 级 = warning
        assert result.severity_of(res) == "warning"  # 成对尖峰不做幅度降级（G5c 决定）
        assert detail["spike_frames"] == [100]  # 物理帧，不是 diff 下标
        assert detail["ref_source"] == "global"  # fps=5 窗口=1，剔除成对尖峰后必走全局回退
        assert detail["thresholds_used"][2] == pytest.approx(6.0)  # floor 折算 1.0×30/5
        assert detail["max_delta"] == pytest.approx(6.2831, abs=0.02)  # 脉冲叠加在噪声上
        # 脉冲帧计数只算物理脉冲帧（100），不是 up/down 两个 diff（99/100）
        pollution = detail["stats_pollution"][0]
        assert pollution["spike_count"] == 1
        assert pollution["n_frames"] == 240

    def test_smooth_ramp_is_not_hit(self, tmp_path):
        """T5：5 个连续超阈 Δ 的平滑斜坡是运动包络，不得命中（修复的反向护栏）。"""
        episodes = {i: _noise_eps(240, seed=300 + i) for i in range(11)}
        ramp = _noise_eps(240, seed=200)
        for k, v in enumerate((1.2, 2.4, 3.6, 4.8, 6.0, 7.2)):
            ramp["action"][100 + k, 0] = v  # 帧 100-105，5 个连续 Δ=1.2 > floor 1.0
        episodes[5] = ramp
        path = build_dataset(tmp_path / "ds", episodes, fps=30)
        _, result = _validate(path)

        res = result.results.get(5)
        assert res is not None
        assert "action_jump" not in res.reasons, f"斜坡被误报：{res.details.get('action_jump')}"


class TestSameSignRampNotRescued:
    """同号 2-diff 是快速斜坡（出去不回来），不能借全局回退被救回。

    pusht fps=10 实测：连续 4 帧单调加速里恰好 2 个 diff 越阈，邻域剔掉同伴后
    只剩 1 个，旧回退拿全局 median（运动尺度）兜底 → 6 集误报。
    """

    def test_unit_same_sign_2run_not_kept(self):
        rng = np.random.default_rng(11)
        vals = rng.normal(0.0, 0.003, size=(240, 1))
        vals[100:, 0] += 8.6
        vals[100, 0] = 4.0  # diff 99 = 4.0、diff 100 = 4.6（同号），阈 3.0
        scan = _scan_spikes(
            vals, [3.0], {}, prefix="action_jump", min_frames=2,
            fps=10.0, diff_ref=[0.0005],
        )
        assert scan is not None
        assert scan.kept == [], f"同号斜坡被全局回退救回：{scan.kept}"
        assert scan.paired_spike == 0  # paired_spike 只统计异号往返脉冲

    def test_end_to_end_fps10_ramp_not_reported(self, tmp_path):
        episodes = {i: _noise_eps(240, seed=500 + i) for i in range(11)}
        ramp = _noise_eps(240, seed=600)
        ramp["action"][100:, 0] = 8.6
        ramp["action"][100, 0] = 4.0
        episodes[4] = ramp
        path = build_dataset(tmp_path / "ds", episodes, fps=10)
        _, result = _validate(path)

        res = result.results.get(4)
        assert res is not None
        assert "action_jump" not in res.reasons, f"斜坡被误报：{res.details.get('action_jump')}"
        assert "state_jump" not in res.reasons


class TestStateJumpEvidence:
    def test_state_jump_reports_ref_source_and_paired(self, tmp_path):
        """state 侧重走同一条回退路径时，证据里必须能看出 ref 来源与脉冲对数。"""
        episodes = {i: _noise_eps(240, seed=700 + i) for i in range(11)}
        pulse = _noise_eps(240, seed=800)
        pulse["state"][100, 1] = 6.28308  # fps=5 下 state floor=6.0，且非 ±π 端点
        episodes[3] = pulse
        path = build_dataset(tmp_path / "ds", episodes, fps=5)
        _, result = _validate(path)

        res = result.results.get(3)
        assert res is not None
        assert "state_jump" in res.reasons, f"state 脉冲漏检：{res.reasons}"
        detail = res.details["state_jump"]
        assert detail["ref_source"] == "global"
        assert detail["paired_spike"] == 1


class TestPollutionIssueGate:
    """数据集级污染提示必须有实际抬高才发，否则数字自相矛盾。"""

    def _issue(self, recorded: float, trimmed: float):
        res = types.SimpleNamespace(details={"action_jump": {"stats_pollution": [{
            "dim": 0, "spike_count": 1, "observed_std": max(recorded, trimmed),
            "trimmed_std": trimmed, "n_frames": 100}]}})
        validation = types.SimpleNamespace(results={0: res})
        dataset = types.SimpleNamespace(
            stats={"action": {"std": [recorded]}}, truncated_to=None)
        return _pollution_issue(validation, dataset)

    def test_no_issue_when_std_not_inflated(self):
        # chingmu 实测：记录 0.1256 vs 剔尖峰 0.1676，剔完更大 → 不是污染
        assert self._issue(0.1256, 0.1676) is None

    def test_issue_when_recorded_std_inflated(self):
        # tokyo 实测：0.115 vs 0.0024
        issue = self._issue(0.115275, 0.002367)
        assert issue is not None
        assert issue.code == "stats_pollution"


class TestSpikeScanUnit:
    """T6：helper 级直测 ref 回退分支（端到端构造不出单独的 <2 邻域场景）。"""

    def test_fallback_to_global_when_few_neighbours_left(self):
        # 4 帧、4 个唯一值（避开离散维豁免）：脉冲的 up/down 成对超阈 diff，
        # 每个候选剔除对方后只剩 1 个邻居 → 必须走全局回退
        arr = np.array([[0.001], [6.283], [0.002], [0.004]], dtype=np.float64)
        scan = _scan_spikes(
            arr, [1.0], {}, prefix="action_jump", min_frames=2,
            fps=30.0, diff_ref=[0.003],
        )
        assert scan is not None and len(scan.kept) == 2
        assert scan.global_hits == 2 and scan.local_hits == 0
        assert scan.paired_spike == 1

    def test_local_ref_when_enough_neighbours(self):
        # 阶跃（跳上去不回来）只有 1 个超阈 diff，邻域 6 个全是正常噪声 → local
        rng = np.random.default_rng(7)
        vals = rng.normal(0.0, 0.003, size=(240, 1))
        vals[100:, 0] = 6.283
        scan = _scan_spikes(
            vals, [1.0], {}, prefix="action_jump", min_frames=2,
            fps=30.0, diff_ref=[0.003],
        )
        assert scan is not None and len(scan.kept) == 1
        assert scan.local_hits == 1 and scan.global_hits == 0

    def test_envelope_runs_skipped(self):
        vals = np.zeros((240, 1))
        for k, v in enumerate((1.2, 2.4, 3.6, 4.8, 6.0, 7.2)):
            vals[100 + k, 0] = v
        scan = _scan_spikes(
            vals, [1.0], {}, prefix="action_jump", min_frames=2,
            fps=30.0, diff_ref=[0.003],
        )
        assert scan is not None and scan.kept == []
        assert scan.envelope_runs == 1  # 一段连续 5 diff 的包络
