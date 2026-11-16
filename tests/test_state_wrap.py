"""state ±π 表示卷绕的豁免与 action 尖峰送审（G5b / G5c）。

背景：tokyo 实测 action/state dim5 存在 ±π 卷绕（端点精确贴 ±π、|Δ|≈2π）。
- state 侧：表示层等价（数据自洽），不应报 state_jump；但卷绕是事实，
  以独立 info 规则 state_wrap 留痕，下游按连续量使用需自行 unwrap。
  端点判据刻意不用「整维值域 ⊆ [−π,π]」——合法摆动维（tokyo dim3）会被误豁免。
- action 侧：delta 语义（dim5 与 next_state−state 逐帧一致），单独修 action
  会破坏 state-action 自洽且真实物理量不可知 → review，不修复；
  归一化统计被脉冲污染的事实以 stats_pollution 证据上报，由下游决定。
"""

from __future__ import annotations

import numpy as np
import pytest

from pipeline.config import load_config
from pipeline.loader import load_dataset
from pipeline.policy import decide
from pipeline.reporter import build_report
from pipeline.validator import validate_dataset

from conftest import build_dataset


def _validate(path, **overrides):
    dataset = load_dataset(path)
    result = validate_dataset(
        dataset, load_config(robot_type="test_robot", overrides=overrides or None)
    )
    decisions = decide(result)
    report = build_report(dataset, result, decisions, dataset_path=path)
    return dataset, result, decisions, report


def _wrap_state_eps(n: int = 50, wrap_frame: int = 20, seed: int = 11) -> dict:
    """state dim2 在 wrap_frame 处从 −π 跨到 +π（两侧端点各贴 ±π 0.01 rad）。"""
    rng = np.random.default_rng(seed)
    state = rng.normal(0.0, 0.003, size=(n, 3))
    state[:wrap_frame, 2] = -3.13 + rng.normal(0.0, 0.002, size=wrap_frame)
    state[wrap_frame:, 2] = 3.13 + rng.normal(0.0, 0.002, size=n - wrap_frame)
    action = rng.normal(0.0, 0.003, size=(n, 3))  # action 独立噪声，避免牵连 action_jump
    return {"state": state, "action": action}


def _nonendpoint_jump_eps(n: int = 50, jump_frame: int = 20, seed: int = 22) -> dict:
    """|Δ|≈2π 但端点不在 ±π（−1.0 → +5.26）：卷绕豁免不得生效。"""
    rng = np.random.default_rng(seed)
    state = rng.normal(0.0, 0.003, size=(n, 3))
    state[:jump_frame, 2] = -1.0 + rng.normal(0.0, 0.002, size=jump_frame)
    state[jump_frame:, 2] = 5.26 + rng.normal(0.0, 0.002, size=n - jump_frame)
    action = rng.normal(0.0, 0.003, size=(n, 3))
    return {"state": state, "action": action}


class TestStateWrap:
    def test_wrap_is_state_wrap_info_not_state_jump(self, tmp_path):
        """T4-A：端点贴 ±π → state_wrap(info) 留痕，state_jump 不出现。"""
        episodes = {i: _wrap_state_eps(seed=900 + i) for i in range(4)}
        episodes[7] = _wrap_state_eps(seed=777, wrap_frame=20)
        path = build_dataset(tmp_path / "ds", episodes)
        dataset, result, decisions, report = _validate(path)

        res = result.results.get(7)
        assert res is not None
        assert "state_wrap" in res.reasons, f"卷绕未留痕：reasons={res.reasons}"
        assert "state_jump" not in res.reasons, "表示等价被误报为突跳"
        ev = res.details["state_wrap"]
        assert ev["wrap_ignored"] == 1
        assert ev["examples"][0]["frame"] == 20  # 卷绕后的物理帧
        assert abs(ev["examples"][0]["delta"] - 2 * np.pi) <= 0.05
        assert result.severity_of(res) == "info"
        assert decisions[7].disposition == "keep"

    def test_nonendpoint_jump_still_state_jump(self, tmp_path):
        """T4-B：|Δ|≈2π 但端点不在 ±π → 不豁免，state_jump 照常、无 state_wrap。"""
        episodes = {i: _wrap_state_eps(seed=910 + i) for i in range(4)}
        episodes[8] = _nonendpoint_jump_eps(seed=888)
        path = build_dataset(tmp_path / "ds", episodes)
        _, result, _, _ = _validate(path)

        res = result.results.get(8)
        assert res is not None
        assert "state_jump" in res.reasons, f"非端点 2π 跳变漏报：reasons={res.reasons}"
        assert "state_wrap" not in res.reasons

    def test_info_codes_reach_report_details(self, tmp_path):
        """info 级 state_wrap 不能被 reporter 丢掉：必须进 episode 明细并给出 unwrap 提示。"""
        episodes = {i: _wrap_state_eps(seed=920 + i) for i in range(4)}
        episodes[7] = _wrap_state_eps(seed=777, wrap_frame=20)
        path = build_dataset(tmp_path / "ds", episodes)
        _, result, _, report = _validate(path)

        detail_eps = [d["episode_index"] for d in report.get("invalid_episode_details") or []]
        assert 7 in detail_eps, "info 级结论被 reporter 丢弃"
        assert any("state_wrap" in (d.get("reasons") or [])
                   for d in report.get("invalid_episode_details") or [])
        assert any("unwrap" in r for r in report.get("recommendations") or [])
