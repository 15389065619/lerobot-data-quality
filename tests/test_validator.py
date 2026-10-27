"""质检规则的单测：每条规则都要有「命中」和「不误报」两侧的用例。"""

from __future__ import annotations

import numpy as np
import pytest

from pipeline.config import load_config
from pipeline.loader import load_dataset
from pipeline.validator import validate_dataset

from conftest import build_dataset, smooth_episode


def _validate(path, **overrides):
    dataset = load_dataset(path)
    return dataset, validate_dataset(dataset, load_config(robot_type="test_robot", overrides=overrides or None))


def test_clean_dataset_produces_no_findings(normal_dataset):
    dataset, result = _validate(normal_dataset)
    assert dataset.num_episodes == 6
    assert result.valid_episodes == 6
    assert result.invalid_episodes == []
    assert result.dataset_issues == []


def test_nan_state_in_middle_is_not_trimmable(tmp_path):
    ep = smooth_episode(40, seed=1)
    ep["state"][15:18, 0] = np.nan
    path = build_dataset(tmp_path / "ds", {0: ep, 1: smooth_episode(40, seed=2)})
    _, result = _validate(path)

    res = result.results[0]
    assert "nan_state" in res.reasons
    detail = res.details["nan_state"]
    assert detail["boundary_only"] is False
    assert detail["trimmable"] is False
    assert detail["nan_count"] == 3
    assert res.trim_plan() is None


def test_boundary_nan_is_trimmable(tmp_path):
    ep = smooth_episode(40, seed=3)
    ep["state"][:3, 1] = np.nan
    path = build_dataset(tmp_path / "ds", {0: ep, 1: smooth_episode(40, seed=4)})
    _, result = _validate(path)

    detail = result.results[0].details["nan_state"]
    assert detail["boundary_only"] is True
    assert detail["trimmable"] is True
    assert detail["head_frames"] == 3
    assert result.results[0].trim_plan() == {"head_frames": 3, "tail_frames": 0}


def test_action_jump_uses_adaptive_threshold(tmp_path):
    """正常抖动不应误报，单帧尖峰必须命中。"""
    normal = smooth_episode(40, seed=5)
    spike = smooth_episode(40, seed=6)
    spike["action"][20, 0] += 12.0

    path = build_dataset(tmp_path / "ds", {0: normal, 1: spike})
    _, result = _validate(path)

    assert "action_jump" not in result.results[0].reasons, "正常轨迹被误报为跳变"
    assert "action_jump" in result.results[1].reasons
    detail = result.results[1].details["action_jump"]
    assert detail["max_delta"] > 10
    assert detail["worst_dim"] == 0
    assert len(detail["thresholds_used"]) == 3


def test_frame_count_outlier_detected(tmp_path):
    episodes = {ep: smooth_episode(40, seed=ep) for ep in range(5)}
    episodes[5] = {"state": np.zeros((3, 3), dtype=np.float32),
                   "action": np.zeros((3, 3), dtype=np.float32)}
    path = build_dataset(tmp_path / "ds", episodes)
    dataset, result = _validate(path)

    bounds = result.resolved_thresholds["frame_count"]
    assert bounds["mode"] == "auto"
    assert bounds["min"] >= 8  # 自适应下限受 floor 约束

    res = result.results[5]
    assert "frame_count_abnormal" in res.reasons
    assert res.details["frame_count_abnormal"]["direction"] == "too_few"
    assert "frame_count_abnormal" not in result.results[0].reasons


def test_timestamp_regression_and_step_deviation(tmp_path):
    bad = smooth_episode(40, seed=7)
    ts = np.arange(40, dtype=np.float32) / 30
    ts[20], ts[21] = ts[21], ts[20] - 0.5

    jitter = smooth_episode(40, seed=8)
    ts_jitter = np.arange(40, dtype=np.float32) / 30
    ts_jitter[20:] += 0.5

    path = build_dataset(tmp_path / "ds", {
        0: {**bad, "timestamp": ts},
        1: {**jitter, "timestamp": ts_jitter},
    })
    _, result = _validate(path)

    assert "timestamp_not_monotonic" in result.results[0].reasons
    # 步长突变只是 info，不应污染 timestamp_not_monotonic
    assert "timestamp_not_monotonic" not in result.results[1].reasons
    assert "timestamp_step_deviation" in result.results[1].reasons
    assert result.severity_of(result.results[1]) == "info"


def test_missing_task(tmp_path):
    ep = smooth_episode(40, seed=9)
    path = build_dataset(tmp_path / "ds", {0: {**ep, "task_index": -1}, 1: smooth_episode(40, seed=10)})
    _, result = _validate(path)
    assert "missing_task" in result.results[0].reasons
    assert "missing_task" not in result.results[1].reasons


def test_static_episode(tmp_path):
    still = {"state": np.full((40, 3), 0.5, dtype=np.float32) + np.random.default_rng(0).normal(0, 1e-8, (40, 3)),
             "action": np.full((40, 3), 0.5, dtype=np.float32)}
    path = build_dataset(tmp_path / "ds", {0: still, 1: smooth_episode(40, seed=11)})
    _, result = _validate(path)
    assert "static_episode" in result.results[0].reasons
    assert "static_episode" not in result.results[1].reasons


def test_duplicate_episode(tmp_path):
    original = smooth_episode(40, seed=12)
    copy = {"state": original["state"].copy(), "action": original["action"].copy()}
    path = build_dataset(tmp_path / "ds", {0: original, 1: copy, 2: smooth_episode(40, seed=13)})
    _, result = _validate(path)

    assert "duplicate_episode" not in result.results[0].reasons, "首条不应被判为重复"
    assert "duplicate_episode" in result.results[1].reasons
    assert result.results[1].details["duplicate_episode"]["duplicate_of"] == 0
    assert "duplicate_episode" not in result.results[2].reasons


def test_stale_stats_detected(tmp_path):
    episodes = {ep: smooth_episode(40, seed=ep) for ep in range(4)}
    path = build_dataset(tmp_path / "ds", episodes, stale_stats=True)
    _, result = _validate(path)

    codes = [i.code for i in result.dataset_issues]
    assert "stats_stale" in codes
    issue = next(i for i in result.dataset_issues if i.code == "stats_stale")
    assert issue.repair == "stats"
    assert issue.detail["max_relative_deviation"] > issue.detail["tolerance"]


def test_fresh_stats_not_flagged(normal_dataset):
    _, result = _validate(normal_dataset)
    assert all(i.code != "stats_stale" for i in result.dataset_issues)


def test_fixed_mode_overrides_adaptive_threshold(tmp_path):
    episodes = {ep: smooth_episode(40, seed=ep) for ep in range(4)}
    episodes[4] = smooth_episode(12, seed=99)
    path = build_dataset(tmp_path / "ds", episodes)
    dataset = load_dataset(path)

    fixed = load_config(overrides={"frame_count": {"mode": "fixed", "floor": 20, "ceiling": 100}})
    result = validate_dataset(dataset, fixed)
    assert result.resolved_thresholds["frame_count"]["mode"] == "fixed"
    assert result.resolved_thresholds["frame_count"]["min"] == 20
    assert "frame_count_abnormal" in result.results[4].reasons


def test_contract_keys_present(normal_dataset):
    """报告消费方依赖的字段契约不能变。"""
    _, result = _validate(normal_dataset)
    payload = result.to_dict()
    assert {"total_episodes", "valid_episodes", "invalid_episodes"} <= set(payload)
    assert payload["total_episodes"] == 6
    for item in payload["invalid_episodes"]:
        assert {"episode_index", "reasons"} <= set(item)


def test_skip_reason_recorded_when_no_video(normal_dataset):
    """无视频特征时必须如实记录跳过的检查，不能假装检查过。"""
    _, result = _validate(normal_dataset)
    assert "frame_mismatch" in result.skipped_checks
