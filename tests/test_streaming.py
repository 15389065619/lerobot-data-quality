"""流式模式：分片扫描与内存模式结论一致，且能识别脏 episode。"""

from __future__ import annotations

import numpy as np

from pipeline.config import load_config
from pipeline.loader import load_dataset
from pipeline.validator import validate_dataset

from conftest import build_dataset, smooth_episode


def _reasons(validation) -> dict[int, set[str]]:
    return {ep: set(res.reasons) for ep, res in validation.results.items()}


def test_streaming_matches_in_memory(normal_dataset):
    config = load_config(robot_type="test_robot")
    full = load_dataset(normal_dataset)
    in_memory = validate_dataset(full, config)
    streamed = validate_dataset(load_dataset(normal_dataset, load_frames=False), config, streaming=True)

    assert streamed.total_episodes == in_memory.total_episodes
    assert streamed.resolved_thresholds["action_jump"] == in_memory.resolved_thresholds["action_jump"]
    assert streamed.resolved_thresholds["frame_count"] == in_memory.resolved_thresholds["frame_count"]
    assert streamed.dataset_issues == in_memory.dataset_issues
    assert _reasons(streamed) == _reasons(in_memory)
    assert streamed.aggregates["total_frames"] == int(len(full.frames))
    assert streamed.aggregates["frames_by_episode"] == {
        int(ep): int(n) for ep, n in full.frames.groupby("episode_index").size().items()
    }


def test_streaming_flags_dirty_episode(tmp_path):
    ok = smooth_episode(40, seed=1)
    dirty = smooth_episode(40, offset=0.5, seed=2)
    dirty_state = dirty["state"].copy()
    dirty_state[10:14, 1] = np.nan  # 中段坏帧：不可裁剪
    dup = smooth_episode(40, seed=1)  # 与 ep0 动作轨迹重复
    episodes = {0: ok, 1: {"state": dirty_state, "action": dirty["action"]}, 2: dup}
    root = build_dataset(tmp_path / "dirty", episodes)

    config = load_config(robot_type="test_robot")
    in_memory = validate_dataset(load_dataset(root), config)
    streamed = validate_dataset(load_dataset(root, load_frames=False), config, streaming=True)

    assert _reasons(streamed) == _reasons(in_memory)
    assert "nan_state" in streamed.results[1].reasons
    assert "duplicate_episode" in streamed.results[2].reasons
    assert streamed.results[2].details["duplicate_episode"]["duplicate_of"] == 0
