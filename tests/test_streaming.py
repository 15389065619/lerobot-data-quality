"""流式模式：分片扫描与内存模式结论一致，且能识别脏 episode。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from pipeline.config import load_config
from pipeline.loader import load_dataset
from pipeline.validator import validate_dataset

from conftest import FPS, build_dataset, smooth_episode


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


def test_streaming_tolerates_mixed_state_dims():
    """不同 episode 的 state 宽度不一致时，流式质检不应在 np.vstack 处抛 ValueError。"""
    from pipeline.validator.streaming import _finite_mean, _stack

    a6 = np.full((3, 6), 1.0)
    a7 = np.full((3, 7), 2.0)

    mean = _finite_mean([a6, a7])
    assert mean is not None and mean.shape == (6,)
    np.testing.assert_allclose(mean, np.ones(6))
    assert _stack([a6, a7]).shape == (3, 6)


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


# --------------------------------------------------------------------- 特殊布局
def _append_shard(root: Path, rows: dict[str, list], file_index: int) -> None:
    """按 conftest 的列结构往 chunk-000 追加一个 data 分片。"""
    table = pa.table({
        "observation.state": pa.array(np.asarray(rows["state"], dtype=np.float32).tolist(),
                                      type=pa.list_(pa.float32(), 3)),
        "action": pa.array(np.asarray(rows["action"], dtype=np.float32).tolist(),
                           type=pa.list_(pa.float32(), 3)),
        "timestamp": pa.array(np.asarray(rows["timestamp"], dtype=np.float32)),
        "frame_index": pa.array(np.asarray(rows["frame_index"], dtype=np.int64)),
        "episode_index": pa.array(np.asarray(rows["episode_index"], dtype=np.int64)),
        "index": pa.array(np.asarray(rows["index"], dtype=np.int64)),
        "task_index": pa.array(np.asarray(rows["task_index"], dtype=np.int64)),
    })
    path = root / "data" / "chunk-000" / f"file-{file_index:03d}.parquet"
    pq.write_table(table, path)


def _split_file_dataset(root: Path) -> Path:
    """ep0 拆到 file-000/file-001（v3 里同一集跨文件是真实场景）。

    缺陷（中段 NaN）埋在第一个分片：逐分片校验会让末分片结论覆盖它。
    """
    ep = smooth_episode(40, seed=3)
    state = ep["state"].copy()
    state[3:5, 1] = np.nan
    base = build_dataset(root, {0: ep})  # meta/info/stats；data 随后重写为两个分片

    def rows(lo: int, hi: int) -> dict[str, list]:
        n = hi - lo
        return {
            "state": state[lo:hi], "action": ep["action"][lo:hi],
            "timestamp": np.arange(lo, hi) / FPS,
            "frame_index": np.arange(lo, hi, dtype=np.int64),
            "episode_index": np.zeros(n, dtype=np.int64),
            "index": np.arange(lo, hi, dtype=np.int64),
            "task_index": np.zeros(n, dtype=np.int64),
        }

    _append_shard(base, rows(0, 20), 0)
    _append_shard(base, rows(20, 40), 1)
    return base


def test_streaming_aggregates_cross_file_episode(tmp_path):
    """跨 parquet 文件的 episode 必须聚合后整体校验。

    逐分片校验会让末分片的结论覆盖前面的：frame_count 变成末分片行数
    （与 Pass 1 全集口径错位 → 误报 frame_count_abnormal），前分片检出的
    nan_state 静默丢失。
    """
    root = _split_file_dataset(tmp_path / "split")
    config = load_config(robot_type="test_robot")
    in_memory = validate_dataset(load_dataset(root), config)
    streamed = validate_dataset(load_dataset(root, load_frames=False), config, streaming=True)

    assert _reasons(streamed) == _reasons(in_memory)
    assert streamed.results[0].frame_count == 40
    assert "nan_state" in streamed.results[0].reasons
    assert "frame_count_abnormal" not in streamed.results[0].reasons


def _per_file_dataset(root: Path) -> Path:
    """两个 episode 各占一个分片（file-000=ep0, file-001=ep1），供损坏注入。"""
    eps = {0: smooth_episode(10, seed=0), 1: smooth_episode(10, offset=1.0, seed=1)}
    base = build_dataset(root, eps)
    for fi, ep in enumerate(sorted(eps)):
        n = len(eps[ep]["state"])
        _append_shard(base, {
            "state": eps[ep]["state"], "action": eps[ep]["action"],
            "timestamp": np.arange(n) / FPS,
            "frame_index": np.arange(n, dtype=np.int64),
            "episode_index": np.full(n, ep, dtype=np.int64),
            "index": np.arange(fi * n, fi * n + n, dtype=np.int64),
            "task_index": np.zeros(n, dtype=np.int64),
        }, fi)
    return base


def test_streaming_degrades_on_corrupt_shard(tmp_path):
    """单个分片损坏不应让整次流式质检中止——与内存模式同样降级：
    可读的照常质检，读不到的 episode 标 episode_read_error 进入处置流程。"""
    root = _per_file_dataset(tmp_path / "ds")
    (root / "data" / "chunk-000" / "file-001.parquet").write_bytes(b"corrupted parquet")

    config = load_config(robot_type="test_robot")
    in_memory = validate_dataset(load_dataset(root), config)
    streamed = validate_dataset(load_dataset(root, load_frames=False), config, streaming=True)

    # 可读分片照常质检；读不到的 episode 不静默消失（内存模式在该场景由
    # 数据集级 schema 审计兜底，流式模式显式标 episode_read_error，更可处置）
    assert 0 in in_memory.results and not in_memory.results[0].reasons
    assert streamed.results[0].frame_count == 10
    assert "episode_read_error" not in streamed.results[0].reasons
    assert "episode_read_error" in streamed.results[1].reasons
    assert streamed.total_episodes == 2


def _string_episode_index_dataset(root: Path) -> Path:
    """把 episode_index 列改写成字符串类型（脏导出的真实形态）。"""
    base = build_dataset(root, {0: smooth_episode(12, seed=0), 1: smooth_episode(12, offset=1, seed=1)})
    path = base / "data" / "chunk-000" / "file-000.parquet"
    table = pq.read_table(path)
    idx = table.schema.get_field_index("episode_index")
    str_col = pa.array([str(v) for v in table.column("episode_index").to_pylist()], type=pa.string())
    table = table.set_column(idx, pa.field("episode_index", pa.string()), str_col)
    pq.write_table(table, path)
    return base


def test_streaming_string_episode_index_with_max_episodes(tmp_path):
    """字符串 episode_index 列 + --max_episodes：白名单过滤前必须做类型归一，
    否则整数 keep 永不命中，流式质检产出 0 个 episode 的空结论。"""
    root = _string_episode_index_dataset(tmp_path / "ds")
    config = load_config(robot_type="test_robot")
    memory = validate_dataset(load_dataset(root, max_episodes=1), config)
    streamed = validate_dataset(load_dataset(root, load_frames=False, max_episodes=1), config,
                                streaming=True)

    assert streamed.total_episodes == memory.total_episodes == 1
    assert streamed.results[0].frame_count == 12
