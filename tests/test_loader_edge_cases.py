"""loader / validator 的边界条件回归：空数据、脏元数据、坏视频/图像、乱序时间戳。

这些用例全部走真实的 load_dataset + validate_dataset（内存与 streaming 各一次），
断言「不抛未捕获异常」，而不是断言某个具体结论——脏数据的正确产出是报告，
不是 traceback。
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import pipeline.validator.rules as rules
from conftest import JOINTS, build_dataset, smooth_episode
from pipeline.loader import DatasetFormatError, load_dataset
from pipeline.validator import validate_dataset
from pipeline.validator.types import EpisodeResult


DATA_REL = Path("data/chunk-000/file-000.parquet")
META_REL = Path("meta/episodes/chunk-000/file-000.parquet")


def _rewrite_data(root: Path, **columns) -> None:
    path = root / DATA_REL
    df = pq.read_table(path).to_pandas()
    for name, values in columns.items():
        df[name] = values
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path)


def _rewrite_meta(root: Path, **columns) -> None:
    path = root / META_REL
    df = pq.read_table(path).to_pandas()
    for name, values in columns.items():
        df[name] = values
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path)


def _patch_info(root: Path, **overrides) -> None:
    path = root / "meta/info.json"
    info = json.loads(path.read_text(encoding="utf-8"))
    info.update(overrides)
    path.write_text(json.dumps(info), encoding="utf-8")


def _add_feature(root: Path, name: str, spec: dict) -> None:
    path = root / "meta/info.json"
    info = json.loads(path.read_text(encoding="utf-8"))
    info["features"][name] = spec
    path.write_text(json.dumps(info), encoding="utf-8")


def _make_empty_dataset(root: Path) -> Path:
    (root / "data/chunk-000").mkdir(parents=True)
    (root / "meta/episodes/chunk-000").mkdir(parents=True)
    vec = pa.list_(pa.float32(), JOINTS)
    pq.write_table(
        pa.table({
            "observation.state": pa.array([], type=vec),
            "action": pa.array([], type=vec),
            "timestamp": pa.array([], type=pa.float32()),
            "frame_index": pa.array([], type=pa.int64()),
            "episode_index": pa.array([], type=pa.int64()),
            "index": pa.array([], type=pa.int64()),
            "task_index": pa.array([], type=pa.int64()),
        }),
        root / DATA_REL,
    )
    pq.write_table(
        pa.table({
            "episode_index": pa.array([], type=pa.int64()),
            "length": pa.array([], type=pa.int64()),
        }),
        root / META_REL,
    )
    info = {
        "codebase_version": "v3.0",
        "robot_type": "test_robot",
        "total_episodes": 0,
        "total_frames": 0,
        "total_tasks": 0,
        "chunks_size": 1000,
        "fps": 30,
        "splits": {"train": "0:0"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {
            "observation.state": {"dtype": "float32", "shape": [JOINTS]},
            "action": {"dtype": "float32", "shape": [JOINTS]},
            "timestamp": {"dtype": "float32", "shape": [1]},
            "frame_index": {"dtype": "int64", "shape": [1]},
            "episode_index": {"dtype": "int64", "shape": [1]},
            "index": {"dtype": "int64", "shape": [1]},
            "task_index": {"dtype": "int64", "shape": [1]},
        },
    }
    (root / "meta/info.json").write_text(json.dumps(info), encoding="utf-8")
    return root


def _load_and_validate(root: Path):
    dataset = load_dataset(root)
    in_memory = validate_dataset(dataset)
    streamed = validate_dataset(load_dataset(root, load_frames=False), streaming=True)
    return dataset, in_memory, streamed


# --------------------------------------------------------------- 1. 空数据集
def test_empty_data_parquet_does_not_crash(tmp_path):
    root = _make_empty_dataset(tmp_path / "ds")
    dataset, in_memory, streamed = _load_and_validate(root)

    assert dataset.num_episodes == 0
    assert dataset.num_frames == 0
    assert in_memory.total_episodes == 0
    assert streamed.total_episodes == 0


def test_info_json_must_be_object(tmp_path):
    root = build_dataset(tmp_path / "ds", {0: smooth_episode(10)})
    (root / "meta/info.json").write_text("[1, 2]", encoding="utf-8")

    with pytest.raises(DatasetFormatError, match="顶层必须是 JSON 对象"):
        load_dataset(root)


def test_non_dict_features_and_bad_fps_fall_back(tmp_path):
    root = build_dataset(tmp_path / "ds", {0: smooth_episode(10)})
    _patch_info(root, features=1, fps="abc")
    # 混入一个非对象特征描述，也不应影响其余特征
    info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
    info["features"] = {"observation.state": {"dtype": "float32", "shape": [JOINTS]},
                        "broken": "not-a-dict"}
    (root / "meta/info.json").write_text(json.dumps(info), encoding="utf-8")

    dataset = load_dataset(root)
    assert dataset.features == {"observation.state": {"dtype": "float32", "shape": [JOINTS]}}
    assert dataset.fps == 30.0
    validate_dataset(dataset)


# ------------------------------------------------- 2. 视频缺失 / 损坏图像字节
def test_dirty_video_meta_values_are_tolerated(tmp_path):
    root = build_dataset(tmp_path / "ds", {0: smooth_episode(10)})
    _add_feature(root, "observation.images.top", {"dtype": "video", "shape": [8, 8, 3]})
    _rewrite_meta(
        root,
        **{
            "videos/observation.images.top/chunk_index": 0,
            "videos/observation.images.top/file_index": 0,
            "videos/observation.images.top/from_timestamp": "oops",
            "videos/observation.images.top/to_timestamp": np.inf,
        },
    )

    dataset, in_memory, _ = _load_and_validate(root)
    loc = dataset.video_location("observation.images.top", 0, require_exists=False)
    assert loc is not None
    _, from_ts, to_ts = loc
    assert from_ts == -1.0 and to_ts == -1.0
    assert math.isfinite(from_ts) and math.isfinite(to_ts)


def test_non_numeric_video_chunk_is_unlocatable_not_crash(tmp_path):
    root = build_dataset(tmp_path / "ds", {0: smooth_episode(10)})
    _add_feature(root, "observation.images.top", {"dtype": "video", "shape": [8, 8, 3]})
    _rewrite_meta(
        root,
        **{
            "videos/observation.images.top/chunk_index": "a",
            "videos/observation.images.top/file_index": None,
            "videos/observation.images.top/from_timestamp": 0.0,
            "videos/observation.images.top/to_timestamp": 10 / 30,
        },
    )

    dataset, _, _ = _load_and_validate(root)
    assert dataset.video_location("observation.images.top", 0, require_exists=False) is None


def test_check_frame_mismatch_handles_infinite_window(tmp_path, monkeypatch):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"")

    class _StubDataset:
        fps = 30.0
        locator_available = True

        def video_location(self, camera, episode_index, *, require_exists=True):
            return video, 0.0, float("inf")

    monkeypatch.setattr(rules, "FFPROBE", True)
    monkeypatch.setattr(rules, "probe_video_frames", lambda path, precise=False: 10)
    res = EpisodeResult(episode_index=0, frame_count=10)

    rules.check_frame_mismatch(res, 0, _StubDataset(), ["cam"], False, {"defaults": {}})
    assert "frame_mismatch" not in res.reasons


def test_broken_image_bytes_do_not_crash(tmp_path):
    root = build_dataset(tmp_path / "ds", {0: smooth_episode(10)})
    _add_feature(root, "observation.images.wrist", {"dtype": "image", "shape": [8, 8, 3]})
    _rewrite_data(root, **{
        "observation.images.wrist": [b"\xff\xd8garbage-bytes"] * 10,
    })

    _, in_memory, streamed = _load_and_validate(root)
    assert in_memory.total_episodes == 1
    assert streamed.total_episodes == 1


def test_multi_camera_with_missing_files_does_not_crash(tmp_path):
    root = build_dataset(tmp_path / "ds", {0: smooth_episode(10)})
    _add_feature(root, "observation.images.top", {"dtype": "video", "shape": [8, 8, 3]})
    _add_feature(root, "observation.images.left", {"dtype": "video", "shape": [8, 8, 3]})
    _rewrite_meta(
        root,
        **{
            "videos/observation.images.top/chunk_index": 0,
            "videos/observation.images.top/file_index": 0,
            "videos/observation.images.top/from_timestamp": 0.0,
            "videos/observation.images.top/to_timestamp": 10 / 30,
            "videos/observation.images.left/chunk_index": None,
            "videos/observation.images.left/file_index": None,
            "videos/observation.images.left/from_timestamp": None,
            "videos/observation.images.left/to_timestamp": None,
        },
    )

    dataset, in_memory, _ = _load_and_validate(root)
    assert dataset.video_location("observation.images.left", 0, require_exists=False) is None
    assert in_memory.total_episodes == 1


# ------------------------------------------------------- 3. 时间戳乱序 / 巨跳
def test_out_of_order_and_huge_timestamp_jump_does_not_crash(tmp_path):
    ep = smooth_episode(10)
    ep["timestamp"] = np.array([0, 0.1, 0.05, 0.2, 5000.0, 0.3, 0.4, 0.5, 0.6, 0.7])
    root = build_dataset(tmp_path / "ds", {0: ep})

    _, in_memory, streamed = _load_and_validate(root)
    assert "timestamp_not_monotonic" in in_memory.results[0].reasons
    assert "timestamp_not_monotonic" in streamed.results[0].reasons


# ------------------------------------------------------- 4. 脏编号 / 脏 index
def test_string_episode_index_does_not_crash(tmp_path):
    root = build_dataset(tmp_path / "ds", {0: smooth_episode(10), 1: smooth_episode(10)})
    _rewrite_data(root, episode_index=["ep0"] * 10 + ["ep1"] * 10)

    dataset, in_memory, streamed = _load_and_validate(root)
    # 数据侧编号全是脏值，退回 meta 编号，保证「meta 有、数据读不到」被如实报告
    assert dataset.episode_indices == [0, 1]
    # meta 里仍有 episode，数据侧读不到 → 标记读取错误而不是 traceback
    assert in_memory.total_episodes == 2
    assert all("episode_read_error" in r.reasons for r in in_memory.results.values())


def test_nan_meta_episode_index_is_dropped(tmp_path):
    root = build_dataset(tmp_path / "ds", {0: smooth_episode(10), 1: smooth_episode(10)})
    _rewrite_meta(root, episode_index=[0, None])

    dataset, in_memory, streamed = _load_and_validate(root)
    assert dataset.num_episodes == 1
    # 数据侧仍有 2 个 episode，以 data 为准；meta 少一行只影响定位，不减少逐帧质检
    assert in_memory.total_episodes == 2
    assert streamed.total_episodes == 2


def test_nan_index_column_does_not_crash(tmp_path):
    root = build_dataset(tmp_path / "ds", {0: smooth_episode(10)})
    _rewrite_data(root, index=[0.0, 1.0, np.nan, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])

    dataset, in_memory, streamed = _load_and_validate(root)
    assert any(issue.code == "meta_index_inconsistent" for issue in in_memory.dataset_issues)
    assert streamed.total_episodes == 1


def test_dirty_tasks_index_is_ignored(tmp_path):
    root = build_dataset(tmp_path / "ds", {0: smooth_episode(10)})
    (root / "meta/tasks.parquet").unlink()
    (root / "meta/tasks.jsonl").write_text(
        "\n".join(json.dumps(x) for x in [
            {"task_index": "bad", "task": "drop me"},
            {"task_index": 0, "task": "pick cube"},
            ["not", "an", "object"],
        ]),
        encoding="utf-8",
    )

    dataset = load_dataset(root)
    assert dataset.tasks == {0: "pick cube"}
