"""测试夹具：在临时目录里造一个最小可用的 LeRobot v3 数据集（不含视频）。

不依赖 ffmpeg/ffprobe：视频相关的 frame_mismatch / blank_video 在集成测试（demo_dataset）里覆盖。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

FPS = 30
JOINTS = 3
TASKS = ["pick cube", "open drawer"]


def build_dataset(
    root: Path,
    episodes: dict[int, dict[str, Any]],
    *,
    fps: int = FPS,
    stale_stats: bool = False,
    collector: dict[int, str] | None = None,
) -> Path:
    """按 {episode_index: {state, action, timestamp?, task_index?}} 写出数据集。"""
    (root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)

    vec = pa.list_(pa.float32(), JOINTS)
    cols: dict[str, list] = {k: [] for k in
                             ("observation.state", "action", "timestamp", "frame_index",
                              "episode_index", "index", "task_index")}
    ep_rows = []
    global_index = 0
    all_state: list[np.ndarray] = []

    for ep in sorted(episodes):
        spec = episodes[ep]
        state = np.asarray(spec["state"], dtype=np.float32)
        action = np.asarray(spec["action"], dtype=np.float32)
        n = len(state)
        ts = spec.get("timestamp")
        if ts is None:
            ts = np.arange(n, dtype=np.float32) / fps
        ep_rows.append({
            "episode_index": ep,
            "length": n,
            "tasks": [TASKS[spec.get("task_index", 0)] if spec.get("task_index", 0) >= 0 else ""],
            "dataset_from_index": global_index,
            "dataset_to_index": global_index + n,
            "data/chunk_index": 0,
            "data/file_index": 0,
            "meta/collector": (collector or {}).get(ep, "alice"),
        })
        cols["observation.state"].extend([x for x in state])
        cols["action"].extend([x for x in action])
        cols["timestamp"].extend(np.asarray(ts, dtype=np.float32))
        cols["frame_index"].extend(np.arange(n, dtype=np.int64))
        cols["episode_index"].extend(np.full(n, ep, dtype=np.int64))
        cols["index"].extend(np.arange(global_index, global_index + n, dtype=np.int64))
        cols["task_index"].extend(np.full(n, spec.get("task_index", 0), dtype=np.int64))
        all_state.append(state)
        global_index += n

    table = pa.table({
        "observation.state": pa.array(cols["observation.state"], type=vec),
        "action": pa.array(cols["action"], type=vec),
        "timestamp": pa.array(np.asarray(cols["timestamp"], dtype=np.float32)),
        "frame_index": pa.array(np.asarray(cols["frame_index"], dtype=np.int64)),
        "episode_index": pa.array(np.asarray(cols["episode_index"], dtype=np.int64)),
        "index": pa.array(np.asarray(cols["index"], dtype=np.int64)),
        "task_index": pa.array(np.asarray(cols["task_index"], dtype=np.int64)),
    })
    pq.write_table(table, root / "data" / "chunk-000" / "file-000.parquet")
    pq.write_table(pa.Table.from_pylist(ep_rows), root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    pq.write_table(
        pa.table({"task_index": pa.array(list(range(len(TASKS))), type=pa.int64()),
                  "task": pa.array(TASKS, type=pa.string())}),
        root / "meta" / "tasks.parquet",
    )

    info = {
        "codebase_version": "v3.0",
        "robot_type": "test_robot",
        "total_episodes": len(episodes),
        "total_frames": global_index,
        "total_tasks": len(TASKS),
        "chunks_size": 1000,
        "fps": fps,
        "splits": {"train": f"0:{len(episodes)}"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {
            "observation.state": {"dtype": "float32", "shape": [JOINTS],
                                  "names": [f"j{i}" for i in range(JOINTS)]},
            "action": {"dtype": "float32", "shape": [JOINTS]},
            "timestamp": {"dtype": "float32", "shape": [1]},
            "frame_index": {"dtype": "int64", "shape": [1]},
            "episode_index": {"dtype": "int64", "shape": [1]},
            "index": {"dtype": "int64", "shape": [1]},
            "task_index": {"dtype": "int64", "shape": [1]},
        },
    }
    (root / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")

    stacked = np.concatenate(all_state, axis=0)
    finite = stacked[np.isfinite(stacked).all(axis=1)]
    mean = finite.mean(axis=0)
    if stale_stats:
        mean = mean * 1.5 + 0.5  # 模拟 stats 没跟着数据更新
    (root / "meta" / "stats.json").write_text(
        json.dumps({
            "observation.state": {
                "mean": [round(float(x), 6) for x in mean],
                "std": [round(float(x), 6) for x in finite.std(axis=0)],
                "min": [round(float(x), 6) for x in finite.min(axis=0)],
                "max": [round(float(x), 6) for x in finite.max(axis=0)],
            }
        }),
        encoding="utf-8",
    )
    return root


def smooth_episode(n: int = 40, offset: float = 0.0, seed: int = 0) -> dict[str, np.ndarray]:
    """一条正常的平滑轨迹（action = 下一帧 state）。"""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / FPS
    state = np.stack([np.sin(2 * np.pi * 0.4 * t + offset + k) for k in range(JOINTS)], axis=1)
    action = np.vstack([state[1:], state[-1:]]) + rng.normal(0, 1e-3, size=state.shape)
    return {"state": state, "action": action}


@pytest.fixture
def normal_dataset(tmp_path: Path) -> Path:
    episodes = {ep: smooth_episode(40, offset=ep * 0.7, seed=ep) for ep in range(6)}
    return build_dataset(tmp_path / "ds", episodes)
