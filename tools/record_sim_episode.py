#!/usr/bin/env python3
"""仿真采集：用二阶跟踪控制器生成机器人轨迹，并写成 LeRobot v3 数据集。

定位是「采集端最小闭环」，不依赖 MuJoCo / Isaac：

- 目标轨迹为正弦关节指令，被控对象是带阻尼的二阶积分器（q'' = kp·e - kd·q'）；
- ``observation.state`` 记录实际关节位置（有跟踪滞后），``action`` 记录下发指令；
- 额外写一列 ``observation.body_pos_w [5,3]``（连杆端点位置），用于覆盖嵌套特征路径。

生成后可直接接主链：

    python tools/record_sim_episode.py --episodes 3 --frames 120 --run
    python main.py --dataset_path outputs/sim_dataset --output_dir outputs/sim_qa --no-cli
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
JOINTS = 8          # 7 关节 + 1 夹爪
LINKS = 5           # 参与 forward kinematics 的连杆端点
STATS_KEYS = ("mean", "std", "min", "max", "q01", "q10", "q50", "q90", "q99")
TASKS = ["reach the target", "follow the trajectory"]


def _simulate_episode(frames: int, fps: int, rng: np.random.Generator,
                      inject_anomaly: bool) -> dict[str, np.ndarray]:
    """二阶跟踪：action 是目标，state 是带滞后的实际位置。"""
    t = np.arange(frames) / fps
    amp = rng.uniform(0.3, 0.9, size=JOINTS)
    freq = rng.uniform(0.2, 0.6, size=JOINTS)
    phase = rng.uniform(0, np.pi, size=JOINTS)

    target = amp * np.sin(2 * np.pi * freq * t[:, None] + phase)
    q = np.zeros(JOINTS)
    qd = np.zeros(JOINTS)
    kp, kd, dt = 60.0, 12.0, 1.0 / (fps * 4)  # 子步积分，避免显式欧拉发散
    state = np.zeros((frames, JOINTS), dtype=np.float32)
    for i in range(frames):
        for _ in range(4):
            qd = qd + (kp * (target[i] - q) - kd * qd) * dt
            q = q + qd * dt
        state[i] = q

    action = target.astype(np.float32)
    if inject_anomaly and frames > 20:
        state[frames // 2:frames // 2 + 3, 2] = np.nan
        action[frames // 2 + 5, 0] += 15.0

    # 平面臂 forward kinematics：每个关节角度累加，得到 LINKS 个端点
    angles = np.cumsum(state[:, :7], axis=1)
    lengths = np.full(7, 0.12)
    body = np.zeros((frames, LINKS, 3), dtype=np.float32)
    x = np.zeros(frames)
    y = np.zeros(frames)
    for k in range(LINKS):
        seg = min(k, 6)
        x = x + lengths[seg] * np.cos(angles[:, seg])
        y = y + lengths[seg] * np.sin(angles[:, seg])
        body[:, k, 0] = x
        body[:, k, 1] = y
        body[:, k, 2] = 0.1 * (k + 1)

    return {"state": state, "action": action, "body_pos_w": body}


def _column(values: np.ndarray) -> pa.Array:
    if values.ndim == 2:
        return pa.array([row.astype(float).tolist() for row in values],
                        type=pa.list_(pa.float32(), values.shape[1]))
    return pa.array([[v.astype(float).tolist() for v in row] for row in values],
                    type=pa.list_(pa.list_(pa.float32(), values.shape[2])))


def _stats_block(arr: np.ndarray) -> dict:
    flat = arr.reshape(arr.shape[0], -1)
    finite = arr[np.isfinite(flat).all(axis=1)]
    out: dict = {"count": int(finite.shape[0])}
    for name, value in (
        ("mean", finite.mean(axis=0)), ("std", finite.std(axis=0)),
        ("min", finite.min(axis=0)), ("max", finite.max(axis=0)),
        ("q01", np.quantile(finite, 0.01, axis=0)),
        ("q10", np.quantile(finite, 0.10, axis=0)),
        ("q50", np.quantile(finite, 0.50, axis=0)),
        ("q90", np.quantile(finite, 0.90, axis=0)),
        ("q99", np.quantile(finite, 0.99, axis=0)),
    ):
        out[name] = np.round(value.reshape(arr.shape[1:]), 6).tolist()
    return out


def record(out: Path, episodes: int = 3, frames: int = 120, fps: int = 30,
           seed: int = 0, inject_anomaly: bool = False) -> Path:
    out = Path(out).resolve()
    (out / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (out / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    records = []
    for ep in range(episodes):
        data = _simulate_episode(frames, fps, rng, inject_anomaly and ep == 0)
        records.append({"episode_index": ep, "task_index": ep % len(TASKS), **data})

    columns: dict[str, list] = {k: [] for k in
                                ("state", "action", "body_pos_w", "timestamp",
                                 "frame_index", "episode_index", "index", "task_index")}
    ep_rows = []
    cursor = 0
    for rec in records:
        n = frames
        columns["state"].append(rec["state"])
        columns["action"].append(rec["action"])
        columns["body_pos_w"].append(rec["body_pos_w"])
        columns["timestamp"].append(np.arange(n, dtype=np.float32) / fps)
        columns["frame_index"].append(np.arange(n, dtype=np.int64))
        columns["episode_index"].append(np.full(n, rec["episode_index"], dtype=np.int64))
        columns["index"].append(np.arange(cursor, cursor + n, dtype=np.int64))
        columns["task_index"].append(np.full(n, rec["task_index"], dtype=np.int64))
        ep_rows.append({
            "episode_index": rec["episode_index"],
            "length": n,
            "tasks": [TASKS[rec["task_index"]]],
            "dataset_from_index": cursor,
            "dataset_to_index": cursor + n,
            "data/chunk_index": 0,
            "data/file_index": 0,
            "meta/collector": "sim",
        })
        cursor += n

    table = pa.table({
        "observation.state": _column(np.concatenate(columns["state"])),
        "action": _column(np.concatenate(columns["action"])),
        "observation.body_pos_w": _column(np.concatenate(columns["body_pos_w"])),
        **{key: pa.array(np.concatenate(columns[key]))
           for key in ("timestamp", "frame_index", "episode_index", "index", "task_index")},
    })
    pq.write_table(table, out / "data" / "chunk-000" / "file-000.parquet")
    pq.write_table(pa.Table.from_pylist(ep_rows), out / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    pq.write_table(
        pa.table({"task_index": pa.array(list(range(len(TASKS))), type=pa.int64()),
                  "task": pa.array(TASKS, type=pa.string())}),
        out / "meta" / "tasks.parquet",
    )

    info = {
        "codebase_version": "v3.0",
        "robot_type": "sim_arm",
        "total_episodes": episodes,
        "total_frames": cursor,
        "total_tasks": len(TASKS),
        "chunks_size": 1000,
        "fps": fps,
        "splits": {"train": f"0:{episodes}"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {
            "observation.state": {"dtype": "float32", "shape": [JOINTS],
                                  "names": [f"joint_{i}" for i in range(JOINTS)]},
            "observation.body_pos_w": {"dtype": "float32", "shape": [LINKS, 3]},
            "action": {"dtype": "float32", "shape": [JOINTS],
                       "names": [f"joint_{i}" for i in range(JOINTS)]},
            **{key: {"dtype": dtype, "shape": [1]} for key, dtype in (
                ("timestamp", "float32"), ("frame_index", "int64"),
                ("episode_index", "int64"), ("index", "int64"), ("task_index", "int64"))},
        },
    }
    (out / "meta" / "info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")

    stats = {
        "observation.state": _stats_block(np.concatenate(columns["state"])),
        "observation.body_pos_w": _stats_block(np.concatenate(columns["body_pos_w"])),
        "action": _stats_block(np.concatenate(columns["action"])),
    }
    (out / "meta" / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="仿真采集一个 LeRobot v3 数据集")
    ap.add_argument("--output", default=str(ROOT / "outputs" / "sim_dataset"))
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--inject-anomaly", action="store_true", help="在 ep0 注入 NaN 与 action 跳变")
    ap.add_argument("--run", action="store_true", help="生成后立刻跑一遍质检主链")
    args = ap.parse_args(argv)

    out = record(Path(args.output), args.episodes, args.frames, args.fps,
                 args.seed, args.inject_anomaly)
    print(f"仿真数据集已生成：{out}")
    print(f"  episodes: {args.episodes}  frames: {args.episodes * args.frames}  fps: {args.fps}")
    print(f"  features: state[{JOINTS}] / body_pos_w[{LINKS},3] / action[{JOINTS}]")

    if args.run:
        qa_dir = out.parent / f"{out.name}_qa"
        cmd = [sys.executable, str(ROOT / "main.py"), "--dataset_path", str(out),
               "--output_dir", str(qa_dir), "--no-cli"]
        print("$ " + " ".join(cmd))
        return subprocess.run(cmd, cwd=ROOT, check=False).returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
