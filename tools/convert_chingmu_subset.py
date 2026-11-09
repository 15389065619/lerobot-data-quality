"""把 ChingMu MotionDecode 的骨骼 CSV 子集转换成 LeRobot v3.0 布局（无视频特征）。

只做结构转换，不改数值：36 维（root 位置 3 + root 四元数 4 + 29 个关节角 rad）
作为 observation.state / action，fps=120，一个 CSV 对应一个 episode。

用法：
    python tools/convert_chingmu_subset.py --episodes 20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TASK_TEXT = "mocap skeletal motion (ChingMu MotionDecode)"


def pick_csvs(src: Path, n: int) -> list[Path]:
    """优先取高强度动作（帧数一致便于对照），其余随机补足，保证 episode 长度有差异。"""
    files = sorted(src.rglob("*.csv"))
    jump = [f for f in files if "Standing_High_Jump" in f.name]
    rest = [f for f in files if f not in jump]
    rng = np.random.default_rng(0)
    rng.shuffle(rest)
    return (jump + rest)[:n]


def frame_feature(name: str, shape: list[int], dtype: str) -> dict:
    return {"dtype": dtype, "shape": shape, "names": None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "data" / "chingmu_motiondecode" / "samples"))
    ap.add_argument("--dst", default=str(ROOT / "data" / "chingmu_subset"))
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--fps", type=float, default=120.0)
    args = ap.parse_args()

    csvs = pick_csvs(Path(args.src), args.episodes)
    dst = Path(args.dst)
    (dst / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (dst / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)

    parts, ep_rows, gidx = [], [], 0
    values_all = []
    for ep, path in enumerate(csvs):
        df = pd.read_csv(path)
        arr = df.to_numpy(dtype=np.float32)
        n = len(arr)
        values_all.append(arr)
        block = pd.DataFrame({
            "observation.state": list(arr),
            "action": list(arr),
            "timestamp": (np.arange(n) / args.fps).astype(np.float32),
            "frame_index": np.arange(n, dtype=np.int64),
            "episode_index": np.full(n, ep, dtype=np.int64),
            "index": np.arange(gidx, gidx + n, dtype=np.int64),
            "task_index": np.zeros(n, dtype=np.int64),
        })
        parts.append(block)
        ep_rows.append({
            "episode_index": ep, "length": n, "task_index": 0,
            "tasks": [TASK_TEXT],
            "dataset_from_index": gidx, "dataset_to_index": gidx + n,
            "data/chunk_index": 0, "data/file_index": 0,
        })
        gidx += n
        print(f"ep{ep:03d} {path.name} frames={n}")

    frames = pd.concat(parts, ignore_index=True)
    frames.to_parquet(dst / "data" / "chunk-000" / "file-000.parquet", index=False)
    pd.DataFrame(ep_rows).to_parquet(dst / "meta" / "episodes" / "chunk-000" / "file-000.parquet", index=False)
    pd.DataFrame({"task_index": [0], "task": ["mocap skeletal motion (ChingMu MotionDecode)"]}).to_parquet(
        dst / "meta" / "tasks.parquet", index=False
    )

    dim = values_all[0].shape[1]
    feats = {
        "observation.state": frame_feature("observation.state", [dim], "float32"),
        "action": frame_feature("action", [dim], "float32"),
        "timestamp": frame_feature("timestamp", [1], "float32"),
        "frame_index": frame_feature("frame_index", [1], "int64"),
        "episode_index": frame_feature("episode_index", [1], "int64"),
        "index": frame_feature("index", [1], "int64"),
        "task_index": frame_feature("task_index", [1], "int64"),
    }
    info = {
        "codebase_version": "v3.0",
        "robot_type": "chingmu_mocap_g1",
        "total_episodes": len(csvs),
        "total_frames": int(gidx),
        "total_tasks": 1,
        "total_videos": 0,
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": args.fps,
        "splits": {"train": f"0:{len(csvs)}"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": None,
        "features": feats,
    }
    (dst / "meta" / "info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")

    stacked = np.concatenate(values_all, axis=0)
    stats = {}
    for key in ("observation.state", "action"):
        stats[key] = {
            "min": np.round(stacked.min(0), 6).tolist(),
            "max": np.round(stacked.max(0), 6).tolist(),
            "mean": np.round(stacked.mean(0), 6).tolist(),
            "std": np.round(stacked.std(0), 6).tolist(),
            "count": [int(stacked.shape[0])],
        }
    (dst / "meta" / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n完成: {dst} | {len(csvs)} episodes / {gidx} frames / {dim} dims / fps={args.fps}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
