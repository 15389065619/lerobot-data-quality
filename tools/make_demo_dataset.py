#!/usr/bin/env python3
"""生成一份「故意带脏数据」的 LeRobot v3 演示数据集。

用途：
1. 端到端验证 pipeline（含视频重建、meta 重写等分支）；
2. 面试演示时有可复现的输入，不必依赖 HuggingFace 下载。

生成的目录结构与 LeRobot v3.0 一致：
    demo_dataset/
    ├── data/chunk-000/file-000.parquet, file-001.parquet
    ├── meta/info.json, stats.json, tasks.parquet
    ├── meta/episodes/chunk-000/file-000.parquet
    └── videos/<camera>/chunk-000/file-000.mp4, file-001.mp4

注入的异常（可通过 --seed 复现）见 ANOMALY_PLAN。
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

FPS = 30
JOINTS = 7
IMG_W, IMG_H = 128, 96
CAMERAS = ("observation.images.top", "observation.images.wrist")
TASKS = ["pick up the red cube", "open the drawer", "stack the blocks"]
STATS_KEYS = ("mean", "std", "min", "max", "q01", "q10", "q50", "q90", "q99")

#: episode_index -> (异常类型, 说明)。刻意覆盖真实采集里最常见的几类问题。
ANOMALY_PLAN = {
    3: ("nan_state", "state 中段写入 NaN（不可修复，只能丢）"),
    5: ("boundary_nan", "state 前 3 帧 NaN（边界坏帧 → 可裁剪修复）"),
    7: ("action_jump", "action 出现单帧 ±18 的跳变（需人审）"),
    9: ("static", "整条几乎无动作，机器人空转"),
    11: ("short_episode", "只有 5 帧，无法构成时序窗口"),
    13: ("duplicate", "与 ep 2 动作轨迹完全重复（合并时重复入库）"),
    15: ("timestamp", "时间戳回退（非单调，可重建）"),
    17: ("blank_video", "画面全黑，镜头被遮挡/相机未推流"),
    19: ("video_truncated", "视频只录了 60 帧，parquet 有 90 帧"),
    21: ("timestamp_step", "中段跳 0.5s，标定时间戳抖动（仅记录）"),
    23: ("missing_task", "task_index 为 -1，任务描述为空"),
    27: ("inf_action", "action 出现 Inf"),
    31: ("multi", "state NaN + action 跳变（多重异常）"),
}

#: 采集员归属：让归因分析有东西可看（bob 的批次明显更脏）
COLLECTORS = ("alice", "bob")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="生成带脏数据的 LeRobot v3 演示数据集")
    p.add_argument("--output", default="demo_dataset", help="输出目录")
    p.add_argument("--episodes", type=int, default=36, help="episode 数量")
    p.add_argument("--seed", type=int, default=7, help="随机种子")
    p.add_argument("--fps", type=int, default=FPS, help="帧率")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.output).resolve()
    rng = np.random.default_rng(args.seed)
    fps = args.fps

    lengths = {ep: int(rng.integers(40, 121)) for ep in range(args.episodes)}
    lengths[11] = 5  # 过短样本

    for sub in ("data/chunk-000", "meta/episodes/chunk-000"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    for cam in CAMERAS:
        (root / "videos" / cam / "chunk-000").mkdir(parents=True, exist_ok=True)

    # 两个数据文件，各占一半 episode（用于验证按文件重写与视频分段重建）
    split = max(1, args.episodes // 2)
    groups = [list(range(0, split)), list(range(split, args.episodes))]

    frames_by_ep: dict[int, dict] = {}
    for ep in range(args.episodes):
        frames_by_ep[ep] = _make_episode(ep, lengths[ep], rng, fps)

    # 重复入库：把 ep 2 整条复制到 ep 13（数据集合并时最容易发生）
    if 13 in frames_by_ep and 2 in frames_by_ep:
        frames_by_ep[13] = {
            "state": frames_by_ep[2]["state"].copy(),
            "action": frames_by_ep[2]["action"].copy(),
            "task_index": frames_by_ep[2]["task_index"],
            "task": frames_by_ep[2]["task"],
        }

    # ---------------- data parquet + episode 元数据 ----------------
    ep_meta_rows: list[dict] = []
    global_index = 0
    video_cursor = {cam: {gi: 0 for gi in range(len(groups))} for cam in CAMERAS}
    video_from_to: dict[tuple[str, int, int], tuple[float, float]] = {}

    for gi, eps in enumerate(groups):
        rows: list[dict] = []
        for ep in eps:
            data = frames_by_ep[ep]
            n = len(data["state"])
            ts = np.arange(n, dtype=np.float32) / fps
            kind = ANOMALY_PLAN.get(ep, (None, ""))[0]
            if kind == "timestamp":
                # 制造时间戳回退：把中段两帧交换并后移，破坏单调性
                mid = max(1, n // 2)
                ts[mid], ts[mid + 1] = ts[mid + 1], ts[mid] - 0.5
            elif kind == "timestamp_step":
                # 单调但步长突变（掉帧/时钟抖动），只应触发 info 级告警
                ts[n // 2:] += 0.5
            rows.append(
                {
                    "observation.state": [x.astype(np.float32) for x in data["state"]],
                    "action": [x.astype(np.float32) for x in data["action"]],
                    "timestamp": ts,
                    "frame_index": np.arange(n, dtype=np.int64),
                    "episode_index": np.full(n, ep, dtype=np.int64),
                    "index": np.arange(global_index, global_index + n, dtype=np.int64),
                    "task_index": np.full(n, data["task_index"], dtype=np.int64),
                }
            )
            global_index += n
        table = _frames_table(rows)
        pq.write_table(table, root / f"data/chunk-000/file-{gi:03d}.parquet")

        for ep in eps:
            data = frames_by_ep[ep]
            n = len(data["state"])
            ep_meta_rows.append(
                {"episode_index": ep, "length": n, "tasks": [data["task"]],
                 "data/chunk_index": 0, "data/file_index": gi}
            )

    # ---------------- videos ----------------
    for cam_i, cam in enumerate(CAMERAS):
        for gi, eps in enumerate(groups):
            all_frames = []
            per_ep_counts: list[tuple[int, int]] = []
            for ep in eps:
                n_data = len(frames_by_ep[ep]["state"])
                n_video = n_data
                if ep == 19:
                    n_video = max(10, n_data - 30)
                frames = _make_video_frames(cam_i, ep, n_video, rng)
                all_frames.append(frames)
                per_ep_counts.append((ep, n_video))
            stack = np.concatenate(all_frames, axis=0)
            _write_mp4(stack, root / "videos" / cam / f"chunk-000/file-{gi:03d}.mp4", fps)

            cursor = 0.0
            for ep, n_video in per_ep_counts:
                video_from_to[(cam, gi, ep)] = (cursor / fps, (cursor + n_video) / fps)
                cursor += n_video

    # ---------------- meta/episodes ----------------
    ep_rows = []
    for row in ep_meta_rows:
        ep, gi = row["episode_index"], row["data/file_index"]
        out = {
            "episode_index": ep,
            "length": row["length"],
            "tasks": row["tasks"],
            "dataset_from_index": int(sum(len(frames_by_ep[e]["state"]) for e in range(ep))),
            "dataset_to_index": int(sum(len(frames_by_ep[e]["state"]) for e in range(ep + 1))),
            "data/chunk_index": 0,
            "data/file_index": gi,
            # 采集元数据：真实数据集常带，用于按批次/操作员归因
            "meta/collector": COLLECTORS[0] if ep % 4 < 2 else COLLECTORS[1],
        }
        for cam in CAMERAS:
            f, t = video_from_to[(cam, gi, ep)]
            out[f"videos/{cam}/chunk_index"] = 0
            out[f"videos/{cam}/file_index"] = gi
            out[f"videos/{cam}/from_timestamp"] = float(f)
            out[f"videos/{cam}/to_timestamp"] = float(t)
        ep_rows.append(out)
    pq.write_table(pa.Table.from_pylist(ep_rows), root / "meta/episodes/chunk-000/file-000.parquet")

    # ---------------- meta/tasks.parquet ----------------
    pq.write_table(
        pa.table({"task_index": pa.array(list(range(len(TASKS))), type=pa.int64()),
                  "task": pa.array(TASKS, type=pa.string())}),
        root / "meta/tasks.parquet",
    )

    # ---------------- meta/info.json + stats.json ----------------
    total_frames = sum(len(frames_by_ep[e]["state"]) for e in range(args.episodes))
    info = {
        "codebase_version": "v3.0",
        "robot_type": "so100_follower",
        "total_episodes": args.episodes,
        "total_frames": total_frames,
        "total_tasks": len(TASKS),
        "chunks_size": 1000,
        "fps": fps,
        "splits": {"train": f"0:{args.episodes}"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {
            "observation.state": {"dtype": "float32", "shape": [JOINTS],
                                  "names": [f"joint_{i}" for i in range(JOINTS)]},
            "action": {"dtype": "float32", "shape": [JOINTS],
                       "names": [f"joint_{i}" for i in range(JOINTS)]},
            **{
                cam: {"dtype": "video", "shape": [3, IMG_H, IMG_W],
                      "info": {"video.codec": "h264", "video.fps": fps, "video.height": IMG_H,
                               "video.width": IMG_W, "video.channels": 3, "video.pix_fmt": "yuv420p"}}
                for cam in CAMERAS
            },
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
    }
    (root / "meta/info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    (root / "meta/stats.json").write_text(
        json.dumps(_compute_stats(frames_by_ep, args.episodes), indent=2), encoding="utf-8"
    )

    _print_summary(root, args.episodes, total_frames, lengths)
    return 0


# ------------------------------------------------------------------ 数据构造
def _make_episode(ep: int, n: int, rng: np.random.Generator, fps: int) -> dict:
    t = np.arange(n) / fps
    amp = rng.uniform(0.4, 1.2, size=JOINTS)
    freq = rng.uniform(0.2, 0.8, size=JOINTS)
    phase = rng.uniform(0, np.pi, size=JOINTS)
    base = rng.uniform(-0.3, 0.3, size=JOINTS)

    state = base + amp * np.sin(2 * np.pi * freq * t[:, None] + phase)
    state += rng.normal(0, 0.01, size=state.shape)

    # action 约等于下一帧的 state（带一点滞后噪声），保证正常 episode 的跳变很小
    action = np.vstack([state[1:], state[-1:]]) + rng.normal(0, 0.005, size=state.shape)

    task_index = ep % len(TASKS)
    task = TASKS[task_index]

    kind = ANOMALY_PLAN.get(ep, (None, ""))[0]

    if kind == "static":
        # 机器人没动：整条 state/action 几乎恒定，属无效采集
        state = np.tile(base, (n, 1)) + rng.normal(0, 1e-7, size=(n, JOINTS))
        action = state + rng.normal(0, 1e-7, size=(n, JOINTS))

    if kind in {"nan_state", "multi"}:
        lo = n // 3
        state[lo:lo + max(3, n // 10), 2] = np.nan
    if kind == "boundary_nan":
        # 只污染开头 3 帧 → 质量校验应判为「可裁剪」而非整条丢弃
        state[:3, 1] = np.nan
    if kind in {"action_jump", "multi"}:
        jump_at = max(1, n // 2)
        action[jump_at, 0] += 18.0
        action[jump_at + 1, 0] -= 16.0
    if kind == "inf_action":
        action[n // 2, 4] = np.inf
    if kind == "missing_task":
        task_index = -1
        task = ""

    return {"state": state, "action": action, "task_index": task_index, "task": task}


def _frames_table(rows: list[dict]) -> pa.Table:
    """把多个 episode 的列拼成一张 arrow 表，向量列用 fixed_size_list<float32>。"""
    vec_type = pa.list_(pa.float32(), JOINTS)
    cols: dict[str, pa.Array] = {}
    cols["observation.state"] = pa.array(
        [v for r in rows for v in r["observation.state"]], type=vec_type
    )
    cols["action"] = pa.array([v for r in rows for v in r["action"]], type=vec_type)
    for key in ("timestamp", "frame_index", "episode_index", "index", "task_index"):
        cols[key] = pa.array(np.concatenate([r[key] for r in rows]))
    return pa.table(cols)


def _make_video_frames(cam_idx: int, ep: int, n: int, rng: np.random.Generator) -> np.ndarray:
    """合成一帧帧「移动的方块」，保证肉眼可区分不同 episode。"""
    if ANOMALY_PLAN.get(ep, (None, ""))[0] == "blank_video":
        # 镜头遮挡/未推流：整段近乎纯黑，内容方差为 0
        return np.full((n, IMG_H, IMG_W, 3), 8, dtype=np.uint8)

    yy, xx = np.mgrid[0:IMG_H, 0:IMG_W]
    frames = np.zeros((n, IMG_H, IMG_W, 3), dtype=np.uint8)
    base_tint = (40 + cam_idx * 30, 60, 90)
    for i in range(n):
        cx = IMG_W / 2 + (IMG_W / 3) * np.sin(2 * np.pi * (i / max(n, 1)) + ep)
        cy = IMG_H / 2 + (IMG_H / 3) * np.cos(2 * np.pi * (i / max(n, 1)) * 1.5 + ep)
        mask = ((xx - cx) ** 2 + (yy - cy) ** 2) < 220
        frame = np.full((IMG_H, IMG_W, 3), base_tint, dtype=np.uint8)
        frame[mask] = (230, 200 - cam_idx * 40, 60)
        frames[i] = frame
    return frames


def _write_mp4(frames: np.ndarray, path: Path, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{IMG_W}x{IMG_H}", "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(path),
    ]
    proc = subprocess.run(cmd, input=frames.tobytes(), capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg 写入失败 {path}: {proc.stderr.decode('utf-8', 'ignore')[-400:]}")


def _compute_stats(frames_by_ep: dict[int, dict], episodes: int) -> dict:
    state = np.concatenate([frames_by_ep[e]["state"] for e in range(episodes)], axis=0)
    action = np.concatenate([frames_by_ep[e]["action"] for e in range(episodes)], axis=0)
    out = {}
    for key, arr in (("observation.state", state), ("action", action)):
        finite = arr[np.isfinite(arr).all(axis=1)]
        out[key] = {
            "mean": np.round(finite.mean(axis=0), 6).tolist(),
            "std": np.round(finite.std(axis=0), 6).tolist(),
            "min": np.round(finite.min(axis=0), 6).tolist(),
            "max": np.round(finite.max(axis=0), 6).tolist(),
            "q01": np.round(np.quantile(finite, 0.01, axis=0), 6).tolist(),
            "q10": np.round(np.quantile(finite, 0.10, axis=0), 6).tolist(),
            "q50": np.round(np.quantile(finite, 0.50, axis=0), 6).tolist(),
            "q90": np.round(np.quantile(finite, 0.90, axis=0), 6).tolist(),
            "q99": np.round(np.quantile(finite, 0.99, axis=0), 6).tolist(),
        }
        # 模拟「stats.json 没跟着数据更新」：这是真实数据集里最坑的问题，
        # 训练时归一化整体偏移，loss 看着正常但策略学不动。
        out[key]["mean"] = [round(v * 1.15 + 0.02, 6) for v in out[key]["mean"]]
        out[key]["std"] = [round(v * 1.3, 6) for v in out[key]["std"]]
    return out


def _print_summary(root: Path, episodes: int, total_frames: int, lengths: dict[int, int]) -> None:
    print(f"演示数据集已生成: {root}")
    print(f"  episode: {episodes}   帧数: {total_frames}   相机: {len(CAMERAS)}   fps: {FPS}")
    print(f"  平均长度: {np.mean(list(lengths.values())):.1f} 帧")
    print("  已注入的异常:")
    for ep, (kind, desc) in sorted(ANOMALY_PLAN.items()):
        print(f"    ep {ep:>3}  {kind:<16} {desc}")


if __name__ == "__main__":
    sys.exit(main())
