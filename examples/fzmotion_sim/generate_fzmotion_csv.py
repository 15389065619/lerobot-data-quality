"""生成 FZMotion 风格的模拟动捕 CSV（含典型缺陷注入）。

FZMotion 等光学动捕系统的导出是长表格式：每行一个 (timestamp, marker_id, x, y, z)，
一个采样帧包含 N 个 marker 行。本脚本生成两个 episode 文件：
  episode_000.csv / episode_001.csv —— 120 FPS × 10 秒 × 6 个 marker。

并注入真实采集中最常见的三类缺陷（供质检/适配层验证容错）：
  1. 丢帧   —— 连续若干帧的所有 marker 行整体缺失（采集丢包）；
  2. NaN    —— 某个 marker 在一小段时间内 x/y/z 为空（遮挡丢点）；
  3. 跳变   —— 某个 marker 的坐标突然平移一段距离（标记点误关联）。

用法：
    python examples/fzmotion_sim/generate_fzmotion_csv.py [--output DIR] [--seed N]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

LOG = logging.getLogger("fzmotion_sim")

FPS = 120            # 采样率
DURATION_S = 10      # 时长（秒）
N_MARKERS = 6        # marker 数量


def _base_trajectory(n_frames: int, marker_id: int, phase: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """生成单个 marker 的平滑基础轨迹（模拟人体关节点的圆周摆动）。

    不同 marker 分布在一条“骨架”上（x 方向等距、高度不同），
    各自以不同半径/相位做圆周运动，保证轨迹平滑且可区分。
    """
    t = np.arange(n_frames, dtype=np.float64) / FPS
    x = marker_id * 0.12 + 0.05 * np.sin(2 * np.pi * 0.5 * t + phase + marker_id * 0.3)
    y = 0.9 + 0.15 * marker_id + 0.03 * np.cos(2 * np.pi * 0.5 * t + phase)
    z = 0.10 * np.sin(2 * np.pi * 0.25 * t + phase + marker_id * 0.7)
    return x, y, z


def _build_episode(seed: int, *, variant: int) -> pd.DataFrame:
    """构建一个 episode 的长表 DataFrame，并按 variant 注入不同的缺陷组合。"""
    rng = np.random.default_rng(seed)
    n_frames = FPS * DURATION_S
    frames: list[pd.DataFrame] = []
    for m in range(N_MARKERS):
        x, y, z = _base_trajectory(n_frames, m, phase=variant * 1.7)
        # 叠加微小的测量噪声（真实动捕不可能完全平滑）
        x = x + rng.normal(0.0, 0.0005, n_frames)
        y = y + rng.normal(0.0, 0.0005, n_frames)
        z = z + rng.normal(0.0, 0.0005, n_frames)
        frames.append(pd.DataFrame({
            "timestamp": np.round(np.arange(n_frames) / FPS, 6),
            "marker_id": m,
            "x": np.round(x, 4),
            "y": np.round(y, 4),
            "z": np.round(z, 4),
        }))
    df = pd.concat(frames, ignore_index=True)

    # ---- 缺陷注入（不同 episode 注入位置不同，模拟真实的不确定性）----
    if variant == 0:
        drop_frames = range(300, 305)        # 丢帧：第 2.5s 附近连续 5 帧
        nan_marker, nan_span = 2, range(500, 510)     # NaN：marker 2 在 ~4.2s 起 10 帧
        jump_marker, jump_span, jump_vec = 1, range(600, 610), (1.5, 0.0, 0.0)  # 跳变
    else:
        drop_frames = range(100, 101)        # 丢帧：单帧
        nan_marker, nan_span = 5, range(700, 705)
        jump_marker, jump_span, jump_vec = 0, range(800, 808), (0.0, 0.0, 1.0)

    # 1) 丢帧：删除这些帧号对应的所有 marker 行（timestamp 出现缺口）
    drop_ts = {round(f / FPS, 6) for f in drop_frames}
    df = df[~df["timestamp"].isin(drop_ts)].reset_index(drop=True)

    # 2) NaN：把指定 marker 在指定帧范围内的坐标置为缺失（写出为空单元格）
    nan_ts = {round(f / FPS, 6) for f in nan_span}
    mask = df["timestamp"].isin(nan_ts) & (df["marker_id"] == nan_marker)
    df.loc[mask, ["x", "y", "z"]] = np.nan

    # 3) 跳变：把指定 marker 在指定帧范围内的坐标整体平移（非物理的突变）
    jump_ts = {round(f / FPS, 6) for f in jump_span}
    mask = df["timestamp"].isin(jump_ts) & (df["marker_id"] == jump_marker)
    df.loc[mask, "x"] = df.loc[mask, "x"] + jump_vec[0]
    df.loc[mask, "y"] = df.loc[mask, "y"] + jump_vec[1]
    df.loc[mask, "z"] = df.loc[mask, "z"] + jump_vec[2]

    # 长表规范排序：先按时间、再按 marker
    return df.sort_values(["timestamp", "marker_id"], ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 FZMotion 风格模拟动捕 CSV")
    parser.add_argument("--output", default=str(Path(__file__).resolve().parent),
                        help="输出目录（默认：脚本所在目录）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子（默认 42）")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(2):
        df = _build_episode(args.seed + i, variant=i)
        path = out_dir / f"episode_{i:03d}.csv"
        df.to_csv(path, index=False)  # NaN 默认写成空单元格，读回即 NaN
        n_nan = int(df[["x", "y", "z"]].isna().any(axis=1).sum())
        LOG.info("已生成 %s：%d 行（%d 个采样帧 × %d marker，含 NaN 行 %d）",
                 path.name, len(df), df["timestamp"].nunique(), N_MARKERS, n_nan)


if __name__ == "__main__":
    main()
