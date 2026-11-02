"""FZMotion 风格动捕 CSV 加载器（可选能力，独立于 pipeline/ 主线）。

FZMotion 等光学动捕系统导出的 CSV 是长表格式：

    timestamp, marker_id, x, y, z
    0.000000,  0,         0.0012, 0.9011, 0.0003
    0.000000,  1,         ...

本模块把它转换为与 LeRobot 帧表同构的宽表 DataFrame，列如下：

    episode_index      int            episode 编号
    frame_index        int            episode 内从 0 开始的帧号
    timestamp          float          原始时间戳（秒），丢帧处表现为缺口，不做重采样/插值
    observation.state  object(ndarray[float32, 3N])  按 marker_id 升序拼接的坐标向量；
                                      某帧缺某个 marker 时，对应三元组补 NaN
    action             object(ndarray[float32, 3N])  定义为下一帧的 state；
                                      最后一帧保持自身（定长契约：action 与 state 同形）

设计约束：
  - 缺陷不在这里裁决：NaN、丢帧、跳变都原样保留进 state/action，
    交给下游质检规则（nan_state / action_jump / timestamp_* 等）标记与处置；
  - 与本仓库 pipeline/ 主线零依赖、零干扰：本包独立安装/导入，
    输出可直接喂给任何接受「episode_index + observation.state + action」帧表的消费者。
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

LOG = logging.getLogger(__name__)

REQUIRED_COLUMNS = ("timestamp", "marker_id", "x", "y", "z")
COORD_COLUMNS = ("x", "y", "z")


def load_fzmotion_csv(path: str | Path, episode_index: int = 0) -> pd.DataFrame:
    """加载单个 FZMotion 长表 CSV，返回 LeRobot 同构的宽表帧 DataFrame。

    Args:
        path: CSV 文件路径。
        episode_index: 写入输出 episode_index 列的编号。

    Returns:
        每行一个采样帧的 DataFrame；列为
        episode_index / frame_index / timestamp / observation.state / action。

    Raises:
        FileNotFoundError: 文件不存在。
        ValueError: 缺少必需列，或文件内没有任何有效帧。
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"CSV 文件不存在: {path}")

    raw = pd.read_csv(path)  # 空单元格自动解析为 NaN，无需特殊处理
    missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"{path.name} 缺少必需列: {missing}（需要 {list(REQUIRED_COLUMNS)}）")

    # marker 集合决定 state 维度：dim = 3 × n_markers，按 marker_id 升序拼接
    markers = sorted(int(m) for m in raw["marker_id"].dropna().unique())
    if not markers:
        raise ValueError(f"{path.name} 中没有任何 marker 行")

    # 长表转宽表：行=timestamp，列=(坐标, marker_id)；缺失组合自动为 NaN
    # （某帧被遮挡丢掉某个 marker 时，对应三元组自然落成 NaN，不崩溃）
    wide = raw.pivot_table(index="timestamp", columns="marker_id",
                           values=list(COORD_COLUMNS), aggfunc="first")
    wide = wide.sort_index()
    timestamps = wide.index.to_numpy(dtype=np.float64)

    # 按 marker 主序拼 state：[m0_x, m0_y, m0_z, m1_x, ...]，缺测为 NaN
    cols = []
    for m in markers:
        for c in COORD_COLUMNS:
            key = (c, m)
            cols.append(wide[key].to_numpy(dtype=np.float32)
                        if key in wide.columns
                        else np.full(len(wide), np.nan, dtype=np.float32))
    mat = np.stack(cols, axis=1) if cols else np.empty((len(wide), 0), dtype=np.float32)

    states = [row.copy() for row in mat]
    # action := 下一帧 state；最后一帧保持自身（定长契约，下游无需分支）
    actions = [states[i + 1].copy() if i + 1 < len(states) else states[i].copy()
               for i in range(len(states))]

    out = pd.DataFrame({
        "episode_index": int(episode_index),
        "frame_index": np.arange(len(wide), dtype=np.int64),
        "timestamp": timestamps,
        "observation.state": states,
        "action": actions,
    })
    LOG.info("加载 %s：%d 帧 × %d marker（state 维度 %d），NaN 元素 %d 个",
             path.name, len(out), len(markers), mat.shape[1],
             int(np.isnan(mat).sum()))
    return out


def load_fzmotion_csv_dir(dir_path: str | Path,
                          pattern: str = "episode_*.csv") -> pd.DataFrame:
    """加载目录下所有匹配 CSV，按 episode 合并为一个帧表。

    episode_index 优先取文件名中最后一组数字（episode_001.csv → 1），
    无法解析时按文件名字典序从 0 顺排。

    Args:
        dir_path: CSV 所在目录。
        pattern: 文件名 glob 模式（默认 episode_*.csv）。

    Raises:
        FileNotFoundError: 目录不存在或没有匹配文件。
        ValueError: 各文件的 marker 集合不一致（state 维度无法对齐）。
    """
    dir_path = Path(dir_path)
    if not dir_path.is_dir():
        raise FileNotFoundError(f"目录不存在: {dir_path}")
    files = sorted(dir_path.glob(pattern))
    if not files:
        raise FileNotFoundError(f"{dir_path} 下没有匹配 {pattern} 的文件")

    frames: list[pd.DataFrame] = []
    dim: int | None = None
    for fallback_idx, path in enumerate(files):
        nums = re.findall(r"\d+", path.stem)
        ep = int(nums[-1]) if nums else fallback_idx
        df = load_fzmotion_csv(path, episode_index=ep)
        state_dim = len(df["observation.state"].iloc[0]) if len(df) else 0
        if dim is None:
            dim = state_dim
        elif state_dim != dim:
            raise ValueError(
                f"{path.name} 的 state 维度（{state_dim}）与之前文件（{dim}）不一致，"
                "多 episode 合并要求同一套 marker 配置")
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.sort_values(["episode_index", "timestamp"], ignore_index=True)
    LOG.info("合并完成：%d 个文件 → %d 帧（episode: %s）",
             len(files), len(merged),
             sorted(merged["episode_index"].unique().tolist()))
    return merged
