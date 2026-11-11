"""MotionDecode（青瞳视觉）Unitree G1 重定向轨迹 CSV 加载器。

MotionDecode 导出的 CSV 是宽表：每行一帧、每列一个通道，没有显式时间戳：

    root_pos_x(m),root_pos_y(m),root_pos_z(m),root_rot_w,...,dof_left_hip_pitch_joint(rad),...
    0.875474,-0.129735,0.774077,0.733805,...,0.0650221,...

本模块把它转换为与 LeRobot 帧表同构的宽表 DataFrame：

    episode_index      int            episode 编号
    frame_index        int            episode 内从 0 开始的帧号
    timestamp          float          frame_index / fps（MotionDecode 默认 120 FPS）
    observation.state  object(ndarray[float32, D])  按 CSV 列顺序拼接的全部数值通道
    action             object(ndarray[float32, D])  下一帧 state；最后一帧保持自身

设计约束与 FZMotion 适配器一致：缺陷（NaN、丢帧、跳变）原样保留，交给下游规则裁决；
非数值列（如注释、标签）会被丢弃并告警，不参与 state 维度。
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd

LOG = logging.getLogger(__name__)

#: 至少要有根位置三列，否则大概率不是 MotionDecode 的 G1 轨迹 CSV
REQUIRED_COLUMNS = ("root_pos_x(m)", "root_pos_y(m)", "root_pos_z(m)")
DEFAULT_FPS = 120.0


def _numeric_frame(raw: pd.DataFrame, path: Path) -> pd.DataFrame:
    """保留数值列（含可转数值的脏字符串），丢弃纯文本列。"""
    kept: dict[str, pd.Series] = {}
    dropped: list[str] = []
    for col in raw.columns:
        series = raw[col]
        if pd.api.types.is_numeric_dtype(series):
            kept[col] = series
            continue
        converted = pd.to_numeric(series, errors="coerce")
        if converted.notna().any():
            kept[col] = converted
        else:
            dropped.append(str(col))
    if dropped:
        LOG.warning("%s 丢弃 %d 个非数值列: %s", path.name, len(dropped), dropped[:5])
    return pd.DataFrame(kept, index=raw.index)


def load_motiondecode_csv(path: str | Path, episode_index: int = 0,
                          fps: float = DEFAULT_FPS) -> pd.DataFrame:
    """加载单个 MotionDecode G1 轨迹 CSV，返回 LeRobot 同构的帧表。

    Args:
        path: CSV 文件路径。
        episode_index: 写入输出 episode_index 列的编号。
        fps: 帧率，用于生成 timestamp；MotionDecode 默认 120。

    Raises:
        FileNotFoundError: 文件不存在。
        ValueError: fps 非法、缺少必需列、没有数据行或没有数值列。
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"CSV 文件不存在: {path}")
    if not math.isfinite(float(fps)) or fps <= 0:
        raise ValueError(f"fps 必须是正有限数，实际为 {fps!r}")

    try:
        raw = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"{path.name} 是空文件，没有任何数据行") from exc

    missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(
            f"{path.name} 缺少必需列: {missing}（需要根位置列 {list(REQUIRED_COLUMNS)}）")
    if len(raw) == 0:
        raise ValueError(f"{path.name} 没有任何数据行")

    numeric = _numeric_frame(raw, path)
    if numeric.shape[1] == 0:
        raise ValueError(f"{path.name} 没有任何可用的数值列")

    mat = numeric.to_numpy(dtype=np.float32, copy=True)
    states = [row.copy() for row in mat]
    # action := 下一帧 state；最后一帧保持自身（与 FZMotion 适配器保持同一契约）
    actions = [states[i + 1].copy() if i + 1 < len(states) else states[i].copy()
               for i in range(len(states))]

    out = pd.DataFrame({
        "episode_index": int(episode_index),
        "frame_index": np.arange(len(mat), dtype=np.int64),
        "timestamp": np.arange(len(mat), dtype=np.float64) / float(fps),
        "observation.state": states,
        "action": actions,
    })
    LOG.info("加载 %s：%d 帧 × %d 维（NaN 元素 %d 个）",
             path.name, len(out), mat.shape[1], int(np.isnan(mat).sum()))
    return out


def load_motiondecode_csv_dir(dir_path: str | Path, pattern: str = "*.csv",
                              fps: float = DEFAULT_FPS,
                              recursive: bool = True) -> pd.DataFrame:
    """加载目录下的 MotionDecode CSV，按路径排序后合并为帧表。

    MotionDecode 的文件编号在不同动作类别下会重复（每个类别都从 00001 开始），
    因此这里按排序后的文件顺序重新分配 episode_index（0..N-1），避免合并冲突。

    Raises:
        FileNotFoundError: 目录不存在或没有匹配文件。
        ValueError: 各文件的 state 维度不一致。
    """
    dir_path = Path(dir_path)
    if not dir_path.is_dir():
        raise FileNotFoundError(f"目录不存在: {dir_path}")
    files = sorted(dir_path.rglob(pattern) if recursive else dir_path.glob(pattern))
    if not files:
        raise FileNotFoundError(f"{dir_path} 下没有匹配 {pattern} 的文件")

    frames: list[pd.DataFrame] = []
    dim: int | None = None
    for ep, path in enumerate(files):
        df = load_motiondecode_csv(path, episode_index=ep, fps=fps)
        state_dim = len(df["observation.state"].iloc[0]) if len(df) else 0
        if dim is None:
            dim = state_dim
        elif state_dim != dim:
            raise ValueError(
                f"{path.name} 的 state 维度（{state_dim}）与之前文件（{dim}）不一致，"
                "多 episode 合并要求同一套通道配置")
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.sort_values(["episode_index", "timestamp"], ignore_index=True)
    LOG.info("合并完成：%d 个文件 → %d 帧", len(files), len(merged))
    return merged
