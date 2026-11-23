"""CHINGMU MotionDecode（青瞳视觉）重定向轨迹 CSV → MotionClip 适配器。

源格式（宽表，每行一帧，无显式时间戳，帧序即时间，MotionDecode 默认 120 FPS）：

    root_pos_x(m), root_pos_y(m), root_pos_z(m),
    root_rot_w, root_rot_x, root_rot_y, root_rot_z,
    dof_left_hip_pitch_joint(rad), ..., dof_right_wrist_yaw_joint(rad)

列名自带语义：29 个 dof_* 通道 = Unitree G1 机器人 1-DOF 关节角（弧度），
即 MotionDecode 软件已把光捕**重定向到 G1 人形骨架**——本适配器不涉及
重定向，只做忠实转换。

与 adapters/motiondecode_csv_loader.py 的关系：
  loaders/ 版本打包成 LeRobot 帧表（VLA 数据域）；本适配器面向步态 IR，
  保留通道名（QC 规则要按关节语义引用通道）。两份解析是有意的领域分离。
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from verticals.gait_unit.motion import MotionClip

LOG = logging.getLogger(__name__)

REQUIRED_COLUMNS = ("root_pos_x(m)", "root_pos_y(m)", "root_pos_z(m)")
DEFAULT_FPS = 120.0


def load_motiondecode_csv(path: str | Path, fps: float = DEFAULT_FPS) -> MotionClip:
    """加载 MotionDecode 宽表 CSV → MotionClip（通道名 = 源列名，原样保留）。

    - 非数值列（注释/标签）丢弃并计数告警，不进 channels；
    - NaN 原样保留（缺测/丢帧是 G1 规则的对象）。

    Raises:
        FileNotFoundError / ValueError：路径无效、缺根位置列、无有效数值通道。
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"CSV 文件不存在: {path}")
    raw = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"{path.name} 缺少必需列: {missing}（不是 MotionDecode 导出？）")

    dropped: list[str] = []
    kept_cols: dict[str, np.ndarray] = {}
    for col in raw.columns:
        series = raw[col]
        if pd.api.types.is_numeric_dtype(series):
            kept_cols[col] = series.to_numpy(dtype=np.float64)
            continue
        converted = pd.to_numeric(series, errors="coerce")
        if converted.notna().mean() > 0.5:      # 多数可转 → 当数值列（容忍脏行）
            kept_cols[col] = converted.to_numpy(dtype=np.float64)
            LOG.warning("%s 列 '%s' 含非数值脏行，已强转（NaN 计入缺测）", path.name, col)
        else:
            dropped.append(col)
    if dropped:
        LOG.warning("%s 丢弃 %d 个非数值列: %s", path.name, len(dropped), dropped[:8])

    if not kept_cols:
        raise ValueError(f"{path.name} 没有任何有效数值通道")

    channels = np.stack([kept_cols[c] for c in kept_cols], axis=1)
    clip = MotionClip(
        fps=fps,
        channel_names=tuple(kept_cols.keys()),
        channels=channels,
        source={"source": "motiondecode_csv", "file": path.name,
                "root_convention": "root_pos(m) + root_rot(wxyz)",
                "dropped_columns": dropped,
                "n_frames_declared": len(raw)},
    )
    LOG.info("加载 %s：%d 帧 × %d 通道，NaN 元素 %d",
             path.name, clip.n_frames, clip.n_channels,
             int(np.isnan(channels).sum()))
    return clip
