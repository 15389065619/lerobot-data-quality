"""FZMotion 光学动捕 marker 长表 CSV → GaitClip **点级子集**适配器。

与 adapters/fzmotion_csv_loader.py 的关系：
  loaders/ 版本把 marker 点打包成 LeRobot 帧表（observation.state 定长向量），
  服务 VLA 数据域；本适配器面向步态 IR，保留 (T, M, 3) 的 marker 结构与
  marker 命名——**点级子集**：positions 有、rotations/parents 恒为 None，
  骨骼级规则（G1.1/G1.3）自动标记不适用。

marker → 骨骼（拟合/配骨骼）是独立跳步，本适配器不做——契约见模块 docstring。
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from verticals.gait_unit.ir import GaitClip

LOG = logging.getLogger(__name__)

REQUIRED_COLUMNS = ("timestamp", "marker_id", "x", "y", "z")


def load_fzmotion_points(path: str | Path) -> GaitClip:
    """加载 FZMotion 长表 marker CSV → 点级 GaitClip。

    - marker_id 升序 → joint_names = ["marker_000", "marker_001", ...]；
    - 某帧缺某个 marker（遮挡）→ 对应位置 NaN，原样保留（G1 规则的对象）；
    - 重复 timestamp 原样保留（盖章规则的对象）。
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"CSV 文件不存在: {path}")
    raw = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"{path.name} 缺少必需列: {missing}")

    marker_ids = sorted(int(m) for m in raw["marker_id"].dropna().unique())
    if not marker_ids:
        raise ValueError(f"{path.name} 中没有任何 marker 行")
    id_to_idx = {mid: i for i, mid in enumerate(marker_ids)}

    ts = raw["timestamp"].to_numpy(dtype=np.float64)
    order = np.argsort(ts, kind="stable")
    ts = ts[order]
    row_m = raw["marker_id"].map(id_to_idx).to_numpy()[order]
    xyz = raw[["x", "y", "z"]].to_numpy(dtype=np.float64)[order]

    # 变化点分帧（与 pico_csv 同逻辑）：重复盖章的帧必须保留为独立帧，
    # unique 去重会吞掉 G1.6 需要的证据。
    change = np.concatenate(([True], np.diff(ts) != 0))
    frame_of_row = np.cumsum(change) - 1
    frame_ts = ts[change]
    T, M = len(frame_ts), len(marker_ids)
    frames_with_repeated_stamp = int((np.diff(frame_ts) == 0).sum())

    positions = np.full((T, M, 3), np.nan)
    seen = np.zeros((T, M), dtype=bool)
    for k in range(len(ts)):
        f, m = frame_of_row[k], row_m[k]
        if not seen[f, m]:
            positions[f, m] = xyz[k]
            seen[f, m] = True

    diffs = np.diff(frame_ts)
    pos_diffs = diffs[diffs > 0]
    clip = GaitClip(
        fps=round(1.0 / float(np.median(pos_diffs)), 6) if len(pos_diffs) else 0.0,
        timestamps=frame_ts,
        joint_names=tuple(f"marker_{mid:03d}" for mid in marker_ids),
        positions=positions,
        parents=None,
        rotations=None,
        source={"source": "fzmotion_points", "file": path.name, "unit": "米（源声明）",
                "frames_with_repeated_stamp": frames_with_repeated_stamp},
    )
    LOG.info("加载 %s：%d 帧 × %d marker，NaN 元素 %d",
             path.name, T, M, int(np.isnan(positions).sum()))
    return clip
