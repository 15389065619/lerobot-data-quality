"""Pico 穿戴动捕导出 CSV → GaitClip 适配器。

字段契约（假设版 v0，真实导出样本到位后核对 REQUIRED_COLUMNS，不动解析逻辑）：

    timestamp, joint, pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z

长表格式（每行一个 timestamp × joint 组合，与 fzmotion loader 的长表惯例一致）：

    timestamp,  joint,     pos_x,   pos_y,   pos_z,   quat_w, quat_x,  quat_y,  quat_z
    0.000000,   Hips,      0.000,   0.950,   0.000,   1.000,  0.000,   0.000,   0.000
    0.000000,   KneeL,     ...

已知与真实 Pico 导出的差异（open questions，拿到样本后核对）：
  - 导出是否长表/宽表未知——若宽表，加一个转置入口即可；
  - 四元数顺序按列名固定为 wxyz（QUAT_COLUMNS）；up 轴未做转换——
    真实导出若为 Z-up 或顺序不同，在适配层补一次性转换；
  - 关节层级 CSV 里不带 → clip.parents = None，G1.3（骨长一致性）标记不适用。
    Pico 官方骨骼层级确认后在此补 PARENT_TABLE。
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from lerobot_data_quality.gait_unit.ir import GaitClip

LOG = logging.getLogger(__name__)

REQUIRED_COLUMNS = ("timestamp", "joint", "pos_x", "pos_y", "pos_z",
                    "quat_w", "quat_x", "quat_y", "quat_z")
QUAT_COLUMNS = ("quat_w", "quat_x", "quat_y", "quat_z")


def load_pico_csv(path: str | Path) -> GaitClip:
    """加载 Pico 长表 CSV → GaitClip。

    - 关节名取文件内出现顺序的 sorted 去重（确定性排序，规则可复现）；
    - 层级未知 → parents=None；
    - 同一 timestamp 出现多行属于源数据现象，原样保留（G1.6 盖章规则的对象）。

    Raises:
        FileNotFoundError / ValueError：路径无效或缺列（缺列 = 契约不符，明确报错
        而不是静默补 NaN——adapter 的职责是忠实转换，不是猜测修复）。
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"CSV 文件不存在: {path}")
    raw = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"{path.name} 缺少必需列: {missing}（契约见模块 docstring）")

    # 关节名按**首次出现顺序**（与文件一致，确定性且保留源排序）
    joint_names_list: list[str] = []
    seen_names: set[str] = set()
    for name in raw["joint"].astype(str):
        if name not in seen_names:
            seen_names.add(name)
            joint_names_list.append(name)
    joint_names = tuple(joint_names_list)
    name_to_idx = {name: i for i, name in enumerate(joint_names)}
    row_j = raw["joint"].astype(str).map(name_to_idx).to_numpy()

    # 长表 → (T, J, ·)：以 timestamp 排序后按**变化点**分帧。
    # 变化点分组（而非 unique 去重）是有意为之：重复盖章的帧必须保留成
    # 独立的帧——它们是 G1.6 盖章规则的质检对象，去重会吞掉证据。
    order = np.argsort(raw["timestamp"].to_numpy(dtype=np.float64), kind="stable")
    ts = raw["timestamp"].to_numpy(dtype=np.float64)[order]
    row_j = row_j[order]

    change = np.concatenate(([True], np.diff(ts) != 0))
    frame_of_row = np.cumsum(change) - 1
    frame_ts = ts[change]
    T, J = len(frame_ts), len(joint_names)
    frames_with_repeated_stamp = int((np.diff(frame_ts) == 0).sum())

    positions = np.full((T, J, 3), np.nan)
    rotations = np.full((T, J, 4), np.nan)

    pos_vals = raw[["pos_x", "pos_y", "pos_z"]].to_numpy(dtype=np.float64)[order]
    quat_vals = raw[list(QUAT_COLUMNS)].to_numpy(dtype=np.float64)[order]
    # 每帧内同一关节出现多次时保留首行（正常导出不会发生；发生即在留痕里可见）
    seen = np.zeros((T, J), dtype=bool)
    for k in range(len(ts)):
        j = row_j[k]
        f = frame_of_row[k]
        if not seen[f, j]:
            positions[f, j] = pos_vals[k]
            rotations[f, j] = quat_vals[k]
            seen[f, j] = True

    diffs = np.diff(frame_ts)
    pos_diffs = diffs[diffs > 0]
    clip = GaitClip(
        fps=round(1.0 / float(np.median(pos_diffs)), 6) if len(pos_diffs) else 0.0,
        timestamps=frame_ts,
        joint_names=joint_names,
        positions=positions,
        parents=None,
        rotations=rotations,
        source={"source": "pico_csv", "file": path.name, "unit": "米（契约假设）",
                "frames_with_repeated_stamp": frames_with_repeated_stamp},
    )
    LOG.info("加载 %s：%d 帧 × %d 关节（重复盖章帧 %d）",
             path.name, T, J, frames_with_repeated_stamp)
    return clip


def write_pico_csv(clip: GaitClip, path: str | Path) -> Path:
    """GaitClip → Pico 契约 CSV（fixture 生成 / 修复结果导出两用）。

    rotations 为 None 时四元数列写 1,0,0,0（单位四元数占位），
    positions 含 NaN 的单元格按契约留空（pandas 自动写空串）。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    T, J = clip.n_frames, clip.n_joints
    for f in range(T):
        for j, name in enumerate(clip.joint_names):
            rot = (clip.rotations[f, j] if clip.rotations is not None
                   else np.array([1.0, 0.0, 0.0, 0.0]))
            rows.append({
                "timestamp": clip.timestamps[f],
                "joint": name,
                "pos_x": clip.positions[f, j, 0],
                "pos_y": clip.positions[f, j, 1],
                "pos_z": clip.positions[f, j, 2],
                "quat_w": rot[0], "quat_x": rot[1], "quat_y": rot[2], "quat_z": rot[3],
            })
    pd.DataFrame(rows).to_csv(path, index=False)
    return path
