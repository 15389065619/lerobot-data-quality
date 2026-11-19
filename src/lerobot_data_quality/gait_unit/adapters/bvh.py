"""BVH（Biovision Hierarchy）→ GaitClip 适配器：人体骨骼层级 + 逐帧运动。

解析约定（文档化，各家导出有差异，遇到新来源先对照本节）：
  - 旋转：CHANNELS 声明顺序 = 内旋顺序，四元数按声明顺序左乘
    （q = q_first ⊗ q_second ⊗ ...，列向量约定 v' = R·v）。内旋 X→Y→Z 等价于
    R = Rx·Ry·Rz；
  - 单位：BVH 不声明单位（常见 cm），IR 一律按"源单位"原样保留，
    source["unit"] = "unverified"，换算到米属于 G3（重定向）的事；
  - End Site 作为叶子关节进 IR（命名 <父关节>_end），骨长一致性规则需要它；
    它没有 CHANNELS，局部旋转恒为单位四元数。
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np

from lerobot_data_quality.gait_unit.ir import (
    GaitClip,
    quat_from_axis_angle,
    quat_mul,
    quat_rotate,
)

LOG = logging.getLogger(__name__)

_AXIS_VEC = {"X": (1.0, 0.0, 0.0), "Y": (0.0, 1.0, 0.0), "Z": (0.0, 0.0, 1.0)}


class _Joint:
    """BVH 层级节点（解析中间态）。"""

    def __init__(self, name: str, parent: int | None):
        self.name = name
        self.parent = parent
        self.offset = (0.0, 0.0, 0.0)
        self.channels: list[str] = []


def _parse_joint(lines: list[str], pos: int, name: str,
                 parent: int | None, joints: list[_Joint]) -> int:
    """递归解析一个 { ... } 块。返回消耗后的行号。joints 按声明序追加（父在子前）。"""
    joint = _Joint(name, parent)
    idx = len(joints)
    joints.append(joint)
    assert lines[pos].strip() == "{", f"第 {pos} 行应为 '{{'，实际: {lines[pos]!r}"
    pos += 1
    while pos < len(lines):
        line = lines[pos].strip()
        if line.startswith("OFFSET"):
            parts = line.split()
            joint.offset = (float(parts[1]), float(parts[2]), float(parts[3]))
            pos += 1
        elif line.startswith("CHANNELS"):
            parts = line.split()
            n = int(parts[1])
            joint.channels = parts[2:2 + n]
            if len(joint.channels) != n:
                raise ValueError(f"关节 {name} 的 CHANNELS 声明 {n} 个，实际 {len(joint.channels)} 个")
            pos += 1
        elif line.startswith("JOINT"):
            pos = _parse_joint(lines, pos + 1, line.split(maxsplit=1)[1].strip(),
                               idx, joints)
        elif line.startswith("End Site"):
            pos = _parse_joint(lines, pos + 1, f"{name}_end", idx, joints)
        elif line == "}":
            return pos + 1
        else:
            pos += 1  # 未知行容错跳过，结构问题交给 validate_clip 兜底
    raise ValueError(f"关节 {name} 的块没有闭合（文件截断）")


def parse_bvh(path: str | Path, scale: float = 1.0, up_axis: str = "y") -> GaitClip:
    """解析 BVH 文件 → GaitClip。

    scale：位置缩放（源单位 → 米）。BVH 不声明单位（LAFAN1 系为 cm，
    传 scale=0.01；合成/米制源用默认 1.0）。缩放只作用于位置，不动旋转。

    up_axis：垂直轴声明（BVH 常见 Y-up；MotionDecode 为 Z-up）。只写入
    source["up_axis"] 供空间规则自动取用，不做坐标变换。

    全局位置由前向运动学算出（root 平移 + 父世界旋转作用于 OFFSET），
    rotations 存各关节的**局部**旋转（相对父关节，含 End Site 的单位四元数）。
    注意：非根关节的 Xposition/Yposition/Zposition 通道若为静态 OFFSET
    回显（LAFAN1 系如此），本解析按 BVH 标准忽略之（FK 用 OFFSET）。

    Raises:
        FileNotFoundError / ValueError：文件缺失、无 MOTION 段、块未闭合、
        帧数据列数与 CHANNELS 总数不符。
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"BVH 文件不存在: {path}")
    lines = [ln.strip() for ln in
             path.read_text(encoding="utf-8", errors="replace").splitlines() if ln.strip()]

    hier_at = next(i for i, ln in enumerate(lines) if ln == "HIERARCHY")
    motion_at = next((i for i, ln in enumerate(lines) if ln == "MOTION"), None)
    if motion_at is None:
        raise ValueError(f"{path.name} 没有 MOTION 段")

    joints: list[_Joint] = []
    # hier_at+1 = "ROOT xxx" 行，hier_at+2 = "{" 行（_parse_joint 的入口契约）
    root_line = lines[hier_at + 1]
    _parse_joint(lines, hier_at + 2, root_line.split(maxsplit=1)[1].strip(),
                 None, joints)

    if lines[motion_at + 1].split()[0] != "Frames:":
        raise ValueError(f"{path.name} MOTION 段格式错误（应为 'Frames: N'）")
    frames_n = int(lines[motion_at + 1].split()[1])
    if lines[motion_at + 2].split()[0] != "Frame":
        raise ValueError(f"{path.name} MOTION 段格式错误（应为 'Frame Time: X'）")
    frame_time = float(lines[motion_at + 2].split()[2])

    ch_total = sum(len(j.channels) for j in joints)
    data_lines = lines[motion_at + 3: motion_at + 3 + frames_n]
    if len(data_lines) < frames_n:
        raise ValueError(f"{path.name} MOTION 帧数据不足：声明 {frames_n} 帧，实得 {len(data_lines)}")

    # 每个关节的通道在整帧行内的下标切片（按声明顺序累计）
    slices: list[list[int]] = []
    cursor = 0
    for j in joints:
        slices.append(list(range(cursor, cursor + len(j.channels))))
        cursor += len(j.channels)

    T, J = frames_n, len(joints)
    positions = np.zeros((T, J, 3))
    rotations = np.zeros((T, J, 4))

    for f, row in enumerate(data_lines):
        vals = [float(v) for v in row.split()]
        if len(vals) != ch_total:
            raise ValueError(f"{path.name} 第 {f + 1} 帧数值 {len(vals)} 个 ≠ 通道总数 {ch_total}")
        world_q: list = [None] * J
        for ji, (j, idxs) in enumerate(zip(joints, slices)):
            motion_pos = np.zeros(3)
            local_q = (1.0, 0.0, 0.0, 0.0)
            ch_i = 0
            for ch in j.channels:
                v = vals[idxs[ch_i]]
                if ch.endswith("position"):
                    motion_pos["XYZ".index(ch[0])] = v
                elif ch.endswith("rotation"):
                    local_q = quat_mul(local_q,
                                       quat_from_axis_angle(_AXIS_VEC[ch[0]], math.radians(v)))
                ch_i += 1
            if j.parent is None:
                world_q[ji] = local_q
                positions[f, ji] = motion_pos
            else:
                world_q[ji] = quat_mul(world_q[j.parent], local_q)
                positions[f, ji] = positions[f, j.parent] + \
                    quat_rotate(world_q[j.parent], j.offset)
            rotations[f, ji] = local_q

    if scale != 1.0:
        positions = positions * scale

    clip = GaitClip(
        fps=round(1.0 / frame_time, 6),
        timestamps=np.arange(T, dtype=np.float64) * frame_time,
        joint_names=tuple(j.name for j in joints),
        positions=positions,
        parents=tuple(-1 if j.parent is None else j.parent for j in joints),
        rotations=rotations,
        source={"source": "bvh", "file": path.name,
                "unit": f"源单位 × scale={scale}" + (" → 米" if scale == 0.01 else ""),
                "up_axis": up_axis,
                "rotation_unit": "度（BVH 标准，解析时转弧度）",
                "rotation_convention": "CHANNELS 声明顺序内旋左乘"},
    )
    LOG.info("解析 %s：%d 帧 × %d 关节，通道总数 %d", path.name, T, J, ch_total)
    return clip
