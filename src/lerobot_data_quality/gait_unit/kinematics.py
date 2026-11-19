"""URDF 运动学：把 G1 的 URDF 解析成骨架规格，并把 MotionClip（关节角通道）
前向运动学成 GaitClip（关节锚点位置）——Tier 3 空间级规则的数据基础。

URDF 约定（官方 g1_29dof.urdf）：
  - 根链接 = pelvis（无父关节的链接）；MotionDecode 的 root_pos/root_rot
    即 pelvis 在世界系的位姿（四元数 wxyz，顺序是**假设**——列名顺序暗示，
    真实导出样本核对：站立帧 pelvis 应大致直立，w 分量占优）；
  - 关节原点 origin xyz 在父链接系内，姿态 rpy = 固定轴 RPY
    （R = Rz(yaw)·Ry(pitch)·Rx(roll)）；
  - 旋转关节绕 <axis>（关节系内）转 angle；
  - 链接位姿 = 父链接位姿 ∘ T(origin_xyz, origin_rpy) ∘ R(axis, angle)，
    关节锚点世界位置 = 父链接位姿作用于 origin_xyz。

诚实声明：
  - MotionDecode 的 29 dof 是 1-DOF 关节角（无逐关节四元数）——FK 后的
    GaitClip rotations=None，骨骼级旋转规则不适用，位置级规则适用；
  - 骨长一致性规则对 FK 输出**恒为零差异**（刚体模型数学上保证），
    它只对测量型骨架（Pico）有意义——对 MotionDecode 数据不要自欺。
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lerobot_data_quality.gait_unit.ir import (
    GaitClip,
    quat_from_axis_angle,
    quat_mul,
    quat_rotate,
)
from lerobot_data_quality.gait_unit.motion import MotionClip

__all__ = ["SkeletonSpec", "load_skeleton", "fk_motion_clip"]


@dataclass
class SkeletonSpec:
    """URDF 解析出的骨架规格（BFS 序，父在前；锚点 = [根链接] + 各关节原点）。"""

    root_link: str
    joint_names: tuple[str, ...]      # (J,) 全部关节（BFS 序，含 fixed）
    anchor_names: tuple[str, ...]     # (J+1,) 锚点名 = [根链接] + 关节系原点
    anchor_parents: tuple[int, ...]   # (J+1,) 锚点层级（-1 = 世界）
    is_actuated: tuple[bool, ...]     # (J,) revolute/continuous = True
    actuated_names: tuple[str, ...]   # 旋转关节名（与 MotionDecode dof 一一对应）
    axes: tuple[tuple[float, float, float], ...]
    origins: tuple[tuple[float, float, float], ...]
    origin_rpy: tuple[tuple[float, float, float], ...]
    parent_link: tuple[str, ...]      # (J,) 关节的父链接
    child_link: tuple[str, ...]       # (J,) 关节的子链接

    def channel_for(self, joint_name: str) -> str:
        """MotionDecode 通道名契约：dof_<URDF 关节名>(rad)。"""
        return f"dof_{joint_name}(rad)"


def _rpy_to_quat(roll: float, pitch: float, yaw: float):
    qx = quat_from_axis_angle((1, 0, 0), roll)
    qy = quat_from_axis_angle((0, 1, 0), pitch)
    qz = quat_from_axis_angle((0, 0, 1), yaw)
    return quat_mul(qz, quat_mul(qy, qx))  # R = Rz·Ry·Rx


def load_skeleton(urdf_path: str | Path) -> SkeletonSpec:
    """解析 URDF → SkeletonSpec。只取链接树与关节几何，忽略惯性/网格。

    Raises:
        FileNotFoundError / ValueError：文件缺失、无关节、根链接不唯一、
        旋转关节缺 <axis>。
    """
    path = Path(urdf_path)
    if not path.is_file():
        raise FileNotFoundError(f"URDF 不存在: {path}")
    root = ET.parse(path).getroot()

    joints_xml = [j for j in root.findall("joint") if j.find("parent") is not None]
    if not joints_xml:
        raise ValueError(f"{path.name} 没有任何关节")

    children = {j.find("child").get("link") for j in joints_xml}
    roots = sorted({j.find("parent").get("link") for j in joints_xml
                    if j.find("parent").get("link") not in children})
    if len(roots) != 1:
        raise ValueError(f"{path.name} 根链接不唯一: {roots}")
    root_link = roots[0]

    joints_by_parent: dict[str, list[ET.Element]] = {}
    for j in joints_xml:
        joints_by_parent.setdefault(j.find("parent").get("link"), []).append(j)

    joint_names: list[str] = []
    anchor_parents: list[int] = []
    anchor_names: list[str] = [f"{root_link}(root)"]
    is_actuated: list[bool] = []
    actuated_names: list[str] = []
    axes: list[tuple] = []
    origins: list[tuple] = []
    origin_rpy: list[tuple] = []
    parent_link: list[str] = []
    child_link: list[str] = []
    link_to_anchor: dict[str, int] = {root_link: 0}

    queue = [root_link]
    while queue:
        link = queue.pop(0)
        for j in joints_by_parent.get(link, []):
            name = j.get("name")
            anchor_idx = len(anchor_names)
            anchor_names.append(name)
            anchor_parents.append(link_to_anchor[link])
            origin = j.find("origin")
            if origin is not None:
                # URDF 契约：xyz/rpy 是**空格分隔字符串**（非 x=/y=/z= 属性）
                xyz_s = origin.get("xyz")
                xyz = (tuple(float(v) for v in xyz_s.split()) if xyz_s
                       else (0.0, 0.0, 0.0))
                rpy_s = origin.get("rpy")
                rpy = (tuple(float(v) for v in rpy_s.split()) if rpy_s
                       else (0.0, 0.0, 0.0))
                if len(xyz) != 3 or len(rpy) != 3:
                    raise ValueError(
                        f"{path.name} 关节 {name} 的 origin xyz/rpy 格式错误: "
                        f"xyz={xyz_s!r} rpy={rpy_s!r}")
            else:
                xyz, rpy = (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
            axis_el = j.find("axis")
            if axis_el is not None:
                # URDF 契约：axis 同样是空格分隔字符串（xyz 属性）
                axis_s = axis_el.get("xyz")
                axis = (tuple(float(v) for v in axis_s.split()) if axis_s
                        else (1.0, 0.0, 0.0))
                if len(axis) != 3:
                    raise ValueError(f"{path.name} 关节 {name} 的 axis 格式错误: {axis_s!r}")
            else:
                axis = (1.0, 0.0, 0.0)  # URDF 缺省轴 = X
            jtype = j.get("type")
            actuated = jtype in ("revolute", "continuous")
            if actuated and axis_el is None:
                raise ValueError(f"旋转关节 {name} 缺 <axis>")

            joint_names.append(name)
            is_actuated.append(actuated)
            if actuated:
                actuated_names.append(name)
            axes.append(axis)
            origins.append(xyz)
            origin_rpy.append(rpy)
            parent_link.append(link)
            child = j.find("child").get("link")
            child_link.append(child)
            link_to_anchor[child] = anchor_idx
            queue.append(child)

    return SkeletonSpec(
        root_link=root_link,
        joint_names=tuple(joint_names),
        anchor_names=tuple(anchor_names),
        # 锚点 = [根] + 各关节：根锚点无父（-1），第 k+1 个锚点的父 = 第 k 个关节的父锚点
        anchor_parents=tuple([-1] + anchor_parents),
        is_actuated=tuple(is_actuated),
        actuated_names=tuple(actuated_names),
        axes=tuple(axes),
        origins=tuple(origins),
        origin_rpy=tuple(origin_rpy),
        parent_link=tuple(parent_link),
        child_link=tuple(child_link),
    )


def fk_motion_clip(clip: MotionClip, spec: SkeletonSpec) -> GaitClip:
    """MotionClip（root 位姿 + dof 角度通道）→ GaitClip（关节锚点位置）。

    通道契约：root_pos_x/y/z(m)、root_rot_w/x/y/z、dof_<关节名>(rad)。
    缺任何一个 actuated 通道都抛错（骨架必须完整，不做静默补零）。
    """
    ch = dict(zip(clip.channel_names, range(clip.n_channels)))
    required = (["root_pos_x(m)", "root_pos_y(m)", "root_pos_z(m)",
                 "root_rot_w", "root_rot_x", "root_rot_y", "root_rot_z"]
                + [spec.channel_for(n) for n in spec.actuated_names])
    missing = [r for r in required if r not in ch]
    if missing:
        raise ValueError(f"缺少通道 {missing[:8]}（骨架不完整，拒绝 FK）")

    root_pos = np.stack([clip.channels[:, ch[f"root_pos_{a}(m)"]] for a in "xyz"], axis=1)
    root_quat = np.stack([clip.channels[:, ch[f"root_rot_{a}"]] for a in "wxyz"], axis=1)

    act_index = {name: i for i, name in enumerate(spec.actuated_names)}
    angle_cols = [clip.channels[:, ch[spec.channel_for(n)]]
                  for n in spec.actuated_names]
    angles = np.stack(angle_cols, axis=1)   # (T, n_act)

    T = clip.n_frames
    positions = np.zeros((T, len(spec.anchor_names), 3))

    for f in range(T):
        link_q = {spec.root_link: tuple(root_quat[f])}
        link_p = {spec.root_link: root_pos[f]}
        positions[f, 0] = root_pos[f]   # 锚点 0 = 根链接原点（不写则恒为 0）
        for ji, name in enumerate(spec.joint_names):
            plink = spec.parent_link[ji]
            q_origin = _rpy_to_quat(*spec.origin_rpy[ji])
            angle = angles[f, act_index[name]] if spec.is_actuated[ji] else 0.0
            link_q_child = quat_mul(
                link_q[plink],
                quat_mul(q_origin, quat_from_axis_angle(spec.axes[ji], angle)))
            link_p_child = link_p[plink] + quat_rotate(link_q[plink], spec.origins[ji])
            link_q[spec.child_link[ji]] = link_q_child
            link_p[spec.child_link[ji]] = link_p_child
            positions[f, ji + 1] = link_p_child  # 关节锚点 = 关节系原点

    return GaitClip(
        fps=clip.fps,
        timestamps=np.arange(T, dtype=np.float64) / clip.fps,
        joint_names=spec.anchor_names,
        positions=positions,
        parents=spec.anchor_parents,
        rotations=None,
        source={"source": "motiondecode_fk", "urdf_root_link": spec.root_link,
                "note": "FK 产物：骨长恒定由刚体模型保证，骨长规则对它无意义"},
    )
