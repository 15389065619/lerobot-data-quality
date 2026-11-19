"""Tier 3 空间级规则：作用于 FK 产物的 GaitClip（需要骨架规格的 FK 先行）。

规则清单：
  compute_foot_contact      足端接触状态机（帮助函数：rules 与导出共用）
  rule_foot_skating         接触期水平速度超阈（脚底打滑——RL 数据经典毒）
  rule_ground_penetration   足端低于地面参考线（需要显式 ground_z；
                            自动地面=min 时本规则恒不触发，见 docstring）
  rule_flight_phase         双脚同时离地的腾空段（信息留痕：跳类动作合法）

约定：
  - 足端锚点名 = URDF 里脚的末端关节锚点（G1 = left/right_ankle_roll_joint）；
  - 地面高度：显式 ground_z 优先；缺省用逐足观测最小 z（此时足端只可能
    等于或高于"地面"，穿透规则退化为 no-op——诚实写进 docstring）；
  - 高动态动作（跳/跑）的腾空与冲击是**动作本身**，findings 定级为
    信息留痕，是否判缺陷由动作类别 profile 决定（同 Tier 2 的哲学）。
"""

from __future__ import annotations

import numpy as np

from lerobot_data_quality.gait_unit.ir import GaitClip

__all__ = [
    "compute_foot_contact", "rule_foot_skating",
    "rule_ground_penetration", "rule_flight_phase",
]

DEFAULT_FEET = ("left_ankle_roll_joint", "right_ankle_roll_joint")


def _foot_anchors(clip: GaitClip, foot_names: tuple[str, ...]) -> list[int]:
    idx = [clip.joint_names.index(n) for n in foot_names]
    return idx


def _runs(flags: np.ndarray) -> list[tuple[int, int]]:
    if not flags.any():
        return []
    idx = np.flatnonzero(flags)
    breaks = np.where(np.diff(idx) > 1)[0]
    starts = np.concatenate(([idx[0]], idx[breaks + 1]))
    ends = np.concatenate((idx[breaks], [idx[-1]])) + 1
    return [(int(s), int(e)) for s, e in zip(starts, ends)]


def compute_foot_contact(clip: GaitClip, foot_names: tuple[str, ...] = DEFAULT_FEET,
                         h_max: float = 0.05) -> dict:
    """足端接触状态机：接触 = 足端高度 ≤ 地面参考 + h_max。

    地面参考 = 各足观测最小 z（站立帧在真实数据里占多数，最小 z 即
    "脚放地上"的高度，自动吸收踝原点到鞋底的未知偏移）。
    """
    idx = _foot_anchors(clip, foot_names)
    contact: dict[str, np.ndarray] = {}
    ground: dict[str, float] = {}
    for i, name in zip(idx, foot_names):
        z = clip.positions[:, i, 2]
        z_ground = float(np.nanmin(z))
        ground[name] = z_ground
        contact[name] = (z - z_ground) <= h_max
    return {"feet": foot_names, "ground_z": ground, "contact": contact}


def rule_foot_skating(clip: GaitClip, foot_names: tuple[str, ...] = DEFAULT_FEET,
                      h_max: float = 0.05, speed_max: float = 0.4) -> list[dict]:
    """接触期水平速度超阈：脚"在地上"却横向移动 = 打滑。

    水平速度用 np.gradient（中央差分）；接触判定复用 compute_foot_contact。
    """
    contact = compute_foot_contact(clip, foot_names, h_max)
    idx = _foot_anchors(clip, foot_names)
    findings = []
    for i, name in zip(idx, foot_names):
        xy = clip.positions[:, i, :2]
        dt = 1.0 / clip.fps
        speed = np.linalg.norm(np.gradient(xy, dt, axis=0), axis=1)
        skate = contact["contact"][name] & (speed > speed_max)
        runs = _runs(skate)
        if runs:
            findings.append({
                "rule": "foot_skating", "foot": name, "speed_max": speed_max,
                "runs": runs, "max_speed": round(float(speed[skate].max()), 3),
                "n_frames": int(skate.sum()),
            })
    return findings


def rule_ground_penetration(clip: GaitClip, foot_names: tuple[str, ...] = DEFAULT_FEET,
                            ground_z: float | None = None, eps: float = 0.005) -> list[dict]:
    """足端低于地面参考线 eps 以上。

    ground_z 必须显式给出（世界系地面高度）；不给出时规则退化为 no-op
    并在 findings 里如实记录——自动地面 = 观测最小 z，数学上不存在
    "低于它"的帧，硬造一个假警报就是自欺。
    """
    if ground_z is None:
        return [{"rule": "ground_penetration", "skipped": True,
                 "reason": "未提供世界系地面高度 ground_z；自动地面=观测最小 z 时穿透不可能发生"}]
    idx = _foot_anchors(clip, foot_names)
    findings = []
    for i, name in zip(idx, foot_names):
        z = clip.positions[:, i, 2]
        bad = np.isfinite(z) & (z < ground_z - eps)
        runs = _runs(bad)
        if runs:
            findings.append({"rule": "ground_penetration", "foot": name,
                             "ground_z": ground_z, "eps": eps,
                             "runs": runs, "n_frames": int(bad.sum()),
                             "min_z": round(float(np.nanmin(z)), 4)})
    return findings


def rule_flight_phase(clip: GaitClip, foot_names: tuple[str, ...] = DEFAULT_FEET,
                      h_max: float = 0.05, min_frames: int = 6) -> list[dict]:
    """双脚同时离地 ≥ min_frames 的腾空段（信息留痕：跳/跑类动作合法）。"""
    contact = compute_foot_contact(clip, foot_names, h_max)
    feet = contact["feet"]
    airborne = np.ones(clip.n_frames, dtype=bool)
    for name in feet:
        airborne &= ~contact["contact"][name]
    runs = _runs(airborne)
    long_runs = [(s, e) for s, e in runs if e - s >= min_frames]
    if not long_runs:
        return []
    return [{"rule": "flight_phase", "feet": feet, "runs": long_runs,
             "longest_s": round((max(e - s for s, e in long_runs)) / clip.fps, 3),
             "n_runs": len(long_runs),
             "note": "信息留痕：跳/跑类动作的腾空是动作本身，是否异常由动作类别定"}]
