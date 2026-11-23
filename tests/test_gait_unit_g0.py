"""G0 验收测试：IR 契约 + 三个 adapter + 四个缺陷注入器。

每条测试都对应设计文档里的一条门禁：
  1. 干净基线的结构不变量（骨长恒定 / 四元数单位范数 / 时间戳严格递增）
  2. Pico CSV 往返一致
  3. BVH FK 已知答案（手算坐标比对）+ 骨长恒定
  4. FZMotion 点级子集（NaN 保留 / 命名 / 重复盖章计数）
  5. 四个注入器的地面真值断言
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from verticals.gait_unit import (
    GaitClip,
    bone_lengths,
    quat_angle,
    validate_clip,
)
from verticals.gait_unit.adapters import (
    load_fzmotion_points,
    load_pico_csv,
    parse_bvh,
    write_pico_csv,
)
from verticals.gait_unit.defects import (
    inject_drift,
    inject_duplicate_stamp,
    inject_frame_drop,
    inject_joint_limit,
    make_clean_clip,
)


# --------------------------------------------------------------------------
# 1. 干净基线
# --------------------------------------------------------------------------

def test_clean_clip_invariants():
    clip = make_clean_clip()
    assert validate_clip(clip) == []
    assert np.all(np.diff(clip.timestamps) > 0), "时间戳应严格递增"
    for q in clip.rotations.reshape(-1, 4):
        assert abs(np.linalg.norm(q) - 1.0) < 1e-9, "四元数应为单位范数"
    lengths = bone_lengths(clip)
    assert len(lengths) == 6  # 7 关节 - 根 = 6 根骨
    for bone, seq in lengths.items():
        assert seq.max() - seq.min() < 1e-9, f"骨 {bone} 长度应恒定（刚体）"


# --------------------------------------------------------------------------
# 2. Pico CSV 往返
# --------------------------------------------------------------------------

def test_pico_csv_roundtrip(tmp_path):
    clip = make_clean_clip(seconds=2.0)
    csv_path = write_pico_csv(clip, tmp_path / "pico_export.csv")
    back = load_pico_csv(csv_path)

    assert validate_clip(back) == []
    assert back.joint_names == clip.joint_names
    assert np.allclose(back.timestamps, clip.timestamps, atol=1e-9)
    assert np.allclose(back.positions, clip.positions, atol=1e-9)
    assert np.allclose(back.rotations, clip.rotations, atol=1e-9)
    # Pico CSV 契约不带层级——parents=None 是契约行为，不是缺陷
    assert back.parents is None


def test_pico_csv_missing_column_fails(tmp_path):
    clip = make_clean_clip(seconds=0.2)
    csv_path = write_pico_csv(clip, tmp_path / "p.csv")
    broken = csv_path.read_text(encoding="utf-8").replace("quat_w", "qw")
    (tmp_path / "broken.csv").write_text(broken, encoding="utf-8")
    with pytest.raises(ValueError, match="缺少必需列"):
        load_pico_csv(tmp_path / "broken.csv")


# --------------------------------------------------------------------------
# 3. BVH FK 已知答案
# --------------------------------------------------------------------------

_BVH = """HIERARCHY
ROOT Hips
{
  OFFSET 0.0 0.0 0.0
  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
  JOINT Spine
  {
    OFFSET 0.0 10.0 0.0
    CHANNELS 3 Zrotation Xrotation Yrotation
    JOINT Head
    {
      OFFSET 0.0 10.0 0.0
      CHANNELS 3 Zrotation Xrotation Yrotation
      End Site
      {
        OFFSET 0.0 5.0 0.0
      }
    }
  }
}
MOTION
Frames: 2
Frame Time: 0.033333
0 0 0   0 0 0   0 0 0   0 0 0
0 5 0   0 0 0   90 0 0   0 0 0
"""


def test_bvh_fk_known_answer(tmp_path):
    bvh_path = tmp_path / "mini.bvh"
    bvh_path.write_text(_BVH, encoding="utf-8")
    clip = parse_bvh(bvh_path)

    assert clip.n_frames == 2
    assert clip.joint_names == ("Hips", "Spine", "Head", "Head_end")
    assert clip.parents == (-1, 0, 1, 2)

    # 帧 1：全零 → 骨架立正，各关节坐标 = OFFSET 沿 Y 累加
    assert np.allclose(clip.positions[0, 0], (0, 0, 0))
    assert np.allclose(clip.positions[0, 1], (0, 10, 0))
    assert np.allclose(clip.positions[0, 2], (0, 20, 0))
    assert np.allclose(clip.positions[0, 3], (0, 25, 0))

    # 帧 2：根平移 (0,5,0) + Spine 的 Zrotation=90（通道序 Z/X/Y → 第一个值是 Z）。
    # Spine = 根 + (0,10,0) = (0,15,0)；Head = Spine + Rz90·(0,10,0) = (-10,15,0)；
    # Head_end = Head + Rz90·(0,5,0) = (-15,15,0)。
    assert np.allclose(clip.positions[1, 0], (0, 5, 0))
    assert np.allclose(clip.positions[1, 1], (0, 15, 0))
    assert np.allclose(clip.positions[1, 2], (-10, 15, 0), atol=1e-9)
    assert np.allclose(clip.positions[1, 3], (-15, 15, 0), atol=1e-9)

    # 骨长在两帧间恒定（刚体）——Z 轴旋转不改变长度
    for bone, seq in bone_lengths(clip).items():
        assert seq.max() - seq.min() < 1e-9, f"骨 {bone} 旋转后长度应恒定"


def test_bvh_scale_and_up_axis(tmp_path):
    """scale（cm→m）与 up_axis 声明——真实 BVH（LAFAN1）适配的两个关键参数。"""
    bvh_path = tmp_path / "mini.bvh"
    bvh_path.write_text(_BVH, encoding="utf-8")
    clip = parse_bvh(bvh_path, scale=0.01, up_axis="y")
    assert np.allclose(clip.positions[0, 1], (0.0, 0.10, 0.0), atol=1e-9)
    assert clip.source["up_axis"] == "y"


def test_bvh_missing_motion_raises(tmp_path):
    bad = tmp_path / "no_motion.bvh"
    bad.write_text("HIERARCHY\nROOT Hips\n{\n OFFSET 0 0 0\n CHANNELS 6 "
                   "Xposition Yposition Zposition Zrotation Xrotation Yrotation\n}\n",
                   encoding="utf-8")
    with pytest.raises(ValueError, match="MOTION"):
        parse_bvh(bad)


# --------------------------------------------------------------------------
# 4. FZMotion 点级子集
# --------------------------------------------------------------------------

def test_fzmotion_points_nan_and_names(tmp_path):
    csv = tmp_path / "episode_000.csv"
    # 帧 1（t=0.0）：3 marker 齐；帧 2（t=0.0333）：marker 1 被遮挡（NaN），
    # 且最后一行 timestamp 与前一行重复（盖章粗化样本）
    csv.write_text(
        "timestamp,marker_id,x,y,z\n"
        "0.0,0,0.10,0.90,0.00\n"
        "0.0,1,0.20,0.90,0.00\n"
        "0.0,2,0.30,0.90,0.00\n"
        "0.0333,0,0.11,0.90,0.01\n"
        "0.0333,1,,,\n"
        "0.0333,2,0.31,0.90,0.01\n",
        encoding="utf-8",
    )
    clip = load_fzmotion_points(csv)
    assert clip.joint_names == ("marker_000", "marker_001", "marker_002")
    assert clip.rotations is None and clip.parents is None
    assert validate_clip(clip) == []
    # 被遮挡 marker：NaN 原样保留（不插值、不裁决——交给 G1 规则）
    assert np.isnan(clip.positions[1, 1]).all()
    assert np.isfinite(clip.positions[0]).all()
    assert clip.source["frames_with_repeated_stamp"] == 0


# --------------------------------------------------------------------------
# 5. 注入器的地面真值
# --------------------------------------------------------------------------

def test_inject_drift_ground_truth():
    clip = make_clean_clip(seconds=10.0)
    drifted, manifest = inject_drift(clip, rate=0.01, axis="x")
    assert validate_clip(drifted) == []
    expect = manifest["total_offset"]
    root_shift = drifted.positions[-1, 0, 0] - clip.positions[-1, 0, 0]
    assert abs(root_shift - expect) < 1e-9
    # 刚体平移：骨长不变——这正是"骨长规则抓不到全局漂移"的文档化证据
    for bone, seq in bone_lengths(drifted).items():
        assert seq.max() - seq.min() < 1e-9


def test_inject_frame_drop_ground_truth():
    clip = make_clean_clip(seconds=5.0)
    dropped, manifest = inject_frame_drop(clip, start=40, n=3)
    assert clip.n_frames - dropped.n_frames == 3
    assert validate_clip(dropped) == []
    # 缺口在新索引 39→40（原 39 直接跳到原 43）
    gap = dropped.timestamps[40] - dropped.timestamps[39]
    assert gap > 2 * np.median(np.diff(dropped.timestamps)), "缺口应远大于中位步长"
    assert abs(gap - manifest["gap_s"]) < 1e-9


def test_inject_duplicate_stamp_ground_truth():
    clip = make_clean_clip(seconds=5.0)
    stamped, manifest = inject_duplicate_stamp(clip)
    assert manifest["repeat_count"] == (clip.n_frames - 1 + 1) // 2
    # 重复盖章是质检对象，不是结构错误——validate 必须放行（单调不减合法）
    assert validate_clip(stamped) == []
    assert (np.diff(stamped.timestamps) == 0).sum() == manifest["repeat_count"]


def test_inject_joint_limit_ground_truth():
    clip = make_clean_clip(seconds=5.0)
    j = clip.joint_names.index("ShankL")
    before = max(quat_angle(tuple(clip.rotations[f, j])) for f in range(clip.n_frames))
    limited, manifest = inject_joint_limit(clip, joint="ShankL",
                                           axis="X", extra_deg=80.0)
    after = max(quat_angle(tuple(limited.rotations[f, j])) for f in range(limited.n_frames))
    assert validate_clip(limited) == []
    assert after > before + math.radians(manifest["extra_deg"]) * 0.9, \
        "附加旋转后关节角应显著增大（超限）"
    assert manifest["angle_after_max_deg"] > 90.0  # 已越过人体膝/踝合理区间
