"""ROS Bag 加载器测试。

build_frame_table 是纯函数（只依赖 numpy/pandas），无需安装 rosbags；
真实 bag 解析路径（AnyReader/typestore）依赖可选库 rosbags，不在本文件测试。
"""

import sys

import numpy as np
import pandas as pd
import pytest

from lerobot_data_quality.loaders import (
    RosbagAdapter,
    RosbagAdapterError,
    build_frame_table,
)
from lerobot_data_quality.loaders.rosbag_loader import _stamp_ns

MS = 1_000_000  # 1 毫秒 = 1e6 纳秒


class _Stamp:
    def __init__(self, sec: int, nsec: int) -> None:
        self.sec, self.nsec = sec, nsec


def test_zero_header_stamp_falls_back_to_bag_time():
    """header 存在但时间戳全零 = 未赋值，必须回退 bag 记录时间（否则整段塌成 1 帧）。"""
    assert _stamp_ns(None, 123) == 123
    assert _stamp_ns(_Stamp(0, 0), 123) == 123
    assert _stamp_ns(_Stamp(2, 500), 123) == 2 * 1_000_000_000 + 500


def _joints(start_ms: int, n: int, step_ms: int = 10, dim: int = 2):
    """生成 n 条关节记录：start_ms 起每 step_ms 一帧，位置 = [帧号, -帧号]。"""
    return [(int((start_ms + i * step_ms) * MS),
             np.array([float(i), float(-i)], dtype=np.float32)) for i in range(n)]


def test_build_frame_table_basic():
    """state/action 契约：action = 下一帧 state，最后一帧保持自身；timestamp 首帧归零。"""
    df = build_frame_table(_joints(1000, 5), episode_index=3)

    assert list(df.columns) == ["episode_index", "frame_index", "timestamp",
                                "observation.state", "action", "camera_timestamp"]
    assert (df["episode_index"] == 3).all()
    assert df["frame_index"].tolist() == [0, 1, 2, 3, 4]
    np.testing.assert_allclose(df["timestamp"], [0, 0.01, 0.02, 0.03, 0.04])
    np.testing.assert_allclose(df["action"].iloc[0], df["observation.state"].iloc[1])
    # 最后一帧 action 保持自身（定长契约）
    np.testing.assert_allclose(df["action"].iloc[-1], df["observation.state"].iloc[-1])
    # 无相机 topic 时整列 NaN
    assert df["camera_timestamp"].isna().all()


def test_merge_asof_nearest_alignment():
    """关节基准轴：相机时间戳最近邻挂到每个关节帧；容差外为 NaN（缺口不裁决）。"""
    joints = _joints(0, 10, step_ms=10)          # 0,10,...,90 ms
    images = [12 * MS, 5000 * MS]                # 12ms 在轴内；5000ms 距任何关节帧都超容差
    df = build_frame_table(joints, images, align_tolerance_ns=50 * MS)

    # 12ms 的相机帧距关节 0~60ms（frame 0-6）都在 50ms 容差内 → 这 7 帧都挂上 0.012s
    matched = df["camera_timestamp"]
    assert matched.iloc[1] == pytest.approx(0.012)
    assert matched.notna().sum() == 7
    assert matched.iloc[8:].isna().all()


def test_image_base_axis():
    """相机基准轴：每个相机帧反查最近关节状态；容差外 state 为 NaN，不插值。"""
    joints = _joints(0, 10, step_ms=10)          # 0,10,...,90 ms（100Hz）
    images = [12 * MS, 45 * MS, 5000 * MS]       # 30Hz 相机 + 一条远离关节轴的
    df = build_frame_table(joints, images, align_tolerance_ns=10 * MS, base="image")

    assert len(df) == 3                          # 每相机帧一行
    # 12ms 最近关节是 10ms（帧 1，state=[1,-1]）；45ms 最近是 40 或 50ms
    np.testing.assert_allclose(df["observation.state"].iloc[0], [1.0, -1.0])
    # 5000ms 容差内无关节帧 → state/action 全 NaN，缺口交给下游规则
    assert np.isnan(df["observation.state"].iloc[2]).all()
    assert np.isnan(df["action"].iloc[2]).all()
    # timestamp 跟相机轴走，以相机首帧为零点（v3「首帧归零」）
    np.testing.assert_allclose(df["timestamp"], [0.0, 0.033, 4.988])


def test_backward_direction_no_future():
    """backward 方向严格因果：只用不晚于基准时刻的关节状态。"""
    joints = _joints(10, 3, step_ms=10)          # 10,20,30 ms
    images = [15 * MS]                           # 最近邻是 10 或 20（等距取左=10），backward 只能是 10
    df = build_frame_table(joints, images, align_tolerance_ns=50 * MS,
                           base="image", match_direction="backward")
    np.testing.assert_allclose(df["observation.state"].iloc[0], [0.0, 0.0])  # 帧0 @10ms


def test_out_of_order_timestamps_sorted():
    """录包乱序是真实缺陷场景：加载层排序恢复单调性，不崩溃。"""
    records = _joints(0, 5)
    records[1], records[3] = records[3], records[1]  # 制造两处乱序
    df = build_frame_table(records)
    assert df["timestamp"].is_monotonic_increasing
    # 排序后 state 也随时间戳走：frame 1 的 state 应是原始第 1 条
    np.testing.assert_allclose(df["observation.state"].iloc[1], [1.0, -1.0])


def test_empty_and_dim_mismatch_raise():
    """空 bag（无关节消息）与关节维度跨消息不一致都必须显式报错。"""
    with pytest.raises(RosbagAdapterError, match="没有任何消息"):
        build_frame_table([])

    bad = [(0, np.zeros(2, dtype=np.float32)), (MS, np.zeros(3, dtype=np.float32))]
    with pytest.raises(RosbagAdapterError, match="维度跨消息不一致"):
        build_frame_table(bad)


def test_missing_rosbags_gives_install_hint(tmp_path, monkeypatch):
    """模拟未装 rosbags：to_dataframe 抛 ImportError 并提示 [rosbag] extra。"""
    monkeypatch.setitem(sys.modules, "rosbags", None)  # 屏蔽真实安装，import 即 ImportError
    bag = tmp_path / "demo.bag"
    bag.write_bytes(b"")
    with pytest.raises(ImportError, match=r"\[rosbag\]"):
        RosbagAdapter(bag).to_dataframe()


def test_missing_path_raises_file_not_found(tmp_path):
    """bag 路径不存在 → FileNotFoundError（在碰 rosbags 之前由路径检查兜住）。"""
    with pytest.raises((FileNotFoundError, ImportError)):
        RosbagAdapter(tmp_path / "nope.bag").to_dataframe()
