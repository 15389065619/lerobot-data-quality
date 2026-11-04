"""ROS Bag 加载器端到端测试：真实 bag 生成 → 解析全链路。

需要可选依赖 rosbags（pip install lerobot-data-quality[rosbag]）；
未安装时整文件跳过，不影响其余纯函数测试。
"""

import numpy as np
import pytest

pytest.importorskip("rosbags", reason="未安装可选依赖 rosbags")

from lerobot_data_quality.loaders import RosbagAdapter  # noqa: E402
from tools.make_demo_rosbag import make_demo_rosbag  # noqa: E402


def test_end_to_end_demo_bag(tmp_path):
    """2.5s 演示 bag（乱序 + 重复 + 相机丢帧）→ 三类异常全被正确处置。"""
    bag = tmp_path / "demo.bag"
    stats = make_demo_rosbag(bag, duration_s=2.5)
    assert (stats["shuffled"], stats["duplicated"]) == (2, 1)
    assert stats["image_written"] == 75 - 4      # 75 帧相机丢 4 帧

    df = RosbagAdapter(bag).to_dataframe()
    assert len(df) == 250                        # 251 条关节消息去掉 1 条重复
    assert df["timestamp"].is_monotonic_increasing   # 乱序已排序恢复
    assert len(df["observation.state"].iloc[0]) == 6
    # action = 下一帧 state
    np.testing.assert_allclose(df["action"].iloc[0], df["observation.state"].iloc[1])
    # 相机缺口（~1.5s，133ms > 50ms 容差）→ 该时段 camera_timestamp 为 NaN
    nan = df["camera_timestamp"].isna()
    assert nan.sum() > 0
    assert df.loc[nan, "timestamp"].between(1.4, 1.6).all()


def test_end_to_end_clean_bag(tmp_path):
    """干净 bag：零 NaN、帧数精确、时间轴单调。"""
    bag = tmp_path / "clean.bag"
    make_demo_rosbag(bag, duration_s=1.0, inject=False)
    df = RosbagAdapter(bag).to_dataframe()
    assert len(df) == 100
    assert df["camera_timestamp"].notna().all()
    assert df["timestamp"].is_monotonic_increasing
