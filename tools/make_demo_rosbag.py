"""生成一份演示用 ROS1 Bag（含注入异常），用于验证 RosbagAdapter 端到端行为。

依赖可选库 rosbags：pip install lerobot-data-quality[rosbag]

生成内容：
  - /joint_states            sensor_msgs/msg/JointState，100Hz，6 关节正弦运动
  - /camera/color/image_raw  sensor_msgs/msg/Image，30Hz（+7ms 相位偏移），64×48 rgb8

注入异常（与 RosbagAdapter / 下游规则一一对应）：
  1. 两条相邻关节消息的 header 时间戳互换 → 乱序（加载层排序恢复，留 warning）
  2. 相机在 ~1.5s 处连丢 4 帧（约 133ms 缺口，超出默认 50ms 对齐容差）→
     该时段的 camera_timestamp 为 NaN（缺口不插值，交给下游规则）
  3. 一条 header 时间戳重复的关节消息 → 加载层去重（保留首条，留 warning）

用法：
    python tools/make_demo_rosbag.py                      # 写 examples/rosbag_sim/episode_000.bag
    python tools/make_demo_rosbag.py --out /tmp/x.bag --duration 1.0
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

LOG = logging.getLogger(__name__)

JOINT_NAMES = [f"joint_{i}" for i in range(6)]
IMG_W, IMG_H = 64, 48
GAP_CENTER_S = 1.5          # 相机丢帧中心（秒）
SHUFFLE_FRAME = 50          # 乱序注入位置（第 50/51 帧互换 header 时间戳）
DUP_FRAME = 200             # 重复时间戳注入位置


def _require_rosbags():
    try:
        from rosbags.rosbag1 import Writer  # noqa: PLC0415
        from rosbags.typesys import Stores, get_typestore  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "生成演示 bag 需要可选依赖 rosbags：pip install lerobot-data-quality[rosbag]"
        ) from exc
    return Writer, get_typestore(Stores.ROS1_NOETIC)


def _image_bytes(t_s: float) -> np.ndarray:
    """64×48 rgb8：黑底 + 随时间水平移动的 8×8 亮块（仅需数据合法，内容不重要）。"""
    img = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
    x = int((t_s * 20) % (IMG_W - 8))
    img[20:28, x:x + 8] = (255, 255, 255)
    return img.reshape(-1)  # step = width*3，data 为按行展开的字节流


def make_demo_rosbag(
    path: str | Path,
    *,
    duration_s: float = 3.0,
    joint_hz: int = 100,
    image_hz: int = 30,
    inject: bool = True,
) -> dict:
    """生成演示 bag，返回统计信息 dict（供 CLI 打印与单测断言复用）。

    Args:
        path: 输出 .bag 路径（父目录自动创建）。
        duration_s: 时长（秒）。
        joint_hz: 关节状态频率。
        image_hz: 相机频率（相位偏移 +7ms，制造跨频率对齐场景）。
        inject: 是否注入上述 3 类异常（False 生成完全干净的 bag）。
    """
    Writer, typestore = _require_rosbags()
    JS = typestore.types["sensor_msgs/msg/JointState"]
    IMG = typestore.types["sensor_msgs/msg/Image"]
    Header = typestore.types["std_msgs/msg/Header"]
    Time = typestore.types["builtin_interfaces/msg/Time"]

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)  # rosbag1.Writer 拒绝覆盖已存在文件，重复生成前先删

    n_joints = int(duration_s * joint_hz)
    n_images = int(duration_s * image_hz)
    joint_period_ns = int(1e9 / joint_hz)
    image_period_ns = int(1e9 / image_hz)
    t0_ns = 1_700_000_000_000_000_000  # 一个固定的「真实采集」时刻，避免 0 附近边界

    # 相机丢帧：缺口中心前后共 4 帧（30Hz 下约 133ms，超出 50ms 对齐容差 → NaN）
    gap_frames = ({int(GAP_CENTER_S * image_hz) + d for d in (-2, -1, 0, 1)}
                  if inject else set())
    stats = {"joint_written": n_joints, "image_written": n_images - len(gap_frames),
             "shuffled": 0, "duplicated": 0}

    with Writer(path) as writer:
        conn_js = writer.add_connection(
            "/joint_states", "sensor_msgs/msg/JointState", typestore=typestore)
        conn_img = writer.add_connection(
            "/camera/color/image_raw", "sensor_msgs/msg/Image", typestore=typestore)

        # 关节消息先算好全部 header 时间戳，注入「互换两条」的乱序
        stamps = [t0_ns + i * joint_period_ns for i in range(n_joints)]
        if inject and n_joints > SHUFFLE_FRAME + 1:
            stamps[SHUFFLE_FRAME], stamps[SHUFFLE_FRAME + 1] = (
                stamps[SHUFFLE_FRAME + 1], stamps[SHUFFLE_FRAME])
            stats["shuffled"] = 2

        def write_joint(seq: int, stamp_ns: int, bag_ts_ns: int) -> None:
            t_s = (stamp_ns - t0_ns) / 1e9
            pos = np.sin(2 * np.pi * 0.5 * t_s + np.arange(6) * 0.3)
            msg = JS(
                Header(seq=np.uint32(seq), stamp=Time(sec=int(stamp_ns // 10**9),
                                                      nanosec=int(stamp_ns % 10**9)),
                       frame_id="arm"),
                JOINT_NAMES,
                pos.astype(np.float64),
                np.zeros(6, dtype=np.float64),
                np.zeros(6, dtype=np.float64),
            )
            writer.write(conn_js, bag_ts_ns,
                         typestore.serialize_ros1(msg, "sensor_msgs/msg/JointState"))

        for i in range(n_joints):
            write_joint(seq=i, stamp_ns=stamps[i], bag_ts_ns=t0_ns + i * joint_period_ns)
            # 重复发布：同一 header 时间戳再来一条（bag 记录时间 +1ns 保持写入单调）
            if inject and i == DUP_FRAME and n_joints > DUP_FRAME:
                write_joint(seq=i, stamp_ns=stamps[i],
                            bag_ts_ns=t0_ns + i * joint_period_ns + 1)
                stats["duplicated"] = 1

        for k in range(n_images):
            if k in gap_frames:
                continue
            stamp_ns = t0_ns + k * image_period_ns + 7_000_000  # +7ms 相位偏移
            msg = IMG(
                Header(seq=np.uint32(k), stamp=Time(sec=int(stamp_ns // 10**9),
                                                    nanosec=int(stamp_ns % 10**9)),
                       frame_id="camera_color_optical_frame"),
                IMG_H, IMG_W, "rgb8", np.uint8(0), IMG_W * 3, _image_bytes(stamp_ns / 1e9),
            )
            writer.write(conn_img, stamp_ns,
                         typestore.serialize_ros1(msg, "sensor_msgs/msg/Image"))

    stats.update(path=str(path), duration_s=duration_s,
                 joint_hz=joint_hz, image_hz=image_hz, inject=inject)
    LOG.info("写出 %s：关节 %d 条（乱序 %d、重复 %d）、图像 %d 条（丢帧 %d）",
             path, n_joints, stats["shuffled"], stats["duplicated"],
             stats["image_written"], len(gap_frames))
    return stats


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="生成演示用 ROS1 Bag（含注入异常）")
    parser.add_argument("--out", default="examples/rosbag_sim/episode_000.bag")
    parser.add_argument("--duration", type=float, default=3.0, help="时长（秒）")
    parser.add_argument("--clean", action="store_true", help="不注入异常，生成干净 bag")
    args = parser.parse_args()
    stats = make_demo_rosbag(args.out, duration_s=args.duration, inject=not args.clean)
    print(stats)


if __name__ == "__main__":
    main()
