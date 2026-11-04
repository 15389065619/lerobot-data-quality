"""ROS Bag 加载器（可选能力，独立于 pipeline/ 主线）。

把机器人最原始的 ROS Bag 文件（rosbag1 `.bag` / rosbag2 目录或 `.db3` / `.mcap`）
解析为与 LeRobot 帧表同构的宽表 DataFrame，列如下：

    episode_index      int            episode 编号（一个 bag = 一个 episode）
    frame_index        int            episode 内从 0 开始的帧号
    timestamp          float          相对时间戳（秒），首帧归零（LeRobot v3 episode 相对时间轴约定）
    observation.state  object(ndarray[float32])  关节位置向量，来自 state_topic
    action             object(ndarray[float32])  定义为下一帧的 state；
                                      最后一帧保持自身（定长契约：action 与 state 同形）
    camera_timestamp   float          最近邻对齐到本帧的相机时间戳（相对秒）；
                                      容差内无匹配帧时为 NaN（无 image_topic 时整列 NaN）

设计约束（与 FZMotion 适配器一致）：
  - 缺陷不在这里裁决：NaN、丢帧、时间戳缺口都原样保留，交给下游质检规则标记；
  - 多传感器时间对齐用 pandas.merge_asof 最近邻匹配，不用 Python for 循环；
  - 只依赖纯 Python 的 rosbags 库，不需要安装 ROS 环境；
  - rosbags 为可选依赖：本模块顶层只 import numpy/pandas，
    仅在实际解析 bag 时才懒加载 rosbags，缺库时报错并给出安装提示。

安装：pip install lerobot-data-quality[rosbag]
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # 仅供 IDE / 类型检查；运行时不引入 rosbags
    from rosbags.typesys.stores.ros2_humble import builtin_interfaces__msg__Time  # noqa: F401

LOG = logging.getLogger(__name__)

DEFAULT_STATE_TOPIC = "/joint_states"
DEFAULT_IMAGE_TOPIC = "/camera/color/image_raw"

# 默认对齐容差（纳秒）：50ms —— 30Hz 相机帧间隔约 33ms，50ms 允许跨一档匹配
DEFAULT_ALIGN_TOLERANCE_NS = 50_000_000


class RosbagAdapterError(RuntimeError):
    """ROS Bag 解析失败（空 bag、缺 topic、消息维度不一致等）。"""


def _require_rosbags() -> Any:
    """懒加载 rosbags.typesys；缺库时给出可选依赖安装提示。"""
    try:
        from rosbags import typesys  # noqa: PLC0415（故意延迟到调用点）
    except ImportError as exc:  # pragma: no cover - 取决于环境是否装 rosbags
        raise ImportError(
            "解析 ROS Bag 需要可选依赖 rosbags：\n"
            "  pip install lerobot-data-quality[rosbag]\n"
            "或 pip install rosbags（纯 Python，无需 ROS 环境）"
        ) from exc
    return typesys


def _stamp_ns(stamp: Any, fallback_ns: int) -> int:
    """从消息 header.stamp 提取纳秒时间戳；字段缺失时回退到 bag 记录时间。

    ROS1 的 Time 字段为 (sec, nsec)，ROS2 为 (sec, nanosec)，两种都兼容。
    """
    sec = getattr(stamp, "sec", None)
    if sec is None:
        return fallback_ns
    nsec = getattr(stamp, "nanosec", getattr(stamp, "nsec", 0))
    return int(sec) * 1_000_000_000 + int(nsec)


def build_frame_table(
    joint_records: Iterable[tuple[int, np.ndarray]],
    image_records: Iterable[int] | None = None,
    *,
    episode_index: int = 0,
    align_tolerance_ns: int | None = DEFAULT_ALIGN_TOLERANCE_NS,
    base: str = "state",
    match_direction: str = "nearest",
) -> pd.DataFrame:
    """把关节/相机采样记录组装为 LeRobot 同构帧表（纯函数，不依赖 rosbags）。

    时间对齐用 pandas.merge_asof，不用 Python for 循环；容差外一律 NaN ——
    缺口原样保留，由下游 timestamp_* / frame_* 规则裁决，加载层不做插值。

    Args:
        joint_records: (时间戳 ns, 关节位置向量) 序列，允许乱序（内部排序并告警）。
        image_records: 相机帧时间戳 ns 序列；None 表示无相机 topic。
        episode_index: 写入输出 episode_index 列的编号。
        align_tolerance_ns: merge_asof 容差；None 表示不限容差。
        base: 基准时间轴。"state"（默认）以关节为轴（控制信号频率高，无信息损失），
            相机时间戳最近邻挂到每帧；"image" 以相机为轴（每相机帧一行，
            关节状态反查匹配，常用于按相机帧率训练的策略）。
        match_direction: merge_asof 方向。"nearest"（默认）最近邻；
            "backward" 只取不晚于基准时刻的样本（严格因果，不用未来状态）。

    Returns:
        每行一个采样帧的 DataFrame（len = 基准轴帧数）。

    Raises:
        RosbagAdapterError: 关节记录为空、关节维度跨消息不一致，或 base/image 参数非法。
    """
    if base not in ("state", "image"):
        raise RosbagAdapterError(f"base 只能是 'state' / 'image'，收到 {base!r}")
    if base == "image" and image_records is None:
        raise RosbagAdapterError("base='image' 需要 image_records（无相机 topic 时请以 state 为轴）")
    joints = list(joint_records)
    if not joints:
        raise RosbagAdapterError("关节状态 topic 没有任何消息（空 bag 或 topic 配错）")

    ts = np.array([t for t, _ in joints], dtype=np.int64)
    dim = len(joints[0][1])
    bad = [i for i, (_, v) in enumerate(joints) if len(v) != dim]
    if bad:
        raise RosbagAdapterError(
            f"关节维度跨消息不一致：首条 {dim} 维，第 {bad[0]} 条 {len(joints[bad[0]][1])} 维"
            "（采集中途改了关节配置，应拆成两个 episode 分别处理）")
    states = np.stack([np.asarray(v, dtype=np.float32) for _, v in joints])

    # 时间戳乱序：采集/录包乱序是真实缺陷场景，排序恢复单调性并留痕
    inversions = int((np.diff(ts) < 0).sum())
    if inversions:
        LOG.warning("关节时间戳存在 %d 处乱序，已按时间戳重新排序", inversions)
        order = np.argsort(ts, kind="stable")
        ts, states = ts[order], states[order]
    # 重复时间戳保留第一条（重复入库交给 duplicate_episode 规则管，加载层不裁决）
    keep = np.concatenate(([True], np.diff(ts) > 0))
    if not keep.all():
        LOG.warning("发现 %d 条重复时间戳的关节消息，保留首条", int((~keep).sum()))
        ts, states = ts[keep], states[keep]

    t0 = ts[0]
    img_ts = (np.sort(np.asarray(list(image_records), dtype=np.int64))
              if image_records is not None else np.empty(0, dtype=np.int64))

    if base == "state":
        # 关节轴为基准：相机时间戳 merge_asof 挂到每个关节帧上
        base_ts, states_out = ts, states
        camera_ts = np.full(len(ts), np.nan, dtype=np.float64)
        if img_ts.size:
            left = pd.DataFrame({"ts": ts})
            right = pd.DataFrame({"ts": img_ts, "img_ts": img_ts})
            merged = pd.merge_asof(left, right, on="ts", direction=match_direction,
                                   tolerance=align_tolerance_ns)
            matched = merged["img_ts"].to_numpy(dtype=np.float64)  # NaN = 容差内无帧
            camera_ts = np.where(np.isnan(matched), np.nan, (matched - t0) / 1e9)
            LOG.info("相机帧 %d 条，容差内对齐 %d / %d 帧关节",
                     img_ts.size, int((~np.isnan(camera_ts)).sum()), len(ts))
    else:
        # 相机轴为基准：每个相机帧反查最近（或不晚于它）的关节状态
        if not img_ts.size:
            raise RosbagAdapterError("相机 topic 没有任何消息，无法以 image 为基准轴")
        left = pd.DataFrame({"ts": img_ts})
        right = pd.DataFrame({"ts": ts, "jidx": np.arange(len(ts))})
        merged = pd.merge_asof(left, right, on="ts", direction=match_direction,
                               tolerance=align_tolerance_ns)
        jidx = merged["jidx"].to_numpy(dtype=np.float64)  # NaN = 容差内无关节帧
        states_out = np.full((len(img_ts), states.shape[1]), np.nan, dtype=np.float32)
        ok = ~np.isnan(jidx)
        states_out[ok] = states[jidx[ok].astype(np.int64)]
        base_ts = img_ts
        camera_ts = (img_ts - t0).astype(np.float64) / 1e9
        LOG.info("相机基准轴 %d 帧，容差内匹配关节状态 %d 帧（其余为 NaN，不插值）",
                 len(img_ts), int(ok.sum()))

    rel_s = (base_ts - t0).astype(np.float64) / 1e9  # 相对秒，首帧归零（v3 约定）

    # action := 下一帧 state；最后一帧保持自身（定长契约，下游无需分支）
    actions = np.empty_like(states_out)
    actions[:-1], actions[-1] = states_out[1:], states_out[-1]

    out = pd.DataFrame({
        "episode_index": int(episode_index),
        "frame_index": np.arange(len(base_ts), dtype=np.int64),
        "timestamp": rel_s,
        "observation.state": [row.copy() for row in states_out],
        "action": [row.copy() for row in actions],
        "camera_timestamp": camera_ts,
    })
    LOG.info("帧表构建完成：%d 帧 × %d 关节，时长 %.3f s（基准轴 %s）",
             len(out), dim, rel_s[-1] if len(rel_s) else 0.0, base)
    return out


class RosbagAdapter:
    """把 ROS Bag 解析为 LeRobot 同构帧表的适配器。

    支持 rosbag1 (.bag)、rosbag2（目录或 .db3）、MCAP，统一走
    rosbags.highlevel.AnyReader；消息类型由 typestore 自动处理，
    不需要在代码里写死任何 msg 定义。

    Args:
        path: bag 路径。.bag/.mcap 传文件；rosbag2 传目录或直接传 .db3 文件
              （自动取父目录，AnyReader 需要读同目录的 metadata.yaml）。
        state_topic: 关节状态 topic（sensor_msgs/msg/JointState）。
        image_topic: 相机 topic（sensor_msgs/msg/Image）；只提取时间戳做对齐，
              设为 None 可跳过。图像解码（rawdata → numpy）留作扩展点，
              不影响帧表结构。
        episode_index: 输出 episode_index 列的编号。
        align_tolerance_ms: 相机对齐容差（毫秒）；None 表示不限容差。
        base: 基准时间轴，"state"（默认，关节轴）或 "image"（相机轴，每相机帧一行）。
        match_direction: merge_asof 方向，"nearest"（默认）或 "backward"（严格因果）。
        ros_version: "ros1" / "ros2"，默认按后缀推断（.bag → ros1，其余 → ros2）。
              MCAP 里装的是 ROS1 序列化数据时需显式传 "ros1"。
        store: 覆盖默认类型库（如 Stores.ROS2_IRON），一般不需要。
    """

    def __init__(
        self,
        path: str | Path,
        *,
        state_topic: str = DEFAULT_STATE_TOPIC,
        image_topic: str | None = DEFAULT_IMAGE_TOPIC,
        episode_index: int = 0,
        align_tolerance_ms: float | None = 50.0,
        base: str = "state",
        match_direction: str = "nearest",
        ros_version: str | None = None,
        store: Any = None,
    ) -> None:
        self.path = Path(path)
        self.state_topic = state_topic
        self.image_topic = image_topic
        self.episode_index = episode_index
        self.align_tolerance_ns = (
            None if align_tolerance_ms is None else int(align_tolerance_ms * 1e6))
        self.base = base
        self.match_direction = match_direction
        self.ros_version = ros_version or ("ros1" if self.path.suffix == ".bag" else "ros2")
        self.store = store

    # -- 内部 ---------------------------------------------------------------

    def _reader_paths(self) -> list[Path]:
        """AnyReader 的路径实参：rosbag2 必须传目录，其余传文件本身。"""
        if not self.path.exists():
            raise FileNotFoundError(f"bag 路径不存在: {self.path}")
        if self.path.is_dir():
            return [self.path]
        if self.path.suffix == ".db3":
            return [self.path.parent]
        return [self.path]

    def _typestore(self) -> Any:
        """按 ros_version 取类型库；store 参数优先（调用方覆盖默认）。"""
        if self.store is not None:
            return self.store
        typesys = _require_rosbags()
        stores = typesys.Stores
        default = stores.ROS1_NOETIC if self.ros_version == "ros1" else stores.ROS2_HUMBLE
        return typesys.get_typestore(default)

    def _deserialize(self, typestore: Any, rawdata: bytes, msgtype: str) -> Any:
        """ROS1 走 rosbag 序列化格式，ROS2 走 CDR。"""
        if self.ros_version == "ros1":
            return typestore.deserialize_ros1(rawdata, msgtype)
        return typestore.deserialize_cdr(rawdata, msgtype)

    # -- 公开 API ------------------------------------------------------------

    def to_dataframe(self) -> pd.DataFrame:
        """解析 bag，返回 LeRobot 同构帧表。

        Raises:
            FileNotFoundError: bag 路径不存在。
            ImportError: 未安装 rosbags（提示装 [rosbag] extra）。
            RosbagAdapterError: 空 bag、缺 topic、关节维度不一致。
        """
        _require_rosbags()  # 先确认依赖，错误信息比 AnyReader 的模糊报错友好
        from rosbags.highlevel import AnyReader  # noqa: PLC0415（懒加载）

        typestore = self._typestore()
        wanted = {self.state_topic} | ({self.image_topic} if self.image_topic else set())
        joint_records: list[tuple[int, np.ndarray]] = []
        image_records: list[int] = []

        with AnyReader(self._reader_paths()) as reader:
            available = {c.topic for c in reader.connections}
            missing = wanted - available
            if missing:
                raise RosbagAdapterError(
                    f"bag 中缺少 topic {sorted(missing)}；实际包含: {sorted(available)}"
                    "（可用 state_topic/image_topic 参数改配）")
            connections = [c for c in reader.connections if c.topic in wanted]
            for connection, bag_ts, rawdata in reader.messages(connections=connections):
                msg = self._deserialize(typestore, rawdata, connection.msgtype)
                stamp = getattr(getattr(msg, "header", None), "stamp", None)
                ts_ns = _stamp_ns(stamp, bag_ts) if stamp is not None else bag_ts
                if connection.topic == self.state_topic:
                    joint_records.append(
                        (ts_ns, np.asarray(msg.position, dtype=np.float32)))
                else:  # 相机帧：只取时间戳（解码 rawdata → numpy 是预留扩展点）
                    image_records.append(ts_ns)

        return build_frame_table(
            joint_records,
            image_records if self.image_topic else None,
            episode_index=self.episode_index,
            align_tolerance_ns=self.align_tolerance_ns,
            base=self.base,
            match_direction=self.match_direction,
        )


def main() -> None:  # pragma: no cover - 调用示例
    """调用示例：python -m lerobot_data_quality.loaders.rosbag_loader <bag> [--out x.parquet]"""
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="把 ROS Bag 解析为 LeRobot 同构帧表")
    parser.add_argument("bag", help=".bag / rosbag2 目录 / .db3 / .mcap")
    parser.add_argument("--state-topic", default=DEFAULT_STATE_TOPIC)
    parser.add_argument("--image-topic", default=DEFAULT_IMAGE_TOPIC)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--out", help="可选：把帧表写成 parquet")
    args = parser.parse_args()

    df = RosbagAdapter(
        args.bag,
        state_topic=args.state_topic,
        image_topic=args.image_topic,
        episode_index=args.episode_index,
    ).to_dataframe()
    print(df.head())
    print(f"共 {len(df)} 帧，state 维度 {len(df['observation.state'].iloc[0])}")
    if args.out:
        df.to_parquet(args.out, index=False)
        print(f"已写出 {args.out}")


if __name__ == "__main__":  # pragma: no cover
    main()
