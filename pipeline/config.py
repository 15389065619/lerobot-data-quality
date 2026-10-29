"""质检策略配置。

真实业务里阈值不是 CLI 参数，而是按机器人型号/采集批次维护的配置文件：
不同机械臂的关节单位（弧度 vs 米）、不同相机的曝光特性，都不该共用一套硬编码常量。

配置结构（JSON）：
    {
      "version": "1.1",
      "defaults": { ... },              # 所有机器人共用的基线
      "robots": {"so100_follower": {...}},  # 按 robot_type 覆盖（浅层 key 级 merge）
      "severity": {"nan_state": "error"}
    }

merge 规则：defaults <- robots[robot_type] <- 调用方显式覆盖（CLI）。
"""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Any

LOG = logging.getLogger(__name__)

CONFIG_VERSION = "1.1"

#: 基线配置。auto 模式表示「按数据集自身分布推导」，floor/ceiling 只是安全边界。
DEFAULT_CONFIG: dict[str, Any] = {
    "version": CONFIG_VERSION,
    "defaults": {
        # 自适应帧数区间：median ± mad_k * 1.4826 * MAD，再夹到 [floor, ceiling]
        "frame_count": {"mode": "auto", "mad_k_low": 3.0, "mad_k_high": 8.0, "floor": 8, "ceiling": 20000},
        # action 跳变：逐维阈值 = max(abs_floor, median|Δ| + mad_k * 1.4826 * MAD|Δ|)
        "action_jump": {"abs_floor": 1.0, "mad_k": 12.0},
        # 时间戳：允许的单步偏差比例（掉帧/抖动容忍）
        "timestamp": {"step_tolerance_ratio": 0.25},
        # 视频帧对齐容差（浮点累积误差，单位：帧）
        "video_frame_tolerance": 1,
        # 空转检测：整条 episode 的 state/action 标准差低于阈值即视为无效采集
        "static_episode": {"enabled": True, "state_std_threshold": 1e-3, "min_frames_checked": 5},
        # 黑屏/遮挡：抽帧算灰度标准差
        "blank_video": {"enabled": True, "sample_points": [0.15, 0.35, 0.55, 0.75, 0.9],
                        "lum_std_threshold": 4.0, "bad_ratio_threshold": 0.6, "max_probes": 400},
        # 重复采样：action 序列量化后做指纹
        "duplicate_episode": {"enabled": True, "quantize_decimals": 3},
        # stats.json 时效性：与实测均值的相对偏差
        "stale_stats": {"enabled": True, "mean_rel_tolerance": 0.05, "sample_episodes": 200},
        # 边界裁剪：NaN 只出现在首尾时优先修而不删
        "boundary_trim": {"enabled": True, "max_ratio": 0.25},
        # 视频探针；max_workers 是并发上限：机械盘建议 2，SSD 可到 4-8
        "probe": {"precise": False, "max_cameras": 3, "small_file_mb_for_precise": 64, "max_workers": 2},
        # 流式模式：推导 action 阈值时采样的 episode 数（大数据集不必全量扫两遍配阈值）
        "streaming": {"action_sample_episodes": 200},
        # 关节限位（可选，未配置则跳过；数组顺序与 observation.state 维度一致）
        "joint_limits": {"enabled": False, "lower": None, "upper": None, "tolerance": 1e-6},
    },
    "robots": {},
    "severity": {
        "nan_state": "error",
        "nan_action": "error",
        "action_jump": "warning",
        "frame_count_abnormal": "error",
        "timestamp_not_monotonic": "warning",
        "timestamp_step_deviation": "info",
        "frame_mismatch": "error",
        "meta_index_inconsistent": "warning",
        "missing_task": "warning",
        "static_episode": "error",
        "blank_video": "error",
        "duplicate_episode": "error",
        "joint_limit_violation": "error",
        # 数据集级问题
        "stats_stale": "warning",
    },
}


def load_config(
    path: str | Path | None = None,
    robot_type: str | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """加载并合并配置：默认值 <- 配置文件 <- 机器人覆盖 <- 显式覆盖。"""
    cfg = copy.deepcopy(DEFAULT_CONFIG)

    if path:
        user_path = Path(path).expanduser()
        if not user_path.is_file():
            raise FileNotFoundError(f"配置文件不存在: {user_path}")
        with user_path.open("r", encoding="utf-8") as fh:
            user_cfg = json.load(fh)
        cfg = _deep_merge(cfg, user_cfg)
        LOG.info("已加载配置 %s", user_path)

    sections: dict[str, Any] = {}
    if robot_type:
        sections = (cfg.get("robots") or {}).get(robot_type) or {}
        if sections:
            LOG.info("应用 robot_type=%s 的专用阈值", robot_type)
    effective = _deep_merge(cfg.get("defaults", {}), sections)

    if overrides:
        effective = _deep_merge(effective, {k: v for k, v in overrides.items() if v is not None})

    return {
        "version": cfg.get("version", CONFIG_VERSION),
        "robot_type": robot_type,
        "defaults": effective,
        "severity": cfg.get("severity", {}),
        "source_file": str(path) if path else None,
    }


def cfg_get(config: dict[str, Any], dotted: str, default: Any = None) -> Any:
    """按 'a.b.c' 取值，取不到返回 default。"""
    node: Any = config.get("defaults", config)
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node if node is not None else default


def severity_of(config: dict[str, Any], code: str) -> str:
    return (config.get("severity") or {}).get(code, "warning")


def snapshot(config: dict[str, Any]) -> dict[str, Any]:
    """写进报告用的配置快照，保证结论可追溯。"""
    return {
        "version": config.get("version"),
        "robot_type": config.get("robot_type"),
        "source_file": config.get("source_file"),
        "effective": config.get("defaults", {}),
        "severity": config.get("severity", {}),
    }


def write_default_config(path: str | Path) -> Path:
    """导出一份可直接改的配置模板。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        json.dump(DEFAULT_CONFIG, fh, ensure_ascii=False, indent=2)
    return p


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out
