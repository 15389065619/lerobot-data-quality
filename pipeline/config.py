"""质检策略配置。

真实业务里阈值不是 CLI 参数，而是按机器人型号/采集批次维护的配置文件：
不同机械臂的关节单位（弧度 vs 米）、不同相机的曝光特性，都不该共用一套硬编码常量。

配置结构（YAML/JSON，按扩展名识别）：
    version: "1.1"
    defaults: { ... }                 # 所有机器人共用的基线
    robots: {so100_follower: {...}}   # 按 robot_type 覆盖（深层 merge）
    severity: {nan_state: error}
    baseline_whitelist: [<fingerprint>, ...]   # 已确认的已知误报（见 baseline.py）

merge 规则：代码内置 DEFAULT_CONFIG <- 配置文件 <- robots[robot_type] <- 调用方显式覆盖（CLI）。
配置文件缺失或某个键缺失时，自动回退到代码里的默认值（cfg_get 的 default 参数），
因此业务方只需在 YAML 里写「与默认值不同」的那几个键。
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
        # min/max 为业务硬约束（单位：帧）：任一非空即进入 manual 模式，显式值优先，
        # 未指定的一侧仍按数据集分布推导。
        "frame_count": {
            "mode": "auto", "mad_k_low": 3.0, "mad_k_high": 8.0,
            "floor": 8, "ceiling": 20000, "min": None, "max": None,
        },
        # action 跳变：逐维阈值 = max(abs_floor, median(每集最大|Δ|) + k * 1.4826 * MAD(每集最大|Δ|))
        # 以「每集峰值」而非逐帧 |Δ| 建模，避免步进保持型 action 的 median 被大量 0 拉崩
        # min_episodes_for_peak：episode 数不足时退回逐帧非零 |Δ| 的稳健尺度
        # discrete_max_unique：取值种类 ≤ 该值的维视为二值/离散（如夹爪 ±1），不按连续跳变判定
        "action_jump": {
            "abs_floor": 1.0, "max_jump_mad_k": 3.0,
            "min_episodes_for_peak": 8, "min_margin_ratio": 0.05,
            # 超阈值后还需相对邻域突跳（连续快动作的峰值通常是邻域的 1~2.5 倍）
            "spike_ratio": 3.0, "spike_window": 3,
            # abs_floor 按 ref_fps 标定，实际使用时按数据集 fps 折算（每秒位移等价）
            "ref_fps": 30.0,
            # 四元数 q≡-q：同一帧多个维同时符号翻转时视为表示等价
            "sign_flip_tol": 0.25, "sign_flip_min_dims": 2,
            "discrete_max_unique": 3,
            # 超阈幅度分级：阈值本身在分布尾部，1.0~1.5 倍的命中多为正常快动作
            "downgrade_ratio": 1.5, "escalate_ratio": None,
        },
        # state 突跳（软规则）：判据同 action_jump，但只提示复核，永不自动剔除
        "state_jump": {
            "enabled": True, "abs_floor": 1.0, "max_jump_mad_k": 3.0,
            "min_episodes_for_peak": 8, "min_margin_ratio": 0.05,
            "spike_ratio": 3.0, "spike_window": 3, "discrete_max_unique": 3,
        },
        # 时间戳：允许的单步偏差比例（掉帧/抖动容忍）
        "timestamp": {"step_tolerance_ratio": 0.25},
        # 视频帧对齐容差（浮点累积误差，单位：帧）
        "video_frame_tolerance": 1,
        # 多模态对齐：全局时间戳相对视频 from/to 的偏移容忍、视频窗与 episode 时长的相对偏差容忍
        "alignment": {"enabled": True, "timestamp_drift_frames": 2.0, "video_span_tolerance_ratio": 0.05},
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
        # 3D 点云清洗：阈值按点云自身尺度自适应，绝对坐标只作为兜底
        "pointcloud": {
            "feature_key": None,
            "batch_size": 256,
            "sample_frames": 200,
            "min_points": 16,
            "sparse_ratio": 0.02,
            "degenerate_diag": 1e-3,
            "duplicate_decimals": 4,
            "max_duplicate_ratio": 0.05,
            "outlier_radius_k": 2.5,
            "outlier_min_neighbors": 2,
            "max_outlier_ratio": 0.10,
            "workspace": {"mode": "auto", "margin_ratio": 0.10,
                          "max_outside_ratio": 0.05, "lower": None, "upper": None},
            "color": {"max_bad_ratio": 0.01},
            "temporal": {"abs_jump": 0.05, "mad_k": 6.0},
            "max_repair_ratio": 0.35,
            "jitter_ratio": 0.005,
        },
    },
    "robots": {},
    # 基线白名单：md5(dataset_name|episode_index|rule_code) 的已知误报，
    # 命中后该告警被摘除，不再影响健康度（见 pipeline/baseline.py）
    "baseline_whitelist": [],
    "severity": {
        "nan_state": "error",
        "nan_action": "error",
        "action_jump": "warning",
        "state_jump": "warning",
        "frame_count_abnormal": "error",
        "timestamp_not_monotonic": "warning",
        "timestamp_step_deviation": "info",
        "timestamp_drift": "warning",
        "frame_index_gap": "warning",
        "video_span_mismatch": "warning",
        "frame_mismatch": "error",
        "video_missing": "error",
        "meta_index_inconsistent": "warning",
        "missing_task": "warning",
        "static_episode": "error",
        "blank_video": "error",
        "duplicate_episode": "error",
        "joint_limit_violation": "error",
        "episode_read_error": "error",
        # 数据集级问题
        "stats_stale": "warning",
        "meta_episode_mismatch": "error",
        # 点云规则
        "pc_non_finite": "error",
        "pc_empty_or_sparse": "error",
        "pc_degenerate": "error",
        "pc_duplicate": "warning",
        "pc_outliers": "warning",
        "pc_out_of_bounds": "error",
        "pc_color_invalid": "warning",
        "pc_temporal_jump": "warning",
    },
}


def _load_config_file(user_path: Path) -> dict[str, Any]:
    """按扩展名解析配置文件：.yaml/.yml 走 PyYAML，.json 走标准库。

    YAML 是 JSON 的超集且支持注释，适合给人维护的阈值文件；
    JSON 保留给需要程序生成/ diff 的场景。未知扩展名按 YAML 解析。
    """
    text = user_path.read_text(encoding="utf-8")
    suffix = user_path.suffix.lower()
    if suffix == ".json":
        data = json.loads(text)
    else:
        import yaml  # 延迟导入：只用代码默认值时不强求装 PyYAML

        data = yaml.safe_load(text)
    if data is None:  # 空文件 / 只有注释：视为「全部用默认值」
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"配置文件顶层必须是键值映射: {user_path}")
    return data


def load_config(
    path: str | Path | None = None,
    robot_type: str | None = None,
    overrides: dict[str, Any] | None = None,
    strict: bool = False,
) -> dict[str, Any]:
    """加载并合并配置：代码默认值 <- 配置文件(YAML/JSON) <- 机器人覆盖 <- 显式覆盖。

    strict=True 时对配置文件做结构校验（未知键、类型不匹配、未知 severity 规则
    直接报错），避免 `abs_floo` 这类拼写错误被 cfg_get 静默回退成默认值。
    """
    cfg = copy.deepcopy(DEFAULT_CONFIG)

    if path:
        user_path = Path(path).expanduser()
        if not user_path.is_file():
            raise FileNotFoundError(f"配置文件不存在: {user_path}")
        raw = _load_config_file(user_path)
        if strict:
            errors = validate_config(raw, source=str(user_path))
            if errors:
                raise ValueError("配置校验失败：\n- " + "\n- ".join(errors))
        cfg = _deep_merge(cfg, raw)
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
        "baseline_whitelist": list(cfg.get("baseline_whitelist") or []),
        "source_file": str(path) if path else None,
    }


#: 配置文件允许出现的顶层键。拼错的顶层键（如 `robot`）不会报错，只会被静默忽略。
_TOP_LEVEL_KEYS = {"version", "defaults", "robots", "severity", "baseline_whitelist"}
#: severity 允许的取值；写错（如 `err`）会让这条规则的严重度永远不生效。
_ALLOWED_SEVERITY = {"error", "warning", "info"}


def validate_config(data: Any, *, source: str = "<config>") -> list[str]:
    """校验配置文件的结构与类型，返回错误列表（空列表=通过）。

    只做「结构 + 类型」检查，不做取值合理性判断（例如阈值是否过大由下游自适应）：
    - 未知键：`abs_floo` 这类拼写错误会被 cfg_get 静默回退成默认值，必须报错；
    - 类型错误：`"3.0"` 写进数值字段会在下游某处才炸，提前到加载时暴露；
    - 未知 severity 规则：规则名写错时该设置永远不会生效，属于静默失效。
    """
    errors: list[str] = []
    if not isinstance(data, dict):
        return [f"{source}: 顶层必须是键值映射，实际是 {_type_name(data)}"]

    for key in data:
        if key not in _TOP_LEVEL_KEYS:
            errors.append(
                f"{source}: 未知顶层配置项 `{key}`（可用: {', '.join(sorted(_TOP_LEVEL_KEYS))}）"
            )

    if "version" in data and not isinstance(data["version"], (str, int, float)):
        errors.append(f"{source}: version 应为字符串，实际是 {_type_name(data['version'])}")

    template = DEFAULT_CONFIG["defaults"]
    if "defaults" in data:
        if not isinstance(data["defaults"], dict):
            errors.append(f"{source}: defaults 应为键值映射，实际是 {_type_name(data['defaults'])}")
        else:
            _validate_mapping(data["defaults"], template, "defaults", source, errors)

    if "robots" in data:
        robots = data["robots"]
        if not isinstance(robots, dict):
            errors.append(f"{source}: robots 应为键值映射，实际是 {_type_name(robots)}")
        else:
            for name, section in robots.items():
                prefix = f"robots.{name}"
                if not isinstance(section, dict):
                    errors.append(f"{source}: {prefix} 应为键值映射，实际是 {_type_name(section)}")
                    continue
                _validate_mapping(section, template, prefix, source, errors)

    if "severity" in data:
        severity = data["severity"]
        if not isinstance(severity, dict):
            errors.append(f"{source}: severity 应为键值映射，实际是 {_type_name(severity)}")
        else:
            known = DEFAULT_CONFIG["severity"]
            for code, level in severity.items():
                if code not in known:
                    errors.append(f"{source}: severity 中的未知规则 `{code}`")
                elif level not in _ALLOWED_SEVERITY:
                    errors.append(
                        f"{source}: severity.{code} 应为 {'/'.join(sorted(_ALLOWED_SEVERITY))}，"
                        f"实际是 {level!r}"
                    )

    if "baseline_whitelist" in data:
        whitelist = data["baseline_whitelist"]
        if not isinstance(whitelist, list):
            errors.append(f"{source}: baseline_whitelist 应为列表，实际是 {_type_name(whitelist)}")
        else:
            for index, item in enumerate(whitelist):
                if not isinstance(item, str):
                    errors.append(
                        f"{source}: baseline_whitelist[{index}] 应为字符串，实际是 {_type_name(item)}"
                    )

    return errors


def _validate_mapping(
    section: dict[str, Any],
    template: dict[str, Any],
    prefix: str,
    source: str,
    errors: list[str],
) -> None:
    """按模板递归校验一层配置：未知键、嵌套结构、标量类型。"""
    for key, value in section.items():
        path = f"{prefix}.{key}"
        if key not in template:
            errors.append(f"{source}: 未知配置项 `{path}`")
            continue
        expected = template[key]
        if isinstance(expected, dict):
            if not isinstance(value, dict):
                errors.append(f"{source}: {path} 应为键值映射，实际是 {_type_name(value)}")
                continue
            _validate_mapping(value, expected, path, source, errors)
            continue
        # 模板为 None 表示「默认不启用该选项」（如 joint_limits.lower），无从判断类型
        if expected is None:
            continue
        if isinstance(expected, list):
            if not isinstance(value, list):
                errors.append(f"{source}: {path} 应为列表，实际是 {_type_name(value)}")
            continue
        if not _same_scalar_type(value, expected):
            errors.append(f"{source}: {path} 应为 {_type_name(expected)}，实际是 {_type_name(value)}")


def _same_scalar_type(value: Any, expected: Any) -> bool:
    if isinstance(expected, bool):
        return isinstance(value, bool)
    if isinstance(expected, (int, float)):
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if isinstance(expected, str):
        return isinstance(value, str)
    return True


def _type_name(value: Any) -> str:
    if isinstance(value, bool):
        return "布尔值"
    if isinstance(value, (int, float)):
        return "数值"
    if isinstance(value, str):
        return "字符串"
    if isinstance(value, list):
        return "列表"
    if isinstance(value, dict):
        return "键值映射"
    return "null" if value is None else type(value).__name__


def read_config_file(path: str | Path) -> dict[str, Any]:
    """读取原始配置文件（不含默认值合并），供写回白名单等场景使用。"""
    p = Path(path).expanduser()
    if not p.is_file():
        raise FileNotFoundError(f"配置文件不存在: {p}")
    return _load_config_file(p)


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
        "baseline_whitelist": list(config.get("baseline_whitelist") or []),
    }


def write_default_config(path: str | Path) -> Path:
    """导出一份可直接改的配置模板（.yaml/.yml 输出 YAML，其余输出 JSON）。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix.lower() in (".yaml", ".yml"):
        import yaml

        p.write_text(yaml.safe_dump(DEFAULT_CONFIG, allow_unicode=True, sort_keys=False),
                     encoding="utf-8")
    else:
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
