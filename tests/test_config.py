"""配置化（YAML/JSON 解耦硬编码阈值）测试。

设计约束：
  1. 配置文件支持 YAML（给人维护，可写注释）与 JSON（给程序生成），按扩展名识别；
  2. 配置文件缺失 → FileNotFoundError；文件里缺某个键 → 回退代码内置 DEFAULT_CONFIG；
  3. config/default.yaml 模板与代码默认值必须逐位一致（防止双源漂移——
     历史上 default.json 是手工副本，新增规则忘了同步就静默失效）。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline.config import (
    DEFAULT_CONFIG,
    cfg_get,
    load_config,
    severity_of,
    validate_config,
    write_default_config,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_yaml_full_template_matches_code_defaults() -> None:
    """config/default.yaml 模板加载后与代码内置默认值逐项一致。"""
    from_file = load_config(REPO_ROOT / "config" / "default.yaml")
    from_code = load_config()
    assert from_file["defaults"] == from_code["defaults"] == DEFAULT_CONFIG["defaults"]
    assert from_file["severity"] == from_code["severity"] == DEFAULT_CONFIG["severity"]


def test_yaml_partial_override_falls_back_to_code_defaults(tmp_path: Path) -> None:
    """YAML 只写「与默认值不同」的键，其余键自动回退代码默认值。"""
    cfg_file = tmp_path / "my.yaml"
    cfg_file.write_text(
        "# 业务方只需要覆写少数键\n"
        "defaults:\n"
        "  pointcloud:\n"
        "    max_outlier_ratio: 0.20   # 注释不影响解析\n"
        "severity:\n"
        "  pc_outliers: error\n",
        encoding="utf-8",
    )
    cfg = load_config(cfg_file)
    # 覆写生效
    assert cfg_get(cfg, "pointcloud.max_outlier_ratio") == 0.20
    assert severity_of(cfg, "pc_outliers") == "error"
    # 没写的键回退代码默认值
    assert cfg_get(cfg, "pointcloud.outlier_radius_k") == 2.5
    assert cfg_get(cfg, "timestamp.step_tolerance_ratio") == 0.25
    assert severity_of(cfg, "nan_state") == "error"


def test_json_config_still_supported(tmp_path: Path) -> None:
    """JSON 配置继续可用（程序生成/diff 场景）。"""
    cfg_file = tmp_path / "my.json"
    cfg_file.write_text(json.dumps({"defaults": {"frame_count": {"floor": 16}}}),
                        encoding="utf-8")
    cfg = load_config(cfg_file)
    assert cfg_get(cfg, "frame_count.floor") == 16
    assert cfg_get(cfg, "frame_count.ceiling") == 20000  # 深合并：同 section 其他键不受影响


def test_robot_override_via_yaml(tmp_path: Path) -> None:
    """robots[robot_type] 覆盖 defaults，其它机型不受影响。"""
    cfg_file = tmp_path / "robots.yaml"
    cfg_file.write_text(
        "robots:\n"
        "  so100_follower:\n"
        "    action_jump:\n"
        "      abs_floor: 0.5\n",
        encoding="utf-8",
    )
    cfg = load_config(cfg_file, robot_type="so100_follower")
    assert cfg_get(cfg, "action_jump.abs_floor") == 0.5
    assert cfg_get(cfg, "action_jump.max_jump_mad_k") == 3.0  # 未覆盖键回退 defaults

    other = load_config(cfg_file, robot_type="gym-xarm")
    assert cfg_get(other, "action_jump.abs_floor") == 1.0  # 别的机型用 defaults


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")


def test_empty_or_comment_only_yaml_means_all_defaults(tmp_path: Path) -> None:
    cfg_file = tmp_path / "empty.yaml"
    cfg_file.write_text("# 只有注释\n", encoding="utf-8")
    cfg = load_config(cfg_file)
    assert cfg["defaults"] == DEFAULT_CONFIG["defaults"]


def test_cli_overrides_beat_config_file(tmp_path: Path) -> None:
    cfg_file = tmp_path / "my.yaml"
    cfg_file.write_text("defaults:\n  frame_count: {floor: 16}\n", encoding="utf-8")
    cfg = load_config(cfg_file, overrides={"frame_count": {"floor": 32}})
    assert cfg_get(cfg, "frame_count.floor") == 32


def test_write_default_config_yaml_round_trip(tmp_path: Path) -> None:
    """导出的 YAML 模板再加载回来 == 代码默认值（防双源漂移）。"""
    out = write_default_config(tmp_path / "exported.yaml")
    assert yaml_safe_load(out) == DEFAULT_CONFIG
    cfg = load_config(out)
    assert cfg["defaults"] == DEFAULT_CONFIG["defaults"]
    # JSON 导出路徂也保留
    out_json = write_default_config(tmp_path / "exported.json")
    assert json.loads(out_json.read_text(encoding="utf-8")) == DEFAULT_CONFIG


def yaml_safe_load(path: Path) -> dict:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- strict 校验


def test_strict_config_accepts_shipped_templates() -> None:
    """仓库自带的两份配置都必须能通过严格校验（防止模板自身写错键）。"""
    for name in ("default.yaml", "chingmu_manual.yaml"):
        cfg = load_config(REPO_ROOT / "config" / name, strict=True)
        assert cfg["defaults"]


def test_strict_config_rejects_unknown_key(tmp_path: Path) -> None:
    """拼错的键（abs_floo）必须报错，否则会被 cfg_get 静默回退成默认值。"""
    cfg_file = tmp_path / "typo.yaml"
    cfg_file.write_text(
        "defaults:\n  action_jump:\n    abs_floo: 0.5\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="abs_floo"):
        load_config(cfg_file, strict=True)
    # 非 strict 仍保持向后兼容：忽略未知键、不抛异常
    assert cfg_get(load_config(cfg_file), "action_jump.abs_floor") == 1.0


def test_strict_config_rejects_type_mismatch(tmp_path: Path) -> None:
    cfg_file = tmp_path / "type.yaml"
    cfg_file.write_text(
        "defaults:\n  action_jump:\n    abs_floor: \"1.0\"\n",
        encoding="utf-8",
    )
    errors = validate_config(yaml_safe_load(cfg_file), source="type.yaml")
    assert any("defaults.action_jump.abs_floor" in e and "数值" in e for e in errors)
    with pytest.raises(ValueError, match="abs_floor"):
        load_config(cfg_file, strict=True)


def test_strict_config_rejects_unknown_severity_rule(tmp_path: Path) -> None:
    cfg_file = tmp_path / "sev.yaml"
    cfg_file.write_text("severity:\n  nan_states: error\n", encoding="utf-8")
    with pytest.raises(ValueError, match="nan_states"):
        load_config(cfg_file, strict=True)


def test_strict_config_rejects_unknown_top_level_key(tmp_path: Path) -> None:
    cfg_file = tmp_path / "top.yaml"
    cfg_file.write_text("robot:\n  so100: {}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="未知顶层配置项"):
        load_config(cfg_file, strict=True)


def test_validate_config_accepts_none_templates(tmp_path: Path) -> None:
    """模板默认值为 null 的键（joint_limits.lower）不做类型约束。"""
    cfg_file = tmp_path / "joint.yaml"
    cfg_file.write_text(
        "defaults:\n  joint_limits:\n    enabled: true\n    lower: [-1.0, 0.0]\n",
        encoding="utf-8",
    )
    assert validate_config(yaml_safe_load(cfg_file)) == []
    assert load_config(cfg_file, strict=True)["defaults"]["joint_limits"]["lower"] == [-1.0, 0.0]
