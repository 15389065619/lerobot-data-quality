"""基线白名单：指纹稳定性、告警摘除、健康度与配置写回。"""

from __future__ import annotations

import re

import pytest

from pipeline import policy as policy_mod
from pipeline.baseline import apply_whitelist, append_to_config, fingerprint
from pipeline.config import load_config
from pipeline.loader import load_dataset
from pipeline.reporter import build_report
from pipeline.validator import EpisodeResult, ValidationResult, validate_dataset

from conftest import build_dataset, smooth_episode


def _spike_dataset(tmp_path):
    """6 条平滑轨迹，其中 ep3 有一个明显单帧尖峰（action_jump）。"""
    episodes = {ep: smooth_episode(40, offset=ep * 0.7, seed=ep) for ep in range(6)}
    episodes[3]["action"][20, 1] += 14.0
    return build_dataset(tmp_path / "ds", episodes)


def test_fingerprint_is_stable_and_scoped():
    fp = fingerprint("ds", 7, "action_jump")
    assert fp == fingerprint("ds", 7, "action_jump")
    assert re.fullmatch(r"[0-9a-f]{16}", fp)
    assert fp != fingerprint("ds", 8, "action_jump")
    assert fp != fingerprint("ds", 7, "nan_state")
    assert fp != fingerprint("other_ds", 7, "action_jump")


def test_whitelisted_anomaly_no_longer_affects_health(tmp_path):
    path = _spike_dataset(tmp_path)
    dataset = load_dataset(path)
    config = load_config(robot_type="test_robot")
    validation = validate_dataset(dataset, config)

    assert 3 in validation.invalid_indices
    before = build_report(dataset, validation, policy_mod.decide(validation), dataset_path=path)
    assert before["summary"]["invalid_episodes"] == 1
    assert before["summary"]["health_level"] == "moderate"

    fp = fingerprint("ds", 3, "action_jump")
    config["baseline_whitelist"] = [fp]
    apply_whitelist(validation, "ds", config)

    assert 3 not in validation.invalid_indices
    assert validation.baseline_ignored[0]["code"] == "action_jump"
    assert validation.baseline_ignored[0]["fingerprint"] == fp
    assert validation.baseline_ignored[0]["severity"] == "warning"

    decisions = policy_mod.decide(validation)
    assert decisions[3].disposition == "keep"

    after = build_report(dataset, validation, decisions, dataset_path=path)
    assert after["summary"]["invalid_episodes"] == 0
    assert after["summary"]["health_level"] == "clean"
    assert after["summary"]["ignored_by_baseline"] == 1
    assert after["baseline_ignored"][0]["episode_index"] == 3


def test_report_exposes_fingerprint_for_accept(tmp_path):
    path = _spike_dataset(tmp_path)
    dataset = load_dataset(path)
    config = load_config(robot_type="test_robot")
    validation = validate_dataset(dataset, config)

    report = build_report(dataset, validation, policy_mod.decide(validation), dataset_path=path)
    detail = report["invalid_episode_details"][0]
    assert detail["fingerprints"]["action_jump"] == fingerprint("ds", 3, "action_jump")


def test_partial_whitelist_keeps_other_reasons():
    config = load_config(robot_type="test_robot")
    res = EpisodeResult(episode_index=2, frame_count=40)
    res.add("nan_state")
    res.add("action_jump")
    validation = ValidationResult(
        total_episodes=1, results={2: res}, config=config, fps=30, dataset_root="ds"
    )
    config["baseline_whitelist"] = [fingerprint("ds", 2, "action_jump")]

    apply_whitelist(validation, "ds", config)

    assert res.reasons == ["nan_state"]
    assert validation.severity_of(res) == "error"
    assert [item["code"] for item in validation.baseline_ignored] == ["action_jump"]


def test_apply_whitelist_is_idempotent():
    config = load_config(robot_type="test_robot")
    res = EpisodeResult(episode_index=0, frame_count=40)
    res.add("action_jump")
    validation = ValidationResult(
        total_episodes=1, results={0: res}, config=config, fps=30, dataset_root="ds"
    )
    config["baseline_whitelist"] = [fingerprint("ds", 0, "action_jump")]

    apply_whitelist(validation, "ds", config)
    apply_whitelist(validation, "ds", config)

    assert res.reasons == []
    assert len(validation.baseline_ignored) == 1


def test_append_to_config_keeps_existing_yaml(tmp_path):
    path = tmp_path / "policy.yaml"
    path.write_text("defaults:\n  frame_count:\n    min: 600\n", encoding="utf-8")
    fp = fingerprint("ds", 3, "action_jump")

    added, existed = append_to_config(path, [fp])
    assert added == [fp] and existed == []

    text = path.read_text(encoding="utf-8")
    assert "min: 600" in text, "原有配置块不应被覆盖"
    assert "baseline_whitelist:" in text
    assert load_config(path)["baseline_whitelist"] == [fp]

    # 重复添加不再写入；键已存在时走整份重写，但内容保持一致
    added2, existed2 = append_to_config(path, [fp])
    assert added2 == [] and existed2 == [fp]
    assert load_config(path)["baseline_whitelist"] == [fp]
    assert load_config(path)["defaults"]["frame_count"]["min"] == 600


def test_append_to_config_rejects_bad_fingerprint(tmp_path):
    path = tmp_path / "policy.yaml"
    path.write_text("defaults: {}\n", encoding="utf-8")
    with pytest.raises(ValueError):
        append_to_config(path, ["not-a-fingerprint"])


def test_accept_baseline_cli_writes_config(tmp_path):
    from main import main

    path = tmp_path / "policy.yaml"
    path.write_text("defaults: {}\n", encoding="utf-8")
    fp = fingerprint("ds", 3, "action_jump")

    assert main(["--config", str(path), "--accept-baseline", fp]) == 0
    assert load_config(path)["baseline_whitelist"] == [fp]

    # 没有 --config 时给出明确错误而不是静默成功
    assert main(["--accept-baseline", fp]) == 2
