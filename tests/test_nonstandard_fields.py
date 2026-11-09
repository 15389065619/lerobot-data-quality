"""非标准字段名：字段被改名/缺失时应降级跳过，不得 KeyError，报告仍能生成。"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq

from main import main
from pipeline import policy as policy_mod
from pipeline.config import load_config
from pipeline.loader import load_dataset
from pipeline.reporter import build_report
from pipeline.validator import validate_dataset

from conftest import build_dataset, smooth_episode

DATA_FILE = Path("data") / "chunk-000" / "file-000.parquet"


def _basic_dataset(root: Path, count: int = 4) -> Path:
    return build_dataset(root, {ep: smooth_episode(40, seed=ep) for ep in range(count)})


def _rename_columns(root: Path, mapping: dict[str, str]) -> None:
    """同时改 data/*.parquet 的列名与 info.json.features，模拟上游自定义命名。"""
    data_path = root / DATA_FILE
    table = pq.read_table(data_path)
    table = table.rename_columns([mapping.get(name, name) for name in table.column_names])
    pq.write_table(table, data_path)

    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    features = info.get("features") or {}
    for old, new in mapping.items():
        if old in features:
            features[new] = features.pop(old)
    info_path.write_text(json.dumps(info), encoding="utf-8")


def _drop_column(root: Path, name: str) -> None:
    data_path = root / DATA_FILE
    pq.write_table(pq.read_table(data_path).drop([name]), data_path)

    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    (info.get("features") or {}).pop(name, None)
    info_path.write_text(json.dumps(info), encoding="utf-8")


def test_renamed_state_and_action_columns_degrade_gracefully(tmp_path):
    """state/action 被改名：相关规则写入 skipped_checks，不报假异常、不崩溃。"""
    root = _basic_dataset(tmp_path / "renamed")
    _rename_columns(
        root,
        {"observation.state": "observation.joint_positions", "action": "actions"},
    )

    dataset = load_dataset(root)
    config = load_config(robot_type="test_robot")
    validation = validate_dataset(dataset, config)

    assert "observation.state" in validation.skipped_checks["nan_state"]
    assert "action" in validation.skipped_checks["action_jump"]
    for res in validation.results.values():
        assert "nan_state" not in res.reasons
        assert "nan_action" not in res.reasons
        assert "action_jump" not in res.reasons

    report = build_report(dataset, validation, policy_mod.decide(validation), dataset_path=root)
    assert report["summary"]["total_episodes"] == 4
    assert report["skipped_checks"]["nan_state"]
    json.dumps(report, ensure_ascii=False)


def test_missing_timestamp_column_reports_reason_without_crash(tmp_path):
    """timestamp 列缺失：逐集记录「缺少 timestamp 列」，内存/流式两条路径都不崩溃。"""
    root = _basic_dataset(tmp_path / "no_timestamp")
    _drop_column(root, "timestamp")

    dataset = load_dataset(root)
    config = load_config(robot_type="test_robot")

    for streaming in (False, True):
        validation = validate_dataset(dataset, config, streaming=streaming)
        assert validation.total_episodes == 4
        for res in validation.results.values():
            assert "timestamp_not_monotonic" in res.reasons
            assert res.details["timestamp_not_monotonic"]["reason"] == "缺少 timestamp 列"

        report = build_report(dataset, validation, policy_mod.decide(validation), dataset_path=root)
        assert report["summary"]["total_episodes"] == 4
        json.dumps(report, ensure_ascii=False)


def test_renamed_episode_index_marks_episodes_read_error(tmp_path):
    """data 的 episode_index 被改名：无法按 episode 分组，全部兜底为 episode_read_error。"""
    root = _basic_dataset(tmp_path / "renamed_episode")
    _rename_columns(root, {"episode_index": "episode_id"})

    dataset = load_dataset(root)
    assert dataset.episode_indices == [0, 1, 2, 3]
    config = load_config(robot_type="test_robot")

    for streaming in (False, True):
        validation = validate_dataset(dataset, config, streaming=streaming)
        assert validation.total_episodes == 4
        for res in validation.results.values():
            assert "episode_read_error" in res.reasons

        report = build_report(dataset, validation, policy_mod.decide(validation), dataset_path=root)
        assert report["summary"]["total_episodes"] == 4
        assert report["summary"]["valid_episodes"] == 0
        json.dumps(report, ensure_ascii=False)


def test_cli_renamed_episode_index_still_writes_report(tmp_path):
    """端到端：非标准 episode 字段下 CLI 仍以退出码 0 落盘 quality_report.json。"""
    root = _basic_dataset(tmp_path / "cli_renamed_episode")
    _rename_columns(root, {"episode_index": "episode_id"})
    out = tmp_path / "out"

    code = main([
        "--dataset_path", str(root),
        "--output_dir", str(out),
        "--streaming", "--skip-clean", "--skip-pointcloud", "--no-cli", "--skip-plots",
    ])
    assert code == 0

    report = json.loads((out / "quality_report.json").read_text(encoding="utf-8"))
    assert report["summary"]["total_episodes"] == 4
    assert report["summary"]["valid_episodes"] == 0
