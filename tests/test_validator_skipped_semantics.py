"""缺 state / action 列时，相关检查必须标记为 skipped，不能虚报 enabled/0 命中。"""

from __future__ import annotations

import json

import pyarrow.parquet as pq
from conftest import build_dataset, smooth_episode

from pipeline.config import load_config
from pipeline.loader import load_dataset
from pipeline.validator import validate_dataset

DATA = ("data", "chunk-000", "file-000.parquet")


def _drop_column(root, column: str) -> None:
    path = root.joinpath(*DATA)
    pq.write_table(pq.read_table(path).drop([column]), path)
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["features"].pop(column)
    info_path.write_text(json.dumps(info), encoding="utf-8")


def _validate(root):
    dataset = load_dataset(root)
    return validate_dataset(dataset, load_config(robot_type="test_robot"))


def test_missing_state_skips_all_state_dependent_checks(tmp_path):
    root = build_dataset(tmp_path / "ds", {0: smooth_episode(40, seed=0)})
    _drop_column(root, "observation.state")

    result = _validate(root)

    for code in ("nan_state", "state_jump", "joint_limit_violation"):
        assert code in result.skipped_checks, code


def test_missing_action_skips_all_action_dependent_checks(tmp_path):
    root = build_dataset(tmp_path / "ds", {0: smooth_episode(40, seed=0)})
    _drop_column(root, "action")

    result = _validate(root)

    for code in ("nan_action", "action_jump", "duplicate_episode"):
        assert code in result.skipped_checks, code
