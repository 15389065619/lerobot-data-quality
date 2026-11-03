"""训练前预检的测试：干净数据放行、脏数据阻断、strict 抛异常、JSON 落盘。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from conftest import build_dataset, smooth_episode
from pipeline.precheck import DirtyDatasetError, precheck, main


def _dirty_episode(n: int = 40) -> dict[str, np.ndarray]:
    traj = smooth_episode(n)
    traj["state"] = traj["state"].copy()
    traj["state"][n // 2] = np.nan  # 中段掉线 → 无法裁剪修复 → drop
    return traj


def test_clean_dataset_passes(tmp_path: Path) -> None:
    root = build_dataset(tmp_path / "ds", {0: smooth_episode(40), 1: smooth_episode(40, offset=1)})
    result = precheck(root)
    assert result.ok
    assert result.drop == 0
    assert sum((result.keep, result.repair, result.review, result.drop)) == result.total_episodes
    assert "通过" in result.message


def test_dirty_dataset_is_blocked(tmp_path: Path) -> None:
    root = build_dataset(tmp_path / "ds", {0: smooth_episode(40), 1: _dirty_episode()})
    result = precheck(root)
    assert not result.ok
    assert result.drop >= 1
    assert result.blocking and result.blocking[0]["episode_index"] == 1
    assert "未通过" in result.message


def test_strict_raises(tmp_path: Path) -> None:
    root = build_dataset(tmp_path / "ds", {0: _dirty_episode()})
    with pytest.raises(DirtyDatasetError):
        precheck(root, strict=True)


def test_write_json_and_cli_exit_code(tmp_path: Path) -> None:
    root = build_dataset(tmp_path / "ds", {0: _dirty_episode()})
    out = tmp_path / "precheck.json"
    code = main(["--dataset_path", str(root), "--strict", "--json", str(out)])
    assert code == 2
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["ok"] is False
    assert payload["disposition"]["drop"] >= 1
