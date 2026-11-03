"""版本管理测试：快照去重、变更检测、diff、回滚 plan 与 apply。"""

from __future__ import annotations

from pathlib import Path

import pytest

from conftest import build_dataset, smooth_episode
from pipeline.versions import DatasetVersionStore


@pytest.fixture
def store(tmp_path: Path) -> tuple[DatasetVersionStore, Path]:
    root = build_dataset(tmp_path / "ds", {0: smooth_episode(30)})
    return DatasetVersionStore(root, tmp_path / "store"), root


def test_snapshot_and_log(store: tuple[DatasetVersionStore, Path]) -> None:
    vstore, root = store
    v1 = vstore.snapshot("baseline")
    assert v1["file_count"] >= 4
    assert (vstore.versions_dir / f"{v1['version_id']}.json").is_file()
    # 内容寻址：每个文件都能在对象库里找到
    for meta in v1["files"].values():
        assert (vstore.objects_dir / meta["sha1"][:2] / meta["sha1"]).is_file()
    assert vstore.log()[0]["version_id"] == v1["version_id"]


def test_status_and_diff(store: tuple[DatasetVersionStore, Path]) -> None:
    vstore, root = store
    v1 = vstore.snapshot("v1")
    stats = root / "meta" / "stats.json"
    stats.write_text(stats.read_text(encoding="utf-8") + " ", encoding="utf-8")

    v2 = vstore.snapshot("v2")
    diff = vstore.diff(v1["version_id"], v2["version_id"])
    assert diff["modified"] == ["meta/stats.json"]
    assert diff["added"] == []
    assert diff["removed"] == []
    assert v2["parent"] == v1["version_id"]


def test_restore_is_dry_run_by_default(store: tuple[DatasetVersionStore, Path]) -> None:
    vstore, root = store
    v1 = vstore.snapshot("v1")
    target = root / "meta" / "stats.json"
    original = target.read_text(encoding="utf-8")
    target.write_text("{}", encoding="utf-8")

    plan = vstore.restore(v1["version_id"])
    assert plan["applied"] is False
    assert plan["to_write"][0]["path"] == "meta/stats.json"
    assert plan["to_write"][0]["available"] is True
    assert target.read_text(encoding="utf-8") == "{}"  # 未写盘

    done = vstore.restore(v1["version_id"], apply=True)
    assert done["applied"] is True
    assert target.read_text(encoding="utf-8") == original


def test_missing_object_blocks_restore(tmp_path: Path) -> None:
    root = build_dataset(tmp_path / "ds", {0: smooth_episode(30)})
    vstore = DatasetVersionStore(root, tmp_path / "store")
    v1 = vstore.snapshot("v1")
    sha = v1["files"]["meta/stats.json"]["sha1"]
    (vstore.objects_dir / sha[:2] / sha).unlink()
    (root / "meta" / "stats.json").write_text("{}", encoding="utf-8")

    with pytest.raises(RuntimeError):
        vstore.restore(v1["version_id"], apply=True)
