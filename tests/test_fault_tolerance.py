"""容错（毒数据防冲击）与断点续传测试。

如何伪造一个「损坏的数据文件」：合法 parquet 以 b"PAR1" 魔数开头、footer 结尾，
把文件改写成垃圾字节后，pyarrow 读 schema/footer 时必然抛异常——
这就是工业现场「采集中断写了一半 / 磁盘坏块」的典型形态。
（本流水线的点云存在 parquet 列里而不是独立 PCD 文件，所以伪造损坏
针对 parquet；若后续接入独立 PCD 加载，同理：写一段非法 header 即可。）
"""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq
import pytest

from conftest import normal_dataset  # noqa: F401 - pytest fixture
from test_pointcloud import _mixed_frames, _write_pc_dataset

import pipeline.pointcloud as pc_mod
from pipeline.config import load_config
from pipeline.loader import load_dataset
from pipeline.pointcloud import validate_pointcloud_dataset
from pipeline.validator import validate_dataset


def _corrupt(path: Path) -> None:
    """把合法 parquet 改写成垃圾字节：魔数还在，footer 没了。"""
    path.write_bytes(b"PAR1" + b"\x00" * 64 + b"corrupted-tail")


def _two_file_pc_dataset(tmp_path: Path) -> tuple[Path, Path, int]:
    """把 _mixed_frames 拆成两个 parquet 分片，返回 (root, 第二个文件, 第一片帧数)。"""
    frames = _mixed_frames()
    root = _write_pc_dataset(tmp_path / "ds", frames)
    f0 = root / "data" / "chunk-000" / "file-000.parquet"
    table = pq.read_table(f0)
    half = table.num_rows // 2
    pq.write_table(table.slice(0, half), f0, row_group_size=8)
    f1 = f0.with_name("file-001.parquet")
    pq.write_table(table.slice(half), f1, row_group_size=8)
    return root, f1, half


# ------------------------------------------------------------------ 容错
@pytest.mark.parametrize("workers", [1, 4])
def test_pointcloud_tolerates_corrupted_file(tmp_path: Path, workers: int) -> None:
    """一个分片损坏：不中断、记录 corrupted、其余分片照常质检。"""
    root, f1, half = _two_file_pc_dataset(tmp_path)
    _corrupt(f1)

    v = validate_pointcloud_dataset(root, load_config(robot_type="pc_test_robot"),
                                    workers=workers)
    assert len(v.corrupted_files) == 1
    assert v.corrupted_files[0]["file"].endswith("file-001.parquet")
    assert v.total_frames == half  # 好分片的帧全部完成评估
    assert "corrupted_files" in v.to_dict()


@pytest.mark.parametrize("workers", [1, 4])
def test_loader_and_validator_tolerate_corrupted_file(normal_dataset: Path, workers: int) -> None:
    """数据文件损坏：loader 跳过并记账，validator 把受影响 episode 标记 episode_read_error。"""
    _corrupt(normal_dataset / "data" / "chunk-000" / "file-000.parquet")

    dataset = load_dataset(normal_dataset)  # 不抛异常
    assert len(dataset.corrupted_files) == 1

    result = validate_dataset(dataset, load_config(robot_type="test_robot"), workers=workers)
    by_ep = {r.episode_index: r for r in result.invalid_episodes}
    assert set(by_ep) == {0, 1, 2, 3, 4, 5}  # 唯一数据文件损坏 → 全部 episode 受影响
    assert all("episode_read_error" in r.reasons for r in by_ep.values())
    assert any("corrupted" in (r.details["episode_read_error"].get("error") or "")
               or "file-000" in (r.details["episode_read_error"].get("file") or "")
               for r in by_ep.values())


@pytest.mark.parametrize("workers", [1, 4])
def test_corrupted_shard_recovers_after_repair(normal_dataset: Path, workers: int) -> None:
    """损坏 → 精确影响全部 6 集；原样修复 → 结论完全恢复，无残留状态。

    断言「精确集数 + 修复后回到干净」而不是「没抛异常」：磁盘半截文件、
    清单/缓存残留这类问题只有这样才抓得住。
    """
    data_file = normal_dataset / "data" / "chunk-000" / "file-000.parquet"
    original = data_file.read_bytes()
    _corrupt(data_file)

    broken = validate_dataset(load_dataset(normal_dataset),
                              load_config(robot_type="test_robot"), workers=workers)
    assert broken.valid_episodes == 0
    assert [r.episode_index for r in broken.invalid_episodes] == [0, 1, 2, 3, 4, 5]
    assert all("episode_read_error" in r.reasons for r in broken.invalid_episodes)

    data_file.write_bytes(original)  # 修复：原始字节写回
    healed = validate_dataset(load_dataset(normal_dataset),
                              load_config(robot_type="test_robot"), workers=workers)
    assert healed.invalid_episodes == []
    assert healed.valid_episodes == 6


def test_pointcloud_clean_tolerates_corrupted_file(tmp_path: Path) -> None:
    """清洗阶段同样容错：损坏分片原样保留，好分片正常重写。"""
    root, f1, _ = _two_file_pc_dataset(tmp_path)
    config = load_config(robot_type="pc_test_robot")
    v = validate_pointcloud_dataset(root, config)  # 先拿到结论（此时文件还没坏）
    _corrupt(f1)
    decisions = pc_mod.decide(v)
    out = tmp_path / "cleaned"
    result = pc_mod.clean_pointclouds(root, v, decisions, out)
    assert any("file-001" in w for w in result.warnings)  # 有告警
    dst = out / "pointcloud_cleaned" / "data" / "chunk-000"
    assert (dst / "file-001.parquet").is_file()  # 原样复制保留
    assert (dst / "file-000.parquet").is_file()  # 好分片正常写入


# ------------------------------------------------------------------ 断点续传
def test_pointcloud_scan_cache_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """第二次运行零重算；文件指纹变化（修复/改动）后自动重扫。"""
    frames = _mixed_frames()
    root = _write_pc_dataset(tmp_path / "ds", frames)
    cache = tmp_path / "pointcloud_scan_cache.json"
    config = load_config(robot_type="pc_test_robot")

    calls = {"n": 0}
    real_evaluate = pc_mod.evaluate_frame

    def counting(*args, **kwargs):
        calls["n"] += 1
        return real_evaluate(*args, **kwargs)

    monkeypatch.setattr(pc_mod, "evaluate_frame", counting)

    v1 = validate_pointcloud_dataset(root, config, cache_path=cache)
    assert cache.is_file() and calls["n"] == len(frames)

    calls["n"] = 0
    v2 = validate_pointcloud_dataset(root, config, cache_path=cache)
    assert calls["n"] == 0  # 全部命中缓存，没有重扫一帧
    assert v1.to_dict() == v2.to_dict()  # 结论逐位一致

    # 改动文件（重写一次）→ 指纹变化 → 自动重扫
    data_file = root / "data" / "chunk-000" / "file-000.parquet"
    pq.write_table(pq.read_table(data_file), data_file, row_group_size=16)
    calls["n"] = 0
    v3 = validate_pointcloud_dataset(root, config, cache_path=cache)
    assert calls["n"] == len(frames)
    assert v3.to_dict() == v1.to_dict()
