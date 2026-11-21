"""多进程并发一致性测试。

核心断言只有一条：**workers>1 的并发结果与串行结果逐位一致**。
并发只改变「在哪里算」，不改变「算什么」——worker 是纯函数，全局规则
（duplicate 指纹 / pc_temporal_jump）都在主进程归并后执行。

Windows 下 ProcessPoolExecutor 走 spawn，这些测试同时验证了
worker 函数的可 pickle 性与 initializer 注入机制。
"""

from __future__ import annotations

import json
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from conftest import build_dataset, smooth_episode
from test_pointcloud import _mixed_frames, _write_pc_dataset

from pipeline.config import load_config
from pipeline.loader import load_dataset
from pipeline.pointcloud import validate_pointcloud_dataset
from pipeline.validator import validate_dataset


# ------------------------------------------------------------------ episode 级
def _canonical_report_bytes(result) -> bytes:
    """报告结构的规范化字节串：键排序 + 固定分隔符 + UTF-8。

    直接断言字节相等，比比较某个汇总数字强得多——任何一格 reasons/details/
    阈值出现差异都会让字节串变化（产物由这份结构序列化而来）。
    """
    payload = json.dumps(result.to_dict(), sort_keys=True, ensure_ascii=False,
                         separators=(",", ":"))
    return payload.encode("utf-8")


def _episodes_with_faults() -> dict[int, dict[str, np.ndarray]]:
    """6 个正常 episode + 1 个 NaN + 1 个与 ep2 完全重复（覆盖全局指纹规则）。"""
    eps = {ep: smooth_episode(40, offset=ep * 0.7, seed=ep) for ep in range(6)}
    bad = smooth_episode(40, offset=9.9, seed=99)
    bad["state"][5, 0] = np.nan  # nan_state
    eps[6] = bad
    eps[7] = {"state": eps[2]["state"].copy(), "action": eps[2]["action"].copy()}  # duplicate_episode
    return eps


def test_episode_validator_parallel_matches_sequential(tmp_path: Path) -> None:
    root = build_dataset(tmp_path / "ds", _episodes_with_faults())
    dataset = load_dataset(root)
    config = load_config(robot_type="test_robot")

    serial = validate_dataset(dataset, config)
    parallel = validate_dataset(dataset, config, workers=4)

    # 逐位一致：reasons / details / 阈值 / 数据集级问题全部相同（字节级比对）
    assert _canonical_report_bytes(serial) == _canonical_report_bytes(parallel)

    invalid = {r.episode_index: r.reasons for r in parallel.invalid_episodes}
    assert "nan_state" in invalid.get(6, [])
    assert "duplicate_episode" in invalid.get(7, [])


def test_broken_pool_falls_back_to_serial(monkeypatch) -> None:
    """worker 被系统杀死（BrokenProcessPool）时，剩余任务回退串行而不是丢掉整批结论。"""
    from pipeline import parallel

    def _boom(*args, **kwargs):
        raise parallel.BrokenProcessPool("worker killed")

    monkeypatch.setattr(parallel, "wait", _boom)
    results = list(parallel.run_pool([1, 2, 3, 4], _double, None, 2, desc="test", unit="job"))
    assert sorted(results) == [2, 4, 6, 8]


class _FakeFuture:
    def __init__(self, job: int, broken: bool):
        self.job = job
        self._broken = broken

    def result(self) -> int:
        if self._broken:
            raise BrokenProcessPool("worker killed")
        return self.job * 2


class _FakePool:
    """假进程池：真实的中途 worker 死亡难以跨平台伪造，而漏洞在簿记不在池本身，
    用假 executor 精确命中 run_pool 的登记/回退路径（部分 future 已完成再破裂）。"""

    def __init__(self, broken_result_job: int | None, broken_submit_job: int | None):
        self._broken_result_job = broken_result_job
        self._broken_submit_job = broken_submit_job

    def submit(self, fn, job: int) -> _FakeFuture:
        if job == self._broken_submit_job:
            raise BrokenProcessPool("submit on dead pool")
        return _FakeFuture(job, broken=(job == self._broken_result_job))

    def shutdown(self, wait=True, cancel_futures=False) -> None:
        pass


def _run_fake_pool(monkeypatch, jobs, *, broken_result_job, broken_submit_job) -> list[int]:
    """让假池里的 future 在每次 wait 时全部就绪，驱动 run_pool 走完破裂路径。"""
    from pipeline import parallel

    pool = _FakePool(broken_result_job, broken_submit_job)
    monkeypatch.setattr(parallel, "ProcessPoolExecutor", lambda **kw: pool)

    def _fake_wait(fs, return_when=None):
        return list(fs), []

    monkeypatch.setattr(parallel, "wait", _fake_wait)
    return list(parallel.run_pool(jobs, _double, None, 2, desc="test", unit="job"))


def test_broken_pool_midrun_requeues_broken_future_job(monkeypatch) -> None:
    """部分 future 已完成弹出后池才破裂：已 pop 的 job 也要并回串行补跑。

    回归背景：run_pool 曾先 pop 再 result()，破裂时该 job 既不在 in_flight
    也不在 pending，串行回退的 remaining 把它静默丢弃（全量套件偶发点云
    workers=4 测试断言失败的根因，见 2026-11-21 审查记录）。
    """
    results = _run_fake_pool(monkeypatch, [1, 2, 3, 4, 5, 6],
                             broken_result_job=3, broken_submit_job=None)
    assert sorted(results) == [2, 4, 6, 8, 10, 12]


def test_broken_pool_midrun_requeues_submit_failed_job(monkeypatch) -> None:
    """破裂发生在 submit 瞬间：刚 popleft 的 job 放回队首，不静默丢失。"""
    results = _run_fake_pool(monkeypatch, [1, 2, 3, 4, 5, 6],
                             broken_result_job=None, broken_submit_job=5)
    assert sorted(results) == [2, 4, 6, 8, 10, 12]


def _double(value: int) -> int:
    return value * 2


# ------------------------------------------------------------------ 点云帧级
def test_pointcloud_parallel_matches_sequential(tmp_path: Path) -> None:
    frames = _mixed_frames()  # 含 NaN / 稀疏 / 退化 / 重复 / 离群 / 越界 / 坏色 / 时序跳变
    root = _write_pc_dataset(tmp_path / "ds_pc", frames)

    # 默认写法只有一个 row group，并发会退化成单任务；强制小 row group 以覆盖分片路径
    data_file = root / "data" / "chunk-000" / "file-000.parquet"
    table = pq.read_table(data_file)
    pq.write_table(table, data_file, row_group_size=6)
    assert pq.ParquetFile(data_file).metadata.num_row_groups > 1

    config = load_config(robot_type="pc_test_robot")
    serial = validate_pointcloud_dataset(root, config)
    parallel = validate_pointcloud_dataset(root, config, workers=4)

    # 逐位一致：每帧 reasons / metrics / 跨帧 temporal 判定全部相同（字节级比对）
    assert _canonical_report_bytes(serial) == _canonical_report_bytes(parallel)

    freq = parallel.reason_frequency()
    assert freq.get("pc_non_finite", 0) >= 1 and freq.get("pc_duplicate", 0) >= 1
