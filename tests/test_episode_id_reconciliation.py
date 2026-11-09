"""episode 编号空间对齐（behavior1k 那类数据集）的回归测试。

真实场景：meta/episodes 用全局 ID（420010, 420020, …），data/*.parquet 里是 0..N-1。
不先对齐，所有 meta 定位（视频窗口 / 任务 / data 文件）都会落空，进而把整批数据误判
为「帧数不一致 + 任务缺失」并建议全量剔除——这是最危险的误报形态。
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from conftest import build_dataset, smooth_episode

from pipeline.config import load_config
from pipeline.loader import load_dataset
from pipeline.policy import decide
from pipeline.validator import validate_dataset

def _episodes(n: int = 6) -> dict[int, dict[str, np.ndarray]]:
    # 长度各不相同：顺序映射是否可靠，靠「行数与 meta length 逐条对上」来证明
    return {ep: smooth_episode(40 + ep * 5, offset=ep * 0.7, seed=ep) for ep in range(n)}


def _rewrite_meta_ids(root, offset: int = 420010, step: int = 10) -> None:
    """把 meta/episodes 的 episode_index 换成全局编号（数据侧保持 0..N-1）。"""
    path = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    df = pq.read_table(path).to_pandas().sort_values("episode_index").reset_index(drop=True)
    df["episode_index"] = [offset + i * step for i in range(len(df))]
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path)


def test_global_meta_ids_are_mapped_by_order(tmp_path):
    root = build_dataset(tmp_path / "ds", _episodes())
    _rewrite_meta_ids(root)

    dataset = load_dataset(root)
    recon = dataset.id_reconciliation

    assert recon["mode"] == "remapped"
    assert recon["overlap"] == 0
    assert recon["length_agreement"] == 1.0
    # 数据侧仍是 0..N-1，meta 侧是全局编号
    assert dataset.episode_indices == list(range(6))
    assert dataset.meta_episode_index(0) == 420010
    assert dataset.meta_episode_index(5) == 420060

    # 定位能力恢复：任务与视频窗口都能查到
    assert dataset.task_of(0) == "pick cube"


def test_mapping_restores_task_lookup(tmp_path):
    """未对齐时 task_of 返回空串 → 200/200 missing_task 误报；对齐后应能取到任务。"""
    root = build_dataset(tmp_path / "ds", _episodes())
    _rewrite_meta_ids(root)
    dataset = load_dataset(root)
    cfg = load_config(robot_type="test_robot")

    validation = validate_dataset(dataset, cfg)

    assert "missing_task" not in validation.skipped_checks
    assert validation.reason_frequency().get("missing_task", 0) == 0
    assert all(dataset.task_of(ep) for ep in dataset.episode_indices)


def test_unmappable_id_space_is_not_evaluable(tmp_path):
    """两侧编号不同且证据不足时，定位类检查必须标记「不可评估」，不能报成数据异常。"""
    root = build_dataset(tmp_path / "ds", _episodes())
    # 打乱 meta 的 length 顺序：顺序映射后行数对不上，证据不足 → ambiguous
    path = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    df = pq.read_table(path).to_pandas().sort_values("episode_index").reset_index(drop=True)
    df["length"] = df["length"].to_numpy()[::-1]
    df["episode_index"] = [420010 + i * 10 for i in range(len(df))]
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path)

    dataset = load_dataset(root)
    assert dataset.id_reconciliation["mode"] == "ambiguous"
    assert dataset.locator_available is False

    validation = validate_dataset(dataset, load_config(robot_type="test_robot"))
    assert "missing_task" in validation.skipped_checks
    assert validation.reason_frequency().get("missing_task", 0) == 0
    # 任何 episode 都不应因定位失败被判为任务缺失
    assert all("missing_task" not in r.reasons for r in validation.results.values())


def test_systemic_guard_blocks_batch_drop():
    """单条规则近乎全量命中时，批量剔除必须降级为人工复核。"""
    from pipeline.validator.types import EpisodeResult, ValidationResult

    results = {
        ep: EpisodeResult(episode_index=ep, frame_count=40, reasons=["frame_mismatch"])
        for ep in range(10)
    }
    validation = ValidationResult(
        total_episodes=10, results=results, config=load_config(robot_type="test_robot"),
        fps=30.0, dataset_root="", resolved_thresholds={}, dataset_issues=[],
        skipped_checks={}, probe_available=True, aggregates={},
    )

    decisions = decide(validation)

    assert all(d.disposition == "review" for d in decisions.values())
    assert all(d.source == "systemic-guard" for d in decisions.values())
    assert not [d for d in decisions.values() if d.disposition == "drop"]


def test_minority_hits_still_drop():
    """命中率不高时不应触发熔断，正常的剔除策略保持不变。"""
    from pipeline.validator.types import EpisodeResult, ValidationResult

    results = {
        ep: EpisodeResult(episode_index=ep, frame_count=40,
                          reasons=["frame_mismatch"] if ep < 2 else [])
        for ep in range(10)
    }
    validation = ValidationResult(
        total_episodes=10, results=results, config=load_config(robot_type="test_robot"),
        fps=30.0, dataset_root="", resolved_thresholds={}, dataset_issues=[],
        skipped_checks={}, probe_available=True, aggregates={},
    )

    decisions = decide(validation)
    assert [ep for ep, d in decisions.items() if d.disposition == "drop"] == [0, 1]
