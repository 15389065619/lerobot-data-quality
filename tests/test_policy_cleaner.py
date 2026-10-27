"""处置策略与清洗执行的单测。"""

from __future__ import annotations

import csv
import json

import numpy as np
import pyarrow.parquet as pq
import pytest

from pipeline import cleaner as cleaner_mod
from pipeline import policy as policy_mod
from pipeline.config import load_config
from pipeline.loader import load_dataset
from pipeline.reporter import build_report
from pipeline.validator import validate_dataset

from conftest import build_dataset, smooth_episode


def _run(path, **cfg):
    dataset = load_dataset(path)
    result = validate_dataset(dataset, load_config(robot_type="test_robot", overrides=cfg or None))
    decisions = policy_mod.decide(result)
    return dataset, result, decisions


def _mixed_dataset(tmp_path):
    """5 条：中段 NaN / 边界 NaN / 时间戳回退 / action 跳变 / 正常。"""
    mid_nan = smooth_episode(40, seed=1)
    mid_nan["state"][15:18, 0] = np.nan

    head_nan = smooth_episode(40, seed=2)
    head_nan["state"][:3, 0] = np.nan

    bad_ts = smooth_episode(40, seed=3)
    ts = np.arange(40, dtype=np.float32) / 30
    ts[20], ts[21] = ts[21], ts[20] - 0.5

    spike = smooth_episode(40, seed=4)
    spike["action"][20, 1] += 14.0

    return build_dataset(tmp_path / "ds", {
        0: mid_nan, 1: head_nan, 2: {**bad_ts, "timestamp": ts},
        3: spike, 4: smooth_episode(40, seed=5),
    })


def test_dispositions_follow_severity_and_repairability(tmp_path):
    _, result, decisions = _run(_mixed_dataset(tmp_path))
    assert decisions[0].disposition == "drop"      # 中段 NaN 不可修
    assert decisions[1].disposition == "repair"    # 边界 NaN 可裁
    assert decisions[1].repair == "trim"
    assert decisions[1].trim == {"head_frames": 3, "tail_frames": 0}
    assert decisions[2].disposition == "repair"    # 时间戳可重建
    assert decisions[2].repair == "timestamps"
    assert decisions[3].disposition == "review"    # 跳变需人审
    assert decisions[4].disposition == "keep"


def test_on_review_drop(tmp_path):
    dataset = load_dataset(_mixed_dataset(tmp_path))
    result = validate_dataset(dataset, load_config(robot_type="test_robot"))
    decisions = policy_mod.decide(result, on_review="drop")
    assert decisions[3].disposition == "drop"


def test_overrides_take_precedence(tmp_path):
    dataset = load_dataset(_mixed_dataset(tmp_path))
    result = validate_dataset(dataset, load_config(robot_type="test_robot"))
    decisions = policy_mod.decide(result, overrides={
        3: {"disposition": "keep", "note": "回放确认是真实动作突变"},
        0: {"disposition": "drop", "note": "重复确认"},
    })
    assert decisions[3].disposition == "keep"
    assert decisions[3].source == "override"
    assert decisions[3].note == "回放确认是真实动作突变"
    assert decisions[0].disposition == "drop"


def test_invalid_override_is_ignored(tmp_path):
    dataset = load_dataset(_mixed_dataset(tmp_path))
    result = validate_dataset(dataset, load_config(robot_type="test_robot"))
    decisions = policy_mod.decide(result, overrides={0: {"disposition": "whatever"}})
    assert decisions[0].source == "rule"


def test_cleaner_applies_drop_and_repairs(tmp_path):
    dataset, result, decisions = _run(_mixed_dataset(tmp_path))
    out = tmp_path / "out"
    cleaning = cleaner_mod.clean_dataset(dataset, decisions, out, prefer_cli=False)

    assert cleaning.success
    assert cleaning.removed_episodes == [0]
    assert cleaning.repaired_episodes == [1, 2]
    assert cleaning.kept_episodes == 4
    assert cleaning.verification["ok"] is True
    assert cleaning.verification["episodes"] == 4

    cleaned = load_dataset(out / "cleaned_dataset")
    assert cleaned.episode_indices == [0, 1, 2, 3]
    # 索引会被重排：原 ep0 被删，原 ep1（裁帧）-> 新 0，原 ep2（时间轴）-> 新 1
    lengths = {ep: len(cleaned.episode_frame(ep)) for ep in cleaned.episode_indices}
    assert lengths[0] == 37, f"裁帧未生效: {lengths}"

    trimmed = cleaned.episode_frame(0)
    arr = np.asarray(trimmed["observation.state"].to_list(), dtype=float)
    assert not np.isnan(arr).any(), "裁剪后仍残留 NaN"

    # 原 ep2 的时间轴被重建且严格单调
    ts = cleaned.episode_frame(1)["timestamp"].to_numpy()
    assert (np.diff(ts) > 0).all()
    assert np.allclose(ts, np.arange(len(ts)) / 30, atol=1e-6)

    # 全局不变量
    assert cleaned.frames["index"].tolist() == list(range(cleaned.num_frames))
    assert all(
        cleaned.episode_frame(ep)["frame_index"].tolist() == list(range(lengths[ep]))
        for ep in cleaned.episode_indices
    )


def test_cleaner_recomputes_stats(tmp_path):
    dataset, _, decisions = _run(_mixed_dataset(tmp_path))
    out = tmp_path / "out"
    cleaner_mod.clean_dataset(dataset, decisions, out, prefer_cli=False)

    stats = json.loads((out / "cleaned_dataset" / "meta" / "stats.json").read_text(encoding="utf-8"))
    cleaned = load_dataset(out / "cleaned_dataset")
    arr = np.asarray(cleaned.frames["observation.state"].to_list(), dtype=float)
    assert np.allclose(stats["observation.state"]["mean"], arr.mean(axis=0), atol=1e-4)
    assert stats["observation.state"]["count"] == cleaned.num_frames


def test_cleaner_refuses_output_inside_source(tmp_path):
    dataset, _, decisions = _run(_mixed_dataset(tmp_path))
    with pytest.raises(ValueError):
        cleaner_mod.clean_dataset(dataset, decisions, dataset.root, prefer_cli=False)


def test_source_dataset_is_never_modified(tmp_path):
    path = _mixed_dataset(tmp_path)
    before = {p: p.stat().st_mtime_ns for p in sorted(path.rglob("*")) if p.is_file()}
    dataset, _, decisions = _run(path)
    cleaner_mod.clean_dataset(dataset, decisions, tmp_path / "out", prefer_cli=False)
    after = {p: p.stat().st_mtime_ns for p in sorted(path.rglob("*")) if p.is_file()}
    assert before == after


def test_no_findings_copies_dataset(tmp_path):
    episodes = {ep: smooth_episode(40, seed=ep) for ep in range(3)}
    path = build_dataset(tmp_path / "ds", episodes)
    dataset, _, decisions = _run(path)
    cleaning = cleaner_mod.clean_dataset(dataset, decisions, tmp_path / "out", prefer_cli=False)
    assert cleaning.method == "noop-copy"
    assert cleaning.kept_episodes == 3
    assert (tmp_path / "out" / "cleaned_dataset" / "meta" / "info.json").is_file()


def test_report_contains_business_sections(tmp_path):
    dataset, result, decisions = _run(_mixed_dataset(tmp_path))
    report = build_report(dataset, result, decisions, dataset_path=tmp_path / "ds")

    assert report["schema_version"]
    assert report["dataset"]["fingerprint"]
    summary = report["summary"]
    assert summary["disposition"]["drop"] == 1
    assert summary["disposition"]["repair"] == 2
    assert summary["severity"]["error"] >= 1
    assert report["resolved_thresholds"]["frame_count"]["mode"] == "auto"
    assert isinstance(report["attribution"], list)
    assert any(c["code"] == "nan_state" for c in report["checks"])
    assert report["recommendations"]


def test_review_queue_roundtrip(tmp_path):
    dataset, result, decisions = _run(_mixed_dataset(tmp_path))
    queue = policy_mod.write_review_queue(tmp_path / "review.csv", result, decisions, dataset)
    assert queue.is_file()

    # 空表回读 → 无覆盖
    assert policy_mod.read_overrides(queue) == {}

    # 用 csv 模块填一行（detail 字段里有逗号，不能按逗号硬切）
    with queue.open("r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    for row in rows:
        if row["episode_index"] == "3":
            row["reviewer_disposition"] = "drop"
            row["reviewer_note"] = "回放确认是误写入"
    with queue.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    overrides = policy_mod.read_overrides(queue)
    assert overrides[3]["disposition"] == "drop"
    assert overrides[3]["note"] == "回放确认是误写入"


def test_attribution_groups_by_collector(tmp_path):
    episodes = {ep: smooth_episode(40, seed=ep) for ep in range(4)}
    episodes[1]["state"][10:12, 0] = np.nan
    path = build_dataset(tmp_path / "ds", episodes,
                         collector={0: "alice", 1: "bob", 2: "bob", 3: "alice"})
    dataset, result, decisions = _run(path)
    report = build_report(dataset, result, decisions, dataset_path=path)

    dims = {a["dimension"]: a for a in report["attribution"]}
    assert "meta/collector" in dims
    groups = {g["value"]: g for g in dims["meta/collector"]["groups"]}
    assert groups["bob"]["invalid"] == 1
    assert groups["alice"]["invalid"] == 0
