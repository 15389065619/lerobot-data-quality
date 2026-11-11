"""处置策略与清洗执行的单测。"""

from __future__ import annotations

import csv
import json

import numpy as np
import pyarrow as pa
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


def test_cleaner_failure_keeps_previous_output(tmp_path, monkeypatch):
    """清洗中途失败（含 Ctrl+C）不能破坏上一版完整产物，也不能留下暂存目录。"""
    dataset, _, decisions = _run(_mixed_dataset(tmp_path))
    out = tmp_path / "out"
    cleaner_mod.clean_dataset(dataset, decisions, out, prefer_cli=False)
    marker = out / "cleaned_dataset" / "meta" / "info.json"
    before = marker.read_text(encoding="utf-8")

    def _boom(*args, **kwargs):
        raise RuntimeError("simulated crash")

    monkeypatch.setattr(cleaner_mod, "manual_clean", _boom)
    with pytest.raises(RuntimeError):
        cleaner_mod.clean_dataset(dataset, decisions, out, prefer_cli=False)

    assert marker.is_file()
    assert marker.read_text(encoding="utf-8") == before
    assert not list(out.glob("cleaned_dataset.tmp-*")), "失败后暂存目录未清理"
    assert not list(out.glob("cleaned_dataset.prev-*")), "失败后旧产物未回滚"


def test_cleaning_interrupt_cleans_staging(tmp_path, monkeypatch):
    dataset, _, decisions = _run(_mixed_dataset(tmp_path))
    out = tmp_path / "out"

    def _interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(cleaner_mod, "manual_clean", _interrupt)
    with pytest.raises(KeyboardInterrupt):
        cleaner_mod.clean_dataset(dataset, decisions, out, prefer_cli=False)

    assert not (out / "cleaned_dataset").exists()
    assert not list(out.glob("cleaned_dataset.tmp-*"))


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
    # 裁帧改变了帧位置，timestamp 必须一并重建（首帧归零、步长不变）
    ts_trim = trimmed["timestamp"].to_numpy()
    assert np.allclose(ts_trim, np.arange(len(ts_trim)) / 30, atol=1e-6)

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


def test_cleaner_rebuilds_global_index_when_inconsistent(tmp_path):
    """模拟 v2.1 导出：index 每个 episode 从 0 重新编号，清洗应重建为全局连续。"""
    episodes = {ep: smooth_episode(20, seed=ep) for ep in range(3)}
    path = build_dataset(tmp_path / "ds", episodes)
    data = path / "data" / "chunk-000" / "file-000.parquet"
    table = pq.read_table(data)
    frame_idx = np.asarray(table.column("frame_index").to_pylist(), dtype=np.int64)
    table = table.set_column(
        table.schema.get_field_index("index"),
        table.schema.field("index"),
        pa.array(frame_idx),
    )
    pq.write_table(table, data)

    dataset = load_dataset(path)
    result = validate_dataset(dataset, load_config(robot_type="test_robot"))
    assert any(i.code == "meta_index_inconsistent" for i in result.dataset_issues)

    decisions = policy_mod.decide(result)
    cleaning = cleaner_mod.clean_dataset(
        dataset, decisions, tmp_path / "out", prefer_cli=False,
        dataset_issues=result.dataset_issues,
    )
    assert cleaning.method == "index-repair-copy"
    assert cleaning.repaired_episodes == [0, 1, 2]

    cleaned = load_dataset(tmp_path / "out" / "cleaned_dataset")
    assert cleaned.frames["index"].tolist() == list(range(cleaned.num_frames))


def test_index_repair_backfills_missing_frame_index(tmp_path):
    """behavior1k 场景：既没有 frame_index，index 又每集从 0 重启。

    只修 index 不够——episode 内帧号仍然无从得知，下游切窗只能自己猜；
    重建时应一并补出 frame_index=0..length-1。
    """
    episodes = {ep: smooth_episode(15, seed=ep) for ep in range(2)}
    path = build_dataset(tmp_path / "ds", episodes)
    data = path / "data" / "chunk-000" / "file-000.parquet"
    table = pq.read_table(data)
    frame_idx = np.asarray(table.column("frame_index").to_pylist(), dtype=np.int64)
    table = table.set_column(
        table.schema.get_field_index("index"),
        table.schema.field("index"),
        pa.array(frame_idx),
    )
    table = table.drop(["frame_index"])
    pq.write_table(table, data)

    dataset = load_dataset(path)
    assert "frame_index" not in dataset.frames.columns
    result = validate_dataset(dataset, load_config(robot_type="test_robot"))
    assert any(i.code == "meta_index_inconsistent" for i in result.dataset_issues)

    cleaning = cleaner_mod.clean_dataset(
        dataset, policy_mod.decide(result), tmp_path / "out", prefer_cli=False,
        dataset_issues=result.dataset_issues,
    )
    assert cleaning.method == "index-repair-copy"

    out_root = tmp_path / "out" / "cleaned_dataset"
    cleaned = load_dataset(out_root)
    assert "frame_index" in cleaned.frames.columns
    assert cleaned.frames["index"].tolist() == list(range(cleaned.num_frames))
    for ep, group in cleaned.frames.groupby("episode_index"):
        assert group["frame_index"].tolist() == list(range(len(group)))

    info = json.loads((out_root / "meta" / "info.json").read_text(encoding="utf-8"))
    assert "frame_index" in info["features"]
    assert info["splits"] == {"train": "0:2"}


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


def test_short_episode_is_warning_and_review(tmp_path):
    """帧数过短可能只是任务步数少的合法数据（LIBERO），不该按 error 直接剔除。

    三条 40 帧的 episode 让 MAD=0、区间收紧为 [40,40]，再用 15 帧 / 60 帧分别
    构造过短与过长两种情况。
    """
    episodes = {
        0: smooth_episode(40, seed=51),
        1: smooth_episode(40, seed=52),
        2: smooth_episode(40, seed=53),
        3: smooth_episode(15, seed=54),
        4: smooth_episode(60, seed=55),
    }
    path = build_dataset(tmp_path / "ds", episodes)
    dataset, result, decisions = _run(path)

    short = result.results[3]
    assert "frame_count_abnormal" in short.reasons
    assert result.severity_of(short) == "warning"
    assert decisions[3].disposition == "review"
    detail = short.details["frame_count_abnormal"]
    assert detail["direction"] == "too_few"
    assert detail["direction_label"] == "过短"
    assert detail["frame_count"] == 15
    assert detail["min_allowed"] == 40 and detail["max_allowed"] == 40

    long = result.results[4]
    assert "frame_count_abnormal" in long.reasons
    assert result.severity_of(long) == "error"
    assert decisions[4].disposition == "drop"
    assert long.details["frame_count_abnormal"]["direction"] == "too_many"
    assert long.details["frame_count_abnormal"]["direction_label"] == "过长"


def test_report_check_severity_reflects_direction(tmp_path):
    """报告汇总表的级别应跟随逐 episode 的方向分级，而不是静态配置的 error。"""
    path = build_dataset(tmp_path / "ds", {0: smooth_episode(15, seed=91),
                                           1: smooth_episode(15, seed=92)})
    dataset, result, decisions = _run(path, frame_count={"min": 30})

    report = build_report(dataset, result, decisions, dataset_path=path)
    check = next(c for c in report["checks"] if c["code"] == "frame_count_abnormal")
    assert check["severity"] == "warning"
    assert check["affected_episodes"] == 2
    for item in report["invalid_episode_details"]:
        assert item["severity"] == "warning"
        assert item["details"]["frame_count_abnormal"]["direction_label"] == "过短"


def test_manual_frame_bounds_override_auto(tmp_path):
    """业务硬约束（如光捕 1~2 分钟）优先于统计学区间，并记录为 manual 模式。"""
    episodes = {
        0: smooth_episode(40, seed=61),
        1: smooth_episode(40, seed=62),
        2: smooth_episode(15, seed=63),
        3: smooth_episode(60, seed=64),
    }
    path = build_dataset(tmp_path / "ds", episodes)
    dataset, result, decisions = _run(path, frame_count={"min": 30, "max": 50})

    bounds = result.resolved_thresholds["frame_count"]
    assert bounds["mode"] == "manual"
    assert bounds["min"] == 30 and bounds["max"] == 50
    assert bounds["source"] == {"min": "manual", "max": "manual"}
    assert bounds["configured"] == {"min": 30, "max": 50}

    short = result.results[2]
    assert short.details["frame_count_abnormal"]["direction_label"] == "过短"
    assert result.severity_of(short) == "warning"
    assert decisions[2].disposition == "review"

    long = result.results[3]
    assert long.details["frame_count_abnormal"]["direction_label"] == "过长"
    assert result.severity_of(long) == "error"
    assert decisions[3].disposition == "drop"


def test_manual_frame_bounds_single_side(tmp_path):
    """只给一侧时，另一侧仍按数据集分布推导，来源分侧记录。"""
    path = build_dataset(tmp_path / "ds", {0: smooth_episode(20, seed=71),
                                           1: smooth_episode(20, seed=72),
                                           2: smooth_episode(40, seed=73)})
    dataset, result, decisions = _run(path, frame_count={"max": 30})

    bounds = result.resolved_thresholds["frame_count"]
    assert bounds["mode"] == "manual"
    assert bounds["max"] == 30
    assert bounds["source"] == {"min": "auto", "max": "manual"}
    # 40 帧超上限：error + drop
    assert result.severity_of(result.results[2]) == "error"
    assert decisions[2].disposition == "drop"


def test_manual_frame_bounds_invalid_range(tmp_path):
    """min > max 属于配置错误，应直接报错而不是产出自相矛盾的区间。"""
    path = build_dataset(tmp_path / "ds", {0: smooth_episode(20, seed=81)})
    with pytest.raises(ValueError, match="不能大于"):
        _run(path, frame_count={"min": 100, "max": 50})


def test_state_jump_soft_rule_never_drops(tmp_path):
    """软规则即使被配置成 error 也不能据此自动剔除（state_jump 只是提示复核）。"""
    from pipeline.validator.types import EpisodeResult, ValidationResult

    config = load_config(robot_type="test_robot")
    config["severity"]["state_jump"] = "error"
    res = EpisodeResult(episode_index=0, frame_count=60)
    res.add("state_jump")
    validation = ValidationResult(
        total_episodes=1, results={0: res}, config=config, fps=30.0,
        dataset_root=str(tmp_path),
    )

    decisions = policy_mod.decide(validation)
    assert decisions[0].disposition == "review"


def test_mass_drop_guard_downgrades_all(tmp_path):
    """整体剔除比例 ≥80% 且样本足够时，全部降级复核，防止阈值性误判被批量执行。"""
    eps = {}
    for ep in range(20):
        item = smooth_episode(40, seed=ep)
        if ep < 18:
            item["state"][20, 0] = np.nan
        eps[ep] = item
    path = build_dataset(tmp_path / "ds", eps)
    _, result, decisions = _run(path)

    # 18/20 = 90%，低于单规则熔断的 95%，但触发整体熔断
    assert result.severity_of(result.results[0]) == "error"
    assert all(decisions[ep].disposition == "review" for ep in range(18))
    assert decisions[0].source == "mass-drop-guard"


def test_streaming_stats_matches_numpy():
    """std 必须是 sqrt(M2/n)；分位数必须来自当前数据而不是旧 stats。"""
    from pipeline.cleaner.stats_streaming import StreamingStats

    rng = np.random.default_rng(0)
    data = rng.normal(3.0, 2.0, size=(500, 4))
    vec = pa.list_(pa.float64(), 4)
    acc = StreamingStats()
    for start in range(0, 500, 37):
        chunk = data[start:start + 37]
        acc.update_from_table(pa.table({"f": pa.array(list(chunk), type=vec)}), ["f"])
    out = acc.finalize({})["f"]

    assert np.allclose(out["mean"], data.mean(axis=0), atol=1e-5)
    assert np.allclose(out["std"], data.std(axis=0), atol=1e-5)
    for q, name in ((0.01, "q01"), (0.5, "q50"), (0.99, "q99")):
        assert np.allclose(out[name], np.quantile(data, q, axis=0), atol=1e-5)
