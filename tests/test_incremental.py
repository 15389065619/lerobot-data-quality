"""增量模式：manifest 指纹、结论复用与失效条件。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from main import main
from pipeline.config import load_config
from pipeline.loader import load_dataset
from pipeline.manifest import QualityManifest, cleaning_signature
from pipeline.policy import decide
from pipeline.validator import validate_dataset

from conftest import smooth_episode  # noqa: F401 - 与其它测试共用夹具构造方式


def _prepare(normal_dataset, tmp_path):
    dataset = load_dataset(normal_dataset)
    config = load_config(robot_type=dataset.info.get("robot_type"))
    validation = validate_dataset(dataset, config)
    decisions = decide(validation)
    manifest = QualityManifest(tmp_path / "quality_manifest.json")
    manifest.store_results(dataset, config, validation, decisions, manifest.episode_signatures(dataset))
    manifest.save()
    return dataset, config, validation, decisions, manifest


def test_manifest_roundtrip_and_reuse(normal_dataset, tmp_path):
    dataset, config, validation, _, _ = _prepare(normal_dataset, tmp_path)
    manifest = QualityManifest.load(tmp_path / "quality_manifest.json")
    scan = load_dataset(normal_dataset, load_frames=False)

    assert manifest.unchanged_episodes(scan, config) == set(dataset.episode_indices)
    assert manifest.changed_episodes(scan, config) == set()

    cached = manifest.cached_results()
    assert set(cached) == set(validation.results)
    assert cached[3].frame_count == validation.results[3].frame_count
    assert cached[3].reasons == validation.results[3].reasons


def test_changed_episode_detected_by_structure(normal_dataset, tmp_path):
    _, config, _, _, _ = _prepare(normal_dataset, tmp_path)
    manifest = QualityManifest.load(tmp_path / "quality_manifest.json")
    scan = load_dataset(normal_dataset, load_frames=False)

    row = scan.episodes.index[scan.episodes["episode_index"] == 2][0]
    scan.episodes.at[row, "length"] = int(scan.episodes.at[row, "length"]) + 1

    assert manifest.changed_episodes(scan, config) == {2}


def test_config_change_invalidates_cache(normal_dataset, tmp_path):
    _, _, _, _, _ = _prepare(normal_dataset, tmp_path)
    manifest = QualityManifest.load(tmp_path / "quality_manifest.json")
    scan = load_dataset(normal_dataset, load_frames=False)

    fixed = load_config(robot_type="test_robot", overrides={
        "frame_count": {"mode": "fixed", "floor": 8, "ceiling": 20000},
    })
    assert manifest.unchanged_episodes(scan, fixed) == set()


def test_incremental_validation_reuses_unchanged_episodes(normal_dataset):
    dataset = load_dataset(normal_dataset)
    config = load_config(robot_type=dataset.info.get("robot_type"))
    baseline = validate_dataset(dataset, config)
    prior = {ep: res for ep, res in baseline.results.items()}

    reused = validate_dataset(dataset, config, prior=prior, only=set())
    assert set(reused.results) == set(baseline.results)
    for ep, res in baseline.results.items():
        assert reused.results[ep].to_dict() == res.to_dict()


def test_incremental_validation_only_rechecks_changed_episode(normal_dataset):
    dataset = load_dataset(normal_dataset)
    config = load_config(robot_type=dataset.info.get("robot_type"))
    baseline = validate_dataset(dataset, config)
    prior = {ep: res for ep, res in baseline.results.items()}

    mask = dataset.frames["episode_index"] == 0
    row = dataset.frames.index[mask][3]
    vector = np.asarray(dataset.frames.at[row, "observation.state"]).copy()
    vector[0] = np.nan
    dataset.frames.at[row, "observation.state"] = vector

    partial = validate_dataset(dataset, config, prior=prior, only={0})
    assert "nan_state" in partial.results[0].reasons
    assert partial.results[1].to_dict() == baseline.results[1].to_dict()
    assert partial.results[5].to_dict() == baseline.results[5].to_dict()


def test_cleaning_signature_tracks_plan(normal_dataset):
    dataset = load_dataset(normal_dataset)
    config = load_config(robot_type=dataset.info.get("robot_type"))
    decisions = decide(validate_dataset(dataset, config))
    signatures = {ep: f"sig-{ep}" for ep in decisions}

    before = cleaning_signature(decisions, signatures)
    assert cleaning_signature(decisions, signatures) == before

    changed = decisions[0]
    changed.disposition = "drop" if changed.disposition != "drop" else "keep"
    assert cleaning_signature(decisions, signatures) != before


def test_reusable_cleaning_requires_existing_output(tmp_path):
    output = tmp_path / "cleaned_dataset"
    output.mkdir()
    manifest = QualityManifest(tmp_path / "quality_manifest.json")
    manifest.store_cleaning("sig-a", {"output_path": str(output), "verification": {"ok": True}})

    assert manifest.reusable_cleaning("sig-a") is not None
    assert manifest.reusable_cleaning("sig-b") is None

    output.rename(tmp_path / "moved")
    assert manifest.reusable_cleaning("sig-a") is None


def test_manifest_save_is_atomic(tmp_path, monkeypatch):
    """写 manifest 中途失败时，旧文件必须保持完整，且不留 .tmp 残骸。"""
    import pipeline.manifest as manifest_mod

    path = tmp_path / "quality_manifest.json"
    manifest = QualityManifest(path)
    manifest.store_cleaning("sig", {"output_path": "x"})
    manifest.save()
    before = path.read_text(encoding="utf-8")

    def _boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(manifest_mod.json, "dump", _boom)
    manifest.store_cleaning("sig2", {"output_path": "y"})
    with pytest.raises(OSError):
        manifest.save()

    assert path.read_text(encoding="utf-8") == before
    assert not (tmp_path / "quality_manifest.json.tmp").exists()


def test_cleaner_failure_marks_run_degraded(normal_dataset, tmp_path, monkeypatch):
    """清洗阶段被吞掉时，退出码必须是 6，且报告里明确标记 cleaning.success=False。"""
    import main as main_module

    def _boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(main_module.cleaner_mod, "clean_dataset", _boom)
    out = tmp_path / "out"
    code = main([
        "--dataset_path", str(normal_dataset),
        "--output_dir", str(out),
        "--skip-pointcloud",
        "--no-cli",
        "--skip-plots",
    ])

    assert code == 6
    report = json.loads((out / "quality_report.json").read_text(encoding="utf-8"))
    assert report["cleaning"]["success"] is False
    assert "cleaned_dataset" not in {p.name for p in out.iterdir()}

    # Monitor 读 run_summary 就能知道本次是「跑完了但不能当成功」
    summary = json.loads((out / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "degraded"
    assert summary["exit_code"] == 6
    assert summary["degraded_reasons"] == ["cleaner"]


def _run_main(dataset_path, output_dir, *extra):
    return main([
        "--dataset_path", str(dataset_path),
        "--output_dir", str(output_dir),
        "--incremental",
        "--skip-clean",
        "--skip-pointcloud",
        "--no-cli",
        "--skip-plots",
        *extra,
    ])


def test_forced_incremental_run_rewrites_manifest(normal_dataset, tmp_path):
    """--incremental --force 跑完全量后必须写回 manifest，否则旧缓存无法刷新。"""
    out = tmp_path / "out"
    assert _run_main(normal_dataset, out, "--force") == 0

    manifest = QualityManifest.load(out / "quality_manifest.json")
    dataset = load_dataset(normal_dataset)
    assert set(manifest.cached_results()) == set(dataset.episode_indices)


def test_partial_run_does_not_write_manifest(normal_dataset, tmp_path):
    """--max_episodes 是子集，不能写入增量缓存。"""
    out = tmp_path / "out"
    assert _run_main(normal_dataset, out, "--force", "--max_episodes", "4") == 0
    assert not (out / "quality_manifest.json").exists()


# ------------------------------------------------------------- run_summary


def test_run_summary_written_on_success(normal_dataset, tmp_path):
    """成功跑完：run_summary 提供 Monitor 需要的全部字段。"""
    out = tmp_path / "out"
    assert _run_main(normal_dataset, out, "--force") == 0

    summary = json.loads((out / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "ok"
    assert summary["exit_code"] == 0
    assert summary["run_id"]
    assert summary["episodes"] > 0
    assert summary["frames"] > 0
    assert summary["elapsed_sec"] is not None and summary["elapsed_sec"] >= 0
    assert summary["degraded_reasons"] == []
    assert summary["reports"]["quality_report"].endswith("quality_report.json")
    assert Path(summary["reports"]["quality_report"]).is_file()


def test_incremental_reuse_writes_run_summary(normal_dataset, tmp_path):
    """增量复用也要留痕：status=reused，而不是让 Monitor 以为没跑过。"""
    out = tmp_path / "out"
    assert _run_main(normal_dataset, out, "--force") == 0
    assert _run_main(normal_dataset, out) == 0

    summary = json.loads((out / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "reused"
    assert summary["exit_code"] == 0
    assert summary["episodes"] > 0


def test_invalid_output_dir_returns_2(normal_dataset, tmp_path):
    """输出目录建不出来（父路径是文件）时返回 2，而不是抛原生 Traceback。"""
    blocker = tmp_path / "blocked"
    blocker.write_text("not a dir", encoding="utf-8")
    code = main([
        "--dataset_path", str(normal_dataset),
        "--output_dir", str(blocker / "out"),
        "--skip-pointcloud",
        "--skip-plots",
    ])
    assert code == 2


def test_strict_config_failure_returns_2(normal_dataset, tmp_path):
    """配置校验失败属于输入错误：退出码 2，并留下 failure_config.json。"""
    out = tmp_path / "out"
    cfg = tmp_path / "bad.yaml"
    cfg.write_text("defaults:\n  action_jump:\n    abs_floo: 0.5\n", encoding="utf-8")
    code = main([
        "--dataset_path", str(normal_dataset),
        "--output_dir", str(out),
        "--config", str(cfg),
        "--strict-config",
        "--skip-pointcloud",
        "--skip-plots",
    ])

    assert code == 2
    assert (out / "intermediate" / "failure_config.json").is_file()
    summary = json.loads((out / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "failed"
    assert summary["exit_code"] == 2


def test_entrypoint_handles_keyboard_interrupt(monkeypatch):
    """Ctrl+C 统一转成 130，不把 KeyboardInterrupt 的堆栈甩给用户。"""
    import main as main_module

    def _interrupt():
        raise KeyboardInterrupt()

    monkeypatch.setattr(main_module, "main", _interrupt)
    assert main_module._entrypoint() == 130
