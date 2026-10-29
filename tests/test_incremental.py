"""增量模式：manifest 指纹、结论复用与失效条件。"""

from __future__ import annotations

import numpy as np

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
