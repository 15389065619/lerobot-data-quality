from __future__ import annotations

import json

import numpy as np
import pyarrow.parquet as pq

from pipeline.cleaner import clean_dataset
from pipeline.loader import load_dataset
from tools.record_sim_episode import record


def _stats(out_dir):
    return json.loads(
        (out_dir / "cleaned_dataset" / "meta" / "stats.json").read_text(encoding="utf-8")
    )


def test_record_sim_dataset_is_valid_v3(tmp_path):
    root = record(tmp_path / "sim", episodes=2, frames=40, fps=20, seed=1)
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    assert info["total_episodes"] == 2
    assert info["total_frames"] == 80
    assert info["fps"] == 20
    assert info["features"]["observation.state"]["shape"] == [8]
    assert info["features"]["observation.body_pos_w"]["shape"] == [5, 3]

    episodes = pq.read_table(root / "meta" / "episodes" / "chunk-000" / "file-000.parquet").to_pylist()
    assert [row["length"] for row in episodes] == [40, 40]
    assert [row["data/file_index"] for row in episodes] == [0, 0]


def test_sim_dataset_runs_through_cleaning(tmp_path):
    root = record(tmp_path / "sim", episodes=2, frames=40)
    result = clean_dataset(load_dataset(root), [], tmp_path / "out", prefer_cli=False)
    assert result.success is True
    assert result.method == "noop-copy"

    stats = _stats(tmp_path / "out")
    assert stats["observation.state"]["count"] == 80
    assert np.asarray(stats["observation.body_pos_w"]["mean"]).shape == (5, 3)
    assert np.asarray(stats["observation.body_pos_w"]["mean"]).ndim == 2


def test_inject_anomaly_shows_up_in_stats(tmp_path):
    root = record(tmp_path / "sim", episodes=2, frames=40, inject_anomaly=True)
    result = clean_dataset(load_dataset(root), [], tmp_path / "out", prefer_cli=False)
    assert result.success is True
    stats = _stats(tmp_path / "out")
    assert stats["observation.state"]["count"] == 77  # ep0 中段 3 帧 NaN 被剔除
    assert stats["observation.body_pos_w"]["count"] == 77
