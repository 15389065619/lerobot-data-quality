"""多模态对齐规则的测试：时间戳漂移 / 视频时间窗长度 / frame_index 断裂。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from conftest import FPS, attach_video, build_dataset, smooth_episode
from pipeline.config import load_config
from pipeline.loader import load_dataset
from pipeline.validator import validate_dataset

N = 40


def _reasons(root: Path, episode: int = 0) -> set[str]:
    dataset = load_dataset(root)
    result = validate_dataset(dataset, load_config())
    return set(result.results[episode].reasons)


def test_clean_window_has_no_alignment_issue(tmp_path: Path) -> None:
    root = build_dataset(tmp_path / "ds", {0: smooth_episode(N)})
    attach_video(root, {0: (0.0, N / FPS)})
    reasons = _reasons(root)
    assert "timestamp_drift" not in reasons
    assert "video_span_mismatch" not in reasons
    assert "frame_index_gap" not in reasons


def test_timestamp_drift_for_global_timestamps(tmp_path: Path) -> None:
    root = build_dataset(tmp_path / "ds", {0: smooth_episode(N)})
    attach_video(root, {0: (0.0, N / FPS)})
    # 把 timestamp 改成「文件内全局时间」并整体后移 5 帧，视频窗保持不变
    data_path = root / "data" / "chunk-000" / "file-000.parquet"
    df = pq.read_table(data_path).to_pandas()
    df["timestamp"] = df["timestamp"] + 5 / FPS
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), data_path)

    reasons = _reasons(root)
    assert "timestamp_drift" in reasons
    assert "video_span_mismatch" not in reasons  # 长度没变，只有位置错


def test_relative_timestamps_are_not_flagged_as_drift(tmp_path: Path) -> None:
    # v3 常见约定：episode 内 timestamp 从 0 开始，视频窗是文件内绝对时间
    root = build_dataset(tmp_path / "ds", {0: smooth_episode(N)})
    attach_video(root, {0: (5 / FPS, N / FPS + 5 / FPS)})
    reasons = _reasons(root)
    assert "timestamp_drift" not in reasons


def test_video_span_mismatch_when_window_is_shorter(tmp_path: Path) -> None:
    root = build_dataset(tmp_path / "ds", {0: smooth_episode(N)})
    attach_video(root, {0: (0.0, (N / FPS) * 0.8)})
    reasons = _reasons(root)
    assert "video_span_mismatch" in reasons


def test_frame_index_gap_detected(tmp_path: Path) -> None:
    root = build_dataset(tmp_path / "ds", {0: smooth_episode(N)})
    data_path = root / "data" / "chunk-000" / "file-000.parquet"
    df = pq.read_table(data_path).to_pandas()
    df["frame_index"] = np.where(df["frame_index"] >= 20, df["frame_index"] + 1, df["frame_index"])
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), data_path)

    reasons = _reasons(root)
    assert "frame_index_gap" in reasons


def test_alignment_can_be_disabled(tmp_path: Path) -> None:
    root = build_dataset(tmp_path / "ds", {0: smooth_episode(N)})
    attach_video(root, {0: (0.0, N / FPS)})
    data_path = root / "data" / "chunk-000" / "file-000.parquet"
    df = pq.read_table(data_path).to_pandas()
    df["timestamp"] = df["timestamp"] + 5 / FPS
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), data_path)

    enabled = validate_dataset(load_dataset(root), load_config())
    assert "timestamp_drift" in enabled.results[0].reasons

    config = load_config()
    config["defaults"]["alignment"]["enabled"] = False
    disabled = validate_dataset(load_dataset(root), config)
    assert "timestamp_drift" not in disabled.results[0].reasons
