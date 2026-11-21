"""双 take（多段 meta）尺子回归：对齐与帧数规则必须按段计量。

真实场景（so101_bomb ep81-102）：同一 episode_index 在 meta/episodes 里有两行
（每行自带 length 与视频窗，甚至不同视频文件），parquet 行数 = 各段之和。
旧尺子只取第一行窗口比全集行数，把正常双 take 系统性误报成
frame_mismatch(error) + video_span_mismatch——正是 A/B 实验 29 个 drop 集
假阳性的根因（详见 results.md / 论文 §6.2 的 badgood 链路）。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from conftest import FPS, build_dataset, smooth_episode
from pipeline.config import load_config
from pipeline.loader import load_dataset
from pipeline.validator import validate_dataset
from pipeline.validator import rules
from pipeline.validator.types import EpisodeResult

CAM = "observation.images.top"
N1, N2 = 24, 20
TOTAL = N1 + N2
W1 = (0.0, N1 / FPS)
W2 = (0.0, N2 / FPS)


def _dual_take_dataset(
    tmp_path: Path,
    windows: tuple[tuple[float, float], tuple[float, float]] = (W1, W2),
    files: tuple[int, int] = (0, 1),
) -> Path:
    """造一个双 take 数据集：ep0 共 44 行，meta 两行各带 length 与视频窗。"""
    root = build_dataset(tmp_path / "ds", {0: smooth_episode(TOTAL)})
    ep_path = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    base = pd.read_parquet(ep_path).iloc[0].to_dict()
    rows = []
    for length, (frm, to), fi in zip((N1, N2), windows, files):
        row = dict(base)
        row["length"] = length
        row[f"videos/{CAM}/chunk_index"] = 0
        row[f"videos/{CAM}/file_index"] = fi
        row[f"videos/{CAM}/from_timestamp"] = frm
        row[f"videos/{CAM}/to_timestamp"] = to
        rows.append(row)
    pq.write_table(pa.Table.from_pylist(rows), ep_path)
    for fi in set(files):
        video = root / "videos" / CAM / "chunk-000" / f"file-{fi:03d}.mp4"
        video.parent.mkdir(parents=True, exist_ok=True)
        video.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x01" * 1024)
    return root


def _episode_reasons(root: Path) -> EpisodeResult:
    ds = load_dataset(root)
    result = validate_dataset(ds, load_config())
    return result.results[0]


def test_dual_take_clean_window_not_flagged(tmp_path: Path) -> None:
    """两段窗口各自与段长一致、Σ 跨度 = 全集时长 → 不报 span/frame 误判。"""
    root = _dual_take_dataset(tmp_path)
    res = _episode_reasons(root)
    assert "video_span_mismatch" not in res.reasons
    assert res.frame_count == TOTAL


def test_dual_take_span_still_catches_bad_window(tmp_path: Path) -> None:
    """某段窗口写短了 → Σ 跨度偏离全集时长，仍会报，且带按段证据。"""
    root = _dual_take_dataset(tmp_path, windows=(W1, (0.0, N2 / FPS * 0.5)))
    res = _episode_reasons(root)
    assert "video_span_mismatch" in res.reasons
    cameras = res.details["video_span_mismatch"]["cameras"]
    assert cameras[0]["segments"] == 2
    # 两段窗口 from 相同，排序后顺序不定，按集合断言（证据 span 舍入到 4 位小数）
    spans = sorted(p["span"] for p in cameras[0]["per_segment"])
    assert spans == pytest.approx([N2 / FPS * 0.5, N1 / FPS], abs=1e-3)


def test_dual_take_drift_pairs_first_and_last_segment(tmp_path: Path) -> None:
    """全局时间戳整体后移：首/末时间戳分别对首段起点、末段终点判定。"""
    root = _dual_take_dataset(tmp_path)
    data_path = root / "data" / "chunk-000" / "file-000.parquet"
    df = pq.read_table(data_path).to_pandas()
    df["timestamp"] = df["timestamp"] + 5 / FPS
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), data_path)
    res = _episode_reasons(root)
    assert "timestamp_drift" in res.reasons
    cameras = res.details["timestamp_drift"]["cameras"]
    assert cameras[0]["segments"] == 2


def test_dual_take_frame_count_not_flagged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """各段视频帧数与各段窗口一致（Σ = 全集行数）→ 不报 frame_mismatch。"""
    root = _dual_take_dataset(tmp_path)
    ds = load_dataset(root)
    totals = {0: N1, 1: N2}  # file_index -> 视频总帧数
    monkeypatch.setattr(
        rules, "probe_video_frames",
        lambda path, precise=False: totals[int(path.stem.split("-")[-1])],
    )
    res = EpisodeResult(episode_index=0, frame_count=TOTAL)
    rules.check_frame_mismatch(res, 0, ds, cameras=[CAM], precise=False, config=load_config())
    assert "frame_mismatch" not in res.reasons


def test_dual_take_frame_count_still_catches_short_video(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """某段视频真短了（窗口越出文件末尾）→ video_truncated 照报。"""
    root = _dual_take_dataset(tmp_path)
    ds = load_dataset(root)
    totals = {0: N1, 1: N2 - 5}
    monkeypatch.setattr(
        rules, "probe_video_frames",
        lambda path, precise=False: totals[int(path.stem.split("-")[-1])],
    )
    res = EpisodeResult(episode_index=0, frame_count=TOTAL)
    rules.check_frame_mismatch(res, 0, ds, cameras=[CAM], precise=False, config=load_config())
    assert "frame_mismatch" in res.reasons
    problems = res.details["frame_mismatch"]["problems"]
    assert any(p["type"] == "video_truncated" for p in problems)


def test_dual_take_frame_count_count_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Σ 各段窗口帧数 ≠ 全集行数 → count_mismatch，证据带段数。

    场景：take2 的 meta 窗口写短了（10 帧），但视频文件本身与窗口一致
    （也是 10 帧）——视频没坏，是 meta 窗口与行数对不上。
    """
    root = _dual_take_dataset(tmp_path, windows=(W1, (0.0, N2 / FPS * 0.5)))
    ds = load_dataset(root)
    totals = {0: N1, 1: int(round(N2 / FPS * 0.5 * FPS))}  # 10 帧，与窗口一致
    monkeypatch.setattr(
        rules, "probe_video_frames",
        lambda path, precise=False: totals[int(path.stem.split("-")[-1])],
    )
    res = EpisodeResult(episode_index=0, frame_count=TOTAL)
    rules.check_frame_mismatch(res, 0, ds, cameras=[CAM], precise=False, config=load_config())
    assert "frame_mismatch" in res.reasons
    problems = res.details["frame_mismatch"]["problems"]
    assert any(p["type"] == "count_mismatch" and p.get("segments") == 2 for p in problems)


def test_dual_take_parallel_matches_serial(tmp_path: Path) -> None:
    """并行路径也要按段计量：stub 必须拿到未去重的 meta 段行。"""
    root = _dual_take_dataset(tmp_path)
    serial = validate_dataset(load_dataset(root), load_config())
    parallel = validate_dataset(load_dataset(root), load_config(), workers=2)
    assert serial.results[0].reasons == parallel.results[0].reasons
    assert "video_span_mismatch" not in parallel.results[0].reasons
