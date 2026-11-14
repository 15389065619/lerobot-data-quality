"""视频窗缺失 (-1,-1) 时，黑屏检测不能退化成「从文件 0 点抽帧」。"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from pipeline.validator import rules
from pipeline.validator.types import EpisodeResult


def _dataset(path, window: tuple[float, float]) -> SimpleNamespace:
    return SimpleNamespace(video_location=lambda camera, ep: (path, *window), fps=30.0)


def _patch_probe(monkeypatch) -> list[tuple]:
    calls: list[tuple] = []

    def fake_probe(path, start_ts, duration):
        calls.append((path, start_ts, duration))
        return [10.0, 10.0], [5.0, 5.0]  # 足够亮，不触发 blank_video

    monkeypatch.setattr(rules, "FFMPEG", "ffmpeg")
    monkeypatch.setattr(rules, "probe_frame_brightness", fake_probe)
    return calls


@pytest.fixture
def video_file(tmp_path):
    path = tmp_path / "videos" / "cam" / "chunk-000" / "file-000.mp4"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"\x00" * 32)
    return path


def test_missing_video_window_skips_probe(video_file, monkeypatch):
    calls = _patch_probe(monkeypatch)
    res = EpisodeResult(episode_index=0, frame_count=40)

    rules.check_blank_video(res, 0, _dataset(video_file, (-1.0, -1.0)), "cam", {})

    assert calls == []
    assert "blank_video" not in res.reasons


def test_valid_video_window_probes_its_own_segment(video_file, monkeypatch):
    calls = _patch_probe(monkeypatch)
    res = EpisodeResult(episode_index=0, frame_count=60)

    rules.check_blank_video(res, 0, _dataset(video_file, (4.0, 6.0)), "cam", {})

    assert calls == [(video_file, 4.0, 2.0)]
