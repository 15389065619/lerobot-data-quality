"""Dashboard 服务测试：API 契约、视频 Range 转发、路径安全。"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from conftest import FPS, attach_video, build_dataset, smooth_episode
from tools.dashboard_server import DashboardState, make_server

N = 40


@pytest.fixture
def dashboard(tmp_path: Path):
    root = build_dataset(tmp_path / "ds", {0: smooth_episode(N), 1: smooth_episode(N, offset=1)})
    attach_video(root, {0: (0.0, N / FPS), 1: (N / FPS, 2 * N / FPS)})
    state = DashboardState(root, None, max_points=16)
    server = make_server(state, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    yield base, root
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


def _get(url: str, headers: dict[str, str] | None = None):
    return urllib.request.urlopen(urllib.request.Request(url, headers=headers or {}), timeout=10)


def test_summary_contract(dashboard) -> None:
    base, root = dashboard
    with _get(f"{base}/api/summary") as res:
        data = json.load(res)
    assert data["kpis"]["total_episodes"] == 2
    assert data["kpis"]["total_frames"] == 2 * N
    assert [e["episode_index"] for e in data["episodes"]] == [0, 1]
    assert data["dataset"]["cameras"] == ["observation.images.top"]


def test_episode_series_is_downsampled_and_has_video_window(dashboard) -> None:
    base, _ = dashboard
    with _get(f"{base}/api/episode/0") as res:
        data = json.load(res)
    assert len(data["timestamp"]) <= 17  # max_points=16 + 末点
    assert len(data["action"]) == len(data["timestamp"])
    assert len(data["state"]) == len(data["timestamp"])
    assert data["action_names"] == ["dim_0", "dim_1", "dim_2"]
    win = data["videos"]["observation.images.top"]
    assert win["from"] == 0.0
    assert abs(win["to"] - N / FPS) < 1e-3


def test_unknown_episode_returns_404(dashboard) -> None:
    base, _ = dashboard
    with pytest.raises(urllib.error.HTTPError) as exc:
        _get(f"{base}/api/episode/999")
    assert exc.value.code == 404


def test_video_range_request(dashboard) -> None:
    base, root = dashboard
    size = (root / "videos" / "observation.images.top" / "chunk-000" / "file-000.mp4").stat().st_size
    with _get(f"{base}/video/0/observation.images.top", {"Range": "bytes=0-99"}) as res:
        body = res.read()
        assert res.status == 206
        assert res.headers["Content-Range"] == f"bytes 0-99/{size}"
        assert res.headers["Accept-Ranges"] == "bytes"
    assert len(body) == 100

    with _get(f"{base}/video/0/observation.images.top") as res:
        assert res.status == 200
        assert len(res.read()) == size


def test_nan_series_is_valid_json(tmp_path: Path) -> None:
    root = build_dataset(tmp_path / "ds", {0: smooth_episode(N)})
    data_path = root / "data" / "chunk-000" / "file-000.parquet"
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    df = pq.read_table(data_path).to_pandas()
    row = df.index[df["frame_index"] == N // 2][0]
    state_value = np.array(df.at[row, "observation.state"], dtype=np.float32)
    state_value[0] = np.nan
    df.at[row, "observation.state"] = state_value
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), data_path)

    state = DashboardState(root, None)
    server = make_server(state, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with _get(f"http://127.0.0.1:{server.server_address[1]}/api/episode/0") as res:
            body = res.read().decode("utf-8")
        assert "NaN" not in body and "Infinity" not in body  # 浏览器 JSON.parse 会直接报错
        data = json.loads(body)
        assert any(v is None for v in data["state"][N // 2])
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_index_page_and_unknown_camera(dashboard) -> None:
    base, _ = dashboard
    with _get(f"{base}/") as res:
        assert "质检 Dashboard" in res.read().decode("utf-8")
    with pytest.raises(urllib.error.HTTPError) as exc:
        _get(f"{base}/video/0/nonexistent.cam")
    assert exc.value.code == 404
