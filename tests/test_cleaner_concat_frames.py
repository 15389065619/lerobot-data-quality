"""回归测试：分段重编码的帧数精确性。

历史 bug：`_concat_segments` 用 `-t 时长` 截断，而帧 pts 是精确的 k/fps（无限循环
小数）。`-t` 的有限小数只要小于精确时长就把最后一帧排除——`43/30` 格化成
1.433333 必丢末帧、`50/30` 进位成 1.666667 则不丢，表现为「部分集各丢 1 帧、
窗口与 parquet 长度错位，集尾样本读到下一集画面」。修复改为 `-ss` 提前半帧 +
`-frames:v` 按帧数硬截断。

本测试用真 ffmpeg 编码一个 CFR 源视频（3 集 130 帧，长度 [43, 37, 50]——两集命中
向下舍位模式），剔除第 3 集触发分段重编码路径，断言：
  1. 产物视频总帧数 == 保留集 parquet 长度之和（旧代码会得 78 ≠ 80）；
  2. 每集视频窗口宽度 == parquet 长度；
  3. 无「帧数不精确 / 偏差」类警告。
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from pipeline.cleaner import clean_dataset
from pipeline.loader import load_dataset

pytest.importorskip("pyarrow")

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")
pytestmark = pytest.mark.skipif(
    not (FFMPEG and FFPROBE), reason="需要 ffmpeg/ffprobe（真编码视频的集成测试）"
)

FPS = 30
LENGTHS = [43, 37, 50]  # 43/30 与 37/30 在 6 位小数下向下舍位——旧代码必丢帧
CAMERA = "observation.images.top"


def _probe_frames(path: Path) -> int:
    proc = subprocess.run(
        [FFPROBE, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=nb_frames", "-of", "json", str(path)],
        capture_output=True, text=True, check=False,
    )
    return int(json.loads(proc.stdout)["streams"][0]["nb_frames"])


def _probe_pts(path: Path) -> list[float]:
    """逐帧 pts_time（秒）。拼接产物的时间戳对齐用。"""
    proc = subprocess.run(
        [FFPROBE, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "frame=pts_time", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=False,
    )
    return [float(x) for x in proc.stdout.replace(",", " ").split() if x.strip()]


def _encode_cfr_source(path: Path, n_frames: int) -> None:
    """rawvideo 管进 libx264：CFR 30fps、pts 精确 = k/30，帧数精确。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    frame = rng.integers(0, 255, size=(8, 8, 3), dtype=np.uint8)
    cmd = [
        FFMPEG, "-y", "-v", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "8x8", "-r", str(FPS),
        "-i", "-", "-an", "-c:v", "libx264", "-preset", "veryfast",
        "-pix_fmt", "yuv420p", "-fps_mode", "passthrough", str(path),
    ]
    proc = subprocess.run(cmd, input=frame.tobytes() * n_frames,
                          capture_output=True, check=False)
    assert proc.returncode == 0, proc.stderr.decode()[-300:]
    assert _probe_frames(path) == n_frames


def test_segment_reencode_frame_exact(tmp_path: Path) -> None:
    from conftest import build_dataset

    root = build_dataset(tmp_path / "ds", {ep: {"state": np.zeros((n, 3), np.float32),
                                                "action": np.zeros((n, 3), np.float32)}
                                          for ep, n in enumerate(LENGTHS)})
    total = sum(LENGTHS)
    video = root / "videos" / CAMERA / "chunk-000" / "file-000.mp4"
    _encode_cfr_source(video, total)

    # 写入真实视频窗口元数据（三集平铺 [0,43),(43,80),(80,130) / 30）
    ep_path = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    df = pq.read_table(ep_path).to_pandas()
    df[f"videos/{CAMERA}/chunk_index"] = 0
    df[f"videos/{CAMERA}/file_index"] = 0
    cum = 0
    froms, tos = [], []
    for n in LENGTHS:
        froms.append(cum / FPS)
        cum += n
        tos.append(cum / FPS)
    df[f"videos/{CAMERA}/from_timestamp"] = froms
    df[f"videos/{CAMERA}/to_timestamp"] = tos
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), ep_path)

    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    info["features"][CAMERA] = {
        "dtype": "video", "shape": [8, 8, 3], "names": ["h", "w", "c"],
        "info": {"video.fps": FPS, "video.codec": "libx264"},
    }
    (root / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")

    # 剔除第 3 集（file 含被剔集 -> 走分段重编码路径）
    view = load_dataset(root)
    out_root = tmp_path / "cleaned"
    result = clean_dataset(view, [2], tmp_path, prefer_cli=False, overwrite=True,
                           out_root=out_root)
    assert result.success

    # 1) 产物视频总帧数 == 保留集 parquet 长度之和（旧代码 78 != 80）
    out_video = out_root / "videos" / CAMERA / "chunk-000" / "file-000.mp4"
    kept_sum = sum(LENGTHS[:2])
    assert _probe_frames(out_video) == kept_sum

    # 2) 每集窗口宽度 == parquet 长度
    meta = pd.read_parquet(out_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    assert len(meta) == 2
    for _, r in meta.iterrows():
        width = round(float(r[f"videos/{CAMERA}/to_timestamp"]) * FPS) - \
            round(float(r[f"videos/{CAMERA}/from_timestamp"]) * FPS)
        assert width == int(r["length"]), f"ep {r['episode_index']} 窗口宽 {width} != length {r['length']}"

    # 3) 无帧数不精确/偏差类警告（损坏源才允许告警；本源是干净 CFR）
    bad = [w for w in result.warnings if ("帧数不精确" in w or "存在偏差" in w)]
    assert not bad, f"不应出现帧数告警: {bad}"

    # 4) pts 对齐（历史 bug 第二模式：段数 ±1 帧 → 拼接 -c copy 时后续集整体偏移，
    #    Kaggle 训练以 FrameTimestampError 崩溃：首帧 pts 偏 33ms > tolerance 0.01s）
    pts = _probe_pts(out_video)
    assert len(pts) == kept_sum
    assert pts[0] < 0.002, f"首帧 pts={pts[0]}，拼接产物起始时间戳未归零"
    drift = max(abs(p - i / FPS) for i, p in enumerate(pts))
    assert drift < 0.5 / FPS, (
        f"pts 累计漂移 {drift * 1000:.1f}ms 超过半帧，元数据窗口与实际画面将错位"
    )
