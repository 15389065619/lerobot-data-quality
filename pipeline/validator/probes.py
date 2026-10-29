"""视频探针：ffprobe 帧数统计与 ffmpeg 亮度抽样，带进程内缓存。"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Any

import numpy as np

LOG = logging.getLogger(__name__)

FFPROBE = shutil.which("ffprobe")
FFMPEG = shutil.which("ffmpeg")

_FRAME_CACHE: dict[str, int | None] = {}
_LUMA_CACHE: dict[str, Any] = {}
_CACHE_LOCK = threading.Lock()
_KEY_LOCKS: dict[str, threading.Lock] = {}


def _key_lock(key: str) -> threading.Lock:
    """同一个视频文件只允许一个线程探针：避免并发重复解码、机械盘来回寻道。"""
    with _CACHE_LOCK:
        return _KEY_LOCKS.setdefault(key, threading.Lock())


def probe_frame_brightness(path: str | Path, start_ts: float, duration: float) -> tuple[list[float], list[float]] | None:
    """抽样若干帧，返回 (每帧平均亮度, 每帧亮度标准差)。"""
    key = f"{path}@{start_ts:.2f}+{duration:.2f}"
    with _key_lock(key):
        return _probe_frame_brightness(key, path, start_ts, duration)


def _probe_frame_brightness(
    key: str, path: str | Path, start_ts: float, duration: float
) -> tuple[list[float], list[float]] | None:
    if key in _LUMA_CACHE:
        return _LUMA_CACHE[key]
    if not FFMPEG:
        return None
    w, h = 64, 48
    cmd = [
        str(FFMPEG), "-v", "error", "-ss", f"{start_ts:.3f}", "-t", f"{duration:.3f}",
        "-i", str(path), "-vf", f"fps=2,scale={w}:{h}", "-f", "rawvideo", "-pix_fmt", "gray", "-",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=120, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        LOG.debug("亮度抽样失败 %s: %s", path, exc)
        _LUMA_CACHE[key] = None
        return None
    buf = np.frombuffer(proc.stdout, dtype=np.uint8)
    frame_size = w * h
    if proc.returncode != 0 or buf.size < frame_size:
        _LUMA_CACHE[key] = None
        return None
    n = buf.size // frame_size
    frames = buf[: n * frame_size].reshape(n, frame_size).astype(np.float32)
    result = (frames.mean(axis=1).tolist(), frames.std(axis=1).tolist())
    _LUMA_CACHE[key] = result
    return result


def probe_video_frames(path: str | Path, precise: bool = False) -> int | None:
    """统计视频帧数。

    真实数据集上全解码代价极高（几十 GB），因此优先读容器元数据：
    1. stream.nb_frames（最快，h264/mp4 通常有）
    2. duration × fps（无 nb_frames 时估算）
    3. -count_frames（精确但需全解码，仅在 precise=True 时使用）
    """
    key = f"{path}|{precise}"
    with _key_lock(key):
        return _probe_video_frames(key, path, precise)


def _probe_video_frames(key: str, path: str | Path, precise: bool) -> int | None:
    if key in _FRAME_CACHE:
        return _FRAME_CACHE[key]
    if not FFPROBE:
        _FRAME_CACHE[key] = None
        return None

    count = _ffprobe_int(path, ["-show_entries", "stream=nb_frames"])
    if count is None:
        count = _ffprobe_estimate(path)
    if count is None and precise:
        count = _ffprobe_int(path, ["-count_frames", "-show_entries", "stream=nb_read_frames"])

    _FRAME_CACHE[key] = count
    if count is None:
        LOG.debug("无法获取视频帧数: %s", path)
    return count


def _ffprobe_int(path: str | Path, extra: list[str]) -> int | None:
    cmd = [str(FFPROBE), "-v", "error", "-select_streams", "v:0", *extra,
           "-of", "default=nw=1:nk=1", str(path)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    for line in (proc.stdout or "").splitlines():
        line = line.strip()
        if not line or line.upper() == "N/A":
            continue
        try:
            value = float(line)
        except ValueError:
            continue
        if value > 0:
            return int(round(value))
    return None


def _ffprobe_estimate(path: str | Path) -> int | None:
    cmd = [str(FFPROBE), "-v", "error", "-select_streams", "v:0",
           "-show_entries", "stream=avg_frame_rate,duration", "-of", "json", str(path)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    try:
        stream = (json.loads(proc.stdout or "{}").get("streams") or [{}])[0]
        num, _, den = str(stream.get("avg_frame_rate") or "0/0").partition("/")
        fps = float(num) / float(den) if float(den or 0) else 0.0
        duration = float(stream.get("duration") or 0.0)
    except (ValueError, ZeroDivisionError, TypeError, json.JSONDecodeError):
        return None
    if fps <= 0 or duration <= 0:
        return None
    return int(round(fps * duration))
