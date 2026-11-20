"""第 1 步 d：统一归一化统计——用 B 组训练集算一份 stats，覆盖 A/B/eval。

这是两臂 loss 可比的前提：SmolVLA 按 dataset stats 做归一化，三份数据必须共用
同一份统计量。以 B 组（106 集完整清洗训练集）为准：
  - 数值特征（action/state/timestamp/索引列）：rewrite_stats 精确重算（parquet 全量）；
  - 图像特征：管道的流式统计扫不到视频像素（rewrite_stats_streaming 会沿用源
    stats.json 的旧值），这里用 cv2 逐帧解码 B 组训练视频、按 IMAGE_STRIDE 抽帧，
    精确累计 RGB 每通道 mean/std/min/max（与官方 stats.json 的 (3,1,1) 结构一致）。
写完 B 的 stats.json 后按字节复制到 A / eval（md5 必须一致）。

依赖顺序：必须在 transcode_h264.py 之后跑（统计口径 = 训练时实际解码的 h264 像素）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections import defaultdict

import cv2
import numpy as np

import sys as _sys
from pathlib import Path as _P
_SCRIPT_DIR = str(_P(__file__).resolve().parent)
if _SCRIPT_DIR not in _sys.path:
    _sys.path.insert(0, _SCRIPT_DIR)

from common import (
    DS_CLEAN,
    DS_EVAL,
    DS_MINREPAIR,
    IMAGE_STRIDE,
    OUT,
    dump_json,
    ensure_repo_on_path,
)

ensure_repo_on_path()

from pipeline.cleaner.meta import rewrite_stats  # noqa: E402
from pipeline.loader import load_dataset  # noqa: E402


def image_stats(view, cam: str, stride: int) -> tuple[dict, int]:
    """解码某相机的全部视频文件，按步长抽帧累计每通道统计。

    帧归属：meta/episodes 的 from/to timestamp（清洗器按 cum/fps 写入，精确对齐
    parquet 行数）。cv2 读出是 BGR，先转 RGB 再累计——lerobot 归一化按 RGB。
    返回 (官方 stats.json 结构, 实际累计帧数)。
    """
    fps = float(view.fps or 30.0)
    # 按 (文件, 集内帧窗) 分组
    by_file: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for ep in view.episode_indices:
        loc = view.video_location(cam, ep)
        if loc is None:
            raise SystemExit(f"[错误] {cam} episode {ep} 无法定位视频")
        path, from_ts, to_ts = loc
        if to_ts is None or to_ts <= 0 or from_ts < 0:
            raise SystemExit(f"[错误] {cam} episode {ep} 视频窗口缺失: {path.name}")
        start = int(round(from_ts * fps))
        end = int(round(to_ts * fps))
        if end <= start:
            raise SystemExit(f"[错误] {cam} episode {ep} 视频窗口为空: [{start},{end})")
        by_file[str(path)].append((start, end))

    total_sum = np.zeros(3, dtype=np.float64)
    total_sumsq = np.zeros(3, dtype=np.float64)
    running_min = np.full(3, np.inf)
    running_max = np.full(3, -np.inf)
    n_frames = 0
    n_pixels_per_frame = None

    for file_str, windows in sorted(by_file.items()):
        expected = max(e for _, e in windows)
        cap = cv2.VideoCapture(file_str)
        if not cap.isOpened():
            raise SystemExit(f"[错误] cv2 打不开视频: {file_str}")
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx >= expected:
                break  # 尾部多余帧（理论上没有）不进统计
            if idx % stride == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).reshape(-1, 3).astype(np.float64)
                total_sum += rgb.sum(axis=0)
                total_sumsq += (rgb ** 2).sum(axis=0)
                running_min = np.minimum(running_min, rgb.min(axis=0))
                running_max = np.maximum(running_max, rgb.max(axis=0))
                n_pixels_per_frame = rgb.shape[0]
                n_frames += 1
            idx += 1
        cap.release()
        if idx < expected:
            raise SystemExit(f"[错误] {file_str} 实际解码 {idx} 帧 < 元数据 {expected} 帧")

    # mean/std 按【像素】计：除以总像素数 = n_frames * 每帧像素数，
    # 除成帧数会得到百万级的均值（首版 bug，被 verify 的 0-255 范围检查抓住）
    if not n_frames or not n_pixels_per_frame:
        raise SystemExit("[错误] 没有累计到任何抽帧，拒绝发布")
    n_px_total = n_frames * n_pixels_per_frame
    mean = total_sum / n_px_total
    std = np.sqrt(np.maximum(total_sumsq / n_px_total - mean ** 2, 0.0))
    if not ((0 <= mean).all() and (mean <= 255).all() and (std > 0).all()):
        raise SystemExit(f"[错误] 图像统计超出合理范围: mean={mean} std={std}")

    def ch(v: np.ndarray) -> list:
        return np.round(v.reshape(3, 1, 1), 6).tolist()

    stats = {
        "count": int(n_frames),
        "mean": ch(mean),
        "std": ch(std),
        "min": ch(running_min),
        "max": ch(running_max),
    }
    return stats, n_frames


def md5(path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="B 组统一 stats 覆盖三份数据集")
    parser.add_argument("--stride", type=int, default=IMAGE_STRIDE, help="图像统计抽帧步长")
    args = parser.parse_args()

    view = load_dataset(DS_CLEAN)
    print(f"B 组: {len(view.episode_indices)} 集 / {view.num_frames} 帧 / {view.fps}Hz")

    # 1) 数值特征精确重算（直接写 so101_clean/meta/stats.json）
    warnings = rewrite_stats(view, view.frames, DS_CLEAN)
    for w in warnings:
        print(f"[警告] {w}", file=sys.stderr)

    # 2) 图像特征：cv2 解码统计
    stats_path = DS_CLEAN / "meta" / "stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    for cam in view.video_features:
        img_stats, n = image_stats(view, cam, args.stride)
        stats[cam] = img_stats
        mean = np.asarray(img_stats["mean"]).reshape(-1)
        std = np.asarray(img_stats["std"]).reshape(-1)
        print(f"  {cam}: {n} 帧抽样  mean={np.round(mean, 3).tolist()}  std={np.round(std, 3).tolist()}")

    # 3) 数值统计必须是有限值（B 是清洗后数据，出现 NaN 说明上游有问题）
    for key in ("action", "observation.state"):
        arr = np.asarray(stats[key]["mean"], dtype=float)
        if not np.isfinite(arr).all():
            sys.exit(f"[错误] {key} 统计含 NaN/Inf，拒绝发布")

    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    # 4) 按字节复制到 A / eval，md5 必须一致
    for dst in (DS_MINREPAIR, DS_EVAL):
        target = dst / "meta" / "stats.json"
        shutil.copyfile(stats_path, target)
        if md5(stats_path) != md5(target):
            sys.exit(f"[错误] stats.json 复制后 md5 不一致: {target}")
    print(f"已覆盖: {DS_MINREPAIR / 'meta' / 'stats.json'}")
    print(f"        {DS_EVAL / 'meta' / 'stats.json'}")

    covered = sorted(k for k in stats if k not in
                     ("timestamp", "frame_index", "episode_index", "index", "task_index"))
    print(f"特征覆盖: {covered}")
    print(f"md5: {md5(stats_path)}")
    print("下一步: verify_datasets.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
