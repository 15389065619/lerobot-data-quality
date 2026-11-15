#!/usr/bin/env python3
"""一次性脚本：构造模拟 Pico 上传 Bug 的数据集。

基于现有 v3 数据集复制一份，注入两类真实缺陷：

1. **缺相机参数**：从 `meta/info.json` 的 `features` 中删掉某个相机特征，
   但保留磁盘上的视频文件与 `meta/episodes` 里的时间窗——模拟「元数据先丢、数据还在」。
2. **上传中断**：把某个相机最后一个视频文件重编码为前 N% 帧，
   parquet 的 state/action 行数不变，尾部 episode 的视频窗口会越过视频实际结尾。

用法：
    python tools/make_pico_bug_dataset.py
    python tools/make_pico_bug_dataset.py --src demo_dataset --dst pico_bug_dataset --keep-ratio 0.6
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

VIDEO_FEATURE_PREFIX = "observation.images."


def _probe_frames(path: Path) -> int:
    """用 ffprobe 读容器声明的帧数；读不到时按时长 × fps 估算。"""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=nb_frames,duration,avg_frame_rate",
         "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout
    stream = json.loads(out)["streams"][0]
    nb = stream.get("nb_frames")
    if nb and str(nb).isdigit() and int(nb) > 0:
        return int(nb)
    num, _, den = str(stream.get("avg_frame_rate", "30/1")).partition("/")
    fps = float(num) / float(den or 1)
    return int(round(float(stream.get("duration") or 0) * fps))


def inject_missing_feature(info_path: Path, camera: str) -> dict:
    """Bug1：删除相机特征定义，视频与 meta 时间窗保持原样。"""
    info = json.loads(info_path.read_text(encoding="utf-8"))
    features = info.get("features") or {}
    if camera not in features:
        raise SystemExit(f"[bug1] features 中没有 {camera}，无法注入；现有相机："
                         f"{[k for k in features if k.startswith(VIDEO_FEATURE_PREFIX)]}")
    removed = features.pop(camera)
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"camera": camera, "removed_spec": removed}


def inject_truncated_video(root: Path, camera: str, keep_ratio: float) -> dict:
    """Bug2：把该相机最后一个 mp4 重编码为前 N% 帧，模拟上传中断。"""
    files = sorted((root / "videos" / camera).rglob("*.mp4"))
    if not files:
        raise SystemExit(f"[bug2] 找不到 {camera} 的视频文件")
    target = files[-1]
    original = _probe_frames(target)
    keep = max(1, int(original * keep_ratio))
    tmp = target.with_suffix(".trim.mp4")
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(target),
         "-frames:v", str(keep), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an", str(tmp)],
        check=True,
    )
    tmp.replace(target)
    after = _probe_frames(target)
    return {"file": str(target), "frames_before": original, "frames_after": after,
            "keep_ratio": keep_ratio}


def main() -> int:
    parser = argparse.ArgumentParser(description="构造模拟 Pico 上传 Bug 的 v3 数据集")
    parser.add_argument("--src", default="demo_dataset", help="源数据集（v3.0，含 meta/info.json）")
    parser.add_argument("--dst", default="pico_bug_dataset", help="输出数据集目录")
    parser.add_argument("--missing-camera", default="observation.images.wrist",
                        help="Bug1：从 features 中删除的相机")
    parser.add_argument("--truncate-camera", default="observation.images.top",
                        help="Bug2：截断哪个相机的视频")
    parser.add_argument("--keep-ratio", type=float, default=0.6,
                        help="Bug2：保留的帧比例（0<ratio<1）")
    parser.add_argument("--force", action="store_true", help="目标目录已存在时先删除")
    args = parser.parse_args()

    src = Path(args.src).resolve()
    dst = Path(args.dst).resolve()
    if not (src / "meta" / "info.json").is_file():
        raise SystemExit(f"源数据集不合法：{src} 缺少 meta/info.json")
    if not 0 < args.keep_ratio < 1:
        raise SystemExit("--keep-ratio 必须在 (0, 1) 之间")
    if dst.exists():
        if not args.force:
            raise SystemExit(f"目标已存在：{dst}（加 --force 覆盖）")
        shutil.rmtree(dst)

    shutil.copytree(src, dst)

    bug1 = inject_missing_feature(dst / "meta" / "info.json", args.missing_camera)
    bug2 = inject_truncated_video(dst, args.truncate_camera, args.keep_ratio)

    print("=" * 72)
    print(f"已生成：{dst}")
    print(f"Bug1 缺相机参数：features 删除 {bug1['camera']}")
    print(f"      保留的视频文件：{(dst / 'videos' / bug1['camera']).as_posix()}（孤儿数据）")
    print(f"Bug2 上传中断：{bug2['file']}")
    print(f"      帧数 {bug2['frames_before']} -> {bug2['frames_after']}"
          f"（保留 {bug2['keep_ratio']:.0%}），parquet 行数未改")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
