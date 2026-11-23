"""第 1 步 c：三份数据集的视频统一转码 h264（原地，逐文件原子替换）。

为什么转码：
  1. 源是 AV1，Kaggle 只有 4 vCPU，软解 AV1 会饿住 2×T4；
  2. clean_dataset 对「含被剔 episode 的文件」走 ffmpeg 分段重编码（h264），
     对整文件保留的文件原样复制（AV1）——不统一转码的话数据集内编码混杂，
     info.json 声明的 codec 也会与实际不符。

做法：ffprobe 探测每个 mp4；已是 h264 的跳过；否则
  ffmpeg -i in -map 0:v:0 -an -c:v libx264 -preset veryfast -crf 20
         -pix_fmt yuv420p -fps_mode passthrough out
  -fps_mode passthrough 逐帧透传不重定时，保帧数；转码后 probe 对比帧数，
  有差集记进 manifest（A 组的坏集视频允许差，属内容级问题，如实留痕）。
最后把 info.json 的 features[cam].info["video.codec"] 补成 h264。

线程池 3 路并发（ffmpeg 自身多线程，受 CPU 核数约束）。
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import sys as _sys
from pathlib import Path as _P
_SCRIPT_DIR = str(_P(__file__).resolve().parent)
if _SCRIPT_DIR not in _sys.path:
    _sys.path.insert(0, _SCRIPT_DIR)

from common import DS_CLEAN, DS_EVAL, DS_MINREPAIR, OUT, dump_json

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")

DATASETS = [DS_MINREPAIR, DS_CLEAN, DS_EVAL]


def ffprobe_int(path: Path, entries: str) -> int | None:
    proc = subprocess.run(
        [FFPROBE, "-v", "error", "-select_streams", "v:0",
         "-show_entries", entries, "-of", "json", str(path)],
        capture_output=True, text=True, timeout=120, check=False,
    )
    if proc.returncode != 0:
        return None
    try:
        streams = json.loads(proc.stdout).get("streams") or [{}]
        if "nb_frames" in entries:
            v = streams[0].get("nb_frames")
            return int(v) if v is not None else None  # ffprobe 返回字符串，必须转 int
        return streams[0].get("codec_name")
    except (json.JSONDecodeError, KeyError, IndexError, TypeError, ValueError):
        return None


def transcode_one(src: Path, dst: Path, fps: float) -> tuple[bool, str | None]:
    """转码单个文件，成功返回 (True, None)。"""
    cmd = [
        FFMPEG, "-y", "-v", "error", "-i", str(src),
        "-map", "0:v:0", "-an",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-fps_mode", "passthrough",
        str(dst),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600, check=False)
    if proc.returncode != 0 or not dst.is_file():
        # 旧版 ffmpeg 没有 -fps_mode，退回 -vsync 0 再试一次
        cmd[cmd.index("-fps_mode")] = "-vsync"
        cmd[cmd.index("passthrough")] = "0"
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600, check=False)
        if proc.returncode != 0 or not dst.is_file():
            return False, (proc.stderr or "")[-300:]
    return True, None


def process_file(rel: str, path: Path, fps: float) -> dict:
    src_codec = ffprobe_int(path, "stream=codec_name")
    frames_before = ffprobe_int(path, "stream=nb_frames")
    if src_codec == "h264":
        return {"file": rel, "action": "skip", "codec": "h264", "frames": frames_before}

    tmp = path.with_suffix(".transcode.mp4")
    ok, err = transcode_one(path, tmp, fps)
    if not ok:
        tmp.unlink(missing_ok=True)
        return {"file": rel, "action": "failed", "codec": src_codec, "error": err}
    frames_after = ffprobe_int(tmp, "stream=nb_frames")
    shutil.move(str(tmp), str(path))
    return {
        "file": rel,
        "action": "transcoded",
        "src_codec": src_codec,
        "codec": "h264",
        "frames_before": frames_before,
        "frames_after": frames_after,
        "frames_delta": (frames_after - frames_before)
        if (frames_before is not None and frames_after is not None) else None,
    }


def transcode_dataset(root: Path, fps: float, workers: int) -> dict:
    files = sorted(root.glob("videos/*/chunk-*/*.mp4"))
    if not files:
        return {"root": str(root), "files": [], "error": "未找到视频文件"}
    print(f"\n=== {root.name}: {len(files)} 个视频文件")
    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(process_file, str(f.relative_to(root)), f, fps): f for f in files
        }
        for fut in as_completed(futures):
            r = fut.result()
            results.append(r)
            if r["action"] == "transcoded":
                delta = r.get("frames_delta")
                mark = "" if delta == 0 else f"  [帧数差 {delta}]"
                print(f"  转码 {r['file']} {r.get('src_codec')}->h264{mark}")
            elif r["action"] == "failed":
                print(f"  失败 {r['file']}: {r.get('error')}", file=sys.stderr)

    # info.json 的 codec 声明与实际对齐
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    for key, spec in (info.get("features") or {}).items():
        if (spec or {}).get("dtype") == "video" and isinstance(spec.get("info"), dict):
            spec["info"]["video.codec"] = "h264"
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")

    transcoded = [r for r in results if r["action"] == "transcoded"]
    delta_files = [r for r in transcoded if r.get("frames_delta") not in (0, None)]
    return {
        "root": str(root),
        "total_files": len(files),
        "transcoded": len(transcoded),
        "skipped_h264": len([r for r in results if r["action"] == "skip"]),
        "failed": len([r for r in results if r["action"] == "failed"]),
        "files_with_frame_delta": len(delta_files),
        "results": sorted(results, key=lambda r: r["file"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="三份数据集视频统一转码 h264")
    parser.add_argument("--workers", type=int, default=3, help="并发 ffmpeg 数（默认 3）")
    args = parser.parse_args()
    if not FFMPEG or not FFPROBE:
        sys.exit("[错误] 找不到 ffmpeg/ffprobe")

    fps = 30.0
    report = {"fps": fps, "datasets": [transcode_dataset(d, fps, args.workers) for d in DATASETS]}
    dump_json(OUT / "transcode_manifest.json", report)

    for ds in report["datasets"]:
        if "error" in ds:
            print(f"[错误] {ds['root']}: {ds['error']}", file=sys.stderr)
            return 1
        print(f"{Path(ds['root']).name}: 转码 {ds['transcoded']} / 跳过 {ds['skipped_h264']} / "
              f"失败 {ds['failed']} / 帧数差文件 {ds['files_with_frame_delta']}")
    print(f"manifest: {OUT / 'transcode_manifest.json'}")
    print("下一步: apply_stats_fixed.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
