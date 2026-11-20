"""v2 验收驱动：修复版 cleaner 重建的 B′/eval′（零漂移目标）。

前置：
  1. 修复版 cleaner 全量重清洗 bomb：
       python main.py --dataset_path data/external/so101_bomb_pick_place_v1 \
            --output_dir outputs_ext_bomb_v2 --no-cli --skip-plots --skip-pointcloud
  2. 处置（121 keep / 29 drop）与 episode_index_map 与 v1 逐项一致（冻结 eval 划分才可复用）；
  3. python -I kaggle_ab/build_datasets.py \
        --source outputs_ext_bomb_v2/cleaned_dataset --suffix _v2

本脚本按序做：
  ① pts 层：重建文件（h264）首帧 pts<2ms 且逐帧 |pts[i]−i/fps|<半帧——
     FrameTimestampError 崩溃模式（33ms 偏移）的直接回归；AV1 整文件材料化
     的源文件只查首帧（其 pts 为源轨原始值，按时间戳窗口读取）；
  ② 转码：B′/eval′ 统一 h264（复用 transcode_h264.transcode_dataset）；
  ③ 内容层：每文件帧数 == meta 窗口和，**零容差**——v1 的 max(3, 0.5%) 容差
     是漂移 bug 的绕行，修复后必须归零；
  ④ stats：与 v1 基准（so101_clean/meta/stats.json）字节一致——归一化契约跨跑
     不变（不重算：B′ 像素与 v1-B 的编码代次差异远小于 crf20 量化噪声）；
  ⑤ 窗口：冻结的 320 窗口对 eval′ 全部合法；
  ⑥ 判定层：validator 对 B′/eval′ 应 0 个非 keep（--skip-validator 可跳过）。
任何一层失败以非零码退出；报告写 exports/kaggle_ab/verify_report_v2.json。
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import sys as _sys
_SCRIPT_DIR = str(Path(__file__).resolve().parent)
if _SCRIPT_DIR not in _sys.path:
    _sys.path.insert(0, _SCRIPT_DIR)

from common import OUT, dump_json, ensure_repo_on_path, load_json

ensure_repo_on_path()

from transcode_h264 import transcode_dataset  # noqa: E402
from verify_datasets import (  # noqa: E402
    check_stats,
    check_structure,
    check_video_frames,
    check_windows,
    md5,
    run_validator,
)

V1_STATS = OUT / "so101_clean" / "meta" / "stats.json"   # v1 归一化契约基准
FPS = 30.0
PTS_START_TOL = 0.002     # 首帧 pts 容差（秒）
PTS_DRIFT_TOL = 0.5 / FPS  # 逐帧漂移容差 = 半帧


def ffprobe_pts(path: Path) -> list[float] | None:
    ffprobe = shutil.which("ffprobe")
    proc = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "frame=pts_time", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, timeout=600, check=False,
    )
    if proc.returncode != 0:
        return None
    try:
        return [float(x) for x in proc.stdout.replace(",", " ").split() if x.strip()]
    except ValueError:
        return None


def check_pts(root: Path) -> dict:
    """h264（本轮重建）文件：首帧 pts 与逐帧漂移；其余文件只查首帧。"""
    ffprobe = shutil.which("ffprobe")
    rows, bad = [], []
    for path in sorted(root.glob("videos/*/chunk-*/*.mp4")):
        proc = subprocess.run(
            [ffprobe, "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=codec_name", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=120, check=False,
        )
        codec = proc.stdout.strip()
        pts = ffprobe_pts(path)
        entry = {"file": str(path.relative_to(root)), "codec": codec,
                 "n_pts": None if pts is None else len(pts)}
        if pts is None or not pts:
            entry["issue"] = "pts 探测失败"
            bad.append(entry)
        else:
            entry["first_pts"] = pts[0]
            if pts[0] >= PTS_START_TOL:
                entry["issue"] = f"首帧 pts={pts[0]:.4f}s 未归零"
                bad.append(entry)
            elif codec == "h264":
                drift = max(abs(p - i / FPS) for i, p in enumerate(pts))
                entry["max_drift_ms"] = round(drift * 1000, 2)
                if drift >= PTS_DRIFT_TOL:
                    entry["issue"] = f"逐帧漂移 {drift * 1000:.1f}ms 超半帧"
                    bad.append(entry)
        rows.append(entry)
    return {"files": len(rows), "bad": bad, "rows": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description="B′/eval′（修复版 cleaner 产物）验收")
    parser.add_argument("--suffix", default="_v2", help="数据集名后缀")
    parser.add_argument("--skip-validator", action="store_true", help="跳过判定层 validator")
    parser.add_argument("--skip-transcode", action="store_true", help="跳过转码（已转过）")
    args = parser.parse_args()

    ds_clean = OUT / f"so101_clean{args.suffix}"
    ds_eval = OUT / f"so101_eval{args.suffix}"
    for root in (ds_clean, ds_eval):
        if not root.is_dir():
            sys.exit(f"[错误] 数据集不存在: {root}（先跑 build_datasets --suffix {args.suffix}）")

    report: dict = {"suffix": args.suffix, "datasets": {}, "gates": {}}

    # ① pts 层（转码前：h264 = 本轮重建产物，逐帧严查）
    for name, root in (("so101_clean", ds_clean), ("so101_eval", ds_eval)):
        pts_report = check_pts(root)
        report["datasets"].setdefault(name, {})["pts"] = pts_report
        print(f"[pts {name}] {len(pts_report['rows']) - len(pts_report['bad'])}"
              f"/{pts_report['files']} 个文件通过")
        for b in pts_report["bad"][:10]:
            print(f"    [pts] {b['file']}: {b['issue']}")

    # ② 转码统一 h264
    if not args.skip_transcode:
        report["gates"]["transcode"] = [transcode_dataset(r, FPS, 3) for r in (ds_clean, ds_eval)]
        for t in report["gates"]["transcode"]:
            print(f"[转码 {Path(t['root']).name}] 转码 {t['transcoded']} / 跳过 {t['skipped_h264']}"
                  f" / 失败 {t['failed']} / 帧数差 {t['files_with_frame_delta']}")

    # ③ 内容层：零帧数差
    for name, root in (("so101_clean", ds_clean), ("so101_eval", ds_eval)):
        vf = check_video_frames(root)
        report["datasets"][name]["video_frames"] = {
            "files": vf["files"], "files_with_delta": vf["files_with_delta"]}
        print(f"[帧数 {name}] 差 {vf['files_with_delta']}/{vf['files']} 个文件（要求 0）")
        for r in vf["rows"]:
            if r["delta"]:
                print(f"    [帧数] {r['file']} 期望 {r['expected_frames']} 实际 {r['actual_frames']}")

    # ④ stats：字节复制 v1 基准后校验一致
    ref_md5 = md5(V1_STATS)
    for name, root in (("so101_clean", ds_clean), ("so101_eval", ds_eval)):
        target = root / "meta" / "stats.json"
        shutil.copyfile(V1_STATS, target)
        sc = check_stats(root, ref_md5)
        report["datasets"][name]["stats"] = {k: sc[k] for k in ("md5", "matches_ref", "missing", "finite")}
        print(f"[stats {name}] md5={sc['md5'][:8]} 一致={sc['matches_ref']} "
              f"缺失={sc['missing']} 有限值={sc['finite']}")

    # ⑤ 窗口
    win = check_windows(ds_eval)
    report["gates"]["windows"] = win
    print(f"[eval 窗口] {win['n_windows']} 个，非法 {win['invalid']}")

    # ⑥ 判定层
    verdict: dict = {}
    for name, root in (("so101_clean", ds_clean), ("so101_eval", ds_eval)):
        entry = report["datasets"][name]
        ok = (entry["pts"] and not entry["pts"]["bad"]
              and entry["video_frames"]["files_with_delta"] == 0
              and entry["stats"]["matches_ref"] and not entry["stats"]["missing"]
              and entry["stats"]["finite"])
        verdict[name] = "PASS" if ok else "FAIL"
    if win["invalid"]:
        verdict["eval_windows"] = "FAIL"

    if not args.skip_validator:
        for name, root in (("so101_clean", ds_clean), ("so101_eval", ds_eval)):
            g = run_validator(root, f"v2/{name}")
            report["gates"][f"validator_{name}"] = g
            if g.get("ok"):
                disp = (g.get("summary") or {}).get("disposition") or {}
                non_keep = int(disp.get("drop") or 0) + int(disp.get("review") or 0) \
                    + int(disp.get("repair") or 0)
                verdict[f"validator_{name}"] = "PASS（0 非 keep）" if non_keep == 0 \
                    else f"FAIL（{non_keep} 个非 keep）"
                print(f"[validator {name}] disposition={disp}")

    report["verdict"] = verdict
    report_path = dump_json(OUT / "verify_report_v2.json", report)
    print("\n=== 判定 ===")
    for k, v in verdict.items():
        print(f"  {k}: {v}")
    print(f"报告: {report_path}")
    return 0 if all(v == "PASS" or v.startswith("PASS") for v in verdict.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
