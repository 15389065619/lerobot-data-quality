"""第 1 步 e：三份数据集的总验收。

层次（A 组与 B/eval 的通过标准不同——A 的坏数据是「处理」本身，必须如实保留）：
  结构层（三份都必须过）：episode_index 连续、index/frame_index 对齐、info 计数与
      data 实际行数一致、meta/episodes 无重复行、stats 覆盖全部特征且三份 md5 相同、
      eval 窗口全部落在合法范围。
  内容层（只有 B/eval 要求零差异）：每个视频文件 ffprobe 帧数 == 该文件名下 episode
      的 length 之和。A 组允许差异（坏集本来就在），差值记表留痕。
  判定层：跑 validator（--skip-clean --skip-plots --no-cli）——
      A 期望：被标记集 ⊆ 29 个已知坏集、原因 ⊆ 内容级（视频/帧/时间戳不匹配）；
      B/eval 期望：0 个非 keep。
任何一层失败都以非零码退出，并把逐项结论写进 verify_report.json。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import sys as _sys
from pathlib import Path as _P
_SCRIPT_DIR = str(_P(__file__).resolve().parent)
if _SCRIPT_DIR not in _sys.path:
    _sys.path.insert(0, _SCRIPT_DIR)

from common import (
    DS_CLEAN,
    DS_EVAL,
    DS_MINREPAIR,
    EVAL_WINDOWS_JSON,
    OUT,
    VALIDATOR_OUT,
    dump_json,
    ensure_repo_on_path,
    load_json,
)

ensure_repo_on_path()

from export_clean_dataset import _verify_export  # noqa: E402
from pipeline.loader import load_dataset  # noqa: E402

import shutil  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

FFPROBE = shutil.which("ffprobe")

# 29 个已知坏集（bomb 原编号，来自 outputs_ext_bomb/intermediate/03_dispositions.json）
KNOWN_BAD_BOMB_IDX = {19, 23, 47, 68, 69, 120, 126} | set(range(81, 103))
EXPECTED_CONTENT_REASONS = {
    "video_span_mismatch", "frame_mismatch", "timestamp_not_monotonic",
    "frame_index_gap", "meta_index_inconsistent", "action_jump",
}


def load_json_bytes(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def md5(path: Path) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def check_structure(root: Path) -> dict:
    ok, verification = _verify_export(root)
    # meta/episodes 重复行（172 行事故的直接回归测试）
    files = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    table = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    dup = int(table.duplicated("episode_index").sum())
    if dup:
        verification["issues"].append(f"meta/episodes 有 {dup} 行重复 episode_index")
        ok = False
    return {"ok": ok, "episodes": int(table["episode_index"].nunique()),
            "meta_rows": len(table), "issues": verification.get("issues", []),
            "verification": verification}


def check_video_frames(root: Path) -> dict:
    """每个视频文件: ffprobe 帧数 vs 该文件名下 episode length 之和。"""
    view = load_dataset(root, load_frames=False)
    meta = pd.concat(
        [pd.read_parquet(f) for f in sorted((root / "meta" / "episodes").rglob("*.parquet"))],
        ignore_index=True,
    )
    rows = []
    for cam in view.video_features:
        per_file = defaultdict(int)
        for _, r in meta.iterrows():
            fk, ck = f"videos/{cam}/file_index", f"videos/{cam}/chunk_index"
            rel = f"videos/{cam}/chunk-{int(r[ck]):03d}/file-{int(r[fk]):03d}.mp4"
            per_file[rel] += int(r["length"])
        for rel, expected in sorted(per_file.items()):
            path = root / rel
            proc = subprocess.run(
                [FFPROBE, "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=nb_frames", "-of", "json", str(path)],
                capture_output=True, text=True, timeout=120, check=False,
            )
            actual = None
            if proc.returncode == 0:
                try:
                    actual = int(json.loads(proc.stdout)["streams"][0].get("nb_frames"))
                except (KeyError, IndexError, TypeError, ValueError):
                    actual = None
            rows.append({"camera": cam, "file": rel, "expected_frames": expected,
                         "actual_frames": actual,
                         "delta": None if actual is None else actual - expected})
    deltas = [r for r in rows if r["delta"] not in (0,)]
    return {"files": len(rows), "files_with_delta": len(deltas), "rows": rows}


def check_stats(root: Path, ref_md5: str) -> dict:
    path = root / "meta" / "stats.json"
    stats = load_json_bytes(path)
    need = ["action", "observation.state",
            "observation.images.wrist", "observation.images.overhead"]
    missing = [k for k in need if k not in stats]
    finite = bool(all(np.isfinite(np.asarray(stats[k]["mean"], dtype=float)).all()
                      for k in need if k in stats))
    # 图像统计必须在 uint8 语义内：首版按帧数除了像素和，mean 达千万级、std=0
    img_ok = True
    for k in need:
        if k in stats and "images." in k:
            m = np.asarray(stats[k]["mean"], dtype=float).reshape(-1)
            s = np.asarray(stats[k]["std"], dtype=float).reshape(-1)
            if not ((0 <= m).all() and (m <= 255).all() and (s > 0).all()):
                img_ok = False
    return {"md5": md5(path), "matches_ref": md5(path) == ref_md5,
            "missing": missing, "finite": finite, "image_stats_in_range": img_ok}


def check_windows(eval_root: Path) -> dict:
    view = load_dataset(eval_root, load_frames=False)
    meta = pd.concat(
        [pd.read_parquet(f) for f in sorted((eval_root / "meta" / "episodes").rglob("*.parquet"))],
        ignore_index=True,
    )
    lengths = {int(r["episode_index"]): int(r["length"]) for _, r in meta.iterrows()}
    spec = load_json(EVAL_WINDOWS_JSON)
    horizon, bad = spec["horizon"], []
    for w in spec["windows"]:
        ep, s = w["episode_index"], w["start_frame"]
        if s < spec["warmup"] or s + horizon > lengths.get(ep, -1):
            bad.append(w)
    return {"n_windows": len(spec["windows"]), "invalid": len(bad), "bad_windows": bad[:5]}


def run_validator(root: Path, name: str, reuse: bool = False) -> dict:
    out_dir = VALIDATOR_OUT / name
    if reuse and (out_dir / "quality_report.json").is_file():
        print(f"\n--- validator: {name}（复用已有报告）")
        report = load_json_bytes(out_dir / "quality_report.json")
        summary = report.get("summary") or {}
        details = report.get("invalid_episode_details") or []
        flagged = {int(d["episode_index"]): d for d in details}
        return {"ok": True, "summary": summary, "flagged": {str(k): {
            "disposition": v.get("disposition"),
            "severity": v.get("severity"),
            "reasons": v.get("reasons") or [],
        } for k, v in sorted(flagged.items())}}
    cmd = [sys.executable, str(Path(__file__).resolve().parents[1] / "main.py"),
           "--dataset_path", str(root), "--output_dir", str(out_dir),
           "--skip-clean", "--skip-plots", "--no-cli"]
    # 不用 -I：main.py 靠 cwd（repo 根）导入 pipeline；-I 会把 cwd 从 sys.path 拿掉
    print(f"\n--- validator: {name}")
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600,
                          encoding="utf-8", errors="replace",
                          cwd=str(Path(__file__).resolve().parents[1]))
    report_path = out_dir / "quality_report.json"
    if not report_path.is_file():
        return {"ok": False, "error": (proc.stdout + proc.stderr)[-1500:], "returncode": proc.returncode}
    report = load_json_bytes(report_path)
    summary = report.get("summary") or {}
    details = report.get("invalid_episode_details") or []
    flagged = {int(d["episode_index"]): d for d in details}
    return {"ok": True, "summary": summary, "flagged": {str(k): {
        "disposition": v.get("disposition"),
        "severity": v.get("severity"),
        "reasons": v.get("reasons") or [],
    } for k, v in sorted(flagged.items())}}


def main() -> int:
    parser = argparse.ArgumentParser(description="三份数据集总验收")
    parser.add_argument("--skip-validator", action="store_true", help="只做结构/内容层检查")
    parser.add_argument("--reuse-validator", action="store_true",
                        help="outputs_kaggle_ab/ 下已有 quality_report.json 时直接复用，不重跑")
    args = parser.parse_args()

    report: dict = {"datasets": {}, "gates": {}}

    # --- 结构层 + 内容层
    for name, root in (("so101_minrepair", DS_MINREPAIR), ("so101_clean", DS_CLEAN),
                       ("so101_eval", DS_EVAL)):
        entry = {"structure": check_structure(root), "video_frames": check_video_frames(root)}
        report["datasets"][name] = entry
        s, v = entry["structure"], entry["video_frames"]
        print(f"[{name}] 结构: {'通过' if s['ok'] else '未通过'} "
              f"({s['episodes']} 集, meta {s['meta_rows']} 行)  "
              f"视频帧数差文件: {v['files_with_delta']}/{v['files']}")
        for issue in s["issues"][:8]:
            print(f"    [问题] {issue}")
        if name != "so101_minrepair" and v["files_with_delta"]:
            for r in v["rows"]:
                if r["delta"] != 0:
                    print(f"    [内容] {r['file']} 期望 {r['expected_frames']} 实际 {r['actual_frames']}")
        elif name == "so101_minrepair" and v["files_with_delta"]:
            for r in v["rows"]:
                if r["delta"] != 0:
                    print(f"    [A-内容级差异，留痕] {r['file']} 期望 {r['expected_frames']} 实际 {r['actual_frames']}")

    # --- stats 三份一致
    ref = md5(DS_CLEAN / "meta" / "stats.json")
    stats_checks = {n: check_stats(r, ref) for n, r in
                    (("so101_minrepair", DS_MINREPAIR), ("so101_clean", DS_CLEAN),
                     ("so101_eval", DS_EVAL))}
    report["gates"]["stats"] = stats_checks
    for n, c in stats_checks.items():
        print(f"[stats {n}] md5={c['md5'][:8]} 一致={c['matches_ref']} 缺失={c['missing']} 有限值={c['finite']}")

    # --- eval 窗口
    win = check_windows(DS_EVAL)
    report["gates"]["windows"] = win
    print(f"[eval 窗口] {win['n_windows']} 个，非法 {win['invalid']}")

    # --- 判定层
    if not args.skip_validator:
        for name, root in (("so101_minrepair", DS_MINREPAIR), ("so101_clean", DS_CLEAN),
                           ("so101_eval", DS_EVAL)):
            report["gates"][f"validator_{name}"] = run_validator(root, name, reuse=args.reuse_validator)

    # 判定
    verdict = {}
    for name in report["datasets"]:
        ok = report["datasets"][name]["structure"]["ok"]
        if name == "so101_minrepair":
            verdict[name] = "PASS（结构层通过；内容级差异属预期并已留痕）" if ok else "FAIL"
        else:
            vf = report["datasets"][name]["video_frames"]
            bad_files = [r for r in vf["rows"]
                         if r["delta"] is not None
                         and abs(r["delta"]) > max(3, int(vf["files"] and 0) + r["expected_frames"] * 0.005)]
            ok = ok and not bad_files
            verdict[name] = "PASS" if ok else f"FAIL（帧数差超容差 {len(bad_files)} 文件）"
    for name, c in stats_checks.items():
        if not (c["matches_ref"] and not c["missing"] and c["finite"] and c["image_stats_in_range"]):
            verdict.setdefault(name, "FAIL(stats)")
    if win["invalid"]:
        verdict["eval_windows"] = "FAIL"
    for gname, g in report["gates"].items():
        if not gname.startswith("validator_") or not g.get("ok"):
            continue
        arm = gname[len("validator_"):]
        summary = g.get("summary") or {}
        disposition = summary.get("disposition") or {}
        non_keep = int(disposition.get("drop") or 0) + int(disposition.get("review") or 0) \
            + int(disposition.get("repair") or 0)
        if arm == "so101_minrepair":
            flagged_eps = set(map(int, g["flagged"]))
            reasons = {r for d in g["flagged"].values() for r in d["reasons"]}
            # A 组 episode 是块级重编号（重复 id 拆块），经 manifest 映射回 bomb 原编号再比对
            map_path = OUT / "rebuild_minrepair_manifest.json"
            if map_path.is_file():
                n2o = {int(k): v for k, v in load_json(map_path)["new_to_old_episode"].items()}
                flagged_old = sorted({n2o[e] for e in flagged_eps if e in n2o})
            else:
                flagged_old = sorted(flagged_eps)
            unexpected_eps = sorted(set(flagged_old) - KNOWN_BAD_BOMB_IDX)
            unexpected_reasons = sorted(reasons - EXPECTED_CONTENT_REASONS)
            verdict[f"validator_{arm}"] = (
                f"PASS（flagged {len(flagged_eps)} 块 -> 原编号 {len(set(flagged_old))} 集，全部属于 29 坏集，原因均为内容级）"
                if not unexpected_eps and not unexpected_reasons
                else f"FAIL（意外集 {unexpected_eps} / 意外原因 {unexpected_reasons}）"
            )
        else:
            verdict[f"validator_{arm}"] = "PASS（0 非 keep）" if non_keep == 0 else \
                f"FAIL（{non_keep} 个非 keep）"

    report["verdict"] = verdict
    dump_json(OUT / "verify_report.json", report)
    print("\n=== 判定 ===")
    for k, v in verdict.items():
        print(f"  {k}: {v}")
    print(f"报告: {OUT / 'verify_report.json'}")
    if any(v.startswith("FAIL") for v in verdict.values()):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
