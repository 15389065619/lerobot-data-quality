"""规范数据集（Happy Path）回归脚本。

对 G:\\Lerobot\\data\\baseline_test 下的每个数据集跑一遍流式质检，
汇总 valid_episodes / invalid_episodes / drop_frames / health_level，
并在出现 error 判定时把命中的规则打印出来，方便立刻定位过严的规则。

用法：
    python tools/run_baseline_test.py
    python tools/run_baseline_test.py --datasets-dir G:\\Lerobot\\data\\baseline_test
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_HEALTH = {"ok", "minor"}


def discover(datasets_dir: Path) -> list[Path]:
    """找出带 meta/info.json 的数据集目录（一层）。"""
    return sorted(
        p for p in datasets_dir.iterdir()
        if p.is_dir() and (p / "meta" / "info.json").exists()
    )


def run_one(dataset: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "main.py",
        "--dataset_path", str(dataset),
        "--output_dir", str(output_dir),
        "--streaming",
        "--skip-clean",
        "--no-cli",
        "--skip-plots",
    ]
    print(f"\n>>> 运行: {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        print(proc.stdout[-4000:])
        print(proc.stderr[-4000:])
        raise SystemExit(f"质检失败: {dataset} (exit={proc.returncode})")

    report_path = output_dir / "quality_report.json"
    with report_path.open(encoding="utf-8") as fh:
        return json.load(fh)


def summarize(name: str, report: dict) -> dict:
    sm = report["summary"]
    sev = sm.get("severity", {})
    disp = sm.get("disposition", {})
    row = {
        "dataset": name,
        "total": sm["total_episodes"],
        "valid": sm["valid_episodes"],
        "invalid": sm["invalid_episodes"],
        "drop_frames": sm["drop_frames"],
        "health": sm["health_level"],
        "error": sev.get("error", 0),
        "warning": sev.get("warning", 0),
        "drop": disp.get("drop", 0),
        "review": disp.get("review", 0),
        "mode": report["resolved_thresholds"].get("frame_count", {}).get("mode", "?"),
        "bounds": (
            report["resolved_thresholds"].get("frame_count", {}).get("min"),
            report["resolved_thresholds"].get("frame_count", {}).get("max"),
        ),
    }
    # 异常的 episode 命中了什么规则，直接列出来
    hits: dict[str, list[int]] = {}
    for item in report.get("invalid_episode_details", []):
        for code in item.get("reasons", []):
            hits.setdefault(code, []).append(item["episode_index"])
    row["hits"] = hits
    row["dataset_issues"] = [f"{i['code']}[{i['severity']}]" for i in report.get("dataset_issues", [])]
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets-dir", default=str(ROOT / "data" / "baseline_test"))
    ap.add_argument("--outputs-dir", default=str(ROOT / "outputs_baseline"))
    args = ap.parse_args()

    datasets = discover(Path(args.datasets_dir))
    if not datasets:
        raise SystemExit(f"未在 {args.datasets_dir} 下找到数据集（需含 meta/info.json）")

    print(f"发现 {len(datasets)} 个数据集: {[p.name for p in datasets]}")
    rows = []
    for ds in datasets:
        rows.append(summarize(ds.name, run_one(ds, Path(args.outputs_dir) / ds.name)))

    print("\n" + "=" * 100)
    print(f"{'数据集':<28}{'总集数':>6}{'正常':>6}{'异常':>6}{'剔除帧':>9}{'error':>7}{'warning':>8}{'健康度':>8}{'模式':>7}")
    print("-" * 100)
    for r in rows:
        print(f"{r['dataset']:<28}{r['total']:>6}{r['valid']:>6}{r['invalid']:>6}"
              f"{r['drop_frames']:>9}{r['error']:>7}{r['warning']:>8}{r['health']:>8}{r['mode']:>7}")
    print("=" * 100)

    failed = False
    for r in rows:
        print(f"\n[{r['dataset']}] 帧数阈值: [{r['bounds'][0]}, {r['bounds'][1]}] ({r['mode']})")
        if r["dataset_issues"]:
            print(f"  数据集级问题: {', '.join(r['dataset_issues'])}")
        if r["invalid"] == 0:
            print("  结果: 0 误报 ✔")
            continue
        print(f"  结果: {r['invalid']} 个 episode 被判异常（drop={r['drop']}, review={r['review']}）")
        for code, eps in r["hits"].items():
            shown = ", ".join(map(str, eps[:10])) + (" ..." if len(eps) > 10 else "")
            print(f"    - {code}: {len(eps)} 个 episode -> {shown}")
        if r["health"] not in EXPECTED_HEALTH or r["error"] > 0:
            failed = True

    print("\n判定:", "FAIL —— 规范数据上出现误报/error，需要定位规则" if failed else "PASS —— 规范数据全部保留")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
