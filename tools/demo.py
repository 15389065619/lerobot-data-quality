"""一键演示：质检 → 分级处置 → 清洗 → 报告 → 版本快照 → 数据卡片。

用法::

    python tools/demo.py                 # 使用 demo_dataset，输出到 outputs/demo
    python tools/demo.py --force         # 覆盖已有输出
    python tools/demo.py --dataset <path> --output <dir> --skip-version

只编排已有能力，不重新实现质检逻辑；任何一步失败都会以非零码退出。
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.card import build_card  # noqa: E402


def _prepare_output(out: Path, force: bool) -> None:
    default_out = ROOT / "outputs" / "demo"
    if not out.exists():
        out.mkdir(parents=True, exist_ok=True)
        return
    if not force and out != default_out:
        raise SystemExit(f"输出目录已存在：{out}，加 --force 覆盖")
    shutil.rmtree(out)


def _run(cmd: list[str]) -> int:
    print("$ " + " ".join(cmd))
    return subprocess.run(cmd, cwd=ROOT, check=False).returncode


def _summary(report: dict) -> None:
    ds = report.get("dataset") or {}
    summary = report.get("summary") or {}
    severity = summary.get("severity") or {}
    disposition = summary.get("disposition") or {}
    print("\n== 质检结果 ==")
    print(f"数据集      : {ds.get('name')} ({ds.get('robot_type')}) "
          f"{ds.get('total_episodes')} ep / {ds.get('total_frames')} frames")
    print(f"正常/异常   : {summary.get('valid_episodes')} / {summary.get('invalid_episodes')}")
    print(f"严重度      : error {severity.get('error', 0)} · warning {severity.get('warning', 0)}"
          f" · info {severity.get('info', 0)}")
    print(f"处置        : keep {disposition.get('keep', 0)} · repair {disposition.get('repair', 0)}"
          f" · review {disposition.get('review', 0)} · drop {disposition.get('drop', 0)}")
    issues = [c for c in report.get("checks") or [] if c.get("affected_episodes")]
    for check in issues:
        print(f"  - {check.get('code'):24s} {check.get('affected_episodes')} 个 episode")
    if report.get("skipped_checks"):
        print(f"跳过检查    : {', '.join(report['skipped_checks'])}（数据集缺少对应特征）")
    cleaning = report.get("cleaning") or {}
    print(f"清洗        : {cleaning.get('method')} -> {cleaning.get('output_path')}")
    for rec in (report.get("recommendations") or [])[:5]:
        print(f"  建议: {rec}")


def demo_version(cleaned: Path) -> None:
    from pipeline.versions import DatasetVersionStore

    store = DatasetVersionStore(cleaned)
    record = store.snapshot(note="demo: 清洗产物")
    print("\n== 版本快照 ==")
    print(f"version_id : {record['version_id']}")
    print(f"files      : {record['file_count']} 个 / {record['total_bytes']} 字节")
    plan = store.restore(record["version_id"], apply=False)
    print(f"回滚演练   : 待写 {len(plan['to_write'])} 个文件，缺失对象 "
          f"{len(plan['missing_objects'])} 个（dry-run，未写盘）")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="LeRobot 数据质量 pipeline 一键演示")
    ap.add_argument("--dataset", default=str(ROOT / "demo_dataset"))
    ap.add_argument("--output", default=str(ROOT / "outputs" / "demo"))
    ap.add_argument("--force", action="store_true", help="覆盖已存在的输出目录")
    ap.add_argument("--skip-version", action="store_true", help="跳过版本快照演示")
    args = ap.parse_args(argv)

    dataset = Path(args.dataset).expanduser().resolve()
    if not (dataset / "meta" / "info.json").is_file():
        if dataset == (ROOT / "demo_dataset").resolve():
            print("demo_dataset 不存在，正在生成…")
            if _run([sys.executable, str(ROOT / "tools" / "make_demo_dataset.py")]) != 0:
                return 1
        else:
            print(f"不是有效的 LeRobot v3 数据集：{dataset}")
            return 2

    out = Path(args.output).expanduser().resolve()
    _prepare_output(out, args.force)

    if _run([sys.executable, str(ROOT / "main.py"),
             "--dataset_path", str(dataset), "--output_dir", str(out), "--no-cli"]) != 0:
        print("主流程失败，见上方日志")
        return 1

    report_path = out / "quality_report.json"
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"报告不可读：{exc}")
        return 1
    _summary(report)

    cleaned = out / "cleaned_dataset"
    info = json.loads((cleaned / "meta" / "info.json").read_text(encoding="utf-8"))
    stats_path = cleaned / "meta" / "stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.is_file() else {}
    card_path = out / "DATASET_CARD.md"
    card_path.write_text(build_card(report, info, stats), encoding="utf-8")
    print(f"\n数据卡片    : {card_path}")

    if not args.skip_version and cleaned.is_dir():
        demo_version(cleaned)

    print("\n== 产物 ==")
    for path in (report_path, out / "review_queue.csv", cleaned,
                 out / "plots", card_path):
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
