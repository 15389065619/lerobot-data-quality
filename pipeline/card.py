"""从质检报告生成数据集卡片（Markdown）。

用法::

    python -m pipeline.card --report outputs/demo/quality_report.json \
        --out outputs/demo/DATASET_CARD.md

卡片内容全部来自落盘产物（quality_report.json / meta/info.json / meta/stats.json），
不重新计算、不引入新指标，保证与报告一致、可追溯。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

METADATA_KEYS = {"timestamp", "frame_index", "episode_index", "index", "task_index"}


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _shape_of(spec: dict[str, Any] | None) -> tuple[int, ...]:
    return tuple(int(v) for v in (spec or {}).get("shape") or [])


def _mean_hint(mean: Any, shape: tuple[int, ...]) -> str:
    """把统计量均值压成一行摘要，避免卡片里堆 29×3 个数。"""
    if mean is None:
        return "-"
    if len(shape) > 1:
        return f"`{shape}` 张量"
    values = mean if isinstance(mean, list) else [mean]
    if len(values) > 6:
        return f"{len(values)} 维"
    return ", ".join(f"{float(v):.4g}" for v in values)


def _health(report: dict[str, Any]) -> str:
    severity = (report.get("summary") or {}).get("severity") or {}
    if severity.get("error"):
        return "error"
    if severity.get("warning"):
        return "warning"
    return "clean"


def build_card(
    report: dict[str, Any],
    info: dict[str, Any] | None = None,
    stats: dict[str, Any] | None = None,
) -> str:
    """把报告、info、stats 拼成 Markdown 卡片。缺失字段降级为 ``-``，不抛异常。"""
    ds = report.get("dataset") or {}
    summary = report.get("summary") or {}
    cleaning = report.get("cleaning") or {}
    name = ds.get("name") or Path(str(ds.get("path", "dataset"))).name
    info = info or {}
    stats = stats or {}

    lines = [
        f"# 数据集卡片：{name}",
        "",
        f"> 健康度 **{_health(report)}** · 生成于 {report.get('generated_at', '-')} "
        f"· 指纹 `{ds.get('fingerprint', '-')}`",
        "",
        "## 基本信息",
        "",
        "| 项 | 值 |",
        "|---|---|",
        f"| robot_type | {ds.get('robot_type') or info.get('robot_type', '-')} |",
        f"| fps | {ds.get('fps', info.get('fps', '-'))} |",
        f"| episodes | {ds.get('total_episodes', info.get('total_episodes', '-'))} |",
        f"| frames | {ds.get('total_frames', info.get('total_frames', '-'))} |",
        f"| state 维度 | {ds.get('state_dims', '-')} |",
        f"| 视频特征 | {', '.join(ds.get('video_features') or []) or '无'} |",
        "",
    ]

    severity = summary.get("severity") or {}
    disposition = summary.get("disposition") or {}
    lines += [
        "## 质检结论",
        "",
        f"- 正常 episode：{summary.get('valid_episodes', '-')} / {summary.get('total_episodes', '-')}"
        f"（{float(summary.get('valid_ratio') or 0):.1%}）",
        f"- 严重度：error {severity.get('error', 0)} · warning {severity.get('warning', 0)}"
        f" · info {severity.get('info', 0)}",
        f"- 处置：keep {disposition.get('keep', 0)} · repair {disposition.get('repair', 0)}"
        f" · review {disposition.get('review', 0)} · drop {disposition.get('drop', 0)}",
        f"- 异常帧：{summary.get('invalid_frames', 0)}"
        f"（剔除 {summary.get('drop_frames', 0)} / 裁剪 {summary.get('trim_frames', 0)}"
        f" / 修复 {summary.get('repair_frames', 0)}）",
        "",
    ]

    features = info.get("features") or {}
    if features:
        lines += [
            "## 特征清单",
            "",
            "| 特征 | dtype | shape | count | mean 摘要 |",
            "|---|---|---|---|---|",
        ]
        for key, spec in features.items():
            if key in METADATA_KEYS:
                continue
            shape = _shape_of(spec)
            feature_stats = stats.get(key) or {}
            lines.append(
                f"| `{key}` | {(spec or {}).get('dtype', '-')} | `{shape}` | "
                f"{feature_stats.get('count', '-')} | {_mean_hint(feature_stats.get('mean'), shape)} |"
            )
        lines.append("")

    skipped = report.get("skipped_checks") or []
    if skipped:
        lines += [
            "## 未执行的检查",
            "",
            *[f"- `{item}`" for item in skipped],
            "",
            "> 通常是因为数据集没有视频/点云特征，对应检查被自动跳过。",
            "",
        ]

    lines += [
        "## 清洗",
        "",
        f"- 是否执行：{'是' if cleaning.get('performed') else '否'}",
        f"- 方法：{cleaning.get('method', '-')}",
        f"- 产出：{cleaning.get('output_path', '-')}",
    ]
    warnings = cleaning.get("warnings") or []
    if warnings:
        lines += ["", "清洗告警：", *[f"- {w}" for w in warnings]]
    lines.append("")

    recommendations = report.get("recommendations") or []
    if recommendations:
        lines += ["## 建议", "", *[f"- {item}" for item in recommendations], ""]

    lines += [
        "## 复现",
        "",
        "```bash",
        f"python main.py --dataset_path {ds.get('path', '<dataset>')} --output_dir <out> --no-cli",
        "```",
        "",
        "> 卡片由 `python -m pipeline.card` 根据落盘产物生成，未引入报告之外的指标。",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="从质量报告生成数据集卡片")
    ap.add_argument("--report", required=True, help="quality_report.json 路径")
    ap.add_argument("--dataset", default=None,
                    help="包含 meta/info.json 与 meta/stats.json 的目录（默认报告同级 cleaned_dataset）")
    ap.add_argument("--out", default=None, help="输出 Markdown（默认报告同目录 DATASET_CARD.md）")
    args = ap.parse_args(argv)

    report_path = Path(args.report).expanduser().resolve()
    report = _load_json(report_path)
    if report is None:
        print(f"无法读取报告：{report_path}")
        return 2

    dataset_dir = Path(args.dataset).expanduser().resolve() if args.dataset \
        else report_path.parent / "cleaned_dataset"
    info = _load_json(dataset_dir / "meta" / "info.json")
    stats = _load_json(dataset_dir / "meta" / "stats.json")

    out_path = Path(args.out).expanduser().resolve() if args.out \
        else report_path.parent / "DATASET_CARD.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(build_card(report, info, stats), encoding="utf-8")
    print(f"卡片已生成：{out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
