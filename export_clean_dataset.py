"""从 quality_report.json 导出「只保留 disposition == 'keep'」的物理子集。

与主流程的 cleaner 区别：cleaner 尽量保住数据（repair 修复后保留、review 保留待人工确认），
本脚本是「严格导出」——只把报告中明确 keep 的 episode 物化到新目录，其余全部剔除，
并把 episode_index / index / meta 元数据从 0 开始绝对连续重排，避免 ID 错位带进训练。

用法：
    python export_clean_dataset.py \
        --report outputs/quality_report.json \
        --output_dir ./exported_dataset

    # 视频整文件保留时用硬链接省磁盘（跨盘会自动回退复制）
    python export_clean_dataset.py --report ... --output_dir ... --hardlink

输出目录结构：
    exported_dataset/
    ├── data/chunk-000/file-*.parquet      # 只含保留 episode，index/frame_index 连续
    ├── meta/info.json                     # total_episodes / total_frames 已更新
    ├── meta/episodes/chunk-000/file-000.parquet   # v3 主元数据，episode_index 0..N-1
    ├── meta/episodes.jsonl                # 同步副本，方便人工核对
    ├── meta/stats.json                    # 在导出数据上重算
    ├── meta/tasks.parquet
    ├── videos/...                         # 有视频时按保留区间重建
    └── cleaning_summary.md                # 剔除了哪些 episode、释放了多少帧
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from pipeline.baseline import fingerprint
from pipeline.cleaner import clean_dataset
from pipeline.loader import load_dataset

LOG = logging.getLogger("export_clean_dataset")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="按 quality_report.json 的 keep 处置导出物理子集（episode_index 从 0 连续重排）",
    )
    p.add_argument("--report", default="outputs/quality_report.json",
                   help="质检报告路径（默认 outputs/quality_report.json）")
    p.add_argument("--dataset_path", default=None,
                   help="源数据集根目录；默认取报告里的 dataset.root")
    p.add_argument("--output_dir", default="exported_dataset",
                   help="导出目录（直接作为新数据集根目录，默认 ./exported_dataset）")
    p.add_argument("--overwrite", action="store_true",
                   help="导出目录已存在时先删除再写（默认拒绝覆盖）")
    p.add_argument("--hardlink", action="store_true",
                   help="视频文件整段保留时优先硬链接，失败自动回退复制")
    p.add_argument("--verbose", action="store_true", help="输出调试日志")
    return p


def _load_report(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"质检报告不存在: {path}")
    with path.open("r", encoding="utf-8") as fh:
        report = json.load(fh)
    if not isinstance(report, dict) or "dataset" not in report:
        raise ValueError(f"不是有效的质检报告（缺少 dataset 字段）: {path}")
    return report


def _non_keep(report: dict[str, Any]) -> dict[int, dict[str, Any]]:
    """报告里 disposition != keep 的 episode（keep 本身不会出现在明细里）。"""
    out: dict[int, dict[str, Any]] = {}
    for item in report.get("invalid_episode_details") or []:
        try:
            ep = int(item.get("episode_index"))
        except (TypeError, ValueError):
            continue
        disposition = item.get("disposition") or "review"
        if disposition != "keep":
            out[ep] = item
    return out


def _episode_frame_counts(dataset) -> dict[int, int]:
    series = dataset.frames_by_episode()
    return {int(ep): int(n) for ep, n in series.items()}


def _read_exported_episodes(out_root: Path) -> pa.Table | None:
    files = sorted((out_root / "meta" / "episodes").rglob("*.parquet"))
    if not files:
        return None
    table = pq.read_table(files[0])
    for extra in files[1:]:
        table = pa.concat_tables([table, pq.read_table(extra)], promote_options="permissive")
    return table


def _verify_export(out_root: Path) -> tuple[bool, dict[str, Any]]:
    """校验 episode_index 连续、info 计数与数据一致。"""
    issues: list[str] = []
    table = _read_exported_episodes(out_root)
    if table is None:
        issues.append("未找到 meta/episodes 元数据")
        return False, {"ok": False, "issues": issues}

    eps = sorted(int(v) for v in table.column("episode_index").to_pylist())
    expected = list(range(len(eps)))
    if eps != expected:
        issues.append(f"episode_index 不连续：期望 0..{len(eps) - 1}，实际 {eps[:10]}...")

    lengths = [int(v) for v in table.column("length").to_pylist()]
    table_rows = sum(lengths)

    info_path = out_root / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    if int(info.get("total_episodes", -1)) != len(eps):
        issues.append(f"info.total_episodes={info.get('total_episodes')} 与实际 {len(eps)} 不符")
    if int(info.get("total_frames", -1)) != table_rows:
        issues.append(f"info.total_frames={info.get('total_frames')} 与 episodes.length 之和 {table_rows} 不符")

    splits = (info.get("splits") or {}).get("train")
    if splits != f"0:{len(eps)}":
        issues.append(f"info.splits.train={splits}，应为 0:{len(eps)}")

    return not issues, {
        "ok": not issues,
        "issues": issues,
        "episodes": len(eps),
        "frames": table_rows,
        "episode_index_range": f"0..{len(eps) - 1}" if eps else "empty",
    }


def _write_episodes_jsonl(out_root: Path) -> Path:
    """把 v3 的 episodes.parquet 同步成人类可读的 episodes.jsonl。"""
    table = _read_exported_episodes(out_root)
    path = out_root / "meta" / "episodes.jsonl"
    rows: list[str] = []
    if table is not None:
        cols = table.column_names
        for row in table.to_pylist():
            entry = {k: row[k] for k in ("episode_index", "length", "tasks") if k in cols}
            rows.append(json.dumps(entry, ensure_ascii=False))
    path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    return path


def _format_map(index_map: dict[int, int]) -> str:
    pairs = [f"{old}->{new}" for old, new in sorted(index_map.items())]
    if len(pairs) <= 12:
        return ", ".join(pairs)
    lines = [", ".join(pairs[i:i + 10]) for i in range(0, len(pairs), 10)]
    return "  \n".join(lines)


def _write_summary(
    path: Path,
    *,
    report_path: Path,
    source_root: Path,
    out_root: Path,
    report: dict[str, Any],
    result: Any,
    verification: dict[str, Any],
    source_frames: int,
    exported_frames: int,
    removed: dict[int, dict[str, Any]],
    frame_counts: dict[int, int],
) -> Path:
    dataset_name = str((report.get("dataset") or {}).get("name") or source_root.name)
    removed_frames = sum(frame_counts.get(ep, 0) for ep in removed)
    released = max(source_frames - exported_frames, 0)
    ratio = f" ({released / source_frames:.1%})" if source_frames else ""

    lines = [
        "# 数据集清洗导出说明",
        "",
        f"- 生成时间：{datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"- 源数据集：`{source_root}`",
        f"- 质检报告：`{report_path}`",
        f"- 导出目录：`{out_root}`",
        f"- 导出方式：{result.method}（data parquet 物理重排 + meta 重建）",
        f"- 结果校验：{'通过' if verification.get('ok') else '未通过'}",
        "",
        "## 概览",
        "",
        "| 指标 | 数值 |",
        "| --- | ---: |",
        f"| 总 episode | {len(result.removed_episodes) + result.kept_episodes} |",
        f"| 保留 | {result.kept_episodes} |",
        f"| 剔除 | {len(result.removed_episodes)} |",
        f"| 源帧数 | {source_frames} |",
        f"| 导出帧数 | {exported_frames} |",
        f"| 释放帧数 | {released}{ratio} |",
        "",
        "## 被剔除的 Episode",
        "",
    ]

    if removed:
        lines += [
            "| 原 episode | 严重度 | 处置 | 命中规则 | 帧数 | 建议动作 | 指纹（可加白名单） |",
            "| ---: | --- | --- | --- | ---: | --- | --- |",
        ]
        for ep in sorted(removed):
            item = removed[ep]
            reasons = item.get("reasons") or []
            labels = item.get("reason_labels") or reasons
            fps = "|".join(
                fingerprint(dataset_name, ep, code) for code in reasons
            )
            lines.append(
                f"| {ep} | {item.get('severity') or '-'} | {item.get('disposition') or '-'} | "
                f"{', '.join(labels) or '-'} | {frame_counts.get(ep, '-')} | "
                f"{item.get('recommended_action') or '-'} | {fps or '-'} |"
            )
        skipped = [ep for ep, item in removed.items() if item.get("disposition") in ("repair", "review")]
        if skipped:
            lines += [
                "",
                f"> 其中 {len(skipped)} 个 episode 的处置是 repair/review（可修复或待人工确认），"
                "本次严格按 keep 导出，未包含它们；如需保留请先走主流程清洗。",
            ]
    else:
        lines.append("未剔除任何 episode，导出结果为源数据集的完整副本。")

    lines += [
        "",
        "## 索引重排",
        "",
        f"`episode_index` 已从 0 连续重排（共 {result.kept_episodes} 个），旧 -> 新：",
        "",
        "```",
        _format_map(result.episode_index_map) or "（无）",
        "```",
        "",
        "## 校验",
        "",
        f"- episode_index 连续区间：{verification.get('episode_index_range', '-')}",
        f"- meta/episodes.jsonl：已按新编号同步生成",
        f"- 全局 index / frame_index：由清洗器按真实行数重排",
    ]
    if verification.get("issues"):
        lines += ["- 问题："] + [f"  - {msg}" for msg in verification["issues"]]

    warnings = list(result.warnings) + list((result.verification or {}).get("issues") or [])
    if warnings:
        lines += ["", "## 警告", ""] + [f"- {msg}" for msg in warnings]

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )

    report_path = Path(args.report).expanduser().resolve()
    try:
        report = _load_report(report_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return 3

    source_root = Path(
        args.dataset_path or report["dataset"].get("path") or report["dataset"].get("root") or ""
    ).expanduser()
    if not str(source_root) or not source_root.is_dir():
        print(f"[错误] 源数据集目录不存在: {source_root}", file=sys.stderr)
        return 3
    source_root = source_root.resolve()

    report_root = report["dataset"].get("path") or report["dataset"].get("root") or ""
    if args.dataset_path and report_root and Path(report_root).resolve() != source_root:
        LOG.warning("报告记录的数据集根目录为 %s，与 --dataset_path 不一致，指纹与统计可能对不上", report_root)

    out_root = Path(args.output_dir).expanduser().resolve()
    out_exists = out_root.exists()
    out_non_empty = out_exists and any(out_root.iterdir())
    if out_non_empty and not args.overwrite:
        print(f"[错误] 导出目录已存在且非空: {out_root}（如需覆盖请加 --overwrite）", file=sys.stderr)
        return 3
    # 空目录无需用户额外声明 --overwrite
    overwrite = args.overwrite or (out_exists and not out_non_empty)

    dataset = load_dataset(source_root)
    frame_counts = _episode_frame_counts(dataset)
    source_frames = int(dataset.info.get("total_frames") or dataset.num_frames or sum(frame_counts.values()))
    removed = _non_keep(report)
    unknown = sorted(set(removed) - set(dataset.episode_indices))
    if unknown:
        LOG.warning("报告中有 %d 个 episode 不在源数据集里，已忽略: %s", len(unknown), unknown[:10])
        for ep in unknown:
            removed.pop(ep, None)

    # 严格导出：所有非 keep 的 episode 一律物理剔除（repair/review 也不例外）
    dispositions = {ep: "drop" for ep in removed}
    LOG.info("报告: %s", report_path)
    LOG.info("源数据集: %s（%d 个 episode / %d 帧 / %sHz）",
             source_root, len(dataset.episode_indices), source_frames, dataset.fps)

    try:
        result = clean_dataset(
            dataset,
            dispositions,
            out_root.parent,
            prefer_cli=False,
            overwrite=overwrite,
            dataset_issues=report.get("dataset_issues"),
            out_root=out_root,
            link_mode="hardlink" if args.hardlink else "copy",
        )
    except Exception as exc:  # noqa: BLE001 - 导出脚本给出明确失败原因即可
        LOG.exception("导出失败")
        print(f"[错误] 导出失败: {exc}", file=sys.stderr)
        return 4

    jsonl_path = _write_episodes_jsonl(out_root)
    ok, verification = _verify_export(out_root)
    info = json.loads((out_root / "meta" / "info.json").read_text(encoding="utf-8"))
    exported_frames = int(info.get("total_frames", 0))

    summary_path = _write_summary(
        out_root / "cleaning_summary.md",
        report_path=report_path,
        source_root=source_root,
        out_root=out_root,
        report=report,
        result=result,
        verification=verification,
        source_frames=source_frames,
        exported_frames=exported_frames,
        removed=removed,
        frame_counts=frame_counts,
    )

    print(f"保留 {result.kept_episodes} / 剔除 {len(result.removed_episodes)} 个 episode，"
          f"释放 {max(source_frames - exported_frames, 0)} 帧")
    print(f"导出目录: {out_root}")
    print(f"episode 元数据: {jsonl_path}")
    print(f"清洗说明: {summary_path}")
    if not ok:
        for msg in verification.get("issues") or []:
            print(f"[校验失败] {msg}", file=sys.stderr)
        return 5
    print("校验通过: episode_index 从 0 连续，info 计数与数据一致")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
