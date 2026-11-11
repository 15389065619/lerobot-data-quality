"""从 quality_report.json 导出物理子集（默认只剔除 disposition == 'drop'）。

与主流程的 cleaner 区别：cleaner 尽量保住数据（repair 修复后保留、review 保留待人工确认），
本脚本是「按报告导出」——默认只把 drop 的 episode 物理剔除，review/repair 原样保留
（review 未经人工确认、repair 未应用修复），除非显式加 --drop-review / --drop-repair。
导出的 episode_index / index / meta 元数据从 0 开始绝对连续重排，避免 ID 错位带进训练。

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

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from pipeline.baseline import fingerprint
from pipeline.cleaner import clean_dataset
from pipeline.loader import DEFAULT_VIDEO_PATH, load_dataset

LOG = logging.getLogger("export_clean_dataset")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="按 quality_report.json 导出物理子集（默认只剔除 drop，episode_index 从 0 连续重排）",
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
    p.add_argument("--drop-review", action="store_true",
                   help="把 review（待人工复核）也物理剔除；默认保留，避免误杀未确认的数据")
    p.add_argument("--drop-repair", action="store_true",
                   help="把 repair（可自动修复）也物理剔除；默认原样保留，不应用修复")
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


def _non_keep(
    report: dict[str, Any],
    *,
    drop_review: bool = False,
    drop_repair: bool = False,
) -> dict[int, dict[str, Any]]:
    """返回要物理剔除的 episode。

    默认只剔除 disposition == drop；review/repair 原样保留，避免把
    "待人工确认"或"可修复"的数据当成坏数据静默删除。
    """
    out: dict[int, dict[str, Any]] = {}
    for item in report.get("invalid_episode_details") or []:
        try:
            ep = int(item.get("episode_index"))
        except (TypeError, ValueError):
            continue
        disposition = item.get("disposition") or "review"
        drop = disposition == "drop" \
            or (disposition == "review" and drop_review) \
            or (disposition == "repair" and drop_repair)
        if drop:
            out[ep] = item
    return out


def _kept_soft_episodes(report: dict[str, Any], removed: dict[int, dict[str, Any]]) -> list[int]:
    """未被剔除、但处置为 review/repair 的 episode（会在导出中原样保留）。"""
    kept: list[int] = []
    for item in report.get("invalid_episode_details") or []:
        try:
            ep = int(item.get("episode_index"))
        except (TypeError, ValueError):
            continue
        disposition = item.get("disposition") or "review"
        if ep not in removed and disposition in ("review", "repair"):
            kept.append(ep)
    return sorted(set(kept))


def _check_report_coverage(report: dict[str, Any]) -> str | None:
    """报告必须覆盖数据集全集，否则未质检的 episode 会被当成合格数据导出。

    `.invalid_episode_details` 只列非 keep，其余 episode 在导出时会被隐式视为 keep。
    因此「报告覆盖范围 == 数据集全集」是导出可执行的前提，必须先验。
    """
    ds = report.get("dataset") or {}
    summary = report.get("summary") or {}

    truncated = ds.get("max_episodes_applied")
    if truncated:
        return (
            f"报告只覆盖前 {truncated} 个 episode（--max_episodes 截断），"
            "其余 episode 从未质检；导出会把它们当作合格数据，请用全量报告重跑质检"
        )

    total = ds.get("total_episodes")
    scanned = summary.get("total_episodes")
    if total and scanned and int(scanned) != int(total):
        return f"报告评估了 {scanned} 个 episode，数据集共 {total} 个，覆盖不完整"

    disposition = summary.get("disposition") or {}
    if total and disposition:
        keep = int(disposition.get("keep") or 0)
        listed = len(report.get("invalid_episode_details") or [])
        if keep + listed != int(total):
            return (
                f"报告明细不完整：keep {keep} + 非 keep 明细 {listed} ≠ 数据集 {total} 个 episode，"
                "存在未评估的 episode，不能据此导出"
            )
    return None


def _episode_frame_counts(dataset) -> dict[int, int]:
    series = dataset.frames_by_episode()
    return {int(ep): int(n) for ep, n in series.items()}


def _read_exported_episodes(out_root: Path) -> pa.Table | None:
    files = sorted((out_root / "meta" / "episodes").rglob("*.parquet"))
    if files:
        table = pq.read_table(files[0])
        for extra in files[1:]:
            table = pa.concat_tables([table, pq.read_table(extra)], promote_options="permissive")
        return table
    # 未重排的 v2.1 副本只有 jsonl，必须读回来，否则会误判「元数据缺失」
    jsonl = out_root / "meta" / "episodes.jsonl"
    if not jsonl.is_file():
        return None
    rows = []
    with jsonl.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    if not rows:
        return None
    return pa.table({
        "episode_index": pa.array(
            [int(r.get("episode_index", i)) for i, r in enumerate(rows)], type=pa.int64()
        ),
        "length": pa.array([int(r.get("length") or 0) for r in rows], type=pa.int64()),
        "tasks": pa.array([list(r.get("tasks") or []) for r in rows], type=pa.list_(pa.string())),
    })


def _check_index_alignment(
    out_root: Path,
    eps: list[int],
    lengths: list[int],
    table: pa.Table,
    issues: list[str],
    *,
    v2: bool = False,
) -> None:
    """独立校验 index/frame_index 与 meta 偏移，避免 DatasetReader 静默错位。

    v2.x 的 index 是 episode 内帧号（也兼容全局编号写法），没有统一契约，
    因此只校验 frame_index；v3 才要求全局 index 从 0 连续。
    """
    data_files = sorted((out_root / "data").rglob("*.parquet"))
    if not data_files:
        if lengths:
            issues.append("meta/episodes 有数据但 data/ 下没有 parquet")
        return
    names = set(pq.ParquetFile(data_files[0]).schema_arrow.names)
    need = ["frame_index", "episode_index"] if v2 else ["index", "frame_index", "episode_index"]
    missing = [c for c in need if c not in names]
    if missing:
        issues.append(f"data parquet 缺少列: {', '.join(missing)}")
        return
    parts = [pq.read_table(f, columns=need) for f in data_files]
    data = pa.concat_tables(parts, promote_options="permissive")
    if not v2:
        index_vals = np.asarray(data.column("index").to_pylist(), dtype=np.int64)
        if not np.array_equal(index_vals, np.arange(len(index_vals), dtype=np.int64)):
            issues.append("data.index 不是从 0 连续的全局行号")

    ep_vals = np.asarray(data.column("episode_index").to_pylist(), dtype=np.int64)
    fi_vals = np.asarray(data.column("frame_index").to_pylist(), dtype=np.int64)
    if not v2 and "dataset_from_index" in table.column_names:
        froms = np.asarray(table.column("dataset_from_index").to_pylist(), dtype=np.int64)
        tos = np.asarray(table.column("dataset_to_index").to_pylist(), dtype=np.int64)
        expected_from = 0
        for ep, start, end, length in zip(eps, froms, tos, lengths):
            if int(start) != expected_from or int(end) - int(start) != length:
                issues.append(
                    f"episode {ep} 的 dataset_from/to_index=({start},{end}) 与累计偏移/长度 {length} 不符"
                )
                break
            expected_from = int(end)
        if expected_from != len(ep_vals):
            issues.append(f"meta 累计帧数 {expected_from} 与 data 实际行数 {len(ep_vals)} 不符")

    for ep, length in zip(eps, lengths):
        seg = fi_vals[ep_vals == ep]
        if len(seg) != length:
            issues.append(f"episode {ep} 实际 {len(seg)} 帧，meta.length={length}")
            continue
        if not np.array_equal(seg, np.arange(length, dtype=np.int64)):
            issues.append(f"episode {ep} 的 frame_index 不是 0..{length - 1}")


def _check_videos_and_stats(out_root: Path, info: dict[str, Any], table: pa.Table, issues: list[str]) -> None:
    """校验声明的视频文件存在、stats 覆盖所有非元数据特征。"""
    features = info.get("features") or {}
    video_keys = [k for k, v in features.items() if (v or {}).get("dtype") == "video"]
    if _is_v2(info):
        # v2.1 原样副本：episodes.jsonl 没有 chunk/file 索引列，按视频模板逐 episode 校验
        _check_v2_videos(out_root, info, table, video_keys, issues)
    else:
        _check_v3_videos(out_root, info, table, video_keys, issues)

    stats_path = out_root / "meta" / "stats.json"
    if not stats_path.is_file():
        issues.append("缺少 meta/stats.json")
        return
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    meta_cols = {"episode_index", "frame_index", "index", "timestamp", "task_index"}
    for key, spec in features.items():
        if key in meta_cols or (spec or {}).get("dtype") == "video":
            continue
        if key not in stats:
            issues.append(f"stats.json 缺少特征 {key}")


def _check_v3_videos(
    out_root: Path, info: dict[str, Any], table: pa.Table, video_keys: list[str], issues: list[str]
) -> None:
    video_path = info.get("video_path") or DEFAULT_VIDEO_PATH
    for cam in video_keys:
        c_idx, c_file = f"videos/{cam}/chunk_index", f"videos/{cam}/file_index"
        if c_idx not in table.column_names or c_file not in table.column_names:
            issues.append(f"视频特征 {cam} 缺少 chunk/file 索引列")
            continue
        for chunk, file_idx in zip(table.column(c_idx).to_pylist(), table.column(c_file).to_pylist()):
            rel = video_path.format(video_key=cam, chunk_index=int(chunk), file_index=int(file_idx))
            if not (out_root / rel).is_file():
                issues.append(f"视频文件缺失: {rel}")
                break


def _check_v2_videos(
    out_root: Path, info: dict[str, Any], table: pa.Table, video_keys: list[str], issues: list[str]
) -> None:
    tmpl = info.get("video_path")
    chunks_size = int(info.get("chunks_size") or 1000) or 1000
    eps = table.column("episode_index").to_pylist() if "episode_index" in table.column_names else []
    for cam in video_keys:
        if not tmpl:
            # 没有模板时只能确认该相机至少有一个视频文件
            cam_dirs = [p for p in (out_root / "videos").rglob(f"*{cam.split('.')[-1]}*") if p.is_dir()]
            cands = [f for d in cam_dirs for f in d.rglob("*.mp4")]
            if not cands:
                issues.append(f"视频特征 {cam} 未找到任何视频文件")
            continue
        for ep in eps:
            ep = int(ep)
            fields = {
                "video_key": cam, "camera": cam, "episode_index": ep,
                "episode_chunk": ep // chunks_size, "chunk_index": ep // chunks_size,
                "file_index": ep,
            }
            try:
                rel = tmpl.format(**fields)
            except (KeyError, IndexError, ValueError):
                issues.append(f"video_path 模板无法按 episode 定位: {tmpl}")
                break
            if not (out_root / rel).is_file():
                issues.append(f"视频文件缺失: {rel}")
                break


def _verify_export(out_root: Path) -> tuple[bool, dict[str, Any]]:
    """校验 episode_index 连续、index/frame_index 对齐、info 计数与数据一致。"""
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

    # 关键：用 data/*.parquet 的实际行数做独立校验。
    # 只比 info 与 meta/episodes 是「自己写的元数据校验自己」，发现不了数据没写全。
    data_files = sorted((out_root / "data").rglob("*.parquet"))
    data_rows = sum(pq.ParquetFile(f).metadata.num_rows for f in data_files)
    if data_rows != table_rows:
        issues.append(f"data 实际 {data_rows} 行，与 meta/episodes.length 之和 {table_rows} 不符")

    info_path = out_root / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    if int(info.get("total_episodes", -1)) != len(eps):
        issues.append(f"info.total_episodes={info.get('total_episodes')} 与实际 {len(eps)} 不符")
    if int(info.get("total_frames", -1)) != data_rows:
        issues.append(f"info.total_frames={info.get('total_frames')} 与 data 实际行数 {data_rows} 不符")

    splits = (info.get("splits") or {}).get("train")
    if splits != f"0:{len(eps)}":
        issues.append(f"info.splits.train={splits}，应为 0:{len(eps)}")

    _check_index_alignment(out_root, eps, lengths, table, issues, v2=_is_v2(info))
    _check_videos_and_stats(out_root, info, table, issues)

    return not issues, {
        "ok": not issues,
        "issues": issues,
        "episodes": len(eps),
        "frames": data_rows,
        "episode_index_range": f"0..{len(eps) - 1}" if eps else "empty",
    }


def _is_v2(info: dict[str, Any]) -> bool:
    return str(info.get("codebase_version") or "").startswith("v2")


def _write_episodes_jsonl(out_root: Path) -> Path:
    """把 v3 的 episodes.parquet 同步成人类可读的 episodes.jsonl。

    没有 v3 parquet（v2.1 原样副本）时不碰源 jsonl：它可能带任务/视频时间窗等
    v3 列之外的字段，重写会丢信息，更不能用空文件覆盖。
    """
    path = out_root / "meta" / "episodes.jsonl"
    files = sorted((out_root / "meta" / "episodes").rglob("*.parquet"))
    if not files:
        return path
    table = pq.read_table(files[0])
    for extra in files[1:]:
        table = pa.concat_tables([table, pq.read_table(extra)], promote_options="permissive")
    rows: list[str] = []
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
                f"> 其中 {len(skipped)} 个 episode 的处置是 repair/review，"
                "本次按 --drop-review/--drop-repair 的显式要求剔除。",
            ]
        kept_soft = _kept_soft_episodes(report, removed)
        if kept_soft:
            lines += [
                "",
                f"> 另有 {len(kept_soft)} 个非 keep episode 已原样保留（repair 未应用修复、"
                "review 未经人工确认）。如需剔除请加 --drop-review / --drop-repair。",
            ]
    else:
        lines.append("未剔除任何 episode，导出结果为源数据集的完整副本。")
        kept_soft = _kept_soft_episodes(report, removed)
        if kept_soft:
            lines += [
                "",
                f"> 有 {len(kept_soft)} 个非 keep episode（repair/review）已原样保留。",
            ]

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

    coverage_error = _check_report_coverage(report)
    if coverage_error:
        print(f"[错误] {coverage_error}", file=sys.stderr)
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
    removed = _non_keep(
        report,
        drop_review=args.drop_review,
        drop_repair=args.drop_repair,
    )
    unknown = sorted(set(removed) - set(dataset.episode_indices))
    if unknown:
        LOG.warning("报告中有 %d 个 episode 不在源数据集里，已忽略: %s", len(unknown), unknown[:10])
        for ep in unknown:
            removed.pop(ep, None)

    kept_soft = _kept_soft_episodes(report, removed)
    if kept_soft:
        LOG.warning(
            "保留 %d 个非 keep episode（review/repair 原样导出）；如需剔除请加 "
            "--drop-review / --drop-repair：%s",
            len(kept_soft), sorted(kept_soft)[:10],
        )

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
