#!/usr/bin/env python3
"""历史 review_queue 抽样标注 + 逐规则精确率/召回率统计。

两个子命令：

    # 1) 从历史 outputs*/review_queue.csv 分层抽样，生成人工标注文件
    python tools/eval_precision.py sample --n 100 --out eval_annotation.csv

    # 2) 人工填写 human_label 后，统计各规则精确率（如有全量标注再算召回率）
    python tools/eval_precision.py report --annotations eval_annotation.csv
    python tools/eval_precision.py report --annotations eval_annotation.csv \\
        --full-labels full_labels.csv

标注取值只有三种：true_positive / false_positive / unknown（预填值，统计时跳过）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import pandas as pd

LABEL_TP = "true_positive"
LABEL_FP = "false_positive"
LABEL_UNKNOWN = "unknown"

_ALIASES = {
    "true_positive": LABEL_TP,
    "tp": LABEL_TP,
    "true": LABEL_TP,
    "1": LABEL_TP,
    "是": LABEL_TP,
    "false_positive": LABEL_FP,
    "fp": LABEL_FP,
    "false": LABEL_FP,
    "0": LABEL_FP,
    "否": LABEL_FP,
    "unknown": LABEL_UNKNOWN,
    "": LABEL_UNKNOWN,
    "?": LABEL_UNKNOWN,
    "不确定": LABEL_UNKNOWN,
    "待定": LABEL_UNKNOWN,
}

# 抽样文件保留的列（存在才保留），顺序尽量贴近人工核对习惯
_OUTPUT_COLUMNS = [
    "annotation_id",
    "dataset",
    "source",
    "episode_index",
    "severity",
    "reasons",
    "suggested_disposition",
    "recommended_action",
    "frame_count",
    "detail",
    "video",
    "window",
    "reviewer_disposition",
    "reviewer_note",
    "human_label",
]

_DEDUP_KEYS = ["dataset", "episode_index", "reasons", "frame_count", "detail"]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def rule_list(reasons: object) -> list[str]:
    """把 reasons 拆成规则列表，兼容 | , ; 三种分隔符。"""
    text = str(reasons or "")
    for sep in (",", ";", " "):
        text = text.replace(sep, "|")
    return [r.strip() for r in text.split("|") if r.strip()]


def _dataset_name(queue_path: Path) -> str:
    """优先用同目录 quality_report.json 的 dataset.name，退回目录名。"""
    report = queue_path.parent / "quality_report.json"
    if report.exists():
        try:
            payload = json.loads(report.read_text(encoding="utf-8"))
            name = (payload.get("dataset") or {}).get("name")
            if name:
                return str(name)
        except Exception:
            pass
    return queue_path.parent.name


def load_corpus(root: Path, pattern: str, excludes: list[str]) -> pd.DataFrame:
    """读取所有 review_queue.csv，按 (数据集, episode, 规则组合, 帧数, detail) 去重。

    同一数据集多次重跑会产生重复行，这里只保留最后一次出现的记录，
    避免把同一处异常重复计入抽样池。
    """
    files = sorted(root.glob(pattern))
    if not files:
        raise SystemExit(f"在 {root} 下没有找到匹配 {pattern} 的文件")

    frames: list[pd.DataFrame] = []
    skipped = 0
    for path in files:
        rel = str(path.relative_to(root)).replace("\\", "/")
        if any(token and token in rel for token in excludes):
            continue
        try:
            frame = pd.read_csv(path)
        except Exception as exc:  # 损坏/编码异常的历史文件不影响整体
            print(f"[warn] 跳过无法读取的 {rel}: {exc}", file=sys.stderr)
            skipped += 1
            continue
        if frame.empty:
            continue
        frame = frame.copy()
        frame["dataset"] = _dataset_name(path)
        frame["source"] = rel
        frames.append(frame)

    if not frames:
        raise SystemExit("所有 review_queue.csv 都是空的（或都被排除了），没有可抽样记录")

    corpus = pd.concat(frames, ignore_index=True)
    for column in ("reasons", "detail", "frame_count", "episode_index"):
        if column not in corpus.columns:
            corpus[column] = ""
    corpus["reasons"] = corpus["reasons"].fillna("").astype(str)
    corpus = corpus[corpus["reasons"].str.strip() != ""]

    before = len(corpus)
    corpus = corpus.drop_duplicates(subset=_DEDUP_KEYS, keep="last")
    corpus = corpus.reset_index(drop=True)

    print(
        f"[corpus] {len(files)} 个 review_queue.csv（跳过 {skipped} 个）→ "
        f"{before} 条原始记录 → 去重后 {len(corpus)} 条"
    )
    if len(corpus):
        counts = (
            corpus["reasons"].map(rule_list).explode().value_counts().sort_values(ascending=False)
        )
        print("[corpus] 规则分布：")
        for rule, count in counts.items():
            print(f"         {rule:<26} {count}")
    return corpus


def stratified_sample(corpus: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """先保证每条规则至少入样 1 条，再随机补足到 n 条。"""
    if len(corpus) <= n:
        return corpus.copy().reset_index(drop=True)

    shuffled = corpus.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    chosen: list[int] = []
    covered: set[str] = set()
    for idx, row in shuffled.iterrows():
        rules = set(rule_list(row["reasons"]))
        if rules - covered:
            chosen.append(int(idx))
            covered |= rules
            if len(chosen) == n:
                break

    if len(chosen) < n:
        rest = [int(i) for i in shuffled.index if i not in set(chosen)]
        k = min(n - len(chosen), len(rest))
        chosen.extend(random.Random(seed).sample(rest, k))

    all_rules = {rule for rules in corpus["reasons"].map(rule_list) for rule in rules}
    missing = sorted(all_rules - covered)
    if missing:
        print(
            f"[warn] 规则数（{len(all_rules)}）超过抽样数或无法全覆盖，未入样规则：{', '.join(missing)}",
            file=sys.stderr,
        )

    sample = shuffled.loc[chosen].copy()
    sort_cols = [c for c in ("dataset", "episode_index", "reasons") if c in sample.columns]
    return sample.sort_values(sort_cols, kind="stable").reset_index(drop=True)


def _annotation_id(row: pd.Series) -> str:
    key = "|".join(
        str(row.get(col, ""))
        for col in ("dataset", "episode_index", "reasons", "frame_count", "detail")
    )
    return hashlib.md5(key.encode("utf-8")).hexdigest()[:12]


def write_annotation(sample: pd.DataFrame, out_path: Path) -> None:
    sample = sample.copy()
    sample.insert(0, "annotation_id", sample.apply(_annotation_id, axis=1))
    sample["human_label"] = LABEL_UNKNOWN
    columns = [c for c in _OUTPUT_COLUMNS if c in sample.columns]
    sample = sample[columns]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"\n[out] 已写出 {out_path}（{len(sample)} 条，human_label 预填 unknown）")
    print(
        "\n填写说明：\n"
        "  1. 只修改 human_label 一列，取值 true_positive 或 false_positive；\n"
        "     拿不准/没时间核对的保持 unknown，统计时会自动跳过。\n"
        "  2. true_positive：记录描述的现象真实存在，且规则归因正确；\n"
        "     false_positive：误报（数据实际正常，或规则归错类）。\n"
        "  3. 核对方式：看 detail 里的实测值与 reasons；必要时回到\n"
        "     source 指向的输出目录打开 quality_report.json，或直接查原始 episode。\n"
        "  4. 同一 episode 有多个规则（reasons 用 | 分隔）时，只需给这条记录一个结论，\n"
        "     统计时会计入其中每一个规则；若只有一个规则成立，请在 reviewer_note 里注明。\n"
        "  5. 填完后运行：\n"
        "     python tools/eval_precision.py report --annotations "
        f"{out_path}\n"
    )


def normalize_label(value: object) -> str:
    text = str(value or "").strip().lower()
    return _ALIASES.get(text, LABEL_UNKNOWN)


def _pct(numerator: int, denominator: int) -> str:
    if denominator <= 0:
        return "-"
    return f"{100.0 * numerator / denominator:.1f}%"


def _flagged_set(root: Path, pattern: str, excludes: list[str]) -> set[tuple[str, str, str]]:
    """(dataset, episode, rule) 集合：历史 review_queue 中出现过即视为被规则命中。"""
    corpus = load_corpus(root, pattern, excludes)
    flagged: set[tuple[str, str, str]] = set()
    for _, row in corpus.iterrows():
        dataset = str(row["dataset"])
        episode = _norm_episode(row["episode_index"])
        for rule in rule_list(row["reasons"]):
            flagged.add((dataset, episode, rule))
    return flagged


def _norm_episode(value: object) -> str:
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    return text


def _read_present(value: object) -> bool | None:
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "是", "有"}:
        return True
    if text in {"0", "false", "no", "n", "否", "无"}:
        return False
    return None


def compute_report(
    annotations_path: Path,
    full_labels_path: Path | None,
    root: Path,
    pattern: str,
    excludes: list[str],
) -> pd.DataFrame:
    annotations = pd.read_csv(annotations_path, dtype=str).fillna("")
    if "human_label" not in annotations.columns:
        raise SystemExit(f"{annotations_path} 缺少 human_label 列，请先按说明填写")
    if "reasons" not in annotations.columns:
        raise SystemExit(f"{annotations_path} 缺少 reasons 列，不是 sample 子命令生成的文件")

    annotations = annotations.copy()
    annotations["label"] = annotations["human_label"].map(normalize_label)
    annotated = annotations[annotations["label"] != LABEL_UNKNOWN]
    unknown = len(annotations) - len(annotated)

    print(
        f"[labels] {len(annotations)} 条标注：已判定 {len(annotated)} 条"
        f"（TP {(annotated['label'] == LABEL_TP).sum()} / "
        f"FP {(annotated['label'] == LABEL_FP).sum()}），unknown {unknown} 条"
    )
    if annotated.empty:
        print("[warn] 还没有任何已判定记录，精确率会全部显示为 '-'", file=sys.stderr)

    rows: list[dict[str, object]] = []
    all_rules = sorted({rule for reasons in annotations["reasons"] for rule in rule_list(reasons)})
    for rule in all_rules:
        mask = annotations["reasons"].map(lambda text, r=rule: r in rule_list(text))
        subset = annotations[mask]
        known = subset[subset["label"] != LABEL_UNKNOWN]
        tp = int((known["label"] == LABEL_TP).sum())
        fp = int((known["label"] == LABEL_FP).sum())
        rows.append(
            {
                "rule": rule,
                "sampled": len(subset),
                "labeled": tp + fp,
                "tp": tp,
                "fp": fp,
                "unknown": len(subset) - tp - fp,
                "precision": _pct(tp, tp + fp),
                "recall": "-",
            }
        )

    # 记录级（不按规则拆）的总体精确率，便于看整体误报水位
    tp_all = int((annotated["label"] == LABEL_TP).sum())
    fp_all = int((annotated["label"] == LABEL_FP).sum())
    rows.append(
        {
            "rule": "__record_level__",
            "sampled": len(annotations),
            "labeled": len(annotated),
            "tp": tp_all,
            "fp": fp_all,
            "unknown": unknown,
            "precision": _pct(tp_all, tp_all + fp_all),
            "recall": "-",
        }
    )

    if full_labels_path is not None:
        full = pd.read_csv(full_labels_path, dtype=str).fillna("")
        required = {"dataset", "episode_index", "rule_code", "present"}
        missing_cols = required - set(full.columns)
        if missing_cols:
            raise SystemExit(
                f"{full_labels_path} 缺少列：{', '.join(sorted(missing_cols))}；"
                "需要 dataset, episode_index, rule_code, present"
            )
        flagged = _flagged_set(root, pattern, excludes)
        per_rule: dict[str, dict[str, int]] = {}
        invalid = 0
        for _, row in full.iterrows():
            present = _read_present(row["present"])
            if present is None:
                invalid += 1
                continue
            key = (
                str(row["dataset"]).strip(),
                _norm_episode(row["episode_index"]),
                str(row["rule_code"]).strip(),
            )
            stats = per_rule.setdefault(key[2], {"tp": 0, "fp": 0, "fn": 0, "tn": 0})
            hit = key in flagged
            if present and hit:
                stats["tp"] += 1
            elif present and not hit:
                stats["fn"] += 1
            elif not present and hit:
                stats["fp"] += 1
            else:
                stats["tn"] += 1
        if invalid:
            print(f"[warn] 全量标注里有 {invalid} 行 present 无法解析，已跳过", file=sys.stderr)

        for row in rows:
            stats = per_rule.get(str(row["rule"]))
            if not stats:
                continue
            tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
            row["recall"] = _pct(tp, tp + fn)
        for rule, stats in per_rule.items():
            if any(str(row["rule"]) == rule for row in rows):
                continue
            tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
            rows.append(
                {
                    "rule": rule,
                    "sampled": 0,
                    "labeled": 0,
                    "tp": 0,
                    "fp": 0,
                    "unknown": 0,
                    "precision": _pct(tp, tp + fp),
                    "recall": _pct(tp, tp + fn),
                }
            )

    return pd.DataFrame(rows)


def print_table(report: pd.DataFrame, full: bool) -> None:
    rule_width = max([len("rule")] + [len(str(r)) for r in report["rule"]])
    header = (
        f"{'rule':<{rule_width}}  {'sampled':>7}  {'labeled':>7}  {'TP':>4}  {'FP':>4}  "
        f"{'unknown':>7}  {'precision':>9}"
    )
    if full:
        header += f"  {'recall':>7}"
    print("\n" + header)
    print("-" * len(header))
    for _, row in report.iterrows():
        line = (
            f"{str(row['rule']):<{rule_width}}  {int(row['sampled']):>7}  {int(row['labeled']):>7}  "
            f"{int(row['tp']):>4}  {int(row['fp']):>4}  {int(row['unknown']):>7}  "
            f"{str(row['precision']):>9}"
        )
        if full:
            line += f"  {str(row['recall']):>7}"
        print(line)
    print(
        "\nprecision = TP / (TP + FP)，只统计已判定记录；"
        "recall 需要 --full-labels 全量标注，否则显示 '-'。"
    )


def to_markdown(report: pd.DataFrame) -> str:
    columns = ["rule", "sampled", "labeled", "tp", "fp", "unknown", "precision", "recall"]
    header = "| 规则 | 抽样数 | 已标注 | TP | FP | unknown | 精确率 | 召回率 |"
    sep = "|---|---|---|---|---|---|---|---|"
    lines = [header, sep]
    for _, row in report.iterrows():
        lines.append(
            "| {rule} | {sampled} | {labeled} | {tp} | {fp} | {unknown} | {precision} | {recall} |".format(
                **{col: row.get(col, "-") for col in columns}
            )
        )
    return "\n".join(lines)


def cmd_sample(args: argparse.Namespace) -> int:
    corpus = load_corpus(Path(args.root), args.glob, args.exclude)
    sample = stratified_sample(corpus, args.n, args.seed)
    write_annotation(sample, Path(args.out))
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    full_path = Path(args.full_labels) if args.full_labels else None
    report = compute_report(
        Path(args.annotations), full_path, Path(args.root), args.glob, args.exclude
    )
    print_table(report, full=full_path is not None)
    if args.markdown_out:
        md_path = Path(args.markdown_out)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(to_markdown(report) + "\n", encoding="utf-8")
        print(f"\n[out] Markdown 表格已写出 {md_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="历史 review_queue 抽样标注，并统计各规则精确率/召回率"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    root = repo_root()

    sample = sub.add_parser("sample", help="分层抽样生成人工标注 CSV")
    sample.add_argument("--root", default=str(root), help="仓库根目录（默认脚本上一级）")
    sample.add_argument("--glob", default="**/review_queue.csv", help="review_queue 匹配模式")
    sample.add_argument(
        "--exclude", nargs="*", default=[], help="路径中包含这些片段时跳过，如 cleaned"
    )
    sample.add_argument("--n", type=int, default=100, help="抽样条数（默认 100）")
    sample.add_argument("--seed", type=int, default=42, help="随机种子，保证可复现")
    sample.add_argument("--out", default="eval_annotation.csv", help="输出标注文件")
    sample.set_defaults(func=cmd_sample)

    report = sub.add_parser("report", help="读取标注文件并打印逐规则指标")
    report.add_argument("--annotations", default="eval_annotation.csv")
    report.add_argument(
        "--full-labels",
        default=None,
        help="全量标注 CSV（dataset, episode_index, rule_code, present），提供后计算召回率",
    )
    report.add_argument("--root", default=str(root))
    report.add_argument("--glob", default="**/review_queue.csv")
    report.add_argument("--exclude", nargs="*", default=[])
    report.add_argument("--markdown-out", default=None, help="另存一份 Markdown 表格")
    report.set_defaults(func=cmd_report)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
