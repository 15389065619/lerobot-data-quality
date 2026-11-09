"""精确率抽样工具与读取基准工具的回归测试。

这些用例守的是「数字会不会算错」，不是「脚本能不能跑」：

1. 抽样必须**每条规则至少覆盖一条**——漏掉一条规则，那张精确率表就少一行，
   而少掉的那一行通常正是没人看过的规则；
2. 精确率的分母只能是**已标注**记录，`unknown` 参与分母就会把尚未核对的样本
   算成误报，指标立刻失真；
3. 同一数据集多次重跑产生的重复 review_queue 行必须去重，否则抽样池被重跑
   次数污染（跑得多的数据集被过度采样）；
4. 基准表的吞吐/增量内存口径必须一致，失败项不能冒充 0。
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from tools.benchmark_reader import _row, print_table, write_markdown
from tools.eval_precision import (
    LABEL_FP,
    LABEL_TP,
    _annotation_id,
    _pct,
    compute_report,
    load_corpus,
    normalize_label,
    rule_list,
    stratified_sample,
)


def _queue(root: Path, name: str, rows: list[dict], dataset_name: str | None = None) -> Path:
    """在 root/name/ 下造一个 review_queue.csv（可选同名 quality_report.json）。"""
    out = root / name
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "review_queue.csv", index=False, encoding="utf-8-sig")
    if dataset_name:
        (out / "quality_report.json").write_text(
            json.dumps({"dataset": {"name": dataset_name}}, ensure_ascii=False), encoding="utf-8"
        )
    return out


# --------------------------------------------------------------- 分隔符与标签
def test_rule_list_splits_all_separators():
    assert rule_list("action_jump|nan_state") == ["action_jump", "nan_state"]
    assert rule_list("action_jump, nan_state") == ["action_jump", "nan_state"]
    assert rule_list("action_jump;nan_state") == ["action_jump", "nan_state"]
    assert rule_list("") == []
    assert rule_list(None) == []
    # 空段不能变成空规则名，否则精确率表里会多出一个 "" 行
    assert rule_list("a||b,") == ["a", "b"]


def test_normalize_label_aliases_and_fallbacks():
    assert normalize_label("TP") == LABEL_TP
    assert normalize_label(" true_positive ") == LABEL_TP
    assert normalize_label("是") == LABEL_TP
    assert normalize_label("FP") == LABEL_FP
    assert normalize_label("否") == LABEL_FP
    # 拼错/留空一律当未判定，不能默认成 TP 也不能默认成 FP
    assert normalize_label("") == "unknown"
    assert normalize_label("maybe") == "unknown"


def test_pct_handles_empty_denominator():
    assert _pct(3, 4) == "75.0%"
    assert _pct(1, 0) == "-"


# ------------------------------------------------------------------- 抽样与去重
def test_load_corpus_dedups_reruns_of_same_dataset(tmp_path):
    rows = [{"episode_index": 3, "reasons": "action_jump", "frame_count": 100, "detail": "d"}]
    _queue(tmp_path, "run1", rows, dataset_name="ds_a")
    _queue(tmp_path, "run2", rows, dataset_name="ds_a")  # 同一数据集重跑

    corpus = load_corpus(tmp_path, "**/review_queue.csv", [])
    assert len(corpus) == 1
    # 保留最后一次出现的记录（run2），source 应指向重跑目录
    assert corpus.iloc[0]["source"].endswith("run2/review_queue.csv")


def test_load_corpus_keeps_different_datasets(tmp_path):
    rows = [{"episode_index": 3, "reasons": "action_jump", "frame_count": 100, "detail": "d"}]
    _queue(tmp_path, "a", rows, dataset_name="ds_a")
    _queue(tmp_path, "b", rows, dataset_name="ds_b")

    corpus = load_corpus(tmp_path, "**/review_queue.csv", [])
    assert len(corpus) == 2


def test_stratified_sample_covers_every_rule(tmp_path):
    rules = ["action_jump", "nan_state", "frame_mismatch", "blank_video", "duplicate_episode"]
    rows = [
        {"episode_index": i, "reasons": rule, "frame_count": 100 + i, "detail": ""}
        for i, rule in enumerate(rules)
    ]
    # 再灌入大量同一条规则的记录，制造「随机抽样易漏长尾规则」的池子
    rows += [
        {"episode_index": 100 + i, "reasons": "action_jump", "frame_count": 200 + i, "detail": ""}
        for i in range(50)
    ]
    for i, row in enumerate(rows):
        _queue(tmp_path, f"run{i:03d}", [row], dataset_name=f"ds_{i:03d}")
    corpus = load_corpus(tmp_path, "**/review_queue.csv", [])

    sample = stratified_sample(corpus, n=5, seed=42)
    covered = {rule for reasons in sample["reasons"] for rule in rule_list(reasons)}
    assert len(sample) == 5
    assert covered == set(rules)


def test_stratified_sample_is_reproducible(tmp_path):
    rows = [
        {"episode_index": i, "reasons": "action_jump", "frame_count": 100 + i, "detail": ""}
        for i in range(20)
    ]
    for i, row in enumerate(rows):
        _queue(tmp_path, f"run{i:03d}", [row], dataset_name=f"ds_{i:03d}")
    corpus = load_corpus(tmp_path, "**/review_queue.csv", [])

    a = stratified_sample(corpus, n=6, seed=7)
    b = stratified_sample(corpus, n=6, seed=7)
    assert list(a["episode_index"]) == list(b["episode_index"])


def test_annotation_id_is_stable_and_discriminating():
    row = pd.Series(
        {
            "dataset": "ds_a",
            "episode_index": 3,
            "reasons": "action_jump",
            "frame_count": 100,
            "detail": "x",
        }
    )
    assert _annotation_id(row) == _annotation_id(row.copy())
    other = row.copy()
    other["episode_index"] = 4
    assert _annotation_id(row) != _annotation_id(other)


# ------------------------------------------------------------------- 精确率统计
def _write_annotations(path: Path, rows: list[dict]) -> Path:
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    return path


def test_report_precision_only_counts_labeled_rows(tmp_path, capsys):
    ann = _write_annotations(
        tmp_path / "ann.csv",
        [
            {"dataset": "d", "episode_index": 1, "reasons": "action_jump", "human_label": "TP"},
            {"dataset": "d", "episode_index": 2, "reasons": "action_jump", "human_label": "FP"},
            {"dataset": "d", "episode_index": 3, "reasons": "action_jump", "human_label": "unknown"},
            {"dataset": "d", "episode_index": 4, "reasons": "nan_state", "human_label": "FP"},
        ],
    )
    report = compute_report(ann, None, tmp_path, "**/review_queue.csv", []).set_index("rule")

    # action_jump: 抽样 3 条，已判定 2 条（1 TP / 1 FP）→ 50%，unknown 不计入分母
    assert report.loc["action_jump", "sampled"] == 3
    assert report.loc["action_jump", "labeled"] == 2
    assert report.loc["action_jump", "unknown"] == 1
    assert report.loc["action_jump", "precision"] == "50.0%"

    # nan_state: 只有一条误报 → 0%，而不是 "无数据"
    assert report.loc["nan_state", "precision"] == "0.0%"

    # 记录级：4 条里已判定 3 条（1 TP / 2 FP）
    assert report.loc["__record_level__", "precision"] == "33.3%"
    assert report.loc["__record_level__", "labeled"] == 3


def test_report_without_any_label_shows_dash(tmp_path):
    ann = _write_annotations(
        tmp_path / "ann.csv",
        [{"dataset": "d", "episode_index": 1, "reasons": "action_jump", "human_label": ""}],
    )
    report = compute_report(ann, None, tmp_path, "**/review_queue.csv", []).set_index("rule")
    assert report.loc["action_jump", "precision"] == "-"
    assert report.loc["action_jump", "labeled"] == 0


def test_report_rejects_file_without_human_label_column(tmp_path):
    ann = _write_annotations(
        tmp_path / "ann.csv", [{"dataset": "d", "episode_index": 1, "reasons": "action_jump"}]
    )
    with pytest.raises(SystemExit):
        compute_report(ann, None, tmp_path, "**/review_queue.csv", [])


def test_report_recall_needs_full_labels(tmp_path):
    _queue(
        tmp_path,
        "out",
        [{"episode_index": 1, "reasons": "action_jump", "frame_count": 10, "detail": ""}],
        dataset_name="ds",
    )
    ann = _write_annotations(
        tmp_path / "ann.csv",
        [{"dataset": "ds", "episode_index": 1, "reasons": "action_jump", "human_label": "TP"}],
    )
    full = _write_annotations(
        tmp_path / "full.csv",
        [
            # 规则命中且确实存在 → TP
            {"dataset": "ds", "episode_index": 1, "rule_code": "action_jump", "present": "1"},
            # 规则没报但实际存在 → FN，召回率必须因此下降
            {"dataset": "ds", "episode_index": 2, "rule_code": "action_jump", "present": "1"},
            # 规则没报也确实不存在 → TN，不影响召回率
            {"dataset": "ds", "episode_index": 3, "rule_code": "action_jump", "present": "0"},
        ],
    )
    report = compute_report(ann, full, tmp_path, "**/review_queue.csv", []).set_index("rule")
    assert report.loc["action_jump", "recall"] == "50.0%"


# ------------------------------------------------------------------- 基准表口径
def test_benchmark_row_uses_delta_not_peak():
    result = {
        "method": "streaming",
        "seconds": 0.5,
        "peak_mb": 732.0,
        "baseline_mb": 41.0,
        "rows": 3994,
    }
    row = _row(result, total_bytes=512 * 1024 * 1024)
    assert row["rows"] == "3,994"
    assert row["delta"] == "691"  # 峰值减基线，不能把解释器占用算进数据内存
    assert row["throughput"] == "1024"


def test_benchmark_row_marks_failure_instead_of_zero():
    row = _row({"method": "duckdb", "error": "ModuleNotFoundError: duckdb"}, 1024)
    assert row["seconds"] == "-"
    assert row["throughput"] == "-"


def test_benchmark_markdown_keeps_failed_method_visible(tmp_path):
    out = tmp_path / "bench.md"
    write_markdown(
        [
            {"method": "pandas", "seconds": 1.0, "peak_mb": 100.0, "baseline_mb": 10.0, "rows": 10},
            {"method": "duckdb", "error": "not installed"},
        ],
        out,
        n_files=1,
        total_bytes=10 * 1024 * 1024,
    )
    text = out.read_text(encoding="utf-8")
    assert "| 方式 |" in text
    assert "失败：not installed" in text
    # 失败行不能消失：读者需要知道哪种方式没测成
    assert text.count("duckdb") == 1


def test_benchmark_print_table_survives_all_failures(capsys):
    print_table([{"method": "pandas", "error": "boom"}])
    out = capsys.readouterr().out
    assert "boom" in out
