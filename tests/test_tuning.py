"""策略学习测试：人审数据 → severity 建议 → YAML patch（不自动生效）。"""

from __future__ import annotations

import csv
from pathlib import Path

from pipeline.tuning import analyse, load_reviews, render_yaml

CONFIG = {"severity": {"action_jump": "warning", "timestamp_drift": "warning",
                       "nan_state": "error"}}


def _report(episodes: list[dict]) -> dict:
    return {"invalid_episode_details": episodes}


def test_upgrade_when_humans_always_drop() -> None:
    episodes = [{"episode_index": i, "reasons": ["action_jump"],
                 "details": {"action_jump": {"max_delta": 50}}} for i in range(3)]
    result = analyse(_report(episodes), {i: "drop" for i in range(3)}, CONFIG)
    item = next(r for r in result["rules"] if r["code"] == "action_jump")
    assert item["suggested_severity"] == "error"
    assert item["drop_ratio"] == 1.0
    assert result["patch"]["severity"] == {"action_jump": "error"}


def test_downgrade_only_with_enough_samples() -> None:
    episodes = [{"episode_index": i, "reasons": ["timestamp_drift"]} for i in range(4)]
    result = analyse(_report(episodes), {i: "keep" for i in range(4)}, CONFIG)
    item = next(r for r in result["rules"] if r["code"] == "timestamp_drift")
    assert item["suggested_severity"] == "info"
    assert result["patch"]["severity"]["timestamp_drift"] == "info"

    # 样本不足（2 条）时保持原 severity，避免小样本过拟合
    result_small = analyse(_report(episodes[:2]), {0: "keep", 1: "keep"}, CONFIG)
    item_small = next(r for r in result_small["rules"] if r["code"] == "timestamp_drift")
    assert item_small["suggested_severity"] == "warning"
    assert item_small["changed"] is False


def test_numeric_separability_hint() -> None:
    episodes = [
        {"episode_index": 0, "reasons": ["action_jump"],
         "details": {"action_jump": {"max_delta": 10}}},
        {"episode_index": 1, "reasons": ["action_jump"],
         "details": {"action_jump": {"max_delta": 20}}},
        {"episode_index": 2, "reasons": ["action_jump"],
         "details": {"action_jump": {"max_delta": 100}}},
        {"episode_index": 3, "reasons": ["action_jump"],
         "details": {"action_jump": {"max_delta": 200}}},
    ]
    reviews = {0: "keep", 1: "keep", 2: "drop", 3: "drop"}
    result = analyse(_report(episodes), reviews, CONFIG)
    hint = result["threshold_hints"][0]
    assert hint["separable"] is True
    assert hint["max_kept"] == 20
    assert hint["min_dropped"] == 100


def test_load_reviews_reads_both_column_names(tmp_path: Path) -> None:
    path = tmp_path / "review.csv"
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["episode_index", "reviewer_disposition"])
        writer.writeheader()
        writer.writerow({"episode_index": 3, "reviewer_disposition": "Drop"})
        writer.writerow({"episode_index": 5, "reviewer_disposition": "keep"})
    reviews = load_reviews([path])
    assert reviews == {3: "drop", 5: "keep"}


def test_render_yaml_is_comment_only_when_no_change() -> None:
    assert "severity:" not in render_yaml({})
    text = render_yaml({"severity": {"action_jump": "error"}})
    assert "action_jump: error" in text
