from __future__ import annotations

import json

from pipeline.card import build_card, main

REPORT = {
    "generated_at": "2026-11-12T10:00:00+08:00",
    "dataset": {
        "path": "/data/demo", "name": "demo", "robot_type": "engineai_pm01",
        "fps": 50, "total_episodes": 2, "total_frames": 240,
        "state_dims": 31, "video_features": [], "fingerprint": "abc123",
    },
    "summary": {
        "total_episodes": 2, "valid_episodes": 1, "invalid_episodes": 1,
        "valid_ratio": 0.5,
        "severity": {"error": 1, "warning": 0, "info": 0},
        "disposition": {"keep": 1, "repair": 0, "review": 1, "drop": 0},
        "invalid_frames": 3, "drop_frames": 0, "trim_frames": 3, "repair_frames": 0,
    },
    "skipped_checks": ["frame_mismatch"],
    "cleaning": {"performed": True, "method": "manual", "output_path": "out/cleaned",
                 "warnings": []},
    "recommendations": ["检查 ep 1 的 timestamp 漂移"],
}

INFO = {
    "robot_type": "engineai_pm01", "fps": 50, "total_episodes": 2, "total_frames": 240,
    "features": {
        "observation.state": {"dtype": "float32", "shape": [31]},
        "observation.body_pos_w": {"dtype": "float32", "shape": [29, 3]},
        "timestamp": {"dtype": "float32", "shape": [1]},
    },
}

STATS = {
    "observation.state": {"count": 237, "mean": [0.1, 0.2, 0.3]},
    "observation.body_pos_w": {"count": 237, "mean": [[0.0, 0.0, 0.0]] * 29},
}


def test_card_contains_core_sections():
    md = build_card(REPORT, INFO, STATS)
    assert "# 数据集卡片：demo" in md
    assert "健康度 **error**" in md
    assert "`observation.body_pos_w`" in md
    assert "`(29, 3)`" in md and "张量" in md
    assert "keep 1" in md and "drop 0" in md
    assert "## 未执行的检查" in md and "frame_mismatch" in md
    assert "检查 ep 1 的 timestamp 漂移" in md
    assert "`timestamp`" not in md  # 索引/时间列不进特征表


def test_card_tolerates_missing_info_and_stats():
    md = build_card({"dataset": {"path": "/data/only_path"}, "summary": {}})
    assert "# 数据集卡片：only_path" in md
    assert "## 特征清单" not in md
    assert "健康度 **clean**" in md


def test_cli_writes_markdown(tmp_path):
    report_path = tmp_path / "quality_report.json"
    report_path.write_text(json.dumps(REPORT), encoding="utf-8")
    dataset_dir = tmp_path / "cleaned_dataset"
    (dataset_dir / "meta").mkdir(parents=True)
    (dataset_dir / "meta" / "info.json").write_text(json.dumps(INFO), encoding="utf-8")
    (dataset_dir / "meta" / "stats.json").write_text(json.dumps(STATS), encoding="utf-8")

    out = tmp_path / "CARD.md"
    assert main(["--report", str(report_path), "--out", str(out)]) == 0
    text = out.read_text(encoding="utf-8")
    assert "observation.body_pos_w" in text and "237" in text
