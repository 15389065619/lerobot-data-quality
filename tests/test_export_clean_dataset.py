"""导出脚本：只保留 keep、episode_index 连续重排、meta 与说明文件落地。"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq

from export_clean_dataset import main

from conftest import build_dataset, smooth_episode


def _my_dataset(tmp_path):
    episodes = {ep: smooth_episode(40, offset=ep * 0.5, seed=ep) for ep in range(5)}
    return build_dataset(tmp_path / "ds", episodes)


def _report(tmp_path, ds_path: Path) -> Path:
    report = {
        "dataset": {"name": "ds", "path": str(ds_path), "fps": 30.0},
        "summary": {},
        "invalid_episode_details": [
            {"episode_index": 2, "severity": "error", "disposition": "drop",
             "reasons": ["nan_state"], "reason_labels": ["状态含 NaN"],
             "frame_count": 40, "recommended_action": "剔除"},
            {"episode_index": 3, "severity": "warning", "disposition": "review",
             "reasons": ["action_jump"], "reason_labels": ["动作跳变"],
             "frame_count": 40, "recommended_action": "人工复核后决定"},
        ],
    }
    path = tmp_path / "quality_report.json"
    path.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    return path


def test_export_keeps_only_keep_and_reindexes(tmp_path):
    ds_path = _my_dataset(tmp_path)
    report_path = _report(tmp_path, ds_path)
    out = tmp_path / "exported"

    assert main(["--report", str(report_path), "--output_dir", str(out)]) == 0

    info = json.loads((out / "meta" / "info.json").read_text(encoding="utf-8"))
    assert info["total_episodes"] == 3
    assert info["splits"]["train"] == "0:3"

    data_files = sorted((out / "data").rglob("*.parquet"))
    table = pq.read_table(data_files[0])
    assert sorted(set(table.column("episode_index").to_pylist())) == [0, 1, 2]
    assert table.column("index").to_pylist() == list(range(table.num_rows))

    episodes = pq.read_table(next((out / "meta" / "episodes").rglob("*.parquet")))
    assert episodes.column("episode_index").to_pylist() == [0, 1, 2]
    assert sum(episodes.column("length").to_pylist()) == info["total_frames"]

    jsonl = (out / "meta" / "episodes.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert [json.loads(line)["episode_index"] for line in jsonl] == [0, 1, 2]

    summary = (out / "cleaning_summary.md").read_text(encoding="utf-8")
    assert "| 剔除 | 2 |" in summary
    assert "释放帧数" in summary
    assert "repair/review" in summary
    assert "4->2" in summary


def test_export_refuses_existing_output_without_overwrite(tmp_path):
    ds_path = _my_dataset(tmp_path)
    report_path = _report(tmp_path, ds_path)
    out = tmp_path / "exported"
    out.mkdir()
    (out / "keep.txt").write_text("x", encoding="utf-8")

    assert main(["--report", str(report_path), "--output_dir", str(out)]) == 3
    assert main(["--report", str(report_path), "--output_dir", str(out), "--overwrite"]) == 0
