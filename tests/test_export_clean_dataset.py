"""导出脚本：默认只剔除 drop、episode_index 连续重排、meta 与说明文件落地。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from export_clean_dataset import main

from conftest import build_dataset, smooth_episode


def _write_report(path: Path, ds_path: Path, details: list[dict]) -> Path:
    report = {
        "dataset": {"name": "ds", "path": str(ds_path), "fps": 30.0},
        "summary": {},
        "invalid_episode_details": details,
    }
    path.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    return path


def _my_dataset(tmp_path):
    episodes = {ep: smooth_episode(40, offset=ep * 0.5, seed=ep) for ep in range(5)}
    return build_dataset(tmp_path / "ds", episodes)


def _report(tmp_path, ds_path: Path) -> Path:
    return _write_report(tmp_path / "quality_report.json", ds_path, [
        {"episode_index": 2, "severity": "error", "disposition": "drop",
         "reasons": ["nan_state"], "reason_labels": ["状态含 NaN"],
         "frame_count": 40, "recommended_action": "剔除"},
        {"episode_index": 3, "severity": "warning", "disposition": "review",
         "reasons": ["action_jump"], "reason_labels": ["动作跳变"],
         "frame_count": 40, "recommended_action": "人工复核后决定"},
    ])


def _v21_dataset(tmp_path: Path) -> Path:
    """v2.1 布局：data/chunk-000/episode_XXXXXX.parquet + meta/episodes.jsonl。"""
    root = tmp_path / "ds_v21"
    (root / "data" / "chunk-000").mkdir(parents=True)
    (root / "meta").mkdir()
    vec = pa.list_(pa.float32(), 3)
    episodes = []
    total = 0
    for ep in range(3):
        spec = smooth_episode(40, seed=ep)
        state = np.asarray(spec["state"], dtype=np.float32)
        action = np.asarray(spec["action"], dtype=np.float32)
        n = len(state)
        pq.write_table(pa.table({
            "observation.state": pa.array(list(state), type=vec),
            "action": pa.array(list(action), type=vec),
            "timestamp": pa.array(np.arange(n, dtype=np.float32) / 30),
            "frame_index": pa.array(np.arange(n, dtype=np.int64)),
            "episode_index": pa.array(np.full(n, ep, dtype=np.int64)),
            "index": pa.array(np.arange(n, dtype=np.int64)),  # v2 语义：episode 内行号
            "task_index": pa.array(np.zeros(n, dtype=np.int64)),
        }), root / "data" / "chunk-000" / f"episode_{ep:06d}.parquet")
        episodes.append({"episode_index": ep, "length": n, "tasks": ["pick cube"]})
        total += n
    (root / "meta" / "episodes.jsonl").write_text(
        "\n".join(json.dumps(e) for e in episodes) + "\n", encoding="utf-8")
    (root / "meta" / "tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": "pick cube"}) + "\n", encoding="utf-8")
    info = {
        "codebase_version": "v2.1", "robot_type": "test_robot", "fps": 30,
        "total_episodes": 3, "total_frames": total, "total_tasks": 1, "chunks_size": 1000,
        "splits": {"train": "0:3"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.state": {"dtype": "float32", "shape": [3], "names": ["j0", "j1", "j2"]},
            "action": {"dtype": "float32", "shape": [3]},
            "timestamp": {"dtype": "float32", "shape": [1]},
            "frame_index": {"dtype": "int64", "shape": [1]},
            "episode_index": {"dtype": "int64", "shape": [1]},
            "index": {"dtype": "int64", "shape": [1]},
            "task_index": {"dtype": "int64", "shape": [1]},
        },
    }
    (root / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")
    return root


def test_export_drops_only_drop_and_reindexes(tmp_path):
    ds_path = _my_dataset(tmp_path)
    report_path = _report(tmp_path, ds_path)
    out = tmp_path / "exported"

    assert main(["--report", str(report_path), "--output_dir", str(out)]) == 0

    info = json.loads((out / "meta" / "info.json").read_text(encoding="utf-8"))
    # 默认保留 review（ep3），只剔除 drop（ep2）
    assert info["total_episodes"] == 4
    assert info["splits"]["train"] == "0:4"

    data_files = sorted((out / "data").rglob("*.parquet"))
    table = pq.read_table(data_files[0])
    assert sorted(set(table.column("episode_index").to_pylist())) == [0, 1, 2, 3]
    assert table.column("index").to_pylist() == list(range(table.num_rows))

    episodes = pq.read_table(next((out / "meta" / "episodes").rglob("*.parquet")))
    assert episodes.column("episode_index").to_pylist() == [0, 1, 2, 3]
    assert sum(episodes.column("length").to_pylist()) == info["total_frames"]

    jsonl = (out / "meta" / "episodes.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert [json.loads(line)["episode_index"] for line in jsonl] == [0, 1, 2, 3]

    summary = (out / "cleaning_summary.md").read_text(encoding="utf-8")
    assert "| 剔除 | 1 |" in summary
    assert "释放帧数" in summary
    assert "原样保留" in summary
    assert "3->2" in summary
    assert "4->3" in summary


def test_export_drop_review_removes_review_episode(tmp_path):
    ds_path = _my_dataset(tmp_path)
    report_path = _report(tmp_path, ds_path)
    out = tmp_path / "exported"

    assert main(["--report", str(report_path), "--output_dir", str(out), "--drop-review"]) == 0

    info = json.loads((out / "meta" / "info.json").read_text(encoding="utf-8"))
    assert info["total_episodes"] == 3
    episodes = pq.read_table(next((out / "meta" / "episodes").rglob("*.parquet")))
    assert episodes.column("episode_index").to_pylist() == [0, 1, 2]
    summary = (out / "cleaning_summary.md").read_text(encoding="utf-8")
    assert "| 剔除 | 2 |" in summary


def test_export_refuses_existing_output_without_overwrite(tmp_path):
    ds_path = _my_dataset(tmp_path)
    report_path = _report(tmp_path, ds_path)
    out = tmp_path / "exported"
    out.mkdir()
    (out / "keep.txt").write_text("x", encoding="utf-8")

    assert main(["--report", str(report_path), "--output_dir", str(out)]) == 3
    assert main(["--report", str(report_path), "--output_dir", str(out), "--overwrite"]) == 0


def test_export_from_v21_with_drop_upgrades_to_v3(tmp_path):
    ds_path = _v21_dataset(tmp_path)
    report_path = _write_report(tmp_path / "r.json", ds_path, [
        {"episode_index": 1, "severity": "error", "disposition": "drop",
         "reasons": ["nan_state"], "reason_labels": ["状态含 NaN"],
         "frame_count": 40, "recommended_action": "剔除"},
    ])
    out = tmp_path / "out"

    assert main(["--report", str(report_path), "--output_dir", str(out)]) == 0

    info = json.loads((out / "meta" / "info.json").read_text(encoding="utf-8"))
    assert info["codebase_version"] == "v3.0"
    assert info["total_episodes"] == 2
    assert list((out / "meta" / "episodes").rglob("*.parquet"))
    data = pa.concat_tables([
        pq.read_table(f) for f in sorted((out / "data").rglob("*.parquet"))
    ])
    assert data.column("index").to_pylist() == list(range(data.num_rows))
    assert sorted(set(data.column("episode_index").to_pylist())) == [0, 1]
    jsonl = (out / "meta" / "episodes.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert [json.loads(line)["episode_index"] for line in jsonl] == [0, 1]


def test_export_v21_without_drop_keeps_v21_layout_and_jsonl(tmp_path):
    ds_path = _v21_dataset(tmp_path)
    report_path = _write_report(tmp_path / "r.json", ds_path, [])
    out = tmp_path / "out"
    source_jsonl = (ds_path / "meta" / "episodes.jsonl").read_text(encoding="utf-8")

    assert main(["--report", str(report_path), "--output_dir", str(out)]) == 0

    info = json.loads((out / "meta" / "info.json").read_text(encoding="utf-8"))
    # 原样副本不能强行升级成 v3，否则 info 与真实的 v2 文件布局不一致
    assert info["codebase_version"] == "v2.1"
    assert not list((out / "meta" / "episodes").rglob("*.parquet"))
    assert (out / "meta" / "episodes.jsonl").read_text(encoding="utf-8") == source_jsonl
