"""本轮审计加固的回归测试。

对应 17 个本地数据集全量跑出来的四类问题：

1. v2.1 数据集的 index 语义被按 v3 契约误判为 error（humanoid_everyday_h1）
2. info.json 声明数与 meta/data 实际内容不符时完全静默（gym_xarm_pc）
3. 视频文件缺失被报成「帧数不一致」，`video_missing` 分支不可达（kuavo4）
4. 严格导出接受子集报告，未质检的 episode 被当合格数据写出
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from conftest import build_dataset, smooth_episode
from export_clean_dataset import _verify_export, main as export_main
from pipeline.schema import REQUIRED_FRAME_COLUMNS, audit_schema
from pipeline.validator import rules
from pipeline.validator.types import EpisodeResult


def _summary(starts: list[int], rows: list[int]) -> object:
    import pandas as pd

    return pd.DataFrame(
        {
            "episode_index": list(range(len(rows))),
            "rows": rows,
            "idx_min": starts,
            "idx_max": [s + r - 1 for s, r in zip(starts, rows)],
        }
    )


def _features(keys=REQUIRED_FRAME_COLUMNS) -> dict:
    return {c: {"dtype": "int64", "shape": (1,)} for c in keys}


# ------------------------------------------------------------------ v2.1 index
def test_v2_index_restart_per_episode_is_compliant():
    """v2.1 每集一个 parquet，index 从 0 起是规范，不能按 v3 的全局唯一判。"""
    items = audit_schema(
        features=_features(),
        splits={"train": "0:3"},
        total_episodes=3,
        data_columns=set(REQUIRED_FRAME_COLUMNS),
        data_summary=_summary([0, 0, 0], [10, 20, 30]),
        codebase_version="v2.1",
    )
    assert [i for i in items if i["column"] == "index"] == []


def test_v2_global_offset_index_is_compliant():
    """kuavo4 形态：v2.1 目录结构但沿用 v3 的全局 index（ep1 从 131 起），也合法。"""
    items = audit_schema(
        features=_features(),
        splits={"train": "0:2"},
        total_episodes=2,
        data_columns=set(REQUIRED_FRAME_COLUMNS),
        data_summary=_summary([0, 131], [131, 341]),
        codebase_version="v2.1",
    )
    assert [i for i in items if i["column"] == "index"] == []


def test_v2_index_gap_within_episode_is_error():
    """v2.1 也会坏：episode 内 index 断号（取值范围比行数大）必须报。"""
    import pandas as pd

    broken = pd.DataFrame(
        {
            "episode_index": [0, 1],
            "rows": [10, 20],
            "idx_min": [5, 0],
            "idx_max": [20, 19],  # ep0 的取值范围 16 > 10 行 → 有断号
        }
    )
    items = audit_schema(
        features=_features(),
        splits={"train": "0:2"},
        total_episodes=2,
        data_columns=set(REQUIRED_FRAME_COLUMNS),
        data_summary=broken,
        codebase_version="v2.1",
    )
    index_items = [i for i in items if i["column"] == "index"]
    assert len(index_items) == 1
    assert index_items[0]["severity"] == "error"
    assert "1/2" in index_items[0]["actual"]


def test_v3_index_restart_still_error():
    """v3.0 的同一份形态必须继续报 error，不能被版本判断放过。"""
    items = audit_schema(
        features=_features(),
        splits={"train": "0:3"},
        total_episodes=3,
        data_columns=set(REQUIRED_FRAME_COLUMNS),
        data_summary=_summary([0, 0, 0], [10, 20, 30]),
        codebase_version="v3.0",
    )
    assert any(i["column"] == "index" and i["severity"] == "error" for i in items)


# ------------------------------------------------------- info 与 meta/data 对账
def test_info_episode_count_mismatch_is_error():
    """gym_xarm_pc：info 声明 50 个 episode，meta 只有 5 行。"""
    items = audit_schema(
        features=_features(),
        splits=None,
        total_episodes=50,
        data_columns=set(REQUIRED_FRAME_COLUMNS),
        data_summary=_summary([0] * 5, [10] * 5),
        info_total_episodes=50,
        meta_episode_rows=5,
        info_total_frames=50,
        data_rows=50,
        codebase_version="v3.0",
    )
    hit = [i for i in items if i["column"] == "meta/info.json:total_episodes"]
    assert len(hit) == 1
    assert hit[0]["severity"] == "error"
    assert "45" in hit[0]["actual"]


def test_info_frame_count_mismatch_is_error():
    items = audit_schema(
        features=_features(),
        splits=None,
        total_episodes=1,
        data_columns=set(REQUIRED_FRAME_COLUMNS),
        data_summary=_summary([0], [100]),
        info_total_episodes=1,
        meta_episode_rows=1,
        info_total_frames=1809 * 10,
        data_rows=1809,
        codebase_version="v3.0",
    )
    hit = [i for i in items if i["column"] == "meta/info.json:total_frames"]
    assert len(hit) == 1
    assert hit[0]["severity"] == "error"


def test_count_check_skipped_when_truncated():
    """--max_episodes 时子集必然小于声明值，不能报成数据缺陷。"""
    items = audit_schema(
        features=_features(),
        splits=None,
        total_episodes=5,
        data_columns=set(REQUIRED_FRAME_COLUMNS),
        data_summary=_summary([0] * 5, [10] * 5),
        info_total_episodes=50,
        meta_episode_rows=5,
        info_total_frames=500,
        data_rows=50,
        codebase_version="v3.0",
        truncated=True,
    )
    assert [i for i in items if i["column"].startswith("meta/info.json:total_")] == []


# --------------------------------------------------------- 视频缺失 vs 定位失败
class _FakeDataset:
    locator_available = True
    fps = 30.0

    def __init__(self, locator):
        self._locator = locator

    def video_location(self, camera, episode_index, *, require_exists=True):
        return self._locator(camera, episode_index)


def test_missing_video_is_own_rule_not_frame_mismatch(monkeypatch, tmp_path):
    monkeypatch.setattr(rules, "FFPROBE", "ffprobe")
    absent = tmp_path / "episode_000000.mp4"
    res = EpisodeResult(episode_index=0, frame_count=100)

    rules.check_frame_mismatch(
        res, 0, _FakeDataset(lambda cam, ep: (absent, -1.0, -1.0)),
        ["observation.images.cam_left"], precise=False, config={},
    )

    assert "video_missing" in res.reasons
    assert "frame_mismatch" not in res.reasons
    detail = res.details["video_missing"]
    assert detail["missing_count"] == 1
    assert detail["cameras"] == ["observation.images.cam_left"]


def test_unlocatable_video_degrades_to_info(monkeypatch):
    """定位不到 ≠ 数据异常：不能因为 meta 给不出路径就判 error 并剔除。"""
    monkeypatch.setattr(rules, "FFPROBE", "ffprobe")
    res = EpisodeResult(episode_index=0, frame_count=100)

    rules.check_frame_mismatch(
        res, 0, _FakeDataset(lambda cam, ep: None),
        ["observation.images.cam_left"], precise=False, config={},
    )

    assert "frame_mismatch" in res.reasons
    assert res.severity_overrides["frame_mismatch"] == "info"
    assert "video_missing" not in res.reasons


# ------------------------------------------------------------- action_jump 分级
def _jump_result(peak: float, threshold: float = 1.0) -> EpisodeResult:
    arr = np.array([[0.0], [0.1], [peak], [peak + 0.05], [peak + 0.1], [peak + 0.15]])
    res = EpisodeResult(episode_index=0, frame_count=len(arr))
    rules.check_action_jump(arr, res, [threshold], {})
    return res


def test_action_jump_near_threshold_downgrades_to_info():
    res = _jump_result(peak=1.25)  # 超阈 1.15 倍，属阈值边缘
    assert "action_jump" in res.reasons
    assert res.details["action_jump"]["exceed_ratio"] == pytest.approx(1.15, abs=0.01)
    assert res.severity_overrides["action_jump"] == "info"


def test_action_jump_far_over_threshold_keeps_default_severity():
    res = _jump_result(peak=5.0)  # 超阈约 4.9 倍
    assert "action_jump" in res.reasons
    assert res.details["action_jump"]["exceed_ratio"] >= 3.0
    assert "action_jump" not in res.severity_overrides


# ------------------------------------------------------------- 导出完整性护栏
def _dataset(tmp_path, n: int = 5) -> Path:
    episodes = {ep: smooth_episode(40, offset=ep * 0.5, seed=ep) for ep in range(n)}
    return build_dataset(tmp_path / "ds", episodes)


def _write_report(path: Path, ds_path: Path, **overrides) -> Path:
    report = {
        "dataset": {
            "name": "ds",
            "path": str(ds_path),
            "fps": 30.0,
            "total_episodes": 5,
            "max_episodes_applied": None,
        },
        "summary": {
            "total_episodes": 5,
            "disposition": {"keep": 4, "repair": 0, "review": 1, "drop": 0},
        },
        "invalid_episode_details": [
            {"episode_index": 3, "severity": "warning", "disposition": "review",
             "reasons": ["action_jump"], "frame_count": 40},
        ],
    }
    report.update(overrides)
    path.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    return path


def test_export_refuses_truncated_report(tmp_path, capsys):
    ds_path = _dataset(tmp_path)
    report = _write_report(tmp_path / "r.json", ds_path)
    data = json.loads(report.read_text(encoding="utf-8"))
    data["dataset"]["max_episodes_applied"] = 3
    data["dataset"]["total_episodes"] = 5
    data["summary"]["total_episodes"] = 3
    report.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    assert export_main(["--report", str(report), "--output_dir", str(tmp_path / "out")]) == 3
    assert "从未质检" in capsys.readouterr().err


def test_export_refuses_incomplete_details(tmp_path, capsys):
    ds_path = _dataset(tmp_path)
    # keep 4 + 明细 1 = 5 才完整；这里声称 keep 4 却没有明细
    report = _write_report(tmp_path / "r.json", ds_path, invalid_episode_details=[])

    assert export_main(["--report", str(report), "--output_dir", str(tmp_path / "out")]) == 3
    assert "明细不完整" in capsys.readouterr().err


def test_verify_export_detects_info_vs_data_mismatch(tmp_path):
    ds_path = _dataset(tmp_path)
    report = _write_report(tmp_path / "r.json", ds_path)
    out = tmp_path / "out"
    assert export_main(["--report", str(report), "--output_dir", str(out)]) == 0

    info_path = out / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["total_frames"] += 1  # 只改声明，data 不动
    info_path.write_text(json.dumps(info, ensure_ascii=False), encoding="utf-8")

    ok, detail = _verify_export(out)
    assert not ok
    assert any("data 实际行数" in msg for msg in detail["issues"])


def test_verify_export_rows_come_from_parquet(tmp_path):
    """校验口径必须来自 data 实际行数，而不是 meta 自证。"""
    ds_path = _dataset(tmp_path)
    report = _write_report(tmp_path / "r.json", ds_path)
    out = tmp_path / "out"
    assert export_main(["--report", str(report), "--output_dir", str(out)]) == 0

    ok, detail = _verify_export(out)
    assert ok, detail["issues"]
    real = sum(
        pq.ParquetFile(f).metadata.num_rows for f in sorted((out / "data").rglob("*.parquet"))
    )
    assert detail["frames"] == real


# ------------------------------------------------- 结论持久化（增量模式结论漂移）
def test_severity_override_survives_dict_roundtrip():
    """to_dict/from_dict 必须带上分级覆盖，否则复用的结论会退回规则默认严重度。"""
    res = EpisodeResult(episode_index=7, frame_count=120)
    res.add("action_jump", severity="info", jump_count=1, exceed_ratio=1.2)

    clone = EpisodeResult.from_dict(json.loads(json.dumps(res.to_dict())))
    assert clone.severity_overrides == {"action_jump": "info"}
    assert clone.to_dict() == res.to_dict()


def test_severity_override_survives_manifest_reuse(normal_dataset, tmp_path):
    """增量二跑复用历史结论时，action_jump 的 info 降级不能被冲回 warning。"""
    from pipeline.config import load_config
    from pipeline.manifest import QualityManifest
    from pipeline.policy import decide
    from pipeline.validator import validate_dataset
    from pipeline.loader import load_dataset

    dataset = load_dataset(normal_dataset)
    config = load_config(robot_type=dataset.info.get("robot_type"))
    validation = validate_dataset(dataset, config)
    ep = sorted(validation.results)[0]
    validation.results[ep].add("action_jump", severity="info", jump_count=1, exceed_ratio=1.1)
    decisions = decide(validation)

    manifest = QualityManifest(tmp_path / "quality_manifest.json")
    manifest.store_results(
        dataset, config, validation, decisions, manifest.episode_signatures(dataset)
    )
    manifest.save()

    cached = QualityManifest.load(tmp_path / "quality_manifest.json").cached_results()
    assert cached[ep].severity_overrides == {"action_jump": "info"}
    # 报告取严重度的表达式与 reporter.py:51 一致
    assert (cached[ep].severity_overrides.get("action_jump") or "warning") == "info"


# --------------------------------------------- 流式模式的总帧数必须取 data 实读行数
def test_streaming_total_frames_uses_actual_rows(tmp_path):
    """--streaming 下 num_frames 是 meta.length 之和，用它校验 info 等于自证。"""
    from pipeline.loader import load_dataset

    root = tmp_path / "ds"
    build_dataset(
        root,
        {
            0: smooth_episode(20, seed=1),
            1: smooth_episode(20, seed=2),
        },
    )
    data_file = sorted((root / "data").rglob("*.parquet"))[0]
    table = pq.read_table(data_file)
    pq.write_table(table.slice(0, table.num_rows - 5), data_file)  # info/meta 完全不动

    view = load_dataset(root, load_frames=False)
    assert view.num_frames == 40  # meta.length 求和，仍是旧值
    items = [i for i in view.schema_contract if i["column"] == "meta/info.json:total_frames"]
    assert items, view.schema_contract
    assert f"data 实际 {table.num_rows - 5} 行" in items[0]["actual"]


# ---------------------------------- v2 index「重复与断号互相抵消」不能漏检
def test_v2_index_duplicate_cancelled_by_gap_is_error():
    """[0,1,1,3]：4 行、跨度 4，仅判跨度会通过；唯一值只有 3 个，必须报。"""
    import pandas as pd

    summary = pd.DataFrame(
        {
            "episode_index": [0, 1],
            "rows": [4, 3],
            "idx_min": [0, 0],
            "idx_max": [3, 2],
            "idx_nunique": [3.0, 3.0],
        }
    )
    items = audit_schema(
        features=_features(),
        splits={"train": "0:2"},
        total_episodes=2,
        data_columns=set(REQUIRED_FRAME_COLUMNS),
        data_summary=summary,
        codebase_version="v2.1",
    )
    index_items = [i for i in items if i["column"] == "index"]
    assert len(index_items) == 1
    assert index_items[0]["severity"] == "error"
    assert "1/2" in index_items[0]["actual"]
    assert "唯一值个数=[3.0]" in index_items[0]["actual"]


def test_v2_unknown_nunique_does_not_false_positive():
    """跨文件无法合并唯一值时记 NaN，应跳过唯一性判据而不是猜。"""
    import pandas as pd

    summary = pd.DataFrame(
        {
            "episode_index": [0, 1],
            "rows": [4, 3],
            "idx_min": [0, 4],
            "idx_max": [3, 6],
            "idx_nunique": [float("nan"), float("nan")],
        }
    )
    items = audit_schema(
        features=_features(),
        splits={"train": "0:2"},
        total_episodes=2,
        data_columns=set(REQUIRED_FRAME_COLUMNS),
        data_summary=summary,
        codebase_version="v2.1",
    )
    assert [i for i in items if i["column"] == "index"] == []


def test_loader_exposes_index_uniqueness(tmp_path):
    """真实文件里 index 集内重复（起止仍正确、跨度不变）必须被 v3 契约抓到。"""
    import pyarrow as pa

    from pipeline.loader import load_dataset

    root = tmp_path / "ds"
    build_dataset(root, {0: smooth_episode(20, seed=1), 1: smooth_episode(20, seed=2)})
    data_file = sorted((root / "data").rglob("*.parquet"))[0]
    table = pq.read_table(data_file)
    ep0 = np.asarray(table.column("episode_index").to_pylist()) == 0
    index = np.asarray(table.column("index").to_pylist()).copy()
    rows = np.flatnonzero(ep0)
    # 第 3 帧改成重复第 2 帧：episode 内仍是 0..19 的跨度、仍是 20 行，
    # 只有唯一值个数掉到 19——这正是「跨度判据」看不见的那类损坏。
    index[rows[2]] = index[rows[1]]
    table = table.set_column(
        table.column_names.index("index"), "index", pa.array(index, type=pa.int64())
    )
    pq.write_table(table, data_file)

    view = load_dataset(root)
    items = [i for i in view.schema_contract if i["column"] == "index"]
    assert items, view.schema_contract
    assert items[0]["severity"] == "error"
    assert "重复" in items[0]["rule"]
