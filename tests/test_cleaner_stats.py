"""rewrite_stats 对多维特征的处理：不得静默丢字段，且保留特征原 shape。"""

from __future__ import annotations

import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from pipeline.cleaner import clean_dataset
from pipeline.loader import load_dataset
from pipeline.reporter import attach_cleaning_result

from conftest import build_dataset, smooth_episode


def _add_2d_feature(root, key: str, values: np.ndarray) -> None:
    """把 (N, M, K) 数组作为二维特征列写进 parquet 与 info.json。"""
    path = root / "data" / "chunk-000" / "file-000.parquet"
    table = pq.read_table(path)
    payload = [[[float(v) for v in vec] for vec in row] for row in values]
    column = pa.array(payload, type=pa.list_(pa.list_(pa.float32(), values.shape[2])))
    pq.write_table(table.append_column(key, column), path)

    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["features"][key] = {"dtype": "float32", "shape": list(values.shape[1:])}
    info_path.write_text(json.dumps(info), encoding="utf-8")


def _set_source_stats(root, key: str, count: int) -> None:
    """伪造一份「源 stats.json 有值」的记录，用于验证回退到源统计量时是否告警。"""
    path = root / "meta" / "stats.json"
    stats = json.loads(path.read_text(encoding="utf-8"))
    zeros = [[0.0, 0.0, 0.0] for _ in range(29)]
    stats[key] = {"count": count, "mean": zeros, "std": zeros,
                  "min": zeros, "max": zeros}
    path.write_text(json.dumps(stats), encoding="utf-8")


def test_clean_dataset_keeps_2d_feature_stats(tmp_path):
    root = build_dataset(tmp_path / "ds", {0: smooth_episode(20, seed=1)})
    body = np.random.default_rng(0).normal(size=(20, 29, 3)).astype(np.float32)
    body[3, 1, 2] = np.nan  # 单帧单点缺失：整帧应被剔除
    _add_2d_feature(root, "observation.body_pos_w", body)

    result = clean_dataset(load_dataset(root), [], tmp_path / "out")
    assert result.warnings == []

    stats = json.loads(
        (tmp_path / "out" / "cleaned_dataset" / "meta" / "stats.json").read_text(encoding="utf-8")
    )
    assert "observation.body_pos_w" in stats, "二维特征的 stats 不得被静默丢弃"
    mean = np.asarray(stats["observation.body_pos_w"]["mean"])
    assert mean.shape == (29, 3), "统计量必须保留特征原 shape，不能摊平成 87 维"
    assert stats["observation.body_pos_w"]["count"] == 19
    kept = body[np.isfinite(body.reshape(len(body), -1)).all(axis=1)]
    np.testing.assert_allclose(mean, kept.mean(axis=0), atol=1e-6)

    # 一维特征输出形态与旧实现保持一致（扁平列表）
    assert len(stats["observation.state"]["mean"]) == 3
    assert isinstance(stats["observation.state"]["mean"][0], float)


def test_clean_dataset_warns_when_stats_cannot_be_computed(tmp_path):
    root = build_dataset(tmp_path / "ds", {0: smooth_episode(8, seed=2)})
    _add_2d_feature(root, "observation.body_pos_w", np.full((8, 29, 3), np.nan, dtype=np.float32))

    result = clean_dataset(load_dataset(root), [], tmp_path / "out")
    messages = [w for w in result.warnings if "统计量计算失败" in w]
    assert messages and "observation.body_pos_w" in messages[0]

    stats = json.loads(
        (tmp_path / "out" / "cleaned_dataset" / "meta" / "stats.json").read_text(encoding="utf-8")
    )
    assert "observation.body_pos_w" not in stats


def test_stats_failure_is_highlighted_in_report(tmp_path):
    report_path = tmp_path / "quality_report.json"
    report_path.write_text(
        json.dumps({"summary": {}, "recommendations": []}), encoding="utf-8"
    )
    attach_cleaning_result(report_path, {
        "kept_episodes": 1,
        "warnings": ["特征 observation.body_pos_w 统计量计算失败：全部帧含 NaN/Inf，无有效样本"],
    })

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert any("统计量计算失败" in rec for rec in report["recommendations"])
    assert report["cleaning"]["warnings"]


def test_manual_clean_keeps_2d_shape_and_no_fake_warnings(tmp_path):
    root = build_dataset(tmp_path / "ds", {
        0: smooth_episode(10, seed=3),
        1: smooth_episode(10, offset=1.0, seed=4),
    })
    body = np.random.default_rng(5).normal(size=(20, 29, 3)).astype(np.float32)
    _add_2d_feature(root, "observation.body_pos_w", body)

    result = clean_dataset(load_dataset(root), [0], tmp_path / "out", prefer_cli=False)
    assert result.method == "manual"
    assert [w for w in result.warnings if "统计量" in w] == [], "索引列不应产生伪告警"

    stats = json.loads(
        (tmp_path / "out" / "cleaned_dataset" / "meta" / "stats.json").read_text(encoding="utf-8")
    )
    mean = np.asarray(stats["observation.body_pos_w"]["mean"])
    assert mean.shape == (29, 3), "流式路径也必须保留特征原 shape"
    assert stats["observation.body_pos_w"]["count"] == 10
    np.testing.assert_allclose(mean, body[10:].mean(axis=0), atol=1e-6)


def test_streaming_stale_source_stats_is_reported(tmp_path):
    root = build_dataset(tmp_path / "ds", {
        0: smooth_episode(8, seed=6),
        1: smooth_episode(8, offset=1.0, seed=7),
    })
    _add_2d_feature(
        root, "observation.body_pos_w", np.full((16, 29, 3), np.nan, dtype=np.float32)
    )
    _set_source_stats(root, "observation.body_pos_w", 777)

    result = clean_dataset(load_dataset(root), [0], tmp_path / "out", prefer_cli=False)
    messages = [w for w in result.warnings if "observation.body_pos_w" in w]
    assert messages, "回退到源 stats 时必须告警"
    assert "统计量计算失败" in messages[0] and "沿用源 stats.json" in messages[0]
    assert not [w for w in result.warnings if "timestamp" in w or "frame_index" in w]

    stats = json.loads(
        (tmp_path / "out" / "cleaned_dataset" / "meta" / "stats.json").read_text(encoding="utf-8")
    )
    assert stats["observation.body_pos_w"]["count"] == 777
    assert np.asarray(stats["observation.body_pos_w"]["mean"]).shape == (29, 3)
