"""清洗路径 × 特征形态矩阵：所有组合都必须不崩、不静默丢 stats、shape 与 info 对齐。

覆盖三条真实分支：
    noop        -> copy_untouched      -> rewrite_stats
    index       -> reindex_untouched   -> rewrite_stats
    manual      -> manual_clean        -> StreamingStats / rewrite_stats_streaming

以及 1D / 2D 嵌套特征、无 NaN / 部分 NaN / 全 NaN、源 stats 有 / 无。
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from pipeline.cleaner import clean_dataset
from pipeline.loader import load_dataset

from conftest import build_dataset, smooth_episode

KEY = "observation.extra_feature"
SOURCE_COUNT = 777
EPISODE_FRAMES = 12
TOTAL_FRAMES = EPISODE_FRAMES * 2

PATHS = {
    "noop": ([], "noop-copy"),
    "index": (
        {0: SimpleNamespace(disposition="repair", repair="index")},
        "index-repair-copy",
    ),
    "manual": ([0], "manual"),
}


def _add_feature(root, values: np.ndarray, source_count: int | None) -> None:
    """追加特征列并同步 info.json；source_count 非空时伪造源 stats 记录。"""
    path = root / "data" / "chunk-000" / "file-000.parquet"
    table = pq.read_table(path)
    if values.ndim == 2:
        payload = [[float(v) for v in row] for row in values]
        column = pa.array(payload, type=pa.list_(pa.float32(), values.shape[1]))
    else:
        payload = [[[float(v) for v in vec] for vec in row] for row in values]
        column = pa.array(payload, type=pa.list_(pa.list_(pa.float32(), values.shape[2])))
    pq.write_table(table.append_column(KEY, column), path)

    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["features"][KEY] = {"dtype": "float32", "shape": list(values.shape[1:])}
    info_path.write_text(json.dumps(info), encoding="utf-8")

    if source_count is not None:
        stats_path = root / "meta" / "stats.json"
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        zeros = np.zeros(values.shape[1:]).tolist()
        stats[KEY] = {
            "count": source_count, "mean": zeros, "std": zeros,
            "min": zeros, "max": zeros,
        }
        stats_path.write_text(json.dumps(stats), encoding="utf-8")


def _run_case(tmp_path, path_name, dims, nan_mode, source):
    dispositions, expected_method = PATHS[path_name]
    root = build_dataset(tmp_path / "ds", {
        0: smooth_episode(EPISODE_FRAMES, seed=1),
        1: smooth_episode(EPISODE_FRAMES, offset=0.5, seed=2),
    })
    shape = (5, 3) if dims == 2 else (4,)
    values = np.random.default_rng(0).normal(size=(TOTAL_FRAMES, *shape)).astype(np.float32)
    if nan_mode == "partial":
        values[EPISODE_FRAMES] = np.nan  # ep1 首帧，manual 路径同样会命中
    elif nan_mode == "all":
        values[:] = np.nan
    _add_feature(root, values, SOURCE_COUNT if source == "present" else None)

    result = clean_dataset(
        load_dataset(root), dispositions, tmp_path / "out", prefer_cli=False
    )
    stats = json.loads(
        (tmp_path / "out" / "cleaned_dataset" / "meta" / "stats.json").read_text(encoding="utf-8")
    )
    return result, stats, shape, expected_method


MATRIX_CASES = [
    (path_name, dims, nan_mode, source)
    for path_name in PATHS
    for dims in (1, 2)
    for nan_mode in ("clean", "partial", "all")
    for source in ("absent", "present")
]


@pytest.mark.parametrize("path_name,dims,nan_mode,source", MATRIX_CASES)
def test_cleaning_matrix(tmp_path, path_name, dims, nan_mode, source):
    result, stats, shape, expected_method = _run_case(
        tmp_path, path_name, dims, nan_mode, source
    )
    assert result.success is True
    assert result.method == expected_method

    kept = EPISODE_FRAMES if path_name == "manual" else TOTAL_FRAMES
    feature_warnings = [w for w in result.warnings if KEY in w]

    # 所有路径都不允许出现索引/时间列的伪告警
    assert not [w for w in result.warnings
                if any(token in w for token in ("timestamp", "frame_index", "episode_index"))]

    if nan_mode == "all" and source == "absent":
        assert KEY not in stats
        assert feature_warnings and "统计量计算失败" in feature_warnings[0]
        return

    assert KEY in stats, f"{path_name}/{dims}D/{nan_mode}/{source}: 特征统计量被静默丢弃"
    if nan_mode == "all":
        assert stats[KEY]["count"] == SOURCE_COUNT
        assert feature_warnings and "统计量计算失败" in feature_warnings[0]
    else:
        expected_count = kept - (1 if nan_mode == "partial" else 0)
        assert stats[KEY]["count"] == expected_count
        assert not feature_warnings

    assert np.asarray(stats[KEY]["mean"]).shape == shape
