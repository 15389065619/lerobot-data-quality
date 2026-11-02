"""FZMotion 风格 CSV 加载器测试。

覆盖：
  1. 单文件读取：长表 → 宽表同构转换（列、维度、episode_index、action 移位语义）；
  2. 缺陷容忍：NaN 坐标与缺 marker 行不导致崩溃，缺陷原样保留进 state；
  3. 多 episode 合并：目录加载、编号解析、维度一致性校验。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lerobot_data_quality.loaders import load_fzmotion_csv, load_fzmotion_csv_dir


def _write_csv(path: Path, rows: list[tuple[float, int, object, object, object]]) -> Path:
    """把 (timestamp, marker_id, x, y, z) 行写成长表 CSV；None 写成空单元格（=NaN）。"""
    df = pd.DataFrame(rows, columns=["timestamp", "marker_id", "x", "y", "z"])
    df.to_csv(path, index=False)
    return path


def _make_rows(n_frames: int = 4, n_markers: int = 2) -> list[tuple]:
    """构造规律数据：marker m 帧 f 的坐标 = (m*10+f, m*100+f, m*1000+f)，便于断言拼接顺序。"""
    return [
        (f / 120.0, m, float(m * 10 + f), float(m * 100 + f), float(m * 1000 + f))
        for f in range(n_frames) for m in range(n_markers)
    ]


def test_load_single_csv_wide_schema(tmp_path: Path) -> None:
    path = _write_csv(tmp_path / "episode_000.csv", _make_rows(n_frames=4, n_markers=2))
    df = load_fzmotion_csv(path, episode_index=3)

    # 列与基本形状
    assert list(df.columns) == ["episode_index", "frame_index", "timestamp",
                                "observation.state", "action"]
    assert len(df) == 4
    assert df["episode_index"].unique().tolist() == [3]
    assert df["frame_index"].tolist() == [0, 1, 2, 3]
    assert df["timestamp"].tolist() == pytest.approx([0.0, 1 / 120, 2 / 120, 3 / 120])

    # state = 按 marker_id 升序拼接的 3N 维向量（marker 主序：m0_xyz, m1_xyz）
    s0 = df["observation.state"].iloc[0]
    assert isinstance(s0, np.ndarray) and s0.dtype == np.float32
    np.testing.assert_array_equal(s0, np.array([0, 0, 0, 10, 100, 1000], dtype=np.float32))

    # action = 下一帧 state；最后一帧保持自身（定长契约）
    np.testing.assert_array_equal(df["action"].iloc[0], df["observation.state"].iloc[1])
    np.testing.assert_array_equal(df["action"].iloc[-1], df["observation.state"].iloc[-1])


def test_nan_and_missing_marker_do_not_crash(tmp_path: Path) -> None:
    rows = _make_rows(n_frames=3, n_markers=2)
    # 帧 1 的 marker 0：x/y/z 全部置空（遮挡丢点 → NaN）
    rows = [(ts, m, (None if (round(ts, 6) == round(1 / 120, 6) and m == 0) else x),
             (None if (round(ts, 6) == round(1 / 120, 6) and m == 0) else y),
             (None if (round(ts, 6) == round(1 / 120, 6) and m == 0) else z))
            for ts, m, x, y, z in rows]
    # 帧 2 的 marker 1：整行缺失（该帧少一个 marker）
    rows = [r for r in rows
            if not (round(r[0], 6) == round(2 / 120, 6) and r[1] == 1)]
    path = _write_csv(tmp_path / "episode_000.csv", rows)

    df = load_fzmotion_csv(path)  # 不崩溃即第一条断言
    assert len(df) == 3
    # 帧 1：marker 0 的三元组是 NaN，marker 1 正常
    s1 = df["observation.state"].iloc[1]
    assert np.isnan(s1[:3]).all() and np.isfinite(s1[3:]).all()
    # 帧 2：缺的 marker 1 三元组自动补 NaN，marker 0 正常
    s2 = df["observation.state"].iloc[2]
    assert np.isfinite(s2[:3]).all() and np.isnan(s2[3:]).all()


def test_load_dir_merges_episodes(tmp_path: Path) -> None:
    _write_csv(tmp_path / "episode_000.csv", _make_rows(n_frames=3, n_markers=2))
    _write_csv(tmp_path / "episode_001.csv", _make_rows(n_frames=5, n_markers=2))

    merged = load_fzmotion_csv_dir(tmp_path)
    # 两个 episode、帧数求和、编号来自文件名
    assert len(merged) == 8
    assert merged["episode_index"].unique().tolist() == [0, 1]
    # 每个 episode 的 frame_index 各自从 0 开始
    assert merged.loc[merged["episode_index"] == 1, "frame_index"].tolist() == [0, 1, 2, 3, 4]
    # 合并后按 (episode_index, timestamp) 排序
    keys = list(zip(merged["episode_index"], merged["timestamp"]))
    assert keys == sorted(keys)


def test_load_dir_rejects_inconsistent_marker_sets(tmp_path: Path) -> None:
    _write_csv(tmp_path / "episode_000.csv", _make_rows(n_frames=2, n_markers=2))
    _write_csv(tmp_path / "episode_001.csv", _make_rows(n_frames=2, n_markers=3))
    with pytest.raises(ValueError, match="marker"):
        load_fzmotion_csv_dir(tmp_path)


def test_missing_columns_and_missing_file(tmp_path: Path) -> None:
    bad = tmp_path / "episode_000.csv"
    bad.write_text("timestamp,marker_id,x,y\n0.0,0,1.0,2.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="缺少必需列"):
        load_fzmotion_csv(bad)
    with pytest.raises(FileNotFoundError):
        load_fzmotion_csv(tmp_path / "nope.csv")
    with pytest.raises(FileNotFoundError):
        load_fzmotion_csv_dir(tmp_path / "empty_dir")
