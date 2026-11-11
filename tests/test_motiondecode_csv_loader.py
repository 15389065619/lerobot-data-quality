"""MotionDecode（Unitree G1 重定向轨迹）CSV 加载器测试。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lerobot_data_quality.loaders import (
    load_motiondecode_csv,
    load_motiondecode_csv_dir,
)


def _write_csv(path: Path, n: int = 4, label: str | None = None) -> Path:
    """构造 3 个根位置通道 + 2 个 DOF 的宽表；可选一个纯文本列验证会被丢弃。"""
    df = pd.DataFrame({
        "root_pos_x(m)": np.arange(n, dtype=float),
        "root_pos_y(m)": np.arange(n, dtype=float) + 10,
        "root_pos_z(m)": np.arange(n, dtype=float) + 20,
        "dof_a(rad)": np.arange(n, dtype=float) + 30,
        "dof_b(rad)": np.arange(n, dtype=float) + 40,
    })
    if label is not None:
        df["label"] = label
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def test_load_single_csv_schema_and_shift(tmp_path: Path) -> None:
    path = _write_csv(tmp_path / "episode_00001.csv", n=4, label="jump")
    df = load_motiondecode_csv(path, episode_index=7)

    assert list(df.columns) == ["episode_index", "frame_index", "timestamp",
                                "observation.state", "action"]
    assert len(df) == 4
    assert df["episode_index"].unique().tolist() == [7]
    assert df["frame_index"].tolist() == [0, 1, 2, 3]
    assert df["timestamp"].tolist() == pytest.approx([0.0, 1 / 120, 2 / 120, 3 / 120])

    # 纯文本列被丢弃；state 维度 = 5，且按 CSV 列顺序拼接
    s0 = df["observation.state"].iloc[0]
    assert isinstance(s0, np.ndarray) and s0.dtype == np.float32
    assert s0.tolist() == [0.0, 10.0, 20.0, 30.0, 40.0]

    # action = 下一帧 state；最后一帧保持自身
    np.testing.assert_array_equal(df["action"].iloc[0], df["observation.state"].iloc[1])
    np.testing.assert_array_equal(df["action"].iloc[-1], df["observation.state"].iloc[-1])


def test_dirty_values_are_preserved_as_nan(tmp_path: Path) -> None:
    path = tmp_path / "episode_00002.csv"
    path.write_text(
        "root_pos_x(m),root_pos_y(m),root_pos_z(m),dof_a(rad)\n"
        "0,1,2,3\n"
        "bad,1.5,2.5,3.5\n"
        "4,,6,\n",
        encoding="utf-8",
    )
    df = load_motiondecode_csv(path)
    assert len(df) == 3
    assert np.isnan(df["observation.state"].iloc[1][0])
    assert df["observation.state"].iloc[1][1] == pytest.approx(1.5)
    assert np.isnan(df["observation.state"].iloc[2][1])
    assert np.isnan(df["observation.state"].iloc[2][3])


def test_load_dir_recursive_assigns_unique_episode_ids(tmp_path: Path) -> None:
    # 不同类别下文件编号会重复（都是 00001），目录加载必须重新编号
    _write_csv(tmp_path / "A/BM_Jump_00001.csv", n=2)
    _write_csv(tmp_path / "B/BM_Dance_00001.csv", n=3)

    merged = load_motiondecode_csv_dir(tmp_path)
    assert len(merged) == 5
    assert merged["episode_index"].unique().tolist() == [0, 1]
    assert merged.loc[merged["episode_index"] == 1, "frame_index"].tolist() == [0, 1, 2]


def test_load_dir_rejects_inconsistent_dimensions(tmp_path: Path) -> None:
    _write_csv(tmp_path / "episode_00001.csv", n=2)
    other = tmp_path / "episode_00002.csv"
    pd.DataFrame({
        "root_pos_x(m)": [0.0], "root_pos_y(m)": [0.0], "root_pos_z(m)": [0.0],
    }).to_csv(other, index=False)

    with pytest.raises(ValueError, match="维度"):
        load_motiondecode_csv_dir(tmp_path)


def test_invalid_inputs_raise_clear_errors(tmp_path: Path) -> None:
    path = _write_csv(tmp_path / "episode_00001.csv", n=2)
    with pytest.raises(ValueError, match="fps"):
        load_motiondecode_csv(path, fps=0)
    with pytest.raises(FileNotFoundError):
        load_motiondecode_csv(tmp_path / "nope.csv")
    with pytest.raises(FileNotFoundError):
        load_motiondecode_csv_dir(tmp_path / "empty_dir")

    bad = tmp_path / "bad.csv"
    bad.write_text("a,b,c\n1,2,3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="缺少必需列"):
        load_motiondecode_csv(bad)

    empty = tmp_path / "empty.csv"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="空文件|没有任何数据行"):
        load_motiondecode_csv(empty)
