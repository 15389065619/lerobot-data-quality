"""时间轴修复只作用于目标 episode，不得重写同文件其它 episode。"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from conftest import build_dataset, smooth_episode

from pipeline.cleaner import clean_dataset
from pipeline.loader import load_dataset

DATA = ("data", "chunk-000", "file-000.parquet")


def test_timestamp_repair_is_scoped_to_target_episode(tmp_path):
    root = build_dataset(
        tmp_path / "ds",
        {0: smooth_episode(20, seed=0), 1: smooth_episode(20, offset=2.0, seed=1)},
    )
    path = root.joinpath(*DATA)
    df = pq.read_table(path).to_pandas()
    # ep1 的时间戳故意保留非零起点，用于验证不会被一起重写
    ep1 = df.episode_index == 1
    df.loc[ep1, "timestamp"] = 100.0 + np.arange(int(ep1.sum()))
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path)

    decisions = {0: SimpleNamespace(disposition="repair", repair="timestamps")}
    clean_dataset(load_dataset(root), decisions, tmp_path / "out", prefer_cli=False)

    out = pq.read_table(
        tmp_path / "out" / "cleaned_dataset" / "data" / "chunk-000" / "file-000.parquet"
    ).to_pandas()
    ep1_out = out[out.episode_index == 1].sort_values("frame_index")
    assert ep1_out["timestamp"].iloc[0] == 100.0
