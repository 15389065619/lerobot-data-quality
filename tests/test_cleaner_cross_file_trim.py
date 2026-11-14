"""同一 episode 跨多个 parquet 时，裁帧必须按整集计数。"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from conftest import build_dataset, smooth_episode

from pipeline.cleaner import clean_dataset
from pipeline.loader import load_dataset

CHUNK = ("data", "chunk-000")


def test_trim_spans_all_files_of_one_episode(tmp_path):
    root = build_dataset(tmp_path / "ds", {0: smooth_episode(40, seed=0)})
    chunk = root.joinpath(*CHUNK)
    df = pq.read_table(chunk / "file-000.parquet").to_pandas()
    pq.write_table(pa.Table.from_pandas(df.iloc[:20], preserve_index=False), chunk / "file-000.parquet")
    pq.write_table(pa.Table.from_pandas(df.iloc[20:], preserve_index=False), chunk / "file-001.parquet")

    decisions = {
        0: SimpleNamespace(
            disposition="repair", repair="trim", trim={"head_frames": 2, "tail_frames": 3}
        )
    }
    clean_dataset(load_dataset(root), decisions, tmp_path / "out", prefer_cli=False)

    out_root = tmp_path / "out" / "cleaned_dataset"
    out = pd.concat(
        [pq.read_table(p).to_pandas() for p in sorted((out_root / "data").rglob("*.parquet"))],
        ignore_index=True,
    )
    assert len(out) == 35, "跨文件裁帧应保留 40-2-3=35 帧"

    src = np.stack(df["observation.state"].to_numpy())[2:37]
    got = np.stack(out["observation.state"].to_numpy())
    np.testing.assert_allclose(got, src)
