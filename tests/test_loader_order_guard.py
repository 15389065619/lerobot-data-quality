"""data 中 episode 编号非升序时，顺序配对不可信，必须判 ambiguous。"""

from __future__ import annotations

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from conftest import build_dataset, smooth_episode

from pipeline.loader import load_dataset

DATA = ("data", "chunk-000", "file-000.parquet")
META = ("meta", "episodes", "chunk-000", "file-000.parquet")


def test_unsorted_data_order_is_ambiguous(tmp_path):
    root = build_dataset(
        tmp_path / "ds",
        {ep: smooth_episode(30, offset=ep * 0.5, seed=ep) for ep in range(3)},
    )

    data_path = root.joinpath(*DATA)
    df = pq.read_table(data_path).to_pandas()
    # 物理重排成 0,2,1；三个 episode 等长，行数证据不足以发现错位，只有顺序假设能兜住
    df = pd.concat([df[df.episode_index == ep] for ep in (0, 2, 1)], ignore_index=True)
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), data_path)

    meta_path = root.joinpath(*META)
    meta = pq.read_table(meta_path).to_pandas().sort_values("episode_index").reset_index(drop=True)
    meta["episode_index"] = [420010 + i * 10 for i in range(len(meta))]
    pq.write_table(pa.Table.from_pandas(meta, preserve_index=False), meta_path)

    dataset = load_dataset(root)

    assert dataset.id_reconciliation["mode"] == "ambiguous"
    assert dataset.locator_available is False
