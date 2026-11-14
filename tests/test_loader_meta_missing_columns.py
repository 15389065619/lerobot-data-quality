"""meta/episodes 缺列时不得崩溃（libero 等野外数据集真实缺 length 列）。"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
from conftest import build_dataset, smooth_episode

from pipeline.loader import load_dataset


def _episodes(n: int = 3):
    return {ep: smooth_episode(30 + ep * 5, offset=ep * 0.7, seed=ep) for ep in range(n)}


def test_missing_length_column_does_not_crash(tmp_path):
    root = build_dataset(tmp_path / "ds", _episodes())
    path = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    df = pq.read_table(path).to_pandas().sort_values("episode_index").reset_index(drop=True)
    df = df.drop(columns=["length"])
    df["episode_index"] = [420010 + i * 10 for i in range(len(df))]
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path)

    dataset = load_dataset(root)  # 不应抛 KeyError

    # 缺行数证据 → 只能判 ambiguous，不能猜映射
    assert dataset.id_reconciliation["mode"] == "ambiguous"
    assert dataset.locator_available is False
