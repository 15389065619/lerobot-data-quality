"""episode_index 为字符串列时，--max_episodes 过滤必须发生在类型归一之后。

字符串列上用整数白名单做 parquet 下推过滤（或 pandas isin）都不会命中，
若先过滤再 to_numeric，`--max_episodes` 会静默丢光所有帧。
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
from conftest import build_dataset, smooth_episode

from pipeline.loader import load_dataset

DATA = ("data", "chunk-000", "file-000.parquet")


def _stringified_dataset(tmp_path, n_eps: int = 4, frames_per_ep: int = 20):
    root = build_dataset(
        tmp_path / "ds",
        {ep: smooth_episode(frames_per_ep, offset=ep * 0.5, seed=ep) for ep in range(n_eps)},
    )
    data_path = root.joinpath(*DATA)
    df = pq.read_table(data_path).to_pandas()
    df["episode_index"] = df["episode_index"].astype(str)
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), data_path)
    return root


def test_string_episode_index_survives_max_episodes(tmp_path):
    root = _stringified_dataset(tmp_path)

    dataset = load_dataset(root, max_episodes=2)

    assert dataset.truncated_to == 2
    assert len(dataset.frames) == 40
    assert sorted(int(x) for x in dataset.frames["episode_index"].unique()) == [0, 1]


def test_string_episode_index_full_load_keeps_all(tmp_path):
    root = _stringified_dataset(tmp_path)

    dataset = load_dataset(root)

    assert dataset.truncated_to is None
    assert len(dataset.frames) == 80
    assert sorted(int(x) for x in dataset.frames["episode_index"].unique()) == [0, 1, 2, 3]
