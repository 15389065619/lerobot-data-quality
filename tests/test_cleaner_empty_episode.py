"""判定保留但一行都写不出的 episode 不得让清洗崩溃。"""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pyarrow.parquet as pq
from conftest import build_dataset, smooth_episode

from pipeline.cleaner import clean_dataset
from pipeline.loader import load_dataset


def test_kept_episode_without_rows_is_dropped_with_warning(tmp_path):
    root = build_dataset(
        tmp_path / "ds",
        {ep: smooth_episode(10, offset=ep * 0.3, seed=ep) for ep in range(2)},
    )

    # ep0 判定保留，但裁帧把 10 帧全部裁掉，产物里一行都不会剩下
    decisions = {
        0: SimpleNamespace(
            disposition="repair", repair="trim", trim={"head_frames": 6, "tail_frames": 6}
        )
    }
    result = clean_dataset(load_dataset(root), decisions, tmp_path / "out", prefer_cli=False)

    assert any("没有任何有效行" in w for w in result.warnings)
    out = pd.concat(
        [
            pq.read_table(p).to_pandas()
            for p in sorted((tmp_path / "out" / "cleaned_dataset" / "data").rglob("*.parquet"))
        ],
        ignore_index=True,
    )
    assert out["episode_index"].nunique() == 1
    assert len(out) == 10
