"""数据集指纹：覆盖全部 data 分片、感知 mtime、stat 失败不崩溃。"""

from __future__ import annotations

import os
from pathlib import Path

import pyarrow.parquet as pq
from conftest import build_dataset, smooth_episode

from pipeline.loader import load_dataset
from pipeline.reporter import dataset_fingerprint


def _build(tmp_path: Path) -> tuple[Path, object]:
    root = build_dataset(tmp_path / "ds", {0: smooth_episode(40, seed=0)})
    return root, load_dataset(root)


def test_fingerprint_reacts_to_mtime_change(tmp_path: Path) -> None:
    root, dataset = _build(tmp_path)
    before = dataset_fingerprint(dataset)

    data_file = next((root / "data").rglob("*.parquet"))
    st = data_file.stat()
    os.utime(data_file, ns=(st.st_atime_ns, st.st_mtime_ns - 1_000_000))

    assert dataset_fingerprint(dataset) != before


def test_fingerprint_covers_more_than_twenty_files(tmp_path: Path) -> None:
    root, _ = _build(tmp_path)
    src = root / "data" / "chunk-000" / "file-000.parquet"
    table = pq.read_table(src)
    total = table.num_rows
    count = 21
    for i in range(count):
        start = round(i * total / count)
        stop = round((i + 1) * total / count)
        pq.write_table(table.slice(start, stop - start), src.with_name(f"file-{i:03d}.parquet"))

    dataset = load_dataset(root)
    before = dataset_fingerprint(dataset)

    last = sorted((root / "data").rglob("*.parquet"))[-1]
    assert last.name == f"file-{count - 1:03d}.parquet"
    st = last.stat()
    os.utime(last, ns=(st.st_atime_ns, st.st_mtime_ns - 1_000_000))

    assert dataset_fingerprint(dataset) != before


def test_fingerprint_survives_stat_failure(tmp_path: Path, monkeypatch) -> None:
    _, dataset = _build(tmp_path)
    real_stat = Path.stat

    def flaky(self: Path, *args, **kwargs):
        if self.suffix == ".parquet" and "data" in self.parts:
            raise OSError("permission denied")
        return real_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", flaky)
    assert len(dataset_fingerprint(dataset)) == 16
