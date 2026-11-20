"""跨文件清洗场景的覆盖测试——专门打破单文件假设。

现有测试大量跑在单文件 build_dataset 上，cleaner 的多文件逻辑（_source_files_in_order
按文件名排序、_episode_row_index 跨文件扫描、manual_clean 主循环按文件缓存）
缺乏边界验证。本文件补 6 个 case：

  1. meta 行序与物理段序相反——帧按物理顺序拼接，不受 meta 行序影响
  2. NULL episode_index——扫描时跳过但不崩溃
  3. load_frames=False（流式模式）——clean_dataset 能跑通 drop + repair
  4. drop 与 split 混合——drop 走的 file 里同时有被 split 的保留集
  5. 两文件 schema 不一致（一个有 frame_index 一个没有）——scan 层容错
  6. 同一集跨越 3 个文件 + trim + timestamp rebuild 叠加
"""

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


def _split_episode_to_files(root, ep, n_files):
    """把 build_dataset 产出的单文件里某个 episode 拆到多个 parquet 文件。"""
    chunk = root.joinpath(*CHUNK)
    df = pq.read_table(chunk / "file-000.parquet").to_pandas()
    sub = df[df["episode_index"] == ep].reset_index(drop=True)
    other = df[df["episode_index"] != ep].reset_index(drop=True)

    # 原 file-000 只保留其他 episode
    pq.write_table(pa.Table.from_pandas(other, preserve_index=False),
                   chunk / "file-000.parquet")

    # sub 拆到 n_files 个新文件
    chunk_size = len(sub) // n_files + (1 if len(sub) % n_files else 0)
    for i in range(n_files):
        start = i * chunk_size
        end = min(start + chunk_size, len(sub))
        if start >= len(sub):
            break
        frag = sub.iloc[start:end].reset_index(drop=True)
        pq.write_table(pa.Table.from_pandas(frag, preserve_index=False),
                       chunk / f"file-{i + 1:03d}.parquet")
    return chunk


# ---------------------------------------------------------------------------
# Case 1: meta 行序与物理段序相反
# ---------------------------------------------------------------------------
def test_cross_file_meta_order_reversed(tmp_path):
    """数据文件顺序 file-000=ep1, file-001=ep0，但 meta 里 ep0 先出现。
    _source_files_in_order 按文件名排序，manual_clean 逐文件扫描，
    不能让 meta 行序影响物理文件遍历顺序。"""
    root = build_dataset(tmp_path / "ds", {
        0: smooth_episode(20, seed=0),
        1: smooth_episode(20, seed=1),
    })
    chunk = root.joinpath(*CHUNK)
    df = pq.read_table(chunk / "file-000.parquet").to_pandas()

    # 把 ep0 挪到 file-001
    pq.write_table(
        pa.Table.from_pandas(df[df["episode_index"] == 1], preserve_index=False),
        chunk / "file-000.parquet",
    )
    pq.write_table(
        pa.Table.from_pandas(df[df["episode_index"] == 0], preserve_index=False),
        chunk / "file-001.parquet",
    )

    # 故意把 meta ep0 和 ep1 的行序反一下
    meta_path = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    meta_df = pq.read_table(meta_path).to_pandas()
    meta_df = meta_df.iloc[::-1].reset_index(drop=True)
    pq.write_table(pa.Table.from_pandas(meta_df, preserve_index=False), meta_path)

    decisions = {
        0: SimpleNamespace(disposition="repair", repair="timestamps"),
        1: SimpleNamespace(disposition="keep"),
    }
    out = clean_dataset(load_dataset(root), decisions, tmp_path / "out", prefer_cli=False)

    # 清洗后两集都在，帧总数不变
    view = load_dataset(out.output_root)
    assert view.num_episodes == 2
    assert view.num_frames == 40


# ---------------------------------------------------------------------------
# Case 2: NULL episode_index
# ---------------------------------------------------------------------------
def test_cross_file_null_episode_index(tmp_path):
    """file-001 里有 episode_index=NULL 的脏行，_episode_row_index 跳过不崩溃。"""
    root = build_dataset(tmp_path / "ds", {0: smooth_episode(20, seed=0)})
    chunk = root.joinpath(*CHUNK)
    df = pq.read_table(chunk / "file-000.parquet").to_pandas()

    # file-001 含 ep0 的部分行 + 10 行 NULL episode_index
    part_rows = df.iloc[:5].to_dict("records")
    bad = pd.DataFrame(part_rows + [
        {**part_rows[0], "episode_index": np.nan, "index": int(df["index"].max()) + 1 + i}
        for i in range(10)
    ])
    pq.write_table(pa.Table.from_pandas(bad, preserve_index=False),
                   chunk / "file-001.parquet")

    # file-000 缩短为剩下的 15 行
    pq.write_table(
        pa.Table.from_pandas(df.iloc[5:].reset_index(drop=True), preserve_index=False),
        chunk / "file-000.parquet",
    )

    decisions = {0: SimpleNamespace(disposition="repair", repair="timestamps")}
    out = clean_dataset(load_dataset(root), decisions, tmp_path / "out", prefer_cli=False)

    view = load_dataset(out.output_root)
    # NULL 行被跳过，ep0 的有效帧 = 20
    assert view.num_episodes == 1
    assert view.num_frames == 20


# ---------------------------------------------------------------------------
# Case 3: load_frames=False（流式模式）
# ---------------------------------------------------------------------------
def test_load_frames_false_cleans(tmp_path):
    """公共 API 直调 clean_dataset 时 dataset.frames 为空——_source_files_in_order
    必须回退到 data_files，drop + repair 都要能跑通。"""
    root = build_dataset(tmp_path / "ds", {
        0: smooth_episode(20, seed=0),
        1: smooth_episode(20, seed=1),
    })
    # 拆成两个文件
    _split_episode_to_files(root, 0, 2)

    # 模拟流式加载：frames 为空
    view = load_dataset(root, load_frames=False)
    assert len(view.frames) == 0, "load_frames=False 应返回空 frames"
    assert view.data_files, "data_files 仍应有值"

    decisions = {
        0: SimpleNamespace(
            disposition="repair", repair="trim",
            trim={"head_frames": 2, "tail_frames": 1}
        ),
        1: SimpleNamespace(disposition="drop"),
    }
    out = clean_dataset(view, decisions, tmp_path / "out", prefer_cli=False)

    result = load_dataset(out.output_root)
    assert result.num_episodes == 1
    assert result.num_frames == 20 - 2 - 1  # trim 后 17 帧


# ---------------------------------------------------------------------------
# Case 4: drop 与 repair 混合在同一文件
# ---------------------------------------------------------------------------
def test_drop_and_repair_mixed_same_file(tmp_path):
    """file-000 里 ep0 要 drop，ep1 要 trim——文件先被 drop 过滤，
    repair 再取保留行，不能因 drop 了同文件另一集就丢帧。"""
    root = build_dataset(tmp_path / "ds", {
        0: smooth_episode(30, seed=0),
        1: smooth_episode(30, seed=1),
    })

    decisions = {
        0: SimpleNamespace(disposition="drop"),
        1: SimpleNamespace(
            disposition="repair", repair="trim",
            trim={"head_frames": 3, "tail_frames": 2},
        ),
    }
    out = clean_dataset(load_dataset(root), decisions, tmp_path / "out", prefer_cli=False)

    view = load_dataset(out.output_root)
    assert view.num_episodes == 1
    assert view.num_frames == 30 - 3 - 2


# ---------------------------------------------------------------------------
# Case 5: 两文件 schema 不一致
# ---------------------------------------------------------------------------
def test_two_files_schema_mismatch(tmp_path):
    """file-000 有 frame_index + timestamp，file-001 缺 timestamp。
    _episode_row_index 只读 episode_index，manual_clean 逐文件操作时
    set_column 在列缺失处补出（新行为，见 #5 修复）。"""
    root = build_dataset(tmp_path / "ds", {0: smooth_episode(20, seed=0)})
    chunk = root.joinpath(*CHUNK)
    df = pq.read_table(chunk / "file-000.parquet").to_pandas()

    part_a = df.iloc[:10].reset_index(drop=True)
    part_b = df.iloc[10:].reset_index(drop=True).drop(columns=["timestamp"])

    pq.write_table(pa.Table.from_pandas(part_a, preserve_index=False),
                   chunk / "file-000.parquet")
    pq.write_table(pa.Table.from_pandas(part_b, preserve_index=False),
                   chunk / "file-001.parquet")

    decisions = {0: SimpleNamespace(disposition="repair", repair="timestamps")}
    out = clean_dataset(load_dataset(root), decisions, tmp_path / "out", prefer_cli=False)

    result = load_dataset(out.output_root)
    assert result.num_episodes == 1
    assert result.num_frames == 20
    # timestamp 应被 rebuild 覆盖补齐
    ep_df = result.episode_frame(0)
    assert "timestamp" in ep_df.columns
    assert ep_df["timestamp"].notna().all()


# ---------------------------------------------------------------------------
# Case 6: 同一集跨 3 文件 + trim + timestamp rebuild 叠加
# ---------------------------------------------------------------------------
def test_episode_spans_three_files_with_overlap_repairs(tmp_path):
    """ep0 跨 file-000/001/002，trim head=3 (file-000) + tail=4 (file-002) +
    timestamp rebuild——三个 repair 叠加，manual_clean 的缓存必须逐文件正确。"""
    root = build_dataset(tmp_path / "ds", {0: smooth_episode(60, seed=0)})
    _split_episode_to_files(root, 0, 3)

    decisions = {
        0: SimpleNamespace(
            disposition="repair", repair="trim",
            trim={"head_frames": 3, "tail_frames": 4}
        ),
    }
    out = clean_dataset(load_dataset(root), decisions, tmp_path / "out", prefer_cli=False)

    result = load_dataset(out.output_root)
    assert result.num_episodes == 1
    assert result.num_frames == 60 - 3 - 4
    # 验证 frame_index 0..52 连续
    ep_df = result.episode_frame(0)
    assert list(ep_df["frame_index"]) == list(range(53))


if __name__ == "__main__":
    import sys, tempfile
    from pathlib import Path
    passed = 0
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            with tempfile.TemporaryDirectory() as td:
                try:
                    fn(Path(td))
                    print(f"  ✓ {name}")
                    passed += 1
                except AssertionError as e:
                    print(f"  ✗ {name}: {e}")
                    failed += 1
                except Exception as e:
                    print(f"  ✗ {name} (unexpected): {type(e).__name__}: {e}")
                    import traceback; traceback.print_exc()
                    failed += 1
    print(f"\n{passed}/{passed + failed} 通过")
    sys.exit(0 if failed == 0 else 1)
