"""meta/episodes 同一 episode 出现多行且内容冲突的回归测试。

真实场景：`so101_bomb_pick_place_v1` 多次导出叠加，ep 81–102 在 meta 里各出现
两行，``length`` 与视频窗口完全不同；数据侧还把两次录制拼进同一集。
旧行为是 ``drop_duplicates`` 静默保留首行，冲突被误归因为 22 条 frame_mismatch，
报告里仍写「两侧编号一致」。
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
from conftest import build_dataset, smooth_episode

from pipeline.config import load_config
from pipeline.loader import load_dataset
from pipeline.validator import validate_dataset


def _episodes(n: int = 4) -> dict[int, dict]:
    return {ep: smooth_episode(40 + ep * 5, offset=ep * 0.7, seed=ep) for ep in range(n)}


def _append_meta_rows(root, episodes: list[int], *, length_delta: int) -> None:
    """模拟「同 episode 再次导出」：追加一个 meta 分片，行内容与首行冲突。"""
    path = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    df = pq.read_table(path).to_pandas()
    extra = df[df["episode_index"].isin(episodes)].copy()
    extra["length"] = extra["length"] + length_delta
    pq.write_table(
        pa.Table.from_pandas(extra, preserve_index=False),
        root / "meta" / "episodes" / "chunk-000" / "file-001.parquet",
    )


def test_conflicting_duplicate_meta_rows_are_reported(tmp_path):
    root = build_dataset(tmp_path / "ds", _episodes())
    _append_meta_rows(root, [1, 2], length_delta=7)

    dataset = load_dataset(root)
    # 仍然按去重后的集数工作（保持既有行为）
    assert dataset.num_episodes == 4
    conflicts = dataset.meta_duplicate_conflicts
    assert conflicts["episodes"] == [1, 2]
    assert conflicts["rows"] == 6
    assert conflicts["unique_episodes"] == 4

    result = validate_dataset(dataset, load_config(robot_type="test_robot"))
    issue = next(i for i in result.dataset_issues if i.code == "meta_episode_duplicate")
    assert issue.detail["episodes"] == [1, 2]
    assert issue.detail["duplicate_rows"] == 2


def test_identical_duplicate_meta_rows_stay_silent(tmp_path):
    root = build_dataset(tmp_path / "ds", _episodes())
    _append_meta_rows(root, [1], length_delta=0)

    dataset = load_dataset(root)
    assert dataset.meta_duplicate_conflicts == {}

    result = validate_dataset(dataset, load_config(robot_type="test_robot"))
    assert "meta_episode_duplicate" not in {i.code for i in result.dataset_issues}
