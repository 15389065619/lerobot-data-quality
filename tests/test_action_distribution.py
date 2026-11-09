"""动作分布对比工具：统计量正确性、跨分片跳变、映射缺失时拒绝归因。

这些用例的价值不在「函数能跑」，而在三条容易悄悄错掉的判断：

1. 方差/跳变是否与按帧直接算的结果一致（浮点累加 + 分块读取很容易差一点）；
2. 跨 parquet 分片的那一次 delta 有没有被算进去（少算就会漏掉最脏的边界）；
3. 拿不到清洗前后编号映射时，是否**拒绝**给归因，而不是默认同名硬算。
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from conftest import build_dataset, smooth_episode
from tools.analyze_action_distribution import (
    DEFAULT_SECTION_ID,
    analyze_dataset,
    compare,
    inject_into_eval,
    render_section,
)


def _actions(episodes: dict[int, dict]) -> np.ndarray:
    return np.vstack([np.asarray(spec["action"], dtype=np.float64) for _, spec in sorted(episodes.items())])


# ------------------------------------------------------------------ 统计量正确性
def test_accumulator_matches_frame_level_numpy(tmp_path):
    episodes = {
        0: smooth_episode(30, seed=1),
        1: smooth_episode(40, offset=2.0, seed=2),
        2: smooth_episode(20, offset=-1.0, seed=3),
    }
    root = build_dataset(tmp_path / "ds", episodes)
    summary = analyze_dataset(root, threshold=0.5)

    matrix = _actions(episodes)
    assert summary["frames"] == len(matrix)
    assert summary["episodes"] == 3
    assert summary["var"] == pytest.approx(matrix.var(axis=0).mean(), rel=1e-4)
    assert summary["mean"] == pytest.approx(matrix.mean(), rel=1e-4)

    per_ep = max(
        float(np.abs(np.diff(np.asarray(spec["action"], dtype=np.float64), axis=0)).max())
        for spec in episodes.values()
    )
    assert summary["max_delta"] == pytest.approx(per_ep, rel=1e-4)


def test_episode_contributions_sum_to_total_sse(tmp_path):
    """单集对 SSE 的贡献必须可加，否则 §9.3 的「占比」和总量对不上。"""
    episodes = {ep: smooth_episode(25, offset=ep, seed=ep) for ep in range(4)}
    episodes[2]["action"] = episodes[2]["action"] * 4.0  # 造一个高方差 episode
    root = build_dataset(tmp_path / "ds", episodes)
    summary = analyze_dataset(root, threshold=0.5)

    shares = sum(row["sse"] for row in summary["per_episode"])
    assert shares == pytest.approx(summary["sse"], rel=1e-6)


# ------------------------------------------------------------------ 跨分片跳变
def _split_at(path: Path, row: int, first: Path, second: Path) -> None:
    table = pq.read_table(path)
    pq.write_table(table.slice(0, row), first)
    pq.write_table(table.slice(row, table.num_rows - row), second)


def test_delta_across_parquet_files_is_counted(tmp_path):
    """跳变正好落在文件边界上：少算一次 delta 就会把 5.0 的跳变当成 0。"""
    n = 20
    action = np.zeros((n, 3), dtype=np.float32)
    action[10:] = 5.0  # 第 9→10 帧之间是唯一一次真实跳变
    state = action.copy()
    root = build_dataset(tmp_path / "ds", {0: {"state": state, "action": action}})

    data_dir = root / "data" / "chunk-000"
    original = data_dir / "file-000.parquet"
    _split_at(original, 10, data_dir / "file-000.parquet", data_dir / "file-001.parquet")

    summary = analyze_dataset(root, threshold=1.0)
    assert summary["frames"] == n
    assert summary["max_delta"] == pytest.approx(5.0, rel=1e-6)
    assert summary["over_threshold"] == 1

    # 拆在 episode 内部：episode 0 的两段必须归到同一集，而不是被当成两集
    assert summary["episodes"] == 1


# ------------------------------------------------------------------ 归因前提
def _prepare_identity_pair(tmp_path, scale: float = 1.0) -> tuple[Path, Path]:
    episodes = {ep: smooth_episode(25, offset=ep * 0.3, seed=ep) for ep in range(3)}
    raw = build_dataset(tmp_path / "raw", episodes)
    cleaned = tmp_path / "cleaned"
    shutil.copytree(raw, cleaned)
    (cleaned / "meta" / "quality_cleaning_notes.json").write_text(
        json.dumps({"episode_index_map": {str(ep): ep for ep in episodes}}), encoding="utf-8"
    )
    if scale != 1.0:
        path = cleaned / "data" / "chunk-000" / "file-000.parquet"
        table = pq.read_table(path)
        column = table.column("action").combine_chunks()
        values = np.asarray(column.flatten(), dtype=np.float32) * scale
        mat = values.reshape(len(column), column.type.list_size)
        rebuilt = pa.FixedSizeListArray.from_arrays(pa.array(mat.reshape(-1)), column.type.list_size)
        pq.write_table(table.set_column(table.column_names.index("action"), "action", rebuilt), path)
    return raw, cleaned


def test_identity_pair_has_zero_change(tmp_path):
    raw, cleaned = _prepare_identity_pair(tmp_path)
    cmp = compare(raw, cleaned)

    assert cmp["metrics"]["var_change_pct"] == pytest.approx(0.0, abs=1e-6)
    assert cmp["metrics"]["frames_dropped"] == 0
    assert cmp["dropped"] == []
    assert cmp["map_source"] == "quality_cleaning_notes.json"


def test_missing_mapping_refuses_attribution(tmp_path):
    """没有映射就不知道「保留集」是哪些集，此时给归因等于编造。"""
    episodes = {0: smooth_episode(20, seed=1)}
    raw = build_dataset(tmp_path / "raw", episodes)
    cleaned = tmp_path / "cleaned"
    shutil.copytree(raw, cleaned)
    (cleaned / "meta" / "quality_cleaning_notes.json").unlink(missing_ok=True)

    with pytest.raises(SystemExit) as exc:
        compare(raw, cleaned)
    assert "无法确定" in str(exc.value)


# ------------------------------------------------------------------ 渲染口径
def test_render_states_increase_instead_of_dropping(tmp_path):
    """方差变大时必须照实写「上升」，不能套「方差降低 X%」的模板。"""
    raw, cleaned = _prepare_identity_pair(tmp_path, scale=2.0)
    section = render_section(compare(raw, cleaned))

    assert "方差为 **+" in section
    assert "降至" not in section
    assert "未收窄" in section or "变为" in section


def test_inject_replaces_existing_block(tmp_path):
    eval_path = tmp_path / "EVAL.md"
    eval_path.write_text("# 实测记录\n\n旧内容\n", encoding="utf-8")
    raw, cleaned = _prepare_identity_pair(tmp_path)
    section = render_section(compare(raw, cleaned))

    inject_into_eval(eval_path, section, DEFAULT_SECTION_ID)
    inject_into_eval(eval_path, section, DEFAULT_SECTION_ID)

    text = eval_path.read_text(encoding="utf-8")
    assert text.count(f"<!-- {DEFAULT_SECTION_ID}:BEGIN -->") == 1
    assert text.count(f"<!-- {DEFAULT_SECTION_ID}:END -->") == 1
    assert "旧内容" in text  # 手写部分不能被覆盖
