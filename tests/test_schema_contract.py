"""Schema 契约审计的单元测试。

对应 behavior1k_task0042 的真实缺陷：`index` 退化为 episode 内帧号、
`frame_index` 缺失、`splits` 与 `total_episodes` 不符。
"""
from __future__ import annotations

import pandas as pd

from pipeline.schema import REQUIRED_FRAME_COLUMNS, audit_schema


def _summary(starts: list[int], rows: list[int]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "episode_index": list(range(len(rows))),
            "rows": rows,
            "idx_min": starts,
            "idx_max": [s + r - 1 for s, r in zip(starts, rows)],
        }
    )


def _codes(items: list[dict], severity: str | None = None) -> set[str]:
    return {i["column"] for i in items if severity is None or i["severity"] == severity}


def _compliant_features() -> dict:
    return {c: {"dtype": "int64", "shape": (1,)} for c in REQUIRED_FRAME_COLUMNS}


def test_compliant_dataset_has_no_violation():
    rows = [10, 20, 30]
    starts = [0, 10, 30]  # 每集起点 = 前序累计行数
    items = audit_schema(
        features=_compliant_features(),
        splits={"train": "0:3"},
        total_episodes=3,
        data_columns=set(REQUIRED_FRAME_COLUMNS),
        data_summary=_summary(starts, rows),
    )
    assert items == [], items


def test_index_restart_per_episode_is_error():
    """index 每集从 0 起 → 全局唯一性丢失，必须报 error。"""
    items = audit_schema(
        features=_compliant_features(),
        splits={"train": "0:3"},
        total_episodes=3,
        data_columns=set(REQUIRED_FRAME_COLUMNS),
        data_summary=_summary([0, 0, 0], [10, 20, 30]),
    )
    index_items = [i for i in items if i["column"] == "index"]
    assert len(index_items) == 1
    assert index_items[0]["severity"] == "error"
    assert "60" in index_items[0]["actual"]  # 总行数被写进证据
    assert "29" in index_items[0]["actual"]  # 实际取值范围上限


def test_index_start_not_matching_cumsum_is_error():
    items = audit_schema(
        features=_compliant_features(),
        splits={"train": "0:3"},
        total_episodes=3,
        data_columns=set(REQUIRED_FRAME_COLUMNS),
        data_summary=_summary([0, 100, 200], [10, 20, 30]),  # 起点对不上累计
    )
    assert "index" in _codes(items, "error")


def test_missing_frame_index_is_warning():
    feats = _compliant_features()
    feats.pop("frame_index")
    cols = set(REQUIRED_FRAME_COLUMNS) - {"frame_index"}
    items = audit_schema(
        features=feats,
        splits={"train": "0:2"},
        total_episodes=2,
        data_columns=cols,
        data_summary=_summary([0, 0], [10, 20]),  # index 变成 ep 内帧号
    )
    frame_items = [i for i in items if i["column"] == "frame_index"]
    assert len(frame_items) == 1
    assert frame_items[0]["severity"] == "warning"
    assert "承担了 frame_index" in frame_items[0]["actual"]


def test_splits_length_mismatch_is_warning():
    items = audit_schema(
        features=_compliant_features(),
        splits={"train": "0:10000"},
        total_episodes=200,
        data_columns=set(REQUIRED_FRAME_COLUMNS),
        data_summary=_summary([0, 10], [10, 20]),
    )
    split_items = [i for i in items if i["column"] == "meta/info.json:splits"]
    assert len(split_items) == 1
    assert split_items[0]["severity"] == "warning"
    assert "0:200" in split_items[0]["expected"]


def test_missing_index_column_skips_uniqueness_check():
    """没有 index 列时不能靠极值推断，只报 frame_index/缺失，不误报唯一性。"""
    items = audit_schema(
        features=_compliant_features(),
        splits={},
        total_episodes=2,
        data_columns={"episode_index", "task_index", "timestamp"},
        data_summary=pd.DataFrame({"episode_index": [0, 1], "rows": [10, 20]}),
    )
    assert "index" not in _codes(items)
    assert _codes(items) <= {"frame_index", "index"}
