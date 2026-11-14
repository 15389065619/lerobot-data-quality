"""卡片必须保留 skipped_checks 的跳过原因，不能只显示规则名。"""

from __future__ import annotations

from pipeline.card import build_card


def test_card_keeps_skip_reason():
    report = {
        "dataset": {},
        "summary": {},
        "skipped_checks": {"frame_mismatch": "数据集中没有视频特征"},
    }

    card = build_card(report)

    assert "frame_mismatch" in card
    assert "数据集中没有视频特征" in card
