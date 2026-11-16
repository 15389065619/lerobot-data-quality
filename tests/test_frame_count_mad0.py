"""MAD=0 且存在离散时帧数区间的最小宽度修复（G4）。

背景：mad==0 且 ptp>0 时区间塌缩成单点（旧实现刻意塌缩以抓「唯一离群集」），
但录制工艺稳定的流水线数据帧数天然有 ±1~2 帧抖动（tokyo 实测 237/238/239
三种长度），塌缩区间把 24/50 集全体判越界送复核。修复：该分支改用最小宽度
`median ± max(1, ceil(ratio×median))`（ratio 可配，默认 1%）。

护栏（评审反例）：49×239 + 1×100（上传截断）时区间不得用观测 [min,max]——
那等于用被检样本定义边界，截断集从此永久合规；最小宽度下 100 仍在界外。
"""

from __future__ import annotations

import numpy as np
import pytest

from pipeline.config import load_config
from pipeline.loader import load_dataset
from pipeline.validator import validate_dataset
from pipeline.validator.context import _resolve_frame_bounds

from conftest import build_dataset


def _validate(path, **overrides):
    dataset = load_dataset(path)
    return dataset, validate_dataset(
        dataset, load_config(robot_type="test_robot", overrides=overrides or None)
    )


def _noise_eps(n: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    vals = rng.normal(0.0, 0.003, size=(n, 3))
    return {"state": vals.copy(), "action": vals.copy()}


class TestMad0MinWidth:
    def test_unit_bounds(self):
        """直接验证区间推导：median=239, ratio=1% → half=3 → [236,242]。"""
        counts = np.array([239.0] * 49 + [100.0])
        bounds = _resolve_frame_bounds(counts, load_config(robot_type="test_robot"), fps=5.0)
        assert bounds["min"] == 236 and bounds["max"] == 242
        assert bounds["source"]["min"] == "auto:mad0-min-width"
        # 护栏：绝不允许退化成观测极值 [100, 239]
        assert bounds["min"] != 100

    def test_t2a_jitter_lengths_not_flagged(self, tmp_path):
        """T2a：26×239 / 23×238 / 1×237 全部落在最小宽度区间内，0 条告警。"""
        lengths = [239] * 26 + [238] * 23 + [237]
        episodes = {i: _noise_eps(n, seed=400 + i) for i, n in enumerate(lengths)}
        path = build_dataset(tmp_path / "ds", episodes, fps=5)
        _, result = _validate(path)

        bounds = result.resolved_thresholds["frame_count"]
        assert (bounds["min"], bounds["max"]) == (236, 242)
        assert bounds["source"]["min"] == "auto:mad0-min-width"
        flagged = [r.episode_index for r in result.results.values()
                   if "frame_count_abnormal" in r.reasons]
        assert flagged == [], f"帧数抖动被批量送复核：{flagged}"

    def test_t2b_truncated_still_flagged(self, tmp_path):
        """T2b：49×239 + 1×100，截断集必须命中，其余保留（防 [min,max] 回退）。"""
        lengths = [239] * 49 + [100]
        episodes = {i: _noise_eps(n, seed=500 + i) for i, n in enumerate(lengths)}
        path = build_dataset(tmp_path / "ds", episodes, fps=5)
        _, result = _validate(path)

        bounds = result.resolved_thresholds["frame_count"]
        assert bounds["min"] == 236 and bounds["max"] == 242  # 不是 [100,239]
        flagged = {r.episode_index: r for r in result.results.values()
                   if "frame_count_abnormal" in r.reasons}
        assert set(flagged) == {49}, f"应只有截断集（ep49）命中：{set(flagged)}"
        ev = flagged[49].details["frame_count_abnormal"]
        assert ev["direction"] == "too_few"
        assert ev["frame_count"] == 100
