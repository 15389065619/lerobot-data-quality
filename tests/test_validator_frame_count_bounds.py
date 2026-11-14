"""等长数据集的 MAD=0 不得让帧数允许区间塌成单点。"""

from __future__ import annotations

import numpy as np

from pipeline.config import load_config
from pipeline.validator.context import _resolve_frame_bounds


def test_equal_length_episodes_keep_non_zero_bounds():
    counts = np.array([120] * 8, dtype=np.int64)

    bounds = _resolve_frame_bounds(counts, load_config(robot_type="test_robot"), fps=30.0)

    assert bounds["mode"] == "auto"
    assert bounds["min"] < 120 < bounds["max"]
    assert bounds["max"] - bounds["min"] >= 10
