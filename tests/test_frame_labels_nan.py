"""NaN frame_index 不得让整条 episode 被判为读取失败。"""

from __future__ import annotations

import numpy as np

from pipeline.validator.rules import _frame_labels


def test_frame_labels_skips_non_finite_values():
    frame_col = np.array([0.0, np.nan, 2.0, np.inf, 4.0])
    positions = np.arange(len(frame_col))

    labels = _frame_labels(frame_col, positions, 10)

    assert labels == [0, 2, 4]
