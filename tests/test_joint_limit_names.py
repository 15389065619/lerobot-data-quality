"""joint_limits 配置维度大于 info 声明的关节名数量时，报关节名不能越界。"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from pipeline.validator.rules import check_joint_limits
from pipeline.validator.types import EpisodeResult


def _config(n: int) -> dict:
    return {
        "joint_limits": {
            "enabled": True,
            "lower": [-1.0] * n,
            "upper": [1.0] * n,
            "tolerance": 1e-6,
        }
    }


def _dataset(names: list[str]) -> SimpleNamespace:
    return SimpleNamespace(joint_names=names)


def test_joint_names_beyond_declared_dim_use_fallback():
    """info 只声明 2 个关节名，数据与配置是 5 维：越界维度用 dim_N 兜底。"""
    arr = np.zeros((10, 5))
    arr[:, 2] = 5.0
    arr[:, 4] = -5.0
    res = EpisodeResult(episode_index=0, frame_count=10)

    check_joint_limits(arr, res, _config(5), _dataset(["j0", "j1"]))

    detail = res.details["joint_limit_violation"]
    assert detail["dims"] == [2, 4]
    assert detail["joint_names"] == ["dim_2", "dim_4"]


def test_joint_names_within_declared_dim_keep_original():
    arr = np.zeros((10, 3))
    arr[:, 1] = 5.0
    res = EpisodeResult(episode_index=0, frame_count=10)

    check_joint_limits(arr, res, _config(3), _dataset(["j0", "j1", "j2"]))

    detail = res.details["joint_limit_violation"]
    assert detail["joint_names"] == ["j1"]
