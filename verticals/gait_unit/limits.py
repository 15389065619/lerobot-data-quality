# Copyright (c) 2026 wu
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""G1 关节限位表加载器（Tier 2 关节级 QC 的数据基础）。

限位表 = config/g1_29dof_limits.json（派生资产，入 git），由
data/external/g1_description/g1_29dof.urdf（Unitree 官方公开 URDF，
本体不入 git）解析生成。派生脚本见 2026-11-20 会话记录；表内
_provenance 记录来源与日期。

通道映射契约：MotionDecode 通道名 = "dof_" + URDF 关节名 + "(rad)"。
限位单位 = rad（URDF 原生）。
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]   # verticals/gait_unit/limits.py → 仓库根
DEFAULT_LIMITS_PATH = REPO / "config" / "g1_29dof_limits.json"


def load_g1_limits(path: str | Path = DEFAULT_LIMITS_PATH) -> dict[str, tuple[float, float]]:
    """加载 G1 限位表 → {MotionDecode 通道名: (lower, upper)}。

    只保留能映射到 MotionDecode 通道（dof_ 前缀 + (rad) 后缀）的关节；
    表内其余字段（provenance 等）跳过并核对关节覆盖数。

    Raises:
        FileNotFoundError / ValueError：文件缺失、schema 不对、关节覆盖数
        不足 29（MotionDecode 的 29 dof 必须全覆盖，缺一个 = 限位表与
        导出版本不匹配，宁可拒绝也不静默降级）。
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"限位表不存在: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    joints = raw.get("joints")
    if not isinstance(joints, dict) or not joints:
        raise ValueError(f"{path.name} 缺少 joints 字段或为空")

    limits: dict[str, tuple[float, float]] = {}
    for name, v in joints.items():
        if not isinstance(v, dict) or "lower" not in v or "upper" not in v:
            continue
        channel = f"dof_{name}(rad)"
        lo, hi = float(v["lower"]), float(v["upper"])
        if lo > hi:
            raise ValueError(f"{path.name} 关节 {name} 限位倒挂: [{lo}, {hi}]")
        limits[channel] = (lo, hi)

    if len(limits) < 29:
        raise ValueError(
            f"{path.name} 只覆盖 {len(limits)}/29 个 dof 通道——限位表与导出版本"
            "不匹配（G1 有 29dof 与其他变体），拒绝静默降级")
    return limits
