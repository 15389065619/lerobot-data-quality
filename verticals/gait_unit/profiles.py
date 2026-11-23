"""按动作类别的规则参数与误报基线（数据驱动，LAFAN1 77 段校准）。

两层设计（源自 LAFAN1 校准的教训，2026-11-20）：
  1. **k 保持查全优先**：单全局 k 无法同时服务 walk（本底 21/千帧×关节）和
     ground（49）——把 k 提到"本底 ≤1/千帧"会让 walk 需 k=23、fight 需 k=123，
     中等幅度的真实缺陷（稳健 Z≈6）全部致盲。因此默认 k=6（查全优先），
     基准已验证：k=6 + 基线排除机制下 8 段真实数据 precision=1.0。
  2. **校准表用于分级排序**：CALIBRATED_BASELINES 记录每个类别在 k=6 下的
     本底报警率——复核队列按"超出该类别本底的显著程度"排优先级，
     而不是把本底误当缺陷。

类别解析：从数据路径关键词匹配（样本目录自带类别名）；未知回落 default。
高动态特殊档（MotionDecode 的 High_Dynamic 路径）：k=10 + 跳过 root_pos_*
（校准：jump 样本 root_pos_z 在 k=6 独占 1055/4740 帧，被动作本身支配）。
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CALIBRATION_PATH = REPO / "reports" / "gait" / "lafan1_calibration.json"

__all__ = ["BUILT_IN_PROFILES", "resolve_profile", "profile_params",
           "calibrated_baselines"]

BUILT_IN_PROFILES: dict[str, dict] = {
    "default": {
        "mad_jump_k": 6.0,
        "mad_jump_skip_channel_patterns": [],
    },
    "high_dynamic": {
        # MotionDecode High_Dynamic 路径：跳过被动作支配的 root_pos 通道
        "mad_jump_k": 10.0,
        "mad_jump_skip_channel_patterns": ["root_pos_*"],
    },
}

# 类别关键词 → 档位（首个命中生效；顺序即优先级）
_PROFILE_KEYWORDS = (
    ("High_Dynamic", "high_dynamic"),
    ("walk", "walk"), ("Walk", "walk"),
    ("run", "run"), ("Run", "run"),
    ("sprint", "sprint"), ("Sprint", "sprint"),
    ("dance", "dance"), ("Dance", "dance"),
    ("jumps", "jumps"), ("Jumps", "jumps"),
    ("ground", "ground"),
    ("fight", "fight"),
    ("push", "push"),
    ("fall", "fallAndGetUp"),
    ("obstacles", "obstacles"),
    ("aiming", "aiming"),
)

_CALIBRATED_BASELINES: dict[str, float] | None = None


def calibrated_baselines() -> dict[str, float]:
    """LAFAN1 校准的各类别本底报警率（k=6，每千帧×关节被标帧数）。

    懒加载 reports/gait/lafan1_calibration.json；文件缺失时返回空 dict
    （此时 profile_params 不带 baseline 字段，报告少一列上下文而已）。
    """
    global _CALIBRATED_BASELINES
    if _CALIBRATED_BASELINES is None:
        if CALIBRATION_PATH.is_file():
            raw = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
            cats = raw.get("categories", {})
            _CALIBRATED_BASELINES = {
                c: float(per_k.get("6.0", 0.0)) for c, per_k in cats.items()
            }
        else:
            _CALIBRATED_BASELINES = {}
    return _CALIBRATED_BASELINES


def resolve_profile(path: str | Path) -> str:
    """从数据路径解析动作类别档位；未命中关键词回落 default。"""
    parts = Path(path).parts
    for keyword, profile in _PROFILE_KEYWORDS:
        if any(keyword in part for part in parts):
            return profile
    return "default"


def profile_params(name: str) -> dict:
    """取档位参数；附该类别在 LAFAN1 校准下的本底报警率（k=6，若已知）。"""
    params = dict(BUILT_IN_PROFILES.get(name, BUILT_IN_PROFILES["default"]))
    baselines = calibrated_baselines()
    if name in baselines:
        params["baseline_rate_per_1k"] = baselines[name]
    return params
