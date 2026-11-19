"""按动作类别的规则参数分档（Tier 2 校准）。

背景：Tier 1 的 MAD 跳变规则在**高动态动作**上会把动作本身标成跳变——
校准实验（Standing_High_Jump，n=1）：k=6 时 root_pos_z 独占 1055 帧
（起跳/落地），而膝角的 342 帧在 k=10 下消失（快速屈膝是动作不是缺陷）。

分档参数（校准于 n=1，扩样本后重新校准是义务不是可选）：
  default      k=6，无跳过——常规/低速动作
  high_dynamic k=10 + 跳过 root_pos_* ——根位置在动态类被动作本身支配；
               关节角通道保留（缺陷面在关节上）

类别解析：从数据路径里找关键词（样本目录自带类别名，如
"1.1.1.High_Dynamic_Movement"）。未知类别回落 default——宁松勿紧，
灰区交给复核队列，不静默定罪。
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["BUILT_IN_PROFILES", "resolve_profile", "profile_params"]

BUILT_IN_PROFILES: dict[str, dict] = {
    "default": {
        "mad_jump_k": 6.0,
        "mad_jump_skip_channel_patterns": [],
    },
    "high_dynamic": {
        "mad_jump_k": 10.0,
        "mad_jump_skip_channel_patterns": ["root_pos_*"],
    },
}

_PROFILE_KEYWORDS = (("High_Dynamic", "high_dynamic"),)


def resolve_profile(path: str | Path) -> str:
    """从数据路径解析动作类别档位；未命中关键词回落 default。"""
    parts = Path(path).parts
    for keyword, profile in _PROFILE_KEYWORDS:
        if any(keyword in part for part in parts):
            return profile
    return "default"


def profile_params(name: str) -> dict:
    """取档位参数；未知档位名回落 default（宁松勿紧，不抛错）。"""
    return dict(BUILT_IN_PROFILES.get(name, BUILT_IN_PROFILES["default"]))
