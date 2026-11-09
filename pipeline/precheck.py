"""训练前预检：把质检做成训练脚本前的一道门。

在训练脚本里加几行即可（不依赖任何训练框架）：

    from pipeline.precheck import precheck, DirtyDatasetError

    precheck("/data/lerobot_ds", strict=True)   # 数据集脏 → 直接抛异常，训练不会白跑

也可以当独立命令用：

    python -m pipeline.precheck --dataset_path /data/lerobot_ds --strict
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import load_config
from .loader import load_dataset
from .policy import decide
from .validator import validate_dataset
from .validator.types import RULE_MAP

#: 这些处置会阻断训练（repair / review 不阻断，只告警）
BLOCKING_DISPOSITIONS = ("drop",)


class DirtyDatasetError(RuntimeError):
    """数据集存在必须处理的问题；strict 模式下抛出。"""


@dataclass
class PrecheckResult:
    dataset_path: str
    ok: bool
    total_episodes: int
    total_frames: int
    keep: int
    repair: int
    review: int
    drop: int
    blocking: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[dict[str, Any]] = field(default_factory=list)
    message: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset_path": self.dataset_path,
            "ok": self.ok,
            "total_episodes": self.total_episodes,
            "total_frames": self.total_frames,
            "disposition": {"keep": self.keep, "repair": self.repair,
                            "review": self.review, "drop": self.drop},
            "blocking": self.blocking,
            "warnings": self.warnings,
            "message": self.message,
        }

    def write_json(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return out


def _episode_issues(res, decision, severity: str) -> dict[str, Any]:
    codes = sorted(res.reasons)
    return {
        "episode_index": res.episode_index,
        "frame_count": res.frame_count,
        "disposition": decision.disposition if decision else None,
        "severity": severity,
        "codes": codes,
        "labels": [RULE_MAP[c].label for c in codes if c in RULE_MAP],
        "action": decision.recommended_action if decision else None,
    }


def precheck(
    dataset_path: str | Path,
    *,
    config: dict[str, Any] | None = None,
    max_episodes: int | None = None,
    on_review: str = "keep",
    strict: bool = False,
) -> PrecheckResult:
    """只读预检：不写任何文件、不修改数据集，返回结论。

    Args:
        strict: True 时只要存在 drop 级 episode 就抛 DirtyDatasetError。
        on_review: 未经人工复核的 warning 如何处置（keep/drop），与主流程一致。
    """
    dataset = load_dataset(dataset_path, max_episodes=max_episodes, load_frames=False)
    config = config or load_config(robot_type=dataset.info.get("robot_type"))
    # 流式质检：两遍扫描 parquet 但不驻留整表，50GB 数据集也不会把训练机内存打爆
    validation = validate_dataset(dataset, config, streaming=True)
    decisions = decide(validation, on_review=on_review)

    counts = {"keep": 0, "repair": 0, "review": 0, "drop": 0}
    blocking: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    total_frames = 0
    for res in validation.results.values():
        total_frames += res.frame_count
        decision = decisions.get(res.episode_index)
        disposition = decision.disposition if decision else "keep"
        counts[disposition] = counts.get(disposition, 0) + 1
        severity = validation.severity_of(res)
        if disposition in BLOCKING_DISPOSITIONS:
            blocking.append(_episode_issues(res, decision, severity))
        elif res.reasons:
            warnings.append(_episode_issues(res, decision, severity))

    ok = not blocking
    if ok:
        message = f"预检通过：{validation.total_episodes} 个 episode / {total_frames} 帧可用"
        if warnings:
            message += f"（{len(warnings)} 个 episode 有告警，建议跑完整流水线修复）"
    else:
        # --on-review keep 本来就是默认值，默认路径下它不是「放宽」手段，只在
        # 用户显式传了 drop 时才给出这条提示。
        relax = "，或加 --on-review keep 放宽" if on_review == "drop" else ""
        message = (f"预检未通过：{len(blocking)}/{validation.total_episodes} 个 episode 需丢弃，"
                   f"涉及 {sum(i['frame_count'] for i in blocking)} 帧。"
                   f"先跑 python main.py ... --skip-clean 看完整报告{relax}")
    result = PrecheckResult(
        dataset_path=str(dataset_path),
        ok=ok,
        total_episodes=validation.total_episodes,
        total_frames=total_frames,
        blocking=blocking,
        warnings=warnings,
        message=message,
        **counts,
    )
    if strict and blocking:
        raise DirtyDatasetError(message)
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="训练前数据预检（只读，不修改数据集）")
    ap.add_argument("--dataset_path", required=True)
    ap.add_argument("--config", default=None, help="阈值配置 yaml/json")
    ap.add_argument("--max_episodes", type=int, default=None)
    ap.add_argument("--on-review", choices=("keep", "drop"), default="keep")
    ap.add_argument("--strict", action="store_true", help="存在 drop 级 episode 时以退出码 2 结束")
    ap.add_argument("--json", default=None, help="把结论写成 JSON")
    args = ap.parse_args(argv)

    result = precheck(
        args.dataset_path,
        config=load_config(path=args.config) if args.config else None,
        max_episodes=args.max_episodes,
        on_review=args.on_review,
    )

    print(("[OK] " if result.ok else "[BLOCKED] ") + result.message)
    if args.json:
        result.write_json(args.json)
    return 2 if (args.strict and not result.ok) else 0


if __name__ == "__main__":
    raise SystemExit(main())
