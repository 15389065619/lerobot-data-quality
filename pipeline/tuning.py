"""从历史人工复核结果学习处置策略（阈值 / 严重度建议）。

不自动改配置：输出一份 **YAML patch**，由人评审后合并进 ``config/default.yaml``。
这是刻意的：训练数据的判定规则必须可审计，机器学习只应「提议」，不应「静默生效」。

用法：
    python -m pipeline.tuning --report outputs/quality_report.json \
        --reviews outputs/review_queue_reviewed.csv --out config/tuned.yaml

学习逻辑：
1. 统计每条规则「命中且被人工复核」的 episode 里，人最终判 keep / repair / drop 的比例；
2. drop 比例 ≥ 0.8 且样本 ≥ 2 → 建议升级为 severity=error（全自动剔除）；
   drop 比例 ≤ 0.2 且样本 ≥ 3 → 建议降级为 info（只记录，不阻断）；
3. 其余保持不变——**样本不足时不做任何建议**，避免小样本过拟合。
4. 对带数值证据的规则（目前是 action_jump.max_delta）额外给出可分离性分析。
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any

from .config import load_config, severity_of
from .validator.types import RULE_MAP

LOG = logging.getLogger(__name__)

#: 人给出的处置 → 是否属于「该丢」
DROP_DISPOSITIONS = {"drop", "reject", "trash"}
#: 触发升级/降级的最小样本量
MIN_SAMPLES_UPGRADE = 2
MIN_SAMPLES_DOWNGRADE = 3
UPGRADE_RATIO = 0.8
DOWNGRADE_RATIO = 0.2


def load_reviews(paths: list[str | Path]) -> dict[int, str]:
    """读取若干复核 CSV，合并为 {episode_index: disposition}。"""
    rows: dict[int, str] = {}
    for raw in paths:
        path = Path(raw)
        if not path.is_file():
            raise FileNotFoundError(f"复核文件不存在：{path}")
        with path.open("r", encoding="utf-8-sig", newline="") as fh:
            for row in csv.DictReader(fh):
                raw_ep = (row.get("episode_index") or "").strip()
                disposition = (row.get("reviewer_disposition") or row.get("disposition") or "").strip()
                if not raw_ep or not disposition:
                    continue
                rows[int(float(raw_ep))] = disposition.lower()
    return rows


def analyse(report: dict[str, Any], reviews: dict[int, str],
            config: dict[str, Any] | None = None) -> dict[str, Any]:
    """按规则汇总「命中 vs 人工结论」，产出建议（不改动任何输入）。"""
    config = config or load_config()
    episodes = report.get("episodes") or report.get("invalid_episode_details") or []
    stats: dict[str, dict[str, int]] = {}
    evidence: dict[str, list[tuple[str, float]]] = {}

    for ep in episodes:
        ep_index = ep.get("episode_index")
        if ep_index not in reviews:
            continue
        human = reviews[ep_index]
        for code in ep.get("reasons") or []:
            item = stats.setdefault(code, {"fired": 0, "reviewed": 0, "dropped": 0, "kept": 0})
            item["fired"] += 1
            item["reviewed"] += 1
            if human in DROP_DISPOSITIONS:
                item["dropped"] += 1
            else:
                item["kept"] += 1
            if code == "action_jump":
                value = ((ep.get("details") or {}).get(code) or {}).get("max_delta")
                if isinstance(value, (int, float)):
                    evidence.setdefault(code, []).append((human, float(value)))

    recommendations = []
    fired_all = report.get("reason_frequency") or {}
    for code, item in sorted(stats.items()):
        rule = RULE_MAP.get(code)
        current = severity_of(config, code)
        drop_ratio = item["dropped"] / max(item["reviewed"], 1)
        suggested = current
        reason = ""
        if item["dropped"] >= MIN_SAMPLES_UPGRADE and drop_ratio >= UPGRADE_RATIO:
            suggested = "error"
            reason = f"人工剔除率 {drop_ratio:.0%}（{item['dropped']}/{item['reviewed']}），建议全自动剔除"
        elif item["reviewed"] >= MIN_SAMPLES_DOWNGRADE and drop_ratio <= DOWNGRADE_RATIO:
            suggested = "info"
            reason = f"人工剔除率仅 {drop_ratio:.0%}（{item['dropped']}/{item['reviewed']}），建议只记录不阻断"
        recommendations.append({
            "code": code,
            "label": rule.label if rule else code,
            "fired": int(fired_all.get(code, item["fired"])),
            "reviewed": item["reviewed"],
            "human_drop": item["dropped"],
            "human_keep": item["kept"],
            "drop_ratio": round(drop_ratio, 3),
            "current_severity": current,
            "suggested_severity": suggested,
            "changed": suggested != current,
            "reason": reason,
        })

    numeric = _analyse_numeric(evidence)
    return {
        "reviewed_episodes": len(reviews),
        "rules": recommendations,
        "threshold_hints": numeric,
        "patch": _build_patch(recommendations, numeric),
    }


def _analyse_numeric(evidence: dict[str, list[tuple[str, float]]]) -> list[dict[str, Any]]:
    """检查数值证据能否把人判 drop / keep 的两组分开。"""
    hints = []
    for code, items in evidence.items():
        drops = [v for human, v in items if human in DROP_DISPOSITIONS]
        keeps = [v for human, v in items if human not in DROP_DISPOSITIONS]
        if not drops or not keeps:
            continue
        hint: dict[str, Any] = {
            "code": code,
            "evidence": "max_delta",
            "max_kept": round(max(keeps), 4),
            "min_dropped": round(min(drops), 4),
            "separable": max(keeps) < min(drops),
        }
        if hint["separable"]:
            hint["suggested_cut"] = round((max(keeps) * min(drops)) ** 0.5, 4)
            hint["note"] = "可直接按该值切分；若要映射回 mad_k，需同时看 thresholds_used"
        else:
            hint["note"] = "两组重叠，单纯调阈值无法分开，建议保留 review 并抽样人工"
        hints.append(hint)
    return hints


def _build_patch(recommendations: list[dict[str, Any]], hints: list[dict[str, Any]]) -> dict[str, Any]:
    severity = {r["code"]: r["suggested_severity"] for r in recommendations if r["changed"]}
    patch: dict[str, Any] = {}
    if severity:
        patch["severity"] = severity
    if hints:
        patch["_threshold_hints"] = hints
    return patch


def render_yaml(patch: dict[str, Any]) -> str:
    lines = ["# 由 pipeline.tuning 根据历史复核结果生成；请评审后再合并进 config/default.yaml"]
    if "severity" in patch:
        lines.append("severity:")
        for code, value in patch["severity"].items():
            lines.append(f"  {code}: {value}")
    for hint in patch.get("_threshold_hints", []):
        lines.append(f"# {hint['code']}: max_kept={hint['max_kept']} "
                     f"min_dropped={hint['min_dropped']} separable={hint['separable']}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description="从历史复核学习处置策略（只产出建议，不改配置）")
    ap.add_argument("--report", required=True, help="quality_report.json")
    ap.add_argument("--reviews", nargs="+", required=True, help="已填写 reviewer_disposition 的复核 CSV")
    ap.add_argument("--config", default=None, help="当前阈值配置（用于对比 severity 现状）")
    ap.add_argument("--out", default=None, help="把建议写成 YAML patch")
    ap.add_argument("--json", default=None, help="把完整分析写成 JSON")
    args = ap.parse_args(argv)

    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    reviews = load_reviews(args.reviews)
    if not reviews:
        LOG.warning("复核文件里没有填写的 reviewer_disposition，无法学习")
        return 1
    result = analyse(report, reviews, load_config(path=args.config) if args.config else None)

    print(f"{'规则':<24}{'命中':>6}{'复核':>6}{'人判丢弃':>10}{'丢弃率':>8}  建议")
    for item in result["rules"]:
        mark = "*" if item["changed"] else " "
        print(f"{mark}{item['code']:<23}{item['fired']:>6}{item['reviewed']:>6}"
              f"{item['human_drop']:>10}{item['drop_ratio']:>8.0%}  "
              f"{item['current_severity']} → {item['suggested_severity']}")
    for hint in result["threshold_hints"]:
        print(f"\n[{hint['code']}] {hint['note']}：kept≤{hint['max_kept']} / dropped≥{hint['min_dropped']}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(render_yaml(result["patch"]), encoding="utf-8")
        LOG.info("建议已写入 %s（需人工合并，不会自动生效）", args.out)
    if args.json:
        Path(args.json).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
