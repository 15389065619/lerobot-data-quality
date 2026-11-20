"""第二轮（A@10k vs B′@7k）结果分析——预注册决策规则固化为脚本。

用法：
    python -I kaggle_ab/analyze_round2.py --results <eval_results.json>

输入结构（与 v1 相同）：{"<step>": {"<arm>": [320 个窗口 loss]}}
臂名/步数可用 --a-arm/--b-arm/--steps 覆盖（默认自动探测）。

四项检查（对应 2026-11-19 的预注册预测，跑完按规则走，不许事后改口）：
  1. 主裁决  A@10000 vs B′@007000（同 ≈1.0 epoch）——drop 策略的裁决
  2. 同步骤  A@5000 vs B′@5000——与 v1 的 -0.00745(显著) 对比 = 漂移代价的粗估
  3. 漂移代价  B′@5000 vs v1-B@5000(0.28250)——跨评测集对比，带 ±1-2 帧 eval 差的保留
  4. 复现检查  本轮 A@1000..5000 vs v1-A 曲线（应 ±0.002 内；偏差大 = 隐藏非确定性）

显著性 = |A−B| > 配对差值 SE（逐窗口配对，与 v1 口径一致）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# v1 基准（抄录自 Kaggle 会话，results.md 同源）
V1_A = {"001000": 0.45745, "002000": 0.36188, "003000": 0.30287,
        "004000": 0.27781, "005000": 0.27505}
V1_B = {"005000": 0.28250, "006000": 0.27862, "007000": 0.27693}


def paired(a: np.ndarray, b: np.ndarray) -> tuple[float, float, int]:
    d = a - b
    se = float(d.std(ddof=1) / np.sqrt(len(d)))
    return float(d.mean()), se, len(d)


def pick(res: dict, arm: str, step: str) -> np.ndarray:
    if step not in res or arm not in res[step]:
        sys.exit(f"[错误] results 里没有 {arm}@{step}；现有 keys: "
                 f"{sorted(res.keys())[:12]} / 臂: {sorted({k for d in res.values() for k in d})}")
    return np.asarray(res[step][arm], dtype=float)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="eval_results.json 路径")
    ap.add_argument("--a-arm", default=None, help="A 臂名（默认自动探测 arm_A）")
    ap.add_argument("--b-arm", default=None, help="B′ 臂名（默认自动探测 arm_B）")
    ap.add_argument("--a-step", default="010000")
    ap.add_argument("--b-step", default="007000")
    args = ap.parse_args()

    res = json.load(open(args.results, encoding="utf-8"))
    steps = sorted(res.keys())
    arms = sorted({k for d in res.values() for k in d})
    a_arm = args.a_arm or next((x for x in arms if "A" in x.upper()), arms[0])
    b_arm = args.b_arm or next((x for x in arms if "B" in x.upper() and x != a_arm), arms[-1])
    print(f"steps={steps}\n臂: A={a_arm}  B′={b_arm}\n")

    verdict = {}

    # 1) 主裁决：同 epoch
    a, b = pick(res, a_arm, args.a_step), pick(res, b_arm, args.b_step)
    gap, se, n = paired(a, b)
    sig = abs(gap) > se
    verdict["主裁决(同epoch)"] = (f"A−B′ = {gap:+.5f} (SE {se:.5f}, n={n}) → "
                                 f"{'A 显著更优' if sig and gap < 0 else 'B′ 显著更优' if sig else '平手(不显著)'}")
    # 决策规则（2026-11-19 预注册）：A 显著赢 → drop 政策定罪(改 repair)；平手 → 数据量混淆主导；
    # B′ 显著赢 → v1 的 A 优势是漂移伪影，政策维持 drop 并重审"29 集有害"假设

    # 2) 同步骤对照（若存在 5000）
    if "005000" in res and a_arm in res["005000"] and b_arm in res["005000"]:
        gap5, se5, _ = paired(np.asarray(res["005000"][a_arm]), np.asarray(res["005000"][b_arm]))
        verdict["同步骤@5000"] = (f"A−B′ = {gap5:+.5f} (SE {se5:.5f}, {'显著' if abs(gap5) > se5 else '不显著'})；"
                                 f"v1 同口径为 -0.00745(显著) → 漂移代价粗估 ≈ {abs(gap5) - 0.00745:+.5f}")

    # 3) 漂移代价：B′@5000 vs v1-B@5000（跨评测集，带保留）
    if "005000" in res and b_arm in res["005000"]:
        bp = float(np.mean(res["005000"][b_arm]))
        verdict["漂移代价(B′@5k−v1B@5k)"] = (f"{bp:.5f} vs {V1_B['005000']:.5f} = {bp - V1_B['005000']:+.5f}"
                                            "（负值 = B′ 更好 = 漂移在伤 B；跨 eval 口径 ±1-2 帧保留）")

    # 4) 复现检查：本轮 A@1..5k vs v1 A
    rows = []
    for s, v in V1_A.items():
        if s in res and a_arm in res[s]:
            cur = float(np.mean(res[s][a_arm]))
            rows.append(f"  {s}: 本轮 {cur:.5f} vs v1 {v:.5f} (Δ {cur - v:+.5f})")
    verdict["复现检查(本轮A vs v1A)"] = ("未提供 @001000..005000" if not rows else
                                        f"最大|Δ|={max(abs(float(r.split('Δ ')[1])) for r in rows):.5f}\n" + "\n".join(rows))

    print("=== 预注册检查 ===")
    for k, v in verdict.items():
        print(f"[{k}] {v}")
    print("\n=== 决策规则（不许改口）===")
    print("  A@10k 显著 < B′@7k → drop 政策定罪：结构性不一致改 repair 写进 validator")
    print("  |gap| ≤ SE        → 数据量混淆主导：政策改 repair（保守止损），loss 指标到顶考虑 rollout")
    print("  B′@7k 显著 < A@10k → v1 的 A 优势是漂移伪影：政策维持 drop，重审 29 集有害性")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
