"""multi-seed 复验裁决——results.md「多 seed 复验」节预注册的三档判读，机械执行。

用法：
    python -I kaggle_ab/analyze_seeds.py seed0.json seed1.json seed2.json \
        [--a-step 010000] [--b-step 007000] [--a-arm arm_A] [--b-arm arm_B]

每个输入文件格式与 v1 相同：{"<step>": {"<arm>": [320 个窗口 loss]}}。
文件名/顺序即 seed 标签（seed0 = 2026-11-19 的 v2 主跑，其余为 seed 1/2）。

预注册判读（2026-11-19 写死，不许事后改口）：
  1. 各 seed 的 A−B′ 全部为负 → 方向稳健，报 mean ± cross-seed SD；
  2. 任一 seed 反号 → 降级为「单次 run 的观测」，不再声称稳健；
  3. cross-seed SD ≥ |mean| → 0.021 视为在 seed 噪声下不可分辨。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def gap_of(path: Path, a_arm: str, b_arm: str, a_step: str, b_step: str) -> float:
    res = json.load(open(path, encoding="utf-8"))
    try:
        a = np.asarray(res[a_step][a_arm], dtype=float)
        b = np.asarray(res[b_step][b_arm], dtype=float)
    except KeyError as e:
        sys.exit(f"[错误] {path.name} 缺少 {e}；现有 steps={sorted(res.keys())[:12]} "
                 f"arms={sorted({k for d in res.values() for k in d})}")
    if len(a) != len(b):
        sys.exit(f"[错误] {path.name} 两臂窗口数不一致 {len(a)} vs {len(b)}")
    return float((a - b).mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+", help="各 seed 的 eval_results.json（顺序 = seed0..N）")
    ap.add_argument("--a-step", default="010000")
    ap.add_argument("--b-step", default="007000")
    ap.add_argument("--a-arm", default="arm_A")
    ap.add_argument("--b-arm", default="arm_B")
    args = ap.parse_args()

    paths = [Path(p) for p in args.results]
    gaps = [gap_of(p, args.a_arm, args.b_arm, args.a_step, args.b_step) for p in paths]

    print("=== 各 seed 的 A−B′（同 epoch：A@%s vs B′@%s）===" % (args.a_step, args.b_step))
    for p, g in zip(paths, gaps):
        print(f"  {p.stem}: {g:+.5f}")
    mean, sd = float(np.mean(gaps)), float(np.std(gaps, ddof=1)) if len(gaps) > 1 else 0.0
    print(f"\nmean = {mean:+.5f}   cross-seed SD = {sd:.5f}   n_seeds = {len(gaps)}")

    all_neg = all(g < 0 for g in gaps)
    any_flip = any(g >= 0 for g in gaps)
    if any_flip:
        verdict = ("第 2 档：存在反号 seed → 降级为「单次 run 的观测」，"
                   "不再声称「drop 政策定罪」稳健；repair 政策落地暂缓，先查反号 seed。")
    elif sd >= abs(mean):
        verdict = ("第 3 档：cross-seed SD ≥ |mean| → 0.021 在 seed 噪声下不可分辨；"
                   "维持「方向性证据」，repair 政策落地需再积累。")
    elif all_neg:
        verdict = (f"第 1 档：全部 seed 同向为负 → 方向稳健，结论报 mean±SD = "
                   f"{mean:+.5f}±{sd:.5f}；drop 政策定罪生效 → validator 落地「结构性不一致 → repair」。")
    else:
        verdict = "未预期状态，人工核查。"
    print("\n=== 预注册裁决 ===")
    print(verdict)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
