"""坏/好窗口可学习性结果判读（本地）。

输入：badgood_results.json（kernel 产出拷回后放 kaggle_ab/ 或 G:\\ab）。
输出：分组 gap ± SE、DiD ± SE、预注册判读。
DiD = gap_bad - gap_good ≈ -(A 在坏集上的训练拟合增益)；
SE(DiD) = sqrt(var_bad/n_bad + var_good/n_good)（两组窗口独立抽样）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def main(path: str) -> None:
    res = json.loads(Path(path).read_text(encoding="utf-8"))
    a = np.asarray(res["arm_A_s1"], dtype=float)
    b = np.asarray(res["arm_B2_s1"], dtype=float)
    g = np.asarray([w["group"] for w in res["windows"]])
    d = a - b
    stats = {}
    for group in ("bad", "good"):
        dg = d[g == group]
        stats[group] = dict(n=len(dg), gap=dg.mean(), se=dg.std(ddof=1) / np.sqrt(len(dg)),
                            A=a[g == group].mean(), B=b[g == group].mean())
    did = stats["bad"]["gap"] - stats["good"]["gap"]
    se_did = np.sqrt(stats["bad"]["se"] ** 2 + stats["good"]["se"] ** 2)
    t = did / se_did
    print(json.dumps(stats, indent=1))
    print(f"DiD = {did:+.5f} ± {se_did:.5f}  t = {t:+.2f}")
    if t < -2:
        verdict = "坏集 (s,a) 对是可学习信号：A 从中获得了超出臂基线的拟合增益 → 支持 repair-not-drop"
    elif t > 2:
        verdict = "A 在坏集上拟合反而更差：脏样本含矛盾监督（如双拷贝分歧），逐样本有害 → 支持内容级 drop"
    else:
        verdict = "坏集逐样本拟合增益测不出（≈0）：坏集价值主要是覆盖/多样性，非内容信号"
    print("判读:", verdict)
    print("参考: A 模型 bad vs good 窗口 loss 差 = %.5f（含任务难度混杂，B2 基线已部分控制）"
          % (stats["bad"]["A"] - stats["good"]["A"]))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else r"G:\ab\badgood_results.json")