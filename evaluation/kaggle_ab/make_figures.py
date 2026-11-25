# Copyright (c) 2026 wuliangzhong
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""v2 复跑结果可视化（数据源：results.md 抄录数字；eval_v2_results.json 归档后可替换为逐窗口真值）。

生成 kaggle_ab/figures/ 下三张图：
  fig1_adjudication.png   主裁决：两臂曲线（step 域 + epoch 域双面板）
  fig2_epoch_confusion.png 同步骤 vs 同 epoch 的结论翻转（epoch 混淆的核心证据）
  fig3_schedule_case.png  曲线断崖结案：cosine 终点决定走平位置
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)

C_A, C_B = "#1f77b4", "#ff7f0e"

# ---- results.md 抄录数据 ----
A_V2 = {4000: 0.29138, 5000: 0.27839, 6000: 0.25624, 7000: 0.24818,
        8000: 0.23784, 9000: 0.23858, 10000: 0.23688}
B2_V2 = {4000: 0.28628, 5000: 0.26899, 6000: 0.25910, 7000: 0.25782}
STEP_DIFF = {4000: (0.00510, 0.00584, 0.87), 5000: (0.00940, 0.00699, 1.34),
             6000: (-0.00286, 0.00552, -0.52), 7000: (-0.00963, 0.00386, -2.49)}
ADJ = dict(a=0.23688, b=0.25782, gap=-0.02094, se=0.00384, t=-5.45,
           ci=(-0.0285, -0.0133))
EP_A, EP_B = 10154, 6954          # 步/epoch（帧数/有效 batch 8）

A_V1 = {1000: 0.45745, 2000: 0.36188, 3000: 0.30287, 4000: 0.27781, 5000: 0.27505}
B_V1 = {1000: 0.46392, 2000: 0.37703, 3000: 0.32039, 4000: 0.28483,
        5000: 0.28250, 6000: 0.27862, 7000: 0.27693}


def interp(d: dict, x: float) -> float:
    xs = sorted(d)
    if x <= xs[0]:
        return d[xs[0]]
    if x >= xs[-1]:
        return d[xs[-1]]
    for x0, x1 in zip(xs, xs[1:]):
        if x0 <= x <= x1:
            f = (x - x0) / (x1 - x0)
            return d[x0] + f * (d[x1] - d[x0])
    raise AssertionError


def fig1():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8))

    # 左：step 域
    ax1.axvspan(4000, 5000, color=C_B, alpha=0.07)
    ax1.plot(*zip(*sorted(A_V2.items())), "-o", color=C_A, label="A（minrepair，81,229 帧）")
    ax1.plot(*zip(*sorted(B2_V2.items())), "--s", color=C_B, label="B′（clean_v2，55,632 帧）")
    ax1.axvline(7000, color="gray", lw=0.8, ls=":")
    ax1.annotate("交叉点 ≈5.6k", xy=(5600, 0.262), xytext=(6100, 0.292),
                 arrowprops=dict(arrowstyle="->", color="gray", lw=1), color="gray", fontsize=9)
    ax1.annotate("", xy=(10000, ADJ["b"] + 0.004), xytext=(10000, ADJ["a"] - 0.004),
                 arrowprops=dict(arrowstyle="<->", color="crimson", lw=1.4))
    ax1.plot([10000], [ADJ["a"]], "o", color=C_A, ms=9, mec="crimson", mfc="none", mew=2)
    ax1.plot([7000, 10000], [ADJ["b"], ADJ["b"]], ls=":", color=C_B, lw=1.1)
    ax1.plot([10000], [ADJ["b"]], "s", color=C_B, ms=9, mec="crimson", mfc="none", mew=2)
    ax1.annotate(f"同 epoch 裁决点\nA@10k={ADJ['a']:.5f}\nB′@7k={ADJ['b']:.5f}\n"
                 f"A−B′={ADJ['gap']:+.5f}（t={ADJ['t']:.2f}）",
                 xy=(10000, 0.247), xytext=(7900, 0.2695), fontsize=9, color="crimson")
    ax1.text(4500, 0.2935, "4–5k：B′ 反先（epoch 混淆，均不显著）", fontsize=9,
             color="#555", ha="center")
    ax1.set_ylim(0.232, 0.2985)
    ax1.set_xlabel("optimizer step")
    ax1.set_ylabel("eval′ flow-matching loss")
    ax1.set_title("(a) 同 optimizer step：曲线交叉")
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(alpha=0.25)

    # 右：epoch 域（阴影只画两臂共同覆盖区间 0.575–1.007，不做外推）
    ax2.plot([s / EP_A for s in sorted(A_V2)], [A_V2[s] for s in sorted(A_V2)],
             "-o", color=C_A, label="A（按自身 epoch）")
    ax2.plot([s / EP_B for s in sorted(B2_V2)], [B2_V2[s] for s in sorted(B2_V2)],
             "--s", color=C_B, label="B′（按自身 epoch）")
    lo, hi = 4000 / EP_B, 7000 / EP_B
    xs = [lo + (hi - lo) * i / 60 for i in range(61)]
    ax2.fill_between(xs,
                     [interp({s / EP_A: v for s, v in A_V2.items()}, x) for x in xs],
                     [interp({s / EP_B: v for s, v in B2_V2.items()}, x) for x in xs],
                     color="crimson", alpha=0.10)
    ax2.text(0.685, 0.2455, "阴影 = 同 epoch 下\nA 全域更低\n（0.58/0.72/0.86/1.01 ep\n分别 −0.026/−0.024/−0.021/−0.021）",
             fontsize=9, color="crimson")
    ax2.set_xlabel("epoch（该臂看过数据的遍数）")
    ax2.set_title("(b) 同 epoch 对齐：裁决口径下 A 全线更低")
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(alpha=0.25)

    fig.suptitle("v2 主裁决：同 epoch 下 A@10k 比 B′@7k 低 0.0209（t=−5.45，CI [−0.0285, −0.0133]）",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_adjudication.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig2():
    steps = [4000, 5000, 6000, 7000]
    same_step = [STEP_DIFF[s][0] for s in steps]
    same_step_se = [STEP_DIFF[s][1] for s in steps]
    same_ep = []
    for s in steps:
        ep = s / EP_B
        a_at_ep = interp({k / EP_A: v for k, v in A_V2.items()}, ep)
        same_ep.append(a_at_ep - B2_V2[s])

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    x = range(len(steps))
    w = 0.36
    b1 = ax.bar([i - w / 2 for i in x], same_step, w, yerr=same_step_se,
                capsize=4, color="#9ecae1", edgecolor=C_A, label="同 optimizer step（epoch 混淆）")
    b2 = ax.bar([i + w / 2 for i in x], same_ep, w, color="crimson", alpha=0.55,
                edgecolor="darkred", label="同 epoch（裁决口径）")
    for i, v in enumerate(same_step):
        ax.text(i - w / 2, v + (0.0012 if v >= 0 else -0.0022), f"{v:+.4f}",
                ha="center", fontsize=9)
    for i, v in enumerate(same_ep):
        ax.text(i + w / 2, v - 0.0018, f"{v:+.4f}", ha="center", fontsize=9, color="darkred")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"B′@{s}\n(epoch {s / EP_B:.2f})" for s in steps])
    ax.set_ylabel("A − B′（负 = A 更低/更好）")
    ax.set_ylim(-0.0305, 0.020)
    ax.set_title("结论翻转的机制：同步骤口径下 B′@4–5k 反先（每帧被看更多次），\n"
                 "同 epoch 口径下 A 全线更低——短跑比的是学得快，裁决比的是谁知道得多")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "fig2_epoch_confusion.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig3():
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.plot(*zip(*sorted(A_V1.items())), "-o", color=C_A, alpha=0.45, ms=5,
            label="A v1（cosine→5000，上轮）")
    ax.plot(*zip(*sorted(B_V1.items())), "--s", color=C_B, alpha=0.35, ms=4,
            label="B v1（旧 cleaner，含漂移）")
    ax.plot(*zip(*sorted(A_V2.items())), "-o", color=C_A, lw=2.2,
            label="A v2（cosine→10000，本轮）")
    ax.axvline(5000, color="gray", lw=1, ls="--")
    ax.axvline(10000, color="gray", lw=1, ls="--")
    ax.text(5000, 0.44, " v1 调度终点\n 骤平（−0.003/千步）", fontsize=9, color="#555")
    ax.text(10000, 0.40, " v2 调度终点", fontsize=9, color="#555", ha="right")
    ax.annotate("同一段 4–5k：v2 继续降\n（0.291→0.278），v1 已平\n→ 断崖 = LR 调度，非数据底",
                xy=(4600, 0.282), xytext=(2050, 0.318),
                arrowprops=dict(arrowstyle="->", color="crimson", lw=1.2),
                fontsize=9.5, color="crimson")
    ax.annotate("v2 新平台 ≈0.237（0.79–1.0 ep）\n疑似 aleatoric 底", xy=(9500, 0.2375),
                xytext=(7250, 0.300), fontsize=9,
                arrowprops=dict(arrowstyle="->", color="gray", lw=1))
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("eval loss")
    ax.set_title("曲线断崖结案：走平位置由 cosine 终点决定，与数据/清洗无关（v2 A 5–8k 每千步 −0.014 ~ −0.020）")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_schedule_case.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    for p in sorted(OUT.glob("*.png")):
        print(p)
