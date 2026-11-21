"""
生成论文所需的全部可视化图表。

用法：
    python generate_paper_figures.py

输出：./paper_figures/ 目录下的 8 张 PNG 图片
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# 全局样式
plt.rcParams.update({
    "font.family": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

OUT = Path(__file__).parent / "paper_figures"
OUT.mkdir(exist_ok=True)

# 配色
C_A = "#4C72B0"      # A 臂（最小修复）- 蓝
C_B = "#DD8452"      # B' 臂（清洗产物）- 橙
C_DIFF = "#55A868"   # 差值 - 绿
C_SEED = ["#4C72B0", "#DD8452", "#55A868"]

# 载入多 seed 原始数据
SEED_RESULTS = json.load(open("kaggle_ab/eval_seed_results.json", encoding="utf-8"))

arm_A_s1 = np.asarray(SEED_RESULTS["arm_A_s1"])
arm_B_s1 = np.asarray(SEED_RESULTS["arm_B2_s1"])
arm_A_s2 = np.asarray(SEED_RESULTS["arm_A_s2"])
arm_B_s2 = np.asarray(SEED_RESULTS["arm_B2_s2"])

# seed 0 数据来自 results.md 抄录
seed0 = {"A": 0.23688, "B": 0.25782, "diff": -0.02094, "se": 0.00384}
seed1 = {"A": arm_A_s1.mean(), "B": arm_B_s1.mean(),
         "diff": (arm_A_s1 - arm_B_s1).mean(),
         "se": (arm_A_s1 - arm_B_s1).std(ddof=0) / math.sqrt(len(arm_A_s1))}
seed2 = {"A": arm_A_s2.mean(), "B": arm_B_s2.mean(),
         "diff": (arm_A_s2 - arm_B_s2).mean(),
         "se": (arm_A_s2 - arm_B_s2).std(ddof=0) / math.sqrt(len(arm_A_s2))}

seeds = [seed0, seed1, seed2]
cross_mean = np.mean([s["diff"] for s in seeds])
cross_sd = np.std([s["diff"] for s in seeds], ddof=1)

print(f"cross-seed mean = {cross_mean:+.5f}")
print(f"cross-seed SD   = {cross_sd:.5f}")
print(f"全部同号 = {all(s['diff'] < 0 for s in seeds)}")


# 图 1：多 seed 森林图（Forest Plot）
def fig1_forest():
    fig, ax = plt.subplots(figsize=(8, 4))

    y_positions = [3, 2, 1]
    labels = ["Seed 0（主实验）", "Seed 1", "Seed 2"]

    for i, (seed, y) in enumerate(zip(seeds, y_positions)):
        diff = seed["diff"]
        se = seed["se"]
        # 点估计
        ax.plot(diff, y, "o", color=C_SEED[i], markersize=8, zorder=3)
        # 95% CI 误差棒
        ax.plot([diff - 1.96 * se, diff + 1.96 * se], [y, y],
                color=C_SEED[i], linewidth=2, zorder=2)
        ax.annotate(f"{diff:+.4f}", (diff + 1.96 * se + 0.0005, y),
                    va="center", fontsize=10)

    # 合并均值（虚线）
    ax.axvline(cross_mean, color="black", linestyle="--", linewidth=1.5,
               label=f"合并均值 = {cross_mean:+.4f}")
    # 跨 seed SD 带
    ax.axvspan(cross_mean - cross_sd, cross_mean + cross_sd,
               color="gray", alpha=0.2, label=f"跨 seed SD ±{cross_sd:.4f}")

    # 零线
    ax.axvline(0, color="gray", linestyle=":", linewidth=1)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.set_xlabel("A − B′ Flow-Matching Loss（同 epoch 口径）")
    ax.set_title("图 1：多 seed 森林图——A 臂（最小修复）在 3 个 seed 下全部显著优于 B′")
    ax.legend(loc="lower right")
    ax.set_xlim(-0.032, 0.005)
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT / "fig1_forest_multi_seed.png")
    plt.close(fig)
    print("  fig1_forest_multi_seed.png")


# 图 2：各 seed 两臂最终 loss 柱状对比
def fig2_bar_comparison():
    fig, ax = plt.subplots(figsize=(7, 5))

    x = np.arange(3)
    width = 0.35

    A_vals = [s["A"] for s in seeds]
    B_vals = [s["B"] for s in seeds]

    bars_A = ax.bar(x - width / 2, A_vals, width, label="A（最小修复，~1.0 epoch）",
                    color=C_A, edgecolor="white")
    bars_B = ax.bar(x + width / 2, B_vals, width, label="B′（清洗产物，~1.0 epoch）",
                    color=C_B, edgecolor="white")

    # 差值标注
    for i in range(3):
        diff = A_vals[i] - B_vals[i]
        mid_y = min(A_vals[i], B_vals[i]) - 0.001
        ax.annotate(f"Δ = {diff:+.4f}", (x[i], mid_y), ha="center", fontsize=10,
                    color=C_DIFF, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(["Seed 0", "Seed 1", "Seed 2"])
    ax.set_ylabel("Flow-Matching Loss")
    ax.set_title("图 2：三个 seed 下两臂最终 loss 对比")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT / "fig2_bar_comparison.png")
    plt.close(fig)
    print("  fig2_bar_comparison.png")


# 图 3：逐窗口配对散点图（Seed 1 + Seed 2）
def fig3_paired_scatter():
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    pairs = [(arm_A_s1, arm_B_s1, "Seed 1"), (arm_A_s2, arm_B_s2, "Seed 2")]

    for ax, (A, B, title) in zip(axes, pairs):
        # 对角线（A = B）
        lim_max = max(A.max(), B.max()) * 1.05
        lim_min = min(A.min(), B.min()) * 0.95
        ax.plot([lim_min, lim_max], [lim_min, lim_max],
                "k--", linewidth=1, label="A = B")

        # 散点
        ax.scatter(B, A, c=C_A, alpha=0.7, s=30, edgecolors="white", zorder=3)

        # 均值标记
        ax.axhline(A.mean(), color=C_A, linestyle=":", alpha=0.6, linewidth=1)
        ax.axvline(B.mean(), color=C_B, linestyle=":", alpha=0.6, linewidth=1)

        # 对角线以下 = A 更优（loss 更低）
        n_below = np.sum(A < B)
        n_total = len(A)
        ax.annotate(f"A < B 的窗口：{n_below}/{n_total} ({100*n_below/n_total:.0f}%)",
                    xy=(0.03, 0.97), xycoords="axes fraction",
                    va="top", fontsize=10,
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

        ax.set_xlabel("B′ 臂 Loss")
        ax.set_ylabel("A 臂 Loss")
        ax.set_title(f"({chr(97 if '1' in title else 98)}) {title}")
        ax.legend(loc="lower right", fontsize=9)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)

    fig.suptitle("图 3：逐窗口配对散点图——对角线下方 = A（最小修复）更优", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_paired_scatter.png")
    plt.close(fig)
    print("  fig3_paired_scatter.png")


# 图 4：逐窗口差值直方图（Seed 1 + Seed 2 合并）
def fig4_diff_histogram():
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    diffs_s1 = arm_A_s1 - arm_B_s1  # A - B
    diffs_s2 = arm_A_s2 - arm_B_s2

    for ax, diffs, title, color in zip(
        axes,
        [diffs_s1, diffs_s2],
        ["Seed 1", "Seed 2"],
        [C_SEED[1], C_SEED[2]]
    ):
        # 直方图
        bins = np.linspace(diffs.min() - 0.005, diffs.max() + 0.005, 18)
        ax.hist(diffs, bins=bins, color=color, alpha=0.7, edgecolor="white")

        # 均值线
        mean_d = diffs.mean()
        ax.axvline(mean_d, color="black", linestyle="--", linewidth=1.5,
                   label=f"均值 = {mean_d:+.4f}")
        # 零线
        ax.axvline(0, color="gray", linestyle=":", linewidth=1)

        n_neg = np.sum(diffs < 0)
        ax.annotate(f"A < B 窗口数：{n_neg}/{len(diffs)} ({100*n_neg/len(diffs):.0f}%)",
                    xy=(0.03, 0.97), xycoords="axes fraction",
                    va="top", fontsize=10,
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

        ax.set_xlabel("A − B′ Loss")
        ax.set_ylabel("窗口数")
        ax.set_title(f"({chr(97 if '1' in title else 98)}) {title}")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("图 4：逐窗口 A−B′ 差值分布——整体向负侧偏移（A 更优）", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_diff_histogram.png")
    plt.close(fig)
    print("  fig4_diff_histogram.png")


# 图 5：训练曲线（同 epoch 对齐）
def fig5_training_curves():
    """根据 results.md 的训练曲线数据绘制。"""

    # A 臂（so101_minrepair, 81,229 帧, ~10,154 步/epoch）
    a_steps = [4000, 5000, 6000, 7000, 8000, 9000, 10000]
    a_loss = [0.29138, 0.27839, 0.25624, 0.24818, 0.23784, 0.23858, 0.23688]
    a_epoch = [s / 10154 for s in a_steps]

    # B' 臂（so101_clean_v2, 55,632 帧, ~6,954 步/epoch）
    b_steps = [4000, 5000, 6000, 7000]
    b_loss = [0.28628, 0.26899, 0.25910, 0.25782]
    b_epoch = [s / 6954 for s in b_steps]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # (a) 同 optimizer step
    ax1.plot(a_steps, a_loss, "o-", color=C_A, linewidth=2, markersize=7, label="A 臂")
    ax1.plot(b_steps, b_loss, "s-", color=C_B, linewidth=2, markersize=7, label="B′ 臂")

    # 交叉点标注
    cross_step = 5600  # 内插估算
    ax1.annotate(f"交叉点\n~5600 步", xy=(cross_step, 0.262),
                 fontsize=9, color="red", ha="center",
                 arrowprops=dict(arrowstyle="->", color="red"))

    ax1.set_xlabel("Optimizer Step")
    ax1.set_ylabel("Flow-Matching Loss")
    ax1.set_title("(a) 按 optimizer step 对比")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # (b) 同 epoch 对齐（主裁决口径）
    # 构造 dense epoch 轴用于插值
    epoch_dense = np.linspace(0.38, 1.05, 100)
    a_interp = np.interp(epoch_dense, a_epoch, a_loss)
    b_interp = np.interp(epoch_dense, b_epoch, b_loss)

    ax2.plot(epoch_dense, a_interp, "-", color=C_A, linewidth=2.5, label="A 臂（最小修复）")
    ax2.plot(epoch_dense, b_interp, "-", color=C_B, linewidth=2.5, label="B′ 臂（清洗产物）")

    # 标记实际数据点
    ax2.plot(a_epoch, a_loss, "o", color=C_A, markersize=6)
    ax2.plot(b_epoch, b_loss, "s", color=C_B, markersize=6)

    # 两者差值阴影
    ax2.fill_between(epoch_dense, a_interp, b_interp, alpha=0.15, color=C_A,
                     label="A − B′ 差值")

    # 1.0 epoch 处标注
    ax2.axvline(1.0, color="gray", linestyle=":", linewidth=1)
    ax2.annotate("同 epoch\n裁决口径", xy=(1.0, 0.25), fontsize=9,
                 color="gray", ha="center")

    ax2.set_xlabel("Epoch（已见数据量）")
    ax2.set_ylabel("Flow-Matching Loss")
    ax2.set_title("(b) 按 epoch 对齐——裁决口径")
    ax2.legend(loc="upper right")
    ax2.grid(alpha=0.3)

    fig.suptitle("图 5：训练曲线——同 step 口径有交叉（A 后期反超），同 epoch 口径 A 全线更低",
                 y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig5_training_curves.png")
    plt.close(fig)
    print("  fig5_training_curves.png")


# 图 6：数据量与帧对比
def fig6_data_volume():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    # (a) Episode 数量
    ep_counts = [135, 106]  # A vs B'
    labels_ep = ["A（最小修复）", "B′（清洗产物）"]
    colors_ep = [C_A, C_B]

    bars = ax1.bar(labels_ep, ep_counts, color=colors_ep, edgecolor="white", width=0.5)
    for bar, val in zip(bars, ep_counts):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f"{val} 集", ha="center", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Episode 数量")
    ax1.set_title("(a) Episode 数量对比")
    ax1.grid(axis="y", alpha=0.3)

    # (b) 帧数量
    frame_counts = [81229, 55632]  # A vs B'
    bars2 = ax2.bar(labels_ep, frame_counts, color=colors_ep, edgecolor="white", width=0.5)
    for bar, val in zip(bars2, frame_counts):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 500,
                 f"{val:,} 帧", ha="center", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Frame 数量")
    ax2.set_title("(b) Frame 数量对比")
    ax2.grid(axis="y", alpha=0.3)

    # 差值说明
    frame_diff = (frame_counts[0] - frame_counts[1]) / frame_counts[0] * 100
    fig.text(0.5, -0.02,
             f"B′ 比 A 少 {frame_diff:.1f}% 的帧（约 25,597 帧）——清洗丢集造成的数据覆盖差异",
             ha="center", fontsize=10, style="italic")

    fig.suptitle("图 6：清洗造成的数据量差异——29 个 episode 被 drop", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig6_data_volume.png")
    plt.close(fig)
    print("  fig6_data_volume.png")


# 图 7：光捕模块 Tier 分层架构示意图
def fig7_gait_unit_architecture():
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 12)
    ax.set_ylim(-0.45, 8)
    ax.axis("off")

    # 标题
    fig.suptitle("图 7：光捕质检模块（gait_unit）Tier 分层架构", fontsize=14, fontweight="bold", y=0.98)

    # ---- Tier 1 ----
    tier1_y = 6.5
    rect1 = plt.Rectangle((0.5, tier1_y - 0.6), 11, 1.2,
                           facecolor="#E8F4FD", edgecolor="#1976D2", linewidth=2,
                           transform=ax.transData)
    ax.add_patch(rect1)
    ax.text(1.0, tier1_y + 0.25, "Tier 1：通道级规则（MotionClip IR）",
            fontsize=12, fontweight="bold", color="#1976D2")
    ax.text(1.0, tier1_y - 0.15,
            "· MAD 跳变检测（按动作类别分档校准）    · NaN 段定位    · 常值通道识别    · 值域检查    · 四元数整体翻转豁免",
            fontsize=9, color="#333")
    ax.text(1.0, tier1_y - 0.4,
            "输入：T × D 通道数组 + 通道名（如 dof_left_knee_joint(rad)）",
            fontsize=9, style="italic", color="#666")

    # ---- Tier 2 ----
    tier2_y = 4.5
    rect2 = plt.Rectangle((0.5, tier2_y - 0.6), 11, 1.2,
                           facecolor="#FFF3E0", edgecolor="#F57C00", linewidth=2,
                           transform=ax.transData)
    ax.add_patch(rect2)
    ax.text(1.0, tier2_y + 0.25, "Tier 2：骨架/标记点级规则（GaitClip）",
            fontsize=12, fontweight="bold", color="#F57C00")
    ax.text(1.0, tier2_y - 0.15,
            "· 时间戳间隙/重复盖章    · 关节旋转角超限    · 骨长漂移    · 标记点缺失/跳变",
            fontsize=9, color="#333")
    ax.text(1.0, tier2_y - 0.4,
            "依赖：骨架层级（parents）+ Unitree G1 URDF 限位表",
            fontsize=9, style="italic", color="#666")

    # ---- Tier 3 ----
    tier3_y = 2.2
    rect3 = plt.Rectangle((0.5, tier3_y - 0.6), 11, 1.2,
                           facecolor="#E8F5E9", edgecolor="#388E3C", linewidth=2,
                           transform=ax.transData)
    ax.add_patch(rect3)
    ax.text(1.0, tier3_y + 0.25, "Tier 3：空间级规则（需要 URDF FK → 升级为 GaitClip IR）",
            fontsize=12, fontweight="bold", color="#388E3C")
    ax.text(1.0, tier3_y - 0.15,
            "· 足部滑移    · 触地穿透    · 腾空相位",
            fontsize=9, color="#333")
    ax.text(1.0, tier3_y - 0.4,
            "升级：MotionClip → GaitClip（位置 + 四元数 + 骨架层级）",
            fontsize=9, style="italic", color="#666")

    # ---- 适配器层 ----
    adapt_y = 0.3
    adapt_boxes = [
        ("MotionDecode\n（青瞳视觉 G1）", "#7B1FA2"),
        ("BVH\n（LAFAN1 公开集）", "#7B1FA2"),
        ("FZMotion\n（动捕长表 CSV）", "#7B1FA2"),
        ("PICO CSV\n（设备导出）", "#7B1FA2"),
    ]
    box_w = 2.4
    start_x = 0.5
    gap = 0.15
    for i, (label, color) in enumerate(adapt_boxes):
        x = start_x + i * (box_w + gap)
        rect = plt.Rectangle((x, adapt_y - 0.5), box_w, 1.0,
                               facecolor="#F3E5F5", edgecolor=color, linewidth=1.5,
                               transform=ax.transData)
        ax.add_patch(rect)
        ax.text(x + box_w / 2, adapt_y, label, ha="center", va="center",
                fontsize=9, color=color, fontweight="bold")

    ax.text(0.5, adapt_y + 0.7, "数据源适配器层", fontsize=10, fontweight="bold", color="#6A1B9A")

    # ---- 箭头 ----
    arrow_props = dict(arrowstyle="->", color="#888", lw=1.5)
    for y_from, y_to in [(tier1_y - 0.6, tier2_y + 0.6), (tier2_y - 0.6, tier3_y + 0.6),
                         (adapt_y + 0.5, tier1_y + 0.6)]:
        ax.annotate("", xy=(6, y_to), xytext=(6, y_from), arrowprops=arrow_props)

    # ---- 实测标注 ----
    ax.text(11.5, tier1_y, "已落地", fontsize=9, color="#388E3C", ha="right")
    ax.text(11.5, tier2_y, "已落地", fontsize=9, color="#388E3C", ha="right")
    ax.text(11.5, tier3_y, "已落地", fontsize=9, color="#388E3C", ha="right")

    fig.tight_layout(rect=[0, 0.12, 1, 0.95])
    fig.savefig(OUT / "fig7_gait_unit_architecture.png")
    plt.close(fig)
    print("  fig7_gait_unit_architecture.png")


# 图 8：清洗流水线四档处置 + 光捕校准结果
def fig8_disposition_and_calibration():
    """左：demo_dataset 四档处置饼图；右：LAFAN1 77 段校准本底报警率。"""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # (a) 处置档位
    dispositions = ["保留", "修复", "复核", "剔除"]
    counts = [24, 2, 2, 8]  # demo_dataset 实测（examples/quality_report.json summary.disposition）
    colors_disp = ["#4CAF50", "#2196F3", "#FF9800", "#F44336"]
    explode = (0, 0, 0.05, 0.1)

    wedges, texts, autotexts = ax1.pie(
        counts, labels=dispositions, colors=colors_disp,
        explode=explode, autopct="%1.1f%%", startangle=90,
        textprops={"fontsize": 11}
    )
    ax1.set_title("(a) 质检四档处置（demo_dataset 实测）", fontsize=12)

    # (b) LAFAN1 校准 - 本底报警率（数据源：reports/gait/lafan1_calibration.json，77 段 BVH 实测；
    #     口径 = 每千帧×关节被规则标记数（‰）。k=6 查全优先 / k=12 保守，两档对照）
    categories = ["push\n（低动态）", "walk\n（常速行走）", "dance\n（舞蹈）", "jumps\n（高动态跳跃）"]
    rate_k6 = [8.62, 21.32, 33.52, 45.29]
    rate_k12 = [5.01, 10.94, 24.22, 38.59]

    x = np.arange(len(categories))
    width = 0.35

    ax2.bar(x - width / 2, rate_k6, width, label="k=6（查全优先）",
            color="#90CAF9", edgecolor="white")
    ax2.bar(x + width / 2, rate_k12, width, label="k=12（保守）",
            color="#E57373", edgecolor="white")

    ax2.set_xticks(x)
    ax2.set_xticklabels(categories, fontsize=10)
    ax2.set_ylabel("本底报警率（‰ / 千帧×关节）")
    ax2.set_title("(b) LAFAN1 校准——正常数据本底报警率（77 段实测）", fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("图 8：处置策略与光捕模块校准结果", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig8_disposition_and_calibration.png")
    plt.close(fig)
    print("  fig8_disposition_and_calibration.png")


# main
if __name__ == "__main__":
    print(f"\n生成目录: {OUT.resolve()}\n")
    fig1_forest()
    fig2_bar_comparison()
    fig3_paired_scatter()
    fig4_diff_histogram()
    fig5_training_curves()
    fig6_data_volume()
    fig7_gait_unit_architecture()
    fig8_disposition_and_calibration()
    print(f"\n全部完成！共 8 张图 → {OUT.resolve()}/")
