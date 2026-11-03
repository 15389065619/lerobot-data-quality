"""生成 README 顶部的流水线架构动图（docs/architecture.gif）。

用法：
    python tools/make_architecture_gif.py

只用 matplotlib + Pillow（都已在 requirements 中），不依赖 graphviz / mermaid-cli
/ 网络，因此 CI 与离线环境都能复现同一张图。
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

LOG = logging.getLogger(__name__)

# 每个节点 = (标题, 副标题)。顺序即数据流方向（与 ARCHITECTURE.md / main.py 一致）
STAGES: list[tuple[str, str]] = [
    ("输入", "LeRobot v3\n数据集"),
    ("① loader", "容错加载\n坏文件跳过"),
    ("② validator", "17+8 条规则\n自适应阈值"),
    ("③ policy", "Keep / Repair\nReview / Drop"),
    ("④ reporter", "报告 + 复核队列\n阈值快照"),
    ("⑤ cleaner", "删除 + 修复\n契约不变"),
    ("⑥⑦ 点云", "质检 / 清洗\n定长补点"),
    ("⑧ visualizer", "复核可视化\n人工闭环"),
    ("输出", "干净数据集\n+ 报告"),
]

BG = "#0d1117"        # 深色底
COLOR_PENDING = "#26303d"
COLOR_DONE = "#1f4e6b"
COLOR_ACTIVE = "#3fa9d6"
COLOR_TEXT = "#e6edf3"
COLOR_SUBTEXT = "#9aa7b4"
COLOR_EDGE = "#3fa9d6"

BOX_W = 0.88
BOX_H = 0.86
Y = 1.0


def _draw_frame(ax, active: int, show_feedback: bool) -> None:
    """重绘一帧：已走过的节点=完成色，当前节点=高亮，未到=待执行。"""
    ax.clear()
    ax.set_xlim(-0.08, len(STAGES) + 0.08)
    ax.set_ylim(0, 2)
    ax.axis("off")
    ax.set_facecolor(BG)

    for i, (title, sub) in enumerate(STAGES):
        x = i + 0.5
        if i < active:
            face, edge, lw = COLOR_DONE, COLOR_DONE, 1.0
        elif i == active:
            face, edge, lw = COLOR_ACTIVE, "#8fd6f5", 2.6
        else:
            face, edge, lw = COLOR_PENDING, COLOR_PENDING, 1.0

        ax.add_patch(
            FancyBboxPatch(
                (x - BOX_W / 2, Y - BOX_H / 2), BOX_W, BOX_H,
                boxstyle="round,pad=0.02,rounding_size=0.06",
                facecolor=face, edgecolor=edge, linewidth=lw, zorder=2,
            )
        )
        # 高亮节点加一圈外发光，模拟「正在处理」的效果
        if i == active:
            ax.add_patch(
                FancyBboxPatch(
                    (x - BOX_W / 2 - 0.05, Y - BOX_H / 2 - 0.05), BOX_W + 0.1, BOX_H + 0.1,
                    boxstyle="round,pad=0.02,rounding_size=0.08",
                    facecolor="none", edgecolor=COLOR_ACTIVE, linewidth=1.0, alpha=0.35, zorder=1,
                )
            )

        text_color = COLOR_TEXT if i <= active else COLOR_SUBTEXT
        ax.text(x, Y + 0.14, title, ha="center", va="center",
                color=text_color, fontsize=10, fontweight="bold", zorder=3)
        ax.text(x, Y - 0.13, sub, ha="center", va="center",
                color=COLOR_SUBTEXT if i > active else COLOR_TEXT,
                fontsize=7.2, linespacing=1.5, zorder=3)

        if i < len(STAGES) - 1:
            passed = i < active
            ax.add_patch(
                FancyArrowPatch(
                    (x + BOX_W / 2 + 0.02, Y), (x + 0.5 - BOX_W / 2 - 0.02, Y),
                    arrowstyle="-|>", mutation_scale=9, linewidth=1.4,
                    color=COLOR_EDGE if passed else "#33404f",
                    alpha=0.95 if passed else 0.6, zorder=2,
                )
            )

    # 人工复核回灌：reporter(④) 虚线回指 policy(③)，形成闭环
    if show_feedback:
        bottom = Y - BOX_H / 2
        loop_y = 0.30
        ax.add_patch(FancyArrowPatch((4.5, bottom), (4.5, loop_y),
                                     arrowstyle="-", linewidth=1.1,
                                     linestyle="--", color="#d9a441", zorder=2))
        ax.add_patch(FancyArrowPatch((4.5, loop_y), (3.5, loop_y),
                                     arrowstyle="-", linewidth=1.1,
                                     linestyle="--", color="#d9a441", zorder=2))
        ax.add_patch(FancyArrowPatch((3.5, loop_y), (3.5, bottom),
                                     arrowstyle="-|>", mutation_scale=9, linewidth=1.1,
                                     linestyle="--", color="#d9a441", zorder=2))
        ax.text(4.0, loop_y - 0.135, "review_queue.csv 人工回灌",
                ha="center", va="center", color="#d9a441", fontsize=6.5)

    ax.text(0.03, 1.86, "lerobot-data-quality  ·  8 步数据治理流水线",
            ha="left", va="center", color="#6f7d8c", fontsize=8)


def build_gif(output: str | Path, *, hold_frames: int = 4, fps: int = 2) -> Path:
    """逐帧高亮 9 个节点，最后停在完整架构上。返回写入的 gif 路径。"""
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei", "SimHei", "PingFang SC", "Hiragino Sans GB",
        "WenQuanYi Micro Hei", "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(12.6, 2.6), dpi=100)
    fig.patch.set_facecolor(BG)
    fig.subplots_adjust(left=0.004, right=0.996, top=1.0, bottom=0.0)

    total = len(STAGES) + hold_frames

    def _update(frame: int):
        active = min(frame, len(STAGES) - 1)
        _draw_frame(ax, active, show_feedback=frame >= len(STAGES) - 1 + hold_frames - 2)
        return ax,

    anim = FuncAnimation(fig, _update, frames=total, interval=int(1000 / fps), blit=False)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(path), writer=PillowWriter(fps=fps), dpi=100)
    plt.close(fig)
    LOG.info("已生成 %s（%d 帧，%.1f KB）", path, total, path.stat().st_size / 1024)
    return path


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description="生成 README 架构动图")
    ap.add_argument("--output", default="docs/architecture.gif", help="输出路径（默认 docs/architecture.gif）")
    ap.add_argument("--fps", type=int, default=2, help="帧率（默认 2，即每步停留 0.5 秒）")
    args = ap.parse_args()
    build_gif(args.output, fps=args.fps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
