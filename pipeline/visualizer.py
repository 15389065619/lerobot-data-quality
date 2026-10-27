"""质检结果可视化。所有图输出到 outputs/plots/。

约束：大数据集只采样前 N 个 episode（默认 20），避免内存与渲染时间失控。
matplotlib 缺失时本模块降级为「跳过可视化」，不阻塞主流程。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .loader import ACTION_KEY, STATE_KEY, LeRobotDatasetView
from .validator import ValidationResult

LOG = logging.getLogger(__name__)

DEFAULT_SAMPLE = 20
COLOR_OK = "#2f6f9f"      # 正常：沉稳蓝
COLOR_BAD = "#c0392b"     # 异常：警示红
COLOR_NEUTRAL = "#8a8f98"


def _setup_matplotlib():
    try:
        import matplotlib
    except ImportError:  # pragma: no cover - 环境缺包时降级
        LOG.warning("未安装 matplotlib，跳过可视化（pip install matplotlib 后可生成图表）")
        return None
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 中文字体：Windows 优先微软雅黑，Linux/macOS 依次回退
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei", "SimHei", "PingFang SC", "Hiragino Sans GB",
        "WenQuanYi Micro Hei", "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.25
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    return plt


def generate_plots(
    dataset: LeRobotDatasetView,
    validation: ValidationResult,
    output_dir: str | Path,
    *,
    cleaning: dict[str, Any] | None = None,
    sample_size: int = DEFAULT_SAMPLE,
) -> dict[str, Path]:
    """生成 5 张图，返回 {名称: 路径}。"""
    plt = _setup_matplotlib()
    if plt is None:
        return {}

    out_dir = Path(output_dir) / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_eps = sorted(dataset.episode_indices)[:sample_size]
    invalid = set(validation.invalid_indices)
    paths: dict[str, Path] = {}

    paths["frame_count_distribution"] = _plot_frame_distribution(
        plt, dataset, sample_eps, invalid, out_dir / "01_frame_count_distribution.png"
    )
    paths["valid_vs_invalid"] = _plot_pie(
        plt, validation, out_dir / "02_valid_vs_invalid_pie.png"
    )
    paths["state_curves"] = _plot_state_curves(
        plt, dataset, validation, sample_eps, invalid, out_dir / "03_state_joint_curves.png"
    )
    paths["action_scatter"] = _plot_action_scatter(
        plt, dataset, sample_eps, invalid, out_dir / "04_action_scatter.png"
    )
    paths["before_after"] = _plot_before_after(
        plt, dataset, validation, cleaning, out_dir / "05_before_after_comparison.png"
    )
    LOG.info("已生成 %d 张图 -> %s", len(paths), out_dir)
    return paths


# ------------------------------------------------------------------- 图 1
def _plot_frame_distribution(plt, dataset: LeRobotDatasetView, sample_eps, invalid, path: Path) -> Path:
    counts = dataset.frames[dataset.frames["episode_index"].isin(sample_eps)].groupby("episode_index").size()
    ok_vals = [int(v) for e, v in counts.items() if int(e) not in invalid]
    bad_vals = [int(v) for e, v in counts.items() if int(e) in invalid]

    fig, ax = plt.subplots(figsize=(9, 5))
    lo = min(int(counts.min()), 0) if len(counts) else 0
    hi = max([int(counts.max()) if len(counts) else 100, 1])
    bins = np.linspace(lo, hi, min(30, max(6, hi - lo + 1)))
    ax.hist([ok_vals, bad_vals], bins=bins, stacked=True, color=[COLOR_OK, COLOR_BAD],
            label=[f"正常 ({len(ok_vals)})", f"异常 ({len(bad_vals)})"], edgecolor="white", linewidth=0.6)
    ax.set_xlabel("episode 帧数")
    ax.set_ylabel("episode 数量")
    ax.set_title(f"Episode 帧数分布（采样前 {len(sample_eps)} 个 episode）")
    ax.legend(frameon=False)
    _annotate(ax, f"均值 {counts.mean():.1f} 帧 / 中位数 {counts.median():.0f} 帧" if len(counts) else "")
    _save(fig, path)
    return path


# ------------------------------------------------------------------- 图 2
def _plot_pie(plt, validation: ValidationResult, path: Path) -> Path:
    total = validation.total_episodes
    valid = validation.valid_episodes
    invalid = total - valid
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    if total == 0:
        ax.text(0.5, 0.5, "无数据", ha="center", va="center")
        ax.axis("off")
    elif invalid == 0:
        ax.pie([1], colors=[COLOR_OK], startangle=90,
               wedgeprops={"width": 0.42, "edgecolor": "white"})
        ax.text(0, 0, f"全部正常\n{total} 个 episode", ha="center", va="center", fontsize=13)
    else:
        wedges, _, autotexts = ax.pie(
            [valid, invalid], colors=[COLOR_OK, COLOR_BAD], startangle=90,
            autopct=lambda p: f"{p:.1f}%", pctdistance=0.78,
            wedgeprops={"width": 0.42, "edgecolor": "white"},
            textprops={"color": "white", "fontsize": 11},
        )
        for t in autotexts:
            t.set_fontweight("bold")
        ax.text(0, 0, f"{total}\nepisodes", ha="center", va="center", fontsize=12)
        ax.legend(wedges, [f"正常 {valid}", f"异常 {invalid}"], loc="lower center",
                  bbox_to_anchor=(0.5, -0.08), ncol=2, frameon=False)
    ax.set_title("正常 vs 异常 episode 占比")
    if total:
        sev = getattr(validation, "severity_summary", lambda: {})()
        if sev:
            ax.text(0.5, -0.12, f"error {sev.get('error', 0)} / warning {sev.get('warning', 0)} / "
                                f"info {sev.get('info', 0)}",
                    transform=ax.transAxes, ha="center", fontsize=9, color="#6b7280")
    _save(fig, path)
    return path


# ------------------------------------------------------------------- 图 3
def _plot_state_curves(plt, dataset: LeRobotDatasetView, validation: ValidationResult,
                       sample_eps, invalid, path: Path) -> Path:
    if STATE_KEY not in dataset.frames.columns or not sample_eps:
        return _placeholder(plt, path, "数据中不存在 observation.state，无法绘制关节曲线")

    normal_ep = next((e for e in sample_eps if e not in invalid), None)
    bad_ep = next((e for e in sample_eps if e in invalid), None)
    targets = [("正常样例", normal_ep, COLOR_OK), ("异常样例", bad_ep, COLOR_BAD)]
    targets = [(label, ep, color) for label, ep, color in targets if ep is not None]
    if not targets:
        return _placeholder(plt, path, "采样范围内没有可用 episode")

    names = dataset.joint_names
    dim = min(len(names), 7)
    fig, axes = plt.subplots(1, len(targets), figsize=(6.4 * len(targets), 4.8), squeeze=False)
    for ax, (label, ep, color) in zip(axes[0], targets):
        df = dataset.episode_frame(ep)
        try:
            arr = np.asarray(df[STATE_KEY].to_list(), dtype=float)
        except (TypeError, ValueError):
            return _placeholder(plt, path, "observation.state 无法解析为数值矩阵")
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        x = np.arange(arr.shape[0])
        for d in range(dim):
            ax.plot(x, arr[:, d], linewidth=1.3, alpha=0.9, label=names[d] if len(targets) == 1 else None)
        reasons = ",".join(validation.results[ep].reasons) if ep in validation.results else ""
        title = f"{label} · ep {ep} · {arr.shape[0]} 帧"
        if reasons:
            title += f"\n{reasons}"
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("frame")
        ax.set_ylabel("state")
        if len(targets) == 1:
            ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.suptitle("observation.state 关节曲线对比", fontsize=13)
    fig.tight_layout()
    _save(fig, path)
    return path


# ------------------------------------------------------------------- 图 4
def _plot_action_scatter(plt, dataset: LeRobotDatasetView, sample_eps, invalid, path: Path) -> Path:
    if ACTION_KEY not in dataset.frames.columns or not sample_eps:
        return _placeholder(plt, path, "数据中不存在 action，无法绘制散点图")

    sub = dataset.frames[dataset.frames["episode_index"].isin(sample_eps)]
    try:
        arr = np.asarray(sub[ACTION_KEY].to_list(), dtype=float)
    except (TypeError, ValueError):
        return _placeholder(plt, path, "action 无法解析为数值矩阵")
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.shape[1] < 2:
        return _placeholder(plt, path, "action 维度不足 2，无法绘制二维散点")

    is_bad = sub["episode_index"].isin(invalid).to_numpy()
    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    ax.scatter(arr[~is_bad, 0], arr[~is_bad, 1], s=8, alpha=0.55, c=COLOR_OK,
               label=f"正常帧 ({int((~is_bad).sum())})", linewidths=0)
    if is_bad.any():
        ax.scatter(arr[is_bad, 0], arr[is_bad, 1], s=12, alpha=0.8, c=COLOR_BAD,
                   label=f"异常帧 ({int(is_bad.sum())})", linewidths=0)
    ax.set_xlabel("action[0]")
    ax.set_ylabel("action[1]")
    ax.set_title(f"Action 分布（前 {len(sample_eps)} 个 episode，共 {len(arr)} 帧）")
    ax.legend(frameon=False)
    _save(fig, path)
    return path


# ------------------------------------------------------------------- 图 5
def _plot_before_after(plt, dataset: LeRobotDatasetView, validation: ValidationResult,
                       cleaning: dict[str, Any] | None, path: Path) -> Path:
    before_eps = validation.total_episodes
    before_frames = dataset.num_frames
    after_eps = validation.valid_episodes
    after_frames = None
    if cleaning:
        kept = cleaning.get("kept_episodes")
        if isinstance(kept, int):
            after_eps = kept
        # 以清洗产物的实测帧数为准（修复会改变帧数，不能简单用「总数 - 删除数」）
        verification = cleaning.get("verification") or {}
        if isinstance(verification.get("frames"), int):
            after_frames = verification["frames"]
        else:
            removed = set(cleaning.get("removed_episode_indices") or [])
            if removed:
                after_frames = int(
                    dataset.frames[~dataset.frames["episode_index"].isin(removed)].shape[0]
                )

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))

    ax = axes[0]
    bars = ax.bar(["清洗前", "清洗后"], [before_eps, after_eps], color=[COLOR_NEUTRAL, COLOR_OK], width=0.55)
    ax.set_title("Episode 数量对比")
    ax.set_ylabel("episode")
    for b, v in zip(bars, [before_eps, after_eps]):
        ax.text(b.get_x() + b.get_width() / 2, v, str(v), ha="center", va="bottom", fontsize=11)
    if before_eps:
        ax.text(0.5, 0.92, f"删除 {before_eps - after_eps} 个 ({(before_eps - after_eps) / before_eps:.1%})",
                transform=ax.transAxes, ha="center", fontsize=10, color=COLOR_BAD)

    ax = axes[1]
    frames_pair = [before_frames, after_frames if after_frames is not None else before_frames]
    bars = ax.bar(["清洗前", "清洗后"], frames_pair, color=[COLOR_NEUTRAL, COLOR_OK], width=0.55)
    ax.set_title("总帧数对比")
    ax.set_ylabel("frames")
    for b, v in zip(bars, frames_pair):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,}", ha="center", va="bottom", fontsize=10)
    if after_frames is None:
        ax.text(0.5, 0.02, "未执行清洗，帧数未变化", transform=ax.transAxes, ha="center", fontsize=9, color=COLOR_NEUTRAL)

    fig.suptitle("清洗前后数据规模对比", fontsize=13)
    fig.tight_layout()
    _save(fig, path)
    return path


# ------------------------------------------------------------------- 工具
def _annotate(ax, text: str) -> None:
    if text:
        ax.text(0.99, 0.97, text, transform=ax.transAxes, ha="right", va="top",
                fontsize=9, color="#4a4f57",
                bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "#d8dbe0", "alpha": 0.9})


def _placeholder(plt, path: Path, message: str) -> Path:
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.axis("off")
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=12, color="#6b7280")
    _save(fig, path)
    return path


def _save(fig, path: Path) -> None:
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    fig.clf()
    import matplotlib.pyplot as plt

    plt.close(fig)
