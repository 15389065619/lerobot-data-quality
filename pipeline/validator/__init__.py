"""逐 episode 数据质检。

设计取向（对齐真实数据工程）：
1. **阈值自适应**：帧数区间、action 跳变阈值都由数据集自身分布（median + k·MAD）推导，
   配置里的 abs_floor / floor 只是安全下限。不同机器人的关节单位不同，绝对阈值必然误报。
2. **严重度分级**：error（不可用）/ warning（需人审或可修）/ info（仅记录），
   而不是所有异常一律删除——处置动作由 policy 层决定。
3. **可修性判定**：时间戳错乱、NaN 仅出现在首尾、meta 索引漂移都属于「可修复」，
   在细节里给出修复所需参数（如 head_frames / tail_frames），交由 cleaner 执行。
4. **性能**：视频探针优先读容器元数据（nb_frames / duration），仅在必要时全解码；
   黑屏检测一次 ffmpeg 调用抽多帧，避免逐帧 subprocess。

规则清单见 RULES。类型定义在 types，探针在 probes，数据集级上下文在 context，
逐条规则在 rules。
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import pandas as pd

from ..config import cfg_get, load_config
from ..loader import ACTION_KEY, STATE_KEY, LeRobotDatasetView
from .context import DatasetContext
from .probes import FFMPEG, FFPROBE, probe_frame_brightness, probe_video_frames
from .rules import (
    as_matrix,
    check_action_jump,
    check_blank_video,
    check_duplicate,
    check_finite,
    check_frame_count,
    check_frame_mismatch,
    check_joint_limits,
    check_meta_consistency,
    check_static,
    check_task,
    check_timestamp,
    frame_column,
)
from .types import (
    RULE_MAP,
    RULE_ORDER,
    RULES,
    SEVERITY_ORDER,
    DatasetIssue,
    EpisodeResult,
    Rule,
    ValidationResult,
)

LOG = logging.getLogger(__name__)

__all__ = [
    "DatasetIssue",
    "EpisodeResult",
    "RULE_MAP",
    "RULE_ORDER",
    "RULES",
    "Rule",
    "SEVERITY_ORDER",
    "ValidationResult",
    "probe_frame_brightness",
    "probe_video_frames",
    "validate_dataset",
]


# ------------------------------------------------------------------ 主入口
def validate_dataset(
    dataset: LeRobotDatasetView,
    config: dict[str, Any] | None = None,
    *,
    prior: Mapping[int, EpisodeResult] | None = None,
    only: Iterable[int] | None = None,
    streaming: bool = False,
) -> ValidationResult:
    """逐 episode 质检。

    Args:
        prior: 上一次的结论；增量模式下未变更的 episode 直接复用，避免重复视频探针。
        only: 只重新质检这些 episode，其余使用 prior；None 表示全部重检。
        streaming: 按 parquet 文件分片扫描，不在内存里保留整张逐帧表（大数据集用）。
    """
    if streaming:
        from .streaming import validate_dataset_streaming

        return validate_dataset_streaming(dataset, config, prior=prior, only=only)

    config = config or load_config(robot_type=dataset.info.get("robot_type"))
    skipped: dict[str, str] = {}
    ctx = DatasetContext.build(dataset, config, skipped)

    results: dict[int, EpisodeResult] = {}
    reused = 0
    blank_enabled = bool(cfg_get(config, "blank_video.enabled", True)) and FFMPEG is not None
    if not blank_enabled:
        skipped["blank_video"] = "未找到 ffmpeg 或已禁用视频内容检测"
    precise = bool(cfg_get(config, "probe.precise", False))
    if not FFPROBE:
        skipped["frame_mismatch"] = "未找到 ffprobe，跳过视频帧对齐"
    if not dataset.video_features:
        skipped["frame_mismatch"] = "数据集中没有视频特征"

    cameras = dataset.video_features[: int(cfg_get(config, "probe.max_cameras", 3))]
    max_probes = int(cfg_get(config, "blank_video.max_probes", 400))
    revalidate = None if only is None else {int(ep) for ep in only}

    # 第一阶段：非视频规则 + 决定哪些 episode 需要复用历史结论
    pending_video: list[tuple[EpisodeResult, int, bool]] = []
    for ep_index, ep_df in dataset.frames.groupby("episode_index", sort=True):
        ep_index = int(ep_index)
        if revalidate is not None and ep_index not in revalidate and prior and ep_index in prior:
            res = EpisodeResult.from_dict(prior[ep_index].to_dict())
            _refresh_duplicate(res, ep_df, ctx, config)
            results[ep_index] = res
            reused += 1
            continue

        res = _validate_non_video(ep_df, ep_index, dataset, ctx, config, skipped)
        results[ep_index] = res
        want_blank = blank_enabled and len(pending_video) < max_probes
        pending_video.append((res, ep_index, want_blank))

    # 第二阶段：视频探针（进程等待为主，但有界并发；机械盘上默认只开 2 路）
    _run_video_checks(pending_video, dataset, cameras, precise, config)

    if reused:
        LOG.info("增量质检：复用 %d 个 episode 的历史结论，重检 %d 个", reused, len(results) - reused)

    result = ValidationResult(
        total_episodes=len(results),
        results=results,
        config=config,
        fps=dataset.fps,
        dataset_root=str(dataset.root),
        resolved_thresholds=ctx.resolved,
        dataset_issues=ctx.issues,
        skipped_checks=skipped,
        probe_available=bool(FFPROBE),
    )
    sev = result.severity_summary()
    LOG.info(
        "质检完成：%d 个 episode，正常 %d / 异常 %d（error %d, warning %d, info %d）",
        result.total_episodes, result.valid_episodes, len(result.invalid_episodes),
        sev.get("error", 0), sev.get("warning", 0), sev.get("info", 0),
    )
    return result


def _validate_non_video(
    ep_df: pd.DataFrame,
    ep_index: int,
    dataset: LeRobotDatasetView,
    ctx: DatasetContext,
    config: dict[str, Any],
    skipped: dict[str, str],
) -> EpisodeResult:
    res = EpisodeResult(episode_index=ep_index, frame_count=int(len(ep_df)))
    frame_col = frame_column(ep_df)

    if STATE_KEY in ep_df.columns:
        arr_s = as_matrix(ep_df[STATE_KEY], res, "nan_state")
        if arr_s is not None:
            check_finite(arr_s, "nan_state", res, frame_col, config)
            check_static(arr_s, "state", res, config)
            check_joint_limits(arr_s, res, config, dataset)
    else:
        skipped.setdefault("nan_state", f"不存在 {STATE_KEY} 列")

    if ACTION_KEY in ep_df.columns:
        arr_a = as_matrix(ep_df[ACTION_KEY], res, "nan_action")
        if arr_a is not None:
            check_finite(arr_a, "nan_action", res, frame_col, config)
            check_action_jump(arr_a, res, ctx.action_thresholds, config)
            check_static(arr_a, "action", res, config)
            check_duplicate(res, arr_a, ctx, config)
    else:
        skipped.setdefault("action_jump", f"不存在 {ACTION_KEY} 列")

    check_frame_count(res, ctx.frame_bounds, config)
    check_timestamp(ep_df, res, config, dataset.fps)
    check_meta_consistency(res, ep_index, dataset)
    check_task(res, ep_index, dataset)
    return res


def _refresh_duplicate(
    res: EpisodeResult, ep_df: pd.DataFrame, ctx: DatasetContext, config: dict[str, Any]
) -> None:
    """复用结论时只重算重复项：它依赖全局，其他 episode 增删会影响判定。"""
    if ACTION_KEY not in ep_df.columns:
        return
    try:
        arr = np.asarray(ep_df[ACTION_KEY].to_list(), dtype=float)
    except (TypeError, ValueError):
        return
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    res.reasons = [code for code in res.reasons if code != "duplicate_episode"]
    res.details.pop("duplicate_episode", None)
    check_duplicate(res, arr, ctx, config)


def _run_video_checks(
    pending: list[tuple[EpisodeResult, int, bool]],
    dataset: LeRobotDatasetView,
    cameras: list[str],
    precise: bool,
    config: dict[str, Any],
) -> None:
    if not pending:
        return
    workers = max(1, int(cfg_get(config, "probe.max_workers", 2)))
    workers = min(workers, len(pending), os.cpu_count() or 1)

    def task(item: tuple[EpisodeResult, int, bool]) -> None:
        res, ep_index, want_blank = item
        check_frame_mismatch(res, ep_index, dataset, cameras, precise, config)
        if want_blank:
            check_blank_video(res, ep_index, dataset, cameras[0] if cameras else None, config)

    if workers <= 1:
        for item in pending:
            task(item)
        return
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="video-probe") as pool:
        for future in [pool.submit(task, item) for item in pending]:
            future.result()
