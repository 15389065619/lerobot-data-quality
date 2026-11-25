# Copyright (c) 2026 wuliangzhong
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""内置清洗实现：删除 + 边界裁帧 + 时间轴重建 + 视频重编码。"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ..loader import (
    DEFAULT_DATA_PATH,
    DEFAULT_VIDEO_PATH,
    SOURCE_COL,
    LeRobotDatasetView,
)
from ..validator import probe_video_frames
from .meta import copy_tasks, rewrite_episodes_meta, rewrite_info, rewrite_stats, rewrite_stats_streaming
from .stats_streaming import StreamingStats, collect_feature_keys
from .types import CleaningResult
from .util import prepare_dir, set_column

LOG = logging.getLogger(__name__)

FFMPEG = shutil.which("ffmpeg")


def manual_clean(
    dataset: LeRobotDatasetView,
    kept: list[int],
    out_root: Path,
    *,
    overwrite: bool,
    trim_plan: dict[int, dict[str, int]],
    rebuild_ts: set[int],
    repairs: dict[str, list[int]],
    split_plan: dict[int, list[dict[str, Any]]] | None = None,
    window_plan: dict[int, dict[str, dict[str, float]]] | None = None,
    link_mode: str = "copy",
) -> tuple[CleaningResult, int]:
    split_plan = split_plan or {}
    window_plan = window_plan or {}
    prepare_dir(out_root, overwrite=overwrite)
    warnings: list[str] = []
    if FFMPEG is None and not split_plan and not window_plan:
        warnings.append("未找到 ffmpeg，视频文件将原样复制，可能残留已删除 episode 的画面（需后续重编码）。")

    # 0) 判定保留却一行都写不出的 episode 必须提前剔除：写入后再改编号映射会让
    #    parquet 里的 episode_index 与 meta 不一致，后续按行查元信息还会 KeyError。
    kept, empty_warnings = _surviving_episodes(dataset, kept, trim_plan)
    warnings.extend(empty_warnings)

    # 输出计划：普通 episode 一行；meta 多段的 episode 按段拆成多行（新编号连续）
    output_plan = _build_output_plan(kept, split_plan)

    # 1) 视频窗口（裁帧同步收窄；拆分取该段 meta 行；收窗按真实帧数重算 to）
    windows = _collect_video_windows(
        dataset, output_plan, trim_plan, split_plan, window_plan, warnings
    )

    # 2) 重写 data parquet（含拆分、裁帧、时间轴重建）
    data_meta, stats_acc, kept_lengths, renumber = _rewrite_frames(
        dataset, output_plan, out_root, trim_plan=trim_plan, rebuild_ts=rebuild_ts,
        split_plan=split_plan,
    )

    # 2.5) 收缩输出计划：读取失败 / 段超行数 / 裁到空的 plan 项不会出现在产物里。
    # 不收缩的话 _rebuild_videos 与 rewrite_episodes_meta 会对缺失的
    # kept_lengths[new] / data_meta[new] 直接 KeyError，让整次清洗 abort——
    # 该集本可降级为「跳过 + 告警」。重编号保持 new 连续（rewrite_info 的假设）。
    if len(renumber) != len(output_plan):
        for it in output_plan:
            if int(it["new"]) not in renumber:
                warnings.append(
                    f"episode old={it['old']}（计划输出编号 {it['new']}）未能写入产物"
                    "（数据行缺失或 meta 谎报长度），已剔除"
                )
        output_plan = [
            dict(it, new=renumber[int(it["new"])])
            for it in output_plan if int(it["new"]) in renumber
        ]
        # windows 以计划编号为键：重映射到实际编号，避免对已剔除集重建窗口
        windows = {renumber[int(k)]: v for k, v in windows.items() if int(k) in renumber}
        LOG.warning("输出计划收缩，剩余 %d 个待写 episode", len(output_plan))

    # 3) 重建视频
    new_video_meta, video_warnings = _rebuild_videos(
        dataset, output_plan, out_root, windows, kept_lengths, link_mode=link_mode
    )
    warnings.extend(video_warnings)

    # 4) 重建 meta
    rewrite_episodes_meta(dataset, output_plan, data_meta, new_video_meta, out_root)
    copy_tasks(dataset.root, out_root)
    total_frames = int(sum(m["length"] for m in data_meta.values()))
    rewrite_info(dataset, list(range(len(output_plan))), total_frames, out_root, force_v3=True)
    warnings.extend(rewrite_stats_streaming(dataset, stats_acc, out_root))

    ep_index_map: dict[int, int] = {}
    for item in output_plan:
        ep_index_map.setdefault(int(item["old"]), int(item["new"]))
    repaired = sorted(set(trim_plan) | set(rebuild_ts) | set(split_plan) | set(window_plan))
    result = CleaningResult(
        success=True,
        method="manual",
        output_root=out_root,
        removed_episodes=sorted(set(dataset.episode_indices) - set(kept)),
        repaired_episodes=repaired,
        repairs=repairs,
        kept_episodes=len(output_plan),
        episode_index_map=ep_index_map,
        warnings=warnings,
    )
    LOG.info("清洗完成：%d -> %d 个 episode（修复 %d 个），写入 %s",
             len(dataset.episode_indices), len(output_plan), len(repaired), out_root)
    return result, total_frames


def _build_output_plan(
    kept: list[int], split_plan: dict[int, list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    """按 kept 顺序生成输出 episode 计划；多段 episode 每段占一行。"""
    plan: list[dict[str, Any]] = []
    for old in kept:
        segments = split_plan.get(old)
        if segments:
            for seg_index, segment in enumerate(segments):
                plan.append({
                    "new": len(plan), "old": int(old), "seg": seg_index,
                    "length": int(segment["length"]),
                })
        else:
            plan.append({"new": len(plan), "old": int(old), "seg": None, "length": None})
    return plan


def copy_untouched(dataset: LeRobotDatasetView, out_root: Path, *, overwrite: bool) -> CleaningResult:
    prepare_dir(out_root, overwrite=overwrite)
    # 跳过 .cache：huggingface-cli 下载残留（可占数百 MB），不属于数据集内容
    shutil.copytree(
        dataset.root, out_root, dirs_exist_ok=True, ignore=shutil.ignore_patterns(".cache")
    )
    # 即使没有 episode 需要清洗，也要重算 stats，否则 stats_stale 会被原样带进训练
    warnings = rewrite_stats(dataset, dataset.frames, out_root)
    # 计数/splits 与源保持一致即可，布局原样保留（不强制升级 v3）
    rewrite_info(
        dataset, list(dataset.episode_indices), dataset.num_frames, out_root, force_v3=False
    )
    return CleaningResult(
        success=True,
        method="noop-copy",
        output_root=out_root,
        removed_episodes=[],
        kept_episodes=len(dataset.episode_indices),
        episode_index_map={e: e for e in dataset.episode_indices},
        warnings=warnings,
    )


def reindex_untouched(
    dataset: LeRobotDatasetView,
    out_root: Path,
    *,
    overwrite: bool,
    reason: str = "全局 index 不连续",
) -> CleaningResult:
    """不做删除/裁帧，只按真实行数重建 data/*.parquet 的全局 index 列。

    适用于「数据本身可用，只是 index 被逐 episode 重新编号」的情况（如 v2.1 导出）。
    走 manual_clean 会顺带重排 episode_index 并重编码视频，代价过高，因此单独实现：
    复制数据集后原地改写每个 parquet 的 index（缺失时一并补出 frame_index），
    视频不动，meta 只同步声明与 splits。
    """
    prepare_dir(out_root, overwrite=overwrite)
    shutil.copytree(
        dataset.root, out_root, dirs_exist_ok=True, ignore=shutil.ignore_patterns(".cache")
    )
    episodes, warnings = _rewrite_index_column(dataset, out_root)
    # 文件已原样复制，info 只同步计数/splits 与 frame_index 声明，布局保持源版本
    rewrite_info(dataset, list(dataset.episode_indices), dataset.num_frames, out_root)
    # index 与统计量无关，但清洗产物统一重算 stats，避免把 stats_stale 原样带进训练
    result_warnings = rewrite_stats(dataset, dataset.frames, out_root)

    result = CleaningResult(
        success=True,
        method="index-repair-copy",
        output_root=out_root,
        removed_episodes=[],
        repaired_episodes=episodes,
        repairs={"index": episodes},
        kept_episodes=len(dataset.episode_indices),
        episode_index_map={e: e for e in dataset.episode_indices},
        warnings=[
            f"{reason}：已按真实行数重建 {len(episodes)} 个 episode 的全局 index（视频未重编码）",
            *warnings,
            *result_warnings,
        ],
    )
    LOG.info("index 重建完成：%d 个 episode，写入 %s", len(episodes), out_root)
    return result


def _rewrite_index_column(
    dataset: LeRobotDatasetView, out_root: Path,
) -> tuple[list[int], list[str]]:
    """重写产出目录下所有 data parquet 的 index 列，返回 (涉及的 episode, 警告)。

    分两遍：先只读 episode_index 统计每个 episode 的真实行数并算出起始偏移，
    再逐文件读全表写回 index = 起始偏移 + episode 内的帧序号。
    """
    warnings: list[str] = []
    files: list[Path] = []
    for src in dataset.data_files:
        try:
            dst = out_root / src.relative_to(dataset.root)
        except ValueError:
            continue
        if dst.is_file():
            files.append(dst)
    if not files:
        warnings.append("未找到可重写的数据文件，index 未重建")
        return [], warnings

    counts: dict[int, int] = {}
    frame_ok: dict[int, bool] = {}
    frame_min: dict[int, int] = {}
    frame_max: dict[int, int] = {}
    unique_frame: dict[int, int] = {}
    readable: list[Path] = []

    for path in files:
        try:
            names = pq.ParquetFile(path).schema_arrow.names
            cols = ["episode_index"] + (["frame_index"] if "frame_index" in names else [])
            table = pq.read_table(path, columns=cols)
        except Exception as exc:  # noqa: BLE001 - 单个文件损坏不应中断其余文件的修复
            warnings.append(f"{path.name} 读取失败，index 未重建: {exc}")
            continue
        readable.append(path)
        eps = _index_values(table.column("episode_index").to_pylist())
        if eps.size == 0:
            continue
        uniq, cnt = np.unique(eps, return_counts=True)
        for e, n in zip(uniq, cnt):
            counts[int(e)] = counts.get(int(e), 0) + int(n)
        if "frame_index" not in table.column_names:
            for e in uniq:
                frame_ok[int(e)] = False
            continue
        frames = _index_values(table.column("frame_index").to_pylist())
        for e in uniq:
            pos = np.where(eps == e)[0]
            fv = frames[pos]
            finite = bool(np.isfinite(fv).all())
            uniq_f = np.unique(fv[np.isfinite(fv)])
            ep = int(e)
            unique_frame[ep] = unique_frame.get(ep, 0) + int(uniq_f.size)
            frame_min[ep] = min(frame_min.get(ep, int(uniq_f[0]) if uniq_f.size else 0),
                                int(uniq_f[0]) if uniq_f.size else 0)
            frame_max[ep] = max(frame_max.get(ep, int(uniq_f[-1]) if uniq_f.size else 0),
                                int(uniq_f[-1]) if uniq_f.size else 0)
            frame_ok[ep] = frame_ok.get(ep, True) and finite

    # 帧号可用于定位：每个 episode 的帧号必须不重不漏地铺满 0..len-1
    for ep, n in counts.items():
        frame_ok[ep] = (
            frame_ok.get(ep, False)
            and unique_frame.get(ep, 0) == n
            and frame_min.get(ep, 0) == 0
            and frame_max.get(ep, n - 1) == n - 1
        )

    starts: dict[int, int] = {}
    running = 0
    for ep in sorted(counts):
        starts[ep] = running
        running += counts[ep]

    offset: dict[int, int] = {}
    for path in readable:
        try:
            table = pq.read_table(path)
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"{path.name} 重写 index 失败: {exc}")
            continue
        eps = _index_values(table.column("episode_index").to_pylist())
        if eps.size == 0:
            continue
        frames = (_index_values(table.column("frame_index").to_pylist())
                  if "frame_index" in table.column_names else None)
        idx = np.empty(len(eps), dtype=np.int64)
        frame_out = np.empty(len(eps), dtype=np.int64)
        for e in np.unique(eps):
            ep = int(e)
            pos = np.where(eps == e)[0]
            if frame_ok.get(ep) and frames is not None:
                frame_pos = frames[pos]
            else:
                # frame_index 缺失或断裂时，退化为按文件内行序连续编号
                cur = offset.get(ep, 0)
                frame_pos = cur + np.arange(len(pos), dtype=np.int64)
                offset[ep] = cur + len(pos)
            frame_out[pos] = frame_pos
            idx[pos] = starts[ep] + frame_pos
        table = set_column(table, "index", idx)
        # v3.0 要求 frame_index 每集从 0 铺满 length-1；源数据缺列或断裂时一并重建，
        # 否则 index 修好了、episode 内帧号仍然无解，下游切窗依旧要自己猜。
        if "frame_index" in table.column_names:
            table = set_column(table, "frame_index", frame_out)
        else:
            table = table.append_column("frame_index", pa.array(frame_out, type=pa.int64()))
        try:
            pq.write_table(table, path)
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"{path.name} 写入失败，index 未重建: {exc}")

    return sorted(counts), warnings


def _index_values(values: list) -> np.ndarray:
    """把 parquet 列转成 int64；无效值记为 -1，保证仍能参与编号。"""
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=np.float64)
    return np.where(np.isfinite(arr), arr, -1).astype(np.int64)


def _episode_row_counts(dataset: LeRobotDatasetView) -> dict[int, int]:
    """每个 episode 在 data 中的实际行数。优先复用已加载的帧表，避免重复 IO。"""
    frames = dataset.frames
    if frames is not None and len(frames) and "episode_index" in frames.columns:
        values = pd.to_numeric(frames["episode_index"], errors="coerce").dropna().astype("int64")
        return {int(k): int(v) for k, v in values.value_counts().items()}
    counts: dict[int, int] = {}
    for src in _source_files_in_order(dataset):
        try:
            col = pq.read_table(src, columns=["episode_index"]).column("episode_index").to_pylist()
        except (pa.ArrowInvalid, KeyError):
            continue
        for value in pd.to_numeric(pd.Series(col), errors="coerce").dropna():
            counts[int(value)] = counts.get(int(value), 0) + 1
    return counts


def _surviving_episodes(
    dataset: LeRobotDatasetView, kept: list[int], trim_plan: dict[int, dict[str, int]]
) -> tuple[list[int], list[str]]:
    """剔除「判定保留但一行都写不出」的 episode，返回 (可用 keep, 提示)。"""
    counts = _episode_row_counts(dataset)
    survivors: list[int] = []
    warnings: list[str] = []
    for ep in kept:
        total = counts.get(ep, 0)
        trim = trim_plan.get(ep) or {}
        head = int(trim.get("head_frames", 0))
        tail = int(trim.get("tail_frames", 0))
        if total - head - tail > 0:
            survivors.append(ep)
            continue
        detail = (
            "在 data 中没有任何有效行"
            if total == 0
            else f"裁帧后没有任何有效行（共 {total} 帧，head {head} / tail {tail}）"
        )
        warnings.append(f"episode {ep} 判定为保留，但{detail}，已从清洗产物剔除。")
    return survivors, warnings


def _video_path(
    dataset: LeRobotDatasetView, camera: str, chunk_index: int, file_index: int
) -> Path | None:
    """按 (chunk, file) 定位源视频；优先用已扫描到的索引，回退到 info 模板。"""
    indexed = dataset.video_index.get(camera, {}).get((int(chunk_index), int(file_index)))
    if indexed is not None:
        return indexed
    template = dataset.info.get("video_path") or DEFAULT_VIDEO_PATH
    try:
        rel = template.format(
            video_key=camera, camera=camera, chunk_index=int(chunk_index),
            file_index=int(file_index), episode_index=int(file_index),
            episode_chunk=int(chunk_index),
        )
    except (KeyError, IndexError, ValueError, TypeError):
        return None
    return dataset.root / rel


def _collect_video_windows(
    dataset: LeRobotDatasetView,
    output_plan: list[dict[str, Any]],
    trim_plan: dict[int, dict[str, int]],
    split_plan: dict[int, list[dict[str, Any]]],
    window_plan: dict[int, dict[str, dict[str, float]]],
    warnings: list[str],
) -> dict[int, dict[str, dict[str, Any]]]:
    """{新 episode: {cam: {path, from, to}}}。

    普通集沿用 meta 窗口并按裁帧收窄；拆分集取对应分段的 meta 窗口；
    收窗修复只改 to_timestamp，不动数据行与视频文件。
    """
    fps = dataset.fps or 30.0
    out: dict[int, dict[str, dict[str, Any]]] = {}
    for item in output_plan:
        new = int(item["new"])
        old = int(item["old"])
        per_ep: dict[str, dict[str, Any]] = {}

        if item["seg"] is not None:
            segment = split_plan[old][int(item["seg"])]
            for cam, window in segment["videos"].items():
                path = _video_path(dataset, cam, window["chunk_index"], window["file_index"])
                if path is None or not path.is_file():
                    warnings.append(f"ep {old} 分段 {item['seg']} 相机 {cam} 视频文件无法定位")
                    continue
                per_ep[cam] = {
                    "path": path, "from": float(window["from"]), "to": float(window["to"]),
                }
            if per_ep:
                out[new] = per_ep
            continue

        trim = trim_plan.get(old) or {}
        head = int(trim.get("head_frames", 0))
        tail = int(trim.get("tail_frames", 0))
        narrowing = window_plan.get(old) or {}
        for cam in dataset.video_features:
            loc = dataset.video_location(cam, old)
            if loc is None:
                warnings.append(f"ep {old} 相机 {cam} 无法定位视频文件")
                continue
            path, from_ts, to_ts = loc
            if not path.is_file():
                warnings.append(f"ep {old} 相机 {cam} 视频文件缺失: {path.name}")
                continue
            if cam in narrowing:
                from_ts = float(narrowing[cam].get("from", from_ts))
                to_ts = float(narrowing[cam]["to"])
            elif (head or tail) and to_ts > from_ts >= 0:
                from_ts += head / fps
                to_ts -= tail / fps
            per_ep[cam] = {"path": path, "from": from_ts, "to": to_ts}
        if per_ep:
            out[new] = per_ep
    return out


def _episode_row_index(
    source_files: list[Path], kept_set: set[int]
) -> dict[int, list[tuple[int, np.ndarray]]]:
    """两遍扫描的第一遍：只读 episode_index 列，建立 {源 episode: [(文件序号, 行号数组)]}。

    内存里只留行号（约 8B/行），不驻留任何观测数据。编号列可能是字符串、
    含 NULL 或含小数，统一数值化后只接受非负整数——与 loader 的
    ``pd.to_numeric(errors="coerce")`` 容错口径一致，损坏行直接跳过而不是让
    ``int(None)`` 把整次清洗打崩。
    """
    index: dict[int, list[tuple[int, np.ndarray]]] = {}
    for file_no, src in enumerate(source_files):
        try:
            col = pq.read_table(src, columns=["episode_index"]).column("episode_index")
        except (pa.ArrowInvalid, KeyError, OSError) as exc:
            LOG.warning("扫描 %s 的 episode_index 失败，跳过: %s", src.name, exc)
            continue
        numeric = pd.to_numeric(
            pd.Series(col.to_pylist(), dtype=object), errors="coerce"
        ).to_numpy(dtype=np.float64)
        valid = np.isfinite(numeric)
        ids = np.where(valid, numeric, -1.0).astype(np.int64)
        valid &= (ids >= 0) & (ids == numeric)
        for raw_old in np.unique(ids[valid]):
            old = int(raw_old)
            if old not in kept_set:
                continue
            positions = np.where(valid & (ids == old))[0]
            if positions.size:
                index.setdefault(old, []).append((file_no, positions))
    return index


def _load_episode_rows(
    old: int,
    row_index: dict[int, list[tuple[int, np.ndarray]]],
    source_files: list[Path],
    file_cache: dict[int, pa.Table],
) -> pa.Table | None:
    """按物理行序取出一个源 episode 的全部行；最多驻留一个源文件的表。

    同一 old 被拆成多段时由调用方复用返回值，避免重复 IO。跨文件 schema
    有差异（列集合 / 类型不完全一致）时用 permissive 合并，缺失列补 null，
    否则 ``concat_tables`` 会因 schema 不同直接抛 ArrowInvalid。
    """
    entries = row_index.get(old)
    if not entries:
        return None
    parts: list[pa.Table] = []
    for file_no, positions in entries:
        table = file_cache.get(file_no)
        if table is None:
            try:
                table = pq.read_table(source_files[file_no])
            except (pa.ArrowInvalid, OSError) as exc:
                LOG.warning("读取 %s 失败，跳过: %s", source_files[file_no].name, exc)
                continue
            file_cache.clear()
            file_cache[file_no] = table
        parts.append(table.take(pa.array(positions, type=pa.int64())))
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    return pa.concat_tables(parts, promote_options="permissive")


def _rewrite_frames(
    dataset: LeRobotDatasetView,
    output_plan: list[dict[str, Any]],
    out_root: Path,
    *,
    trim_plan: dict[int, dict[str, int]],
    rebuild_ts: set[int],
    split_plan: dict[int, list[dict[str, Any]]],
) -> tuple[dict[int, dict[str, Any]], StreamingStats, dict[int, int]]:
    """按输出 episode 顺序重写 parquet；两遍扫描，内存只驻留一个 episode。

    第一遍只扫 episode_index 建行索引，第二遍按 output_plan 按需取行、逐集
    重写并及时写盘释放。不能假定源文件顺序等于 output_plan 顺序（episode
    编号是排序后的值），但也不能把全部源行收进内存——那会让 50GB 数据集
    在训练机上 OOM，与整条流水线的流式设计冲突。
    """
    # 清洗产物统一写 v3.0 布局，避免 v2 模板里的 {episode_index} 与 chunk/file 冲突
    template = DEFAULT_DATA_PATH
    fps = dataset.fps or 30.0
    kept_set = {int(item["old"]) for item in output_plan}
    rebuild_causes = set(rebuild_ts) | set(trim_plan) | set(split_plan)
    feature_keys = collect_feature_keys(dataset)
    stats_acc = StreamingStats(feature_shapes={
        key: tuple((dataset.features.get(key) or {}).get("shape") or [])
        for key in feature_keys
    })

    source_files = _source_files_in_order(dataset)
    row_index = _episode_row_index(source_files, kept_set)

    data_meta: dict[int, dict[str, Any]] = {}
    # 计划编号 -> 实际写入编号：读取失败/段超行数/裁到空的 plan 项会被跳过，
    # 跳过项不占号，保证产物编号连续 0..M-1（rewrite_info 依赖该假设）；
    # 调用方按 renumber 收缩 output_plan 并重映射视频窗口
    renumber: dict[int, int] = {}
    next_new = 0
    pending: list[pa.Table] = []
    pending_rows = 0
    file_index = 0
    global_index = 0
    rows_per_file = 20000  # 单个输出 parquet 的行数上限；episode 不跨文件

    def flush() -> None:
        nonlocal pending, pending_rows, file_index
        if not pending:
            return
        table = (pending[0] if len(pending) == 1
                 else pa.concat_tables(pending, promote_options="permissive"))
        rel = template.format(chunk_index=0, file_index=file_index)
        out_path = out_root / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, out_path)
        LOG.debug("写入 %s (%d 行)", out_path, table.num_rows)
        stats_acc.update_from_table(table, feature_keys)
        pending = []
        pending_rows = 0
        file_index += 1

    file_cache: dict[int, pa.Table] = {}
    last_old: int | None = None
    base_frame: pa.Table | None = None

    for item in output_plan:
        old = int(item["old"])
        plan_new = int(item["new"])
        if old not in row_index:
            LOG.warning("episode %s 在 data 中不存在，跳过", old)
            continue
        # split 的多段是连续的，复用同一份源行，避免每段都重新读一遍文件
        if old != last_old:
            base_frame = _load_episode_rows(old, row_index, source_files, file_cache)
            last_old = old
        if base_frame is None:
            LOG.warning("episode %s 的数据行读取失败，跳过", old)
            continue
        total = base_frame.num_rows
        if item["seg"] is not None:
            start = sum(
                int(seg["length"]) for seg in split_plan[old][:int(item["seg"])]
            )
            length = int(item["length"])
            if start + length > total:
                LOG.warning("episode %s 第 %s 段超出实际行数 %d，跳过",
                            old, item["seg"], total)
                continue
            frame = base_frame.slice(start, length)
        else:
            trim = trim_plan.get(old) or {}
            head = int(trim.get("head_frames", 0))
            tail = int(trim.get("tail_frames", 0))
            length = total - head - tail
            if length <= 0:
                LOG.warning("episode %s 裁帧后没有剩余行，跳过", old)
                continue
            frame = base_frame.slice(head, length) if (head or tail) else base_frame

        if frame.num_rows != length:
            LOG.warning("episode %s 实际取到 %d 行，预期 %d，跳过",
                        old, frame.num_rows, length)
            continue

        # 通过全部校验后才占号：实际写入编号连续，且与产物 data parquet 一致
        new = next_new
        next_new += 1
        renumber[plan_new] = new

        if pending and pending_rows + length > rows_per_file:
            flush()
        target_file = file_index
        frame_idx = np.arange(length, dtype=np.int64)
        index_values = np.arange(global_index, global_index + length, dtype=np.int64)
        frame = set_column(frame, "episode_index", np.full(length, new, dtype=np.int64))
        frame = set_column(frame, "frame_index", frame_idx)
        frame = set_column(frame, "index", index_values)
        if old in rebuild_causes:
            if "timestamp" in frame.column_names:
                ts = np.asarray(frame.column("timestamp").to_pylist(), dtype=np.float64)
            else:
                ts = np.zeros(length, dtype=np.float64)
            ts[:] = frame_idx / fps
            frame = set_column(frame, "timestamp", ts)
            LOG.debug("episode %d 重建了时间轴", new)

        data_meta[new] = {
            "length": int(length),
            "from_index": int(global_index),
            "to_index": int(global_index) + int(length),
            "file_index": int(target_file),
            "chunk_index": 0,
        }
        pending.append(frame)
        pending_rows += length
        global_index += length

    file_cache.clear()
    flush()
    kept_lengths = {ep: int(m["length"]) for ep, m in data_meta.items()}
    LOG.info("data parquet 重写完成：%d 个文件 / %d 行", file_index, global_index)
    return data_meta, stats_acc, kept_lengths, renumber


def _rebuild_videos(
    dataset: LeRobotDatasetView,
    output_plan: list[dict[str, Any]],
    out_root: Path,
    windows: dict[int, dict[str, dict[str, Any]]],
    kept_lengths: dict[int, int],
    *,
    link_mode: str = "copy",
) -> tuple[dict[int, dict[str, dict[str, Any]]], list[str]]:
    """按相机重建 mp4：优先整文件物化（窗口已指向真实帧），有删除时才重编码拼接。"""
    warnings: list[str] = []
    result: dict[int, dict[str, dict[str, Any]]] = {}
    # 输出统一 v3.0 布局；v2 的 episode_XXXXXX.mp4 模板无法用 chunk/file 格式化
    template = DEFAULT_VIDEO_PATH
    fps = dataset.fps or 30.0
    old_of_new = {int(item["new"]): int(item["old"]) for item in output_plan}

    for cam in dataset.video_features:
        src_to_eps: dict[Path, list[int]] = {}
        for ep in dataset.episode_indices:
            loc = dataset.video_location(cam, ep)
            if loc is not None and loc[0].is_file():
                src_to_eps.setdefault(loc[0], []).append(ep)
        src_to_new: dict[Path, list[int]] = {}
        for new in sorted(windows):
            info = windows[new].get(cam)
            if info is not None:
                src_to_new.setdefault(info["path"], []).append(new)

        out_index = 0
        # 以「源文件 -> 输出窗口」为主，同时补齐 meta 定位到的源文件（可能只有分段引用）
        all_sources = set(src_to_eps) | set(src_to_new)
        for src_path in sorted(all_sources, key=str):
            all_eps = src_to_eps.get(src_path, [])
            keep_here = sorted(src_to_new.get(src_path, []))
            if not keep_here:
                continue

            rel = template.format(video_key=cam, chunk_index=0, file_index=out_index)
            out_path = out_root / rel
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_index += 1

            def assign(entries: list[tuple[int, int]]) -> None:
                cum = 0
                for ep, frames in entries:
                    result.setdefault(ep, {})[cam] = {
                        "chunk_index": 0,
                        "file_index": out_index - 1,
                        "from": cum / fps,
                        "to": (cum + frames) / fps,
                    }
                    cum += frames

            # 源文件里的每个 episode 都仍有输出窗口（拆分集按段计、收窗集仍存在），
            # 说明没有任何画面需要删除——整文件物化即可，窗口保持指向真实帧，不重编码。
            covered_old = {old_of_new[new] for new in keep_here}
            if set(all_eps) <= covered_old:
                _materialize(src_path, out_path, link_mode)
                for new in keep_here:
                    info = windows[new][cam]
                    result.setdefault(new, {})[cam] = {
                        "chunk_index": 0,
                        "file_index": out_index - 1,
                        "from": float(info["from"]),
                        "to": float(info["to"]),
                    }
                continue

            segments = [
                (new, windows[new][cam]["from"], windows[new][cam]["to"])
                for new in keep_here
            ]
            if not FFMPEG or any(f < 0 or t <= f for _, f, t in segments):
                shutil.copy2(src_path, out_path)
                warnings.append(
                    f"相机 {cam} 的 {src_path.name} 含已删除 episode 且无法按时间戳切分，已原样复制（画面会残留脏 episode）。"
                )
                assign([(new, kept_lengths[new]) for new in keep_here])
                continue

            built = _concat_segments(src_path, segments, out_path, fps, warnings)
            assign(built)
            expected = sum(kept_lengths.get(new, 0) for new, _ in built)
            got = sum(f for _, f in built)
            if abs(got - expected) > len(built):
                warnings.append(
                    f"相机 {cam} 重建后总帧数 {got} 与 state 行数 {expected} 存在偏差，建议人工抽查该视频。"
                )
    return result, warnings


def _concat_segments(
    src_path: Path, segments: list[tuple[int, float, float]], out_path: Path, fps: float, warnings: list[str]
) -> list[tuple[int, int]]:
    """把若干 [from, to) 区间抽成独立片段，再无损拼接。返回 [(ep, 实际帧数)]。

    帧数必须精确等于 parquet 行数：LeRobotDataset 按窗口时间戳取帧，视频每少一帧，
    集尾样本就会读到下一集的第一帧（跨集画面污染）。因此这里不用 `-t 时长` 截断——
    帧 pts 是精确的 k/fps（无限循环小数），`-t` 的任何有限小数只要小于精确时长，
    就会把最后一帧排除在外（`430/30` 格化成 14.333333 必丢末帧、`485/30` 进位则不丢，
    历史上表现为「部分集各丢 1 帧」）。改用：
      - `-ss` 提前半帧定位：阈值落在 (上一帧 pts, 本帧 pts] 的安全区间中点，
        6 位小数的舍入误差（~3e-7）远小于半帧（1/(2·fps)），首帧不会跳过；
      - `-frames:v 期望帧数` 按帧数硬截断，完全不依赖浮点时长；
      - `-vf setpts=PTS-STARTPTS` 强制输出 pts 从 0 起：实测寻址后 ffmpeg 不重置
        起始时间戳，from_ts>0 的段首帧 pts 恰偏 1 帧（33ms@30fps，恒定不累计），
        超过 LeRobotDataset tolerance_s=0.01 直接 FrameTimestampError 崩溃。
    对时间戳不规则的损坏源，截断后帧数仍可能与期望不符——由调用方按探测值告警。
    """
    built: list[tuple[int, int]] = []
    with tempfile.TemporaryDirectory(prefix="lerobot_clean_") as tmp:
        tmp_dir = Path(tmp)
        seg_files: list[Path] = []
        for ep, from_ts, to_ts in segments:
            seg = tmp_dir / f"seg_{ep:06d}.mp4"
            expected = max(int(round((to_ts - from_ts) * fps)), 1)
            ss = max(from_ts - 0.5 / fps, 0.0)
            cmd = [
                str(FFMPEG), "-y", "-v", "error",
                "-ss", f"{ss:.6f}", "-i", str(src_path), "-frames:v", str(expected),
                "-an", "-vf", "setpts=PTS-STARTPTS",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                "-pix_fmt", "yuv420p", "-r", f"{fps:g}", str(seg),
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, check=False)
            if proc.returncode != 0 or not seg.is_file():
                warnings.append(f"切片失败 ep {ep} ({src_path.name}): {(proc.stderr or '')[-300:]}")
                continue
            frames = probe_video_frames(seg)
            if frames is None:
                warnings.append(
                    f"切片帧数探测失败 ep {ep} ({src_path.name})，按期望值 {expected} 记账，请人工核验。"
                )
                frames = expected
            elif frames != expected:
                warnings.append(
                    f"切片帧数不精确 ep {ep} ({src_path.name}): 期望 {expected} 实得 {frames}，"
                    "源视频时间戳可能不规则。"
                )
            built.append((ep, frames))
            seg_files.append(seg)

        if not seg_files:
            shutil.copy2(src_path, out_path)
            warnings.append(f"{src_path.name} 切片全部失败，已原样复制。")
            return built

        if len(seg_files) == 1:
            shutil.copy2(seg_files[0], out_path)
            return built

        list_file = tmp_dir / "concat.txt"
        list_file.write_text("\n".join(f"file '{p.as_posix()}'" for p in seg_files), encoding="utf-8")
        cmd = [
            str(FFMPEG), "-y", "-v", "error", "-f", "concat", "-safe", "0",
            "-i", str(list_file), "-c", "copy", "-movflags", "+faststart", str(out_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, check=False)
        if proc.returncode != 0 or not out_path.is_file():
            shutil.copy2(seg_files[0], out_path)
            warnings.append(f"视频拼接失败 ({out_path.name}): {(proc.stderr or '')[-300:]}")
            return built[:1]
    return built


def _materialize(src: Path, dst: Path, link_mode: str) -> None:
    """整文件不需要改动时按需复制或建硬链接（跨盘/权限不足时自动回退复制）。"""
    if link_mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError as exc:
            LOG.debug("硬链接失败(%s)，回退为复制: %s", exc, src)
    shutil.copy2(src, dst)


def _source_files_in_order(dataset: LeRobotDatasetView) -> list[Path]:
    # 按文件名排序，而不是「首次出现顺序」：frames 会按 (episode, frame_index) 排过序，
    # 同一集的两段可能让后一个文件先出现，按首次出现顺序拼接会打乱物理行序，
    # 拆分时就会把两段的行接错。这里与 loader.episode_frame_raw 的文件顺序保持一致。
    paths = ({Path(v) for v in dataset.frames[SOURCE_COL].dropna().tolist()}
             if SOURCE_COL in dataset.frames.columns else set())
    if not paths:
        # 流式模式（load_frames=False）下 frames 为空，此时必须回退到 data_files，
        # 否则 _surviving_episodes 会把所有 episode 判为「没有有效行」而产出空数据集。
        paths = set(dataset.data_files)
    return sorted(paths, key=lambda p: str(p))

