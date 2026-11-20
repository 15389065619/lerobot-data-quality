"""清洗流程共用的文件系统与表操作工具。"""

from __future__ import annotations

import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa

from ..loader import load_dataset

LOG = logging.getLogger(__name__)


def assert_safe_output(src_root: Path, out_root: Path) -> None:
    src_root = src_root.resolve()
    out_root = out_root.resolve()
    if out_root == src_root or src_root in out_root.parents:
        raise ValueError(f"输出目录 {out_root} 位于源数据集 {src_root} 内部，拒绝写入（源数据集必须保持只读）")
    if out_root in src_root.parents:
        raise ValueError(f"输出目录 {out_root} 包含源数据集 {src_root}，发布旧产物时会连带删除源数据集，"
                         "拒绝写入（源数据集必须保持只读）")


def prepare_dir(out_root: Path, *, overwrite: bool) -> None:
    if out_root.exists():
        if not overwrite:
            raise FileExistsError(f"输出目录已存在: {out_root}")
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)


def staging_dir(final_root: Path) -> Path:
    """返回与最终目录同级的暂存目录（同一磁盘，rename 才可能是原子的）。

    命名带 pid 与时间戳，避免两个进程共用同一个暂存目录；不以点开头，
    因为该名字可能被传给 `lerobot-edit-dataset` 的 repo-id。
    """
    stamp = f"{os.getpid()}-{int(time.time() * 1000)}"
    return final_root.with_name(f"{final_root.name}.tmp-{stamp}")


def publish_dir(stage: Path, dest: Path) -> None:
    """把已写完并校验通过的暂存目录发布到最终路径。

    Windows 上 `os.replace` 不能覆盖非空目录，因此先把旧目录改名为 `.prev-*`，
    新目录改名到位后再删除旧目录；新目录改名失败则把旧目录退回原位。
    这样任何一步失败，最终路径上要么是完整的旧产物，要么是完整的新产物。
    """
    previous: Path | None = None
    if dest.exists():
        previous = dest.with_name(f"{dest.name}.prev-{int(time.time() * 1000)}")
        dest.rename(previous)
    try:
        stage.rename(dest)
    except BaseException:
        if previous is not None and previous.exists() and not dest.exists():
            previous.rename(dest)
        raise
    if previous is not None:
        shutil.rmtree(previous, ignore_errors=True)


def set_column(table: pa.Table, name: str, values: np.ndarray | list) -> pa.Table:
    """保持原 arrow 类型替换列；列不存在时 append 新列（不再静默 noop）。"""
    idx = table.schema.get_field_index(name)
    if idx < 0:
        # 列不存在：append 新列。manual.py 的「重建时间轴」路径曾因 timestamp
        # 列缺失误入此处并静默跳过，导致 ts 数组白算。这里改为显式 append + warning。
        LOG.warning("set_column: %s 不在 schema 中，追加新列（原文档承诺「原样返回」已废弃）", name)
        arr = pa.array(values)
        return table.append_column(name, arr)
    field = table.schema.field(idx)
    try:
        arr = pa.array(values, type=field.type)
    except (pa.lib.ArrowInvalid, pa.lib.ArrowNotImplementedError, TypeError) as exc:
        LOG.warning("列 %s 类型转换失败(%s)，尝试无类型推断", name, exc)
        arr = pa.array([float(v) if isinstance(v, np.floating) else v for v in values])
    return table.set_column(idx, field, arr)


def verify(out_root: Path, expected_episodes: int, expected_frames: int | None) -> dict[str, Any]:
    try:
        view = load_dataset(out_root)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"重新加载产出物失败: {exc}"}
    ok = view.num_episodes == expected_episodes and (expected_frames is None or view.num_frames == expected_frames)
    if view.num_episodes != expected_episodes:
        LOG.warning("校验：产出 %d 个 episode，预期 %d", view.num_episodes, expected_episodes)
    contiguous = view.episode_indices == list(range(view.num_episodes))
    result: dict[str, Any] = {
        "ok": bool(ok and contiguous),
        "episodes": view.num_episodes,
        "frames": view.num_frames,
        "episode_indices_contiguous": contiguous,
    }
    # 逐集断言修复结果：帧号连续、时间轴步长、视频窗口跨度与文件时长。
    # 这一步必须能失败，否则错误修复（例如只改了 data 没改 meta 窗口）会静默通过。
    try:
        violations = _verify_structure(view)
        result["structure_violations"] = violations
        if violations:
            LOG.warning("校验：产出物有 %d 条结构违规", len(violations))
            result["ok"] = False
    except Exception as exc:  # noqa: BLE001
        LOG.warning("校验：结构复验执行失败（%s）", exc)
        result["structure_violations"] = [f"结构复验执行失败: {exc}"]
        result["ok"] = False
    # 结构之外再重跑一遍规则引擎：修复本身可能引入新问题（如裁剪后时间窗错位），
    # 只验结构查不出来。strict=False 保证复检是告警性质，不阻断发布流程。
    try:
        from ..precheck import precheck

        recheck = precheck(out_root, strict=False)
        result["rule_recheck"] = {
            "ok": recheck.ok,
            "keep": recheck.keep,
            "repair": recheck.repair,
            "review": recheck.review,
            "drop": recheck.drop,
            "blocking": recheck.blocking,
            "message": recheck.message,
        }
        if not recheck.ok:
            LOG.warning("复检：清洗产物存在 %d 个 drop 级 episode（修复可能引入了新问题）", recheck.drop)
            result["ok"] = False
    except Exception as exc:  # noqa: BLE001
        # 复检自身失败不吞掉校验结论，只降级为告警
        LOG.warning("清洗后规则复检执行失败，跳过（%s）", exc)
        result["rule_recheck"] = {"ok": None, "error": str(exc)}
    return result


def _verify_structure(view) -> list[str]:
    """逐集复验帧号 / 时间轴 / meta length / 视频窗口，返回违规描述列表。"""
    from ..validator.probes import probe_video_frames

    violations: list[str] = []
    fps = float(view.fps or 30.0)
    frame_tolerance = 1.5 / fps
    # 时间轴步长：重建过的集必然精确等于 1/fps；未重建的集保留的是原始采集
    # 时间戳，有亚帧级抖动是正常的。LeRobot 自己用 tolerance_s=0.01 匹配帧
    # 时间戳，早期版本用 1ms 绝对阈值会把 ±2ms 抖动的合法数据判成违规，
    # 而且对 120Hz 数据（1/fps≈8.3ms）相当于 12% 的相对误差。这里改用
    # 「半帧」容差：既能抓到丢帧 / 重复帧这类真异常，又不会误伤正常抖动。
    step_tolerance = min(0.01, 0.5 / fps)
    duration_cache: dict[Path, int | None] = {}

    for ep in view.episode_indices:
        meta = view.meta_row(ep)
        if meta is None:
            violations.append(f"ep {ep}: 缺少 meta 行")
            continue
        ep_df = view.episode_frame(ep)
        rows = len(ep_df)
        length = int(meta.get("length", -1))
        if rows != length:
            violations.append(f"ep {ep}: meta length={length} 与 data 行数 {rows} 不一致")
        if rows == 0:
            violations.append(f"ep {ep}: 没有任何数据行")
            continue

        if "frame_index" in ep_df.columns:
            frame_index = pd.to_numeric(ep_df["frame_index"], errors="coerce").to_numpy(dtype=float)
            if np.isnan(frame_index).any() or not np.array_equal(
                frame_index.astype(np.int64), np.arange(rows)
            ):
                violations.append(f"ep {ep}: frame_index 不是严格 0..{rows - 1}")

        if "timestamp" in ep_df.columns and rows > 1:
            ts = pd.to_numeric(ep_df["timestamp"], errors="coerce").to_numpy(dtype=float)
            if np.isnan(ts).any():
                violations.append(f"ep {ep}: timestamp 含 NaN")
            else:
                diff = np.diff(ts)
                if not bool((diff > 0).all()):
                    violations.append(f"ep {ep}: timestamp 非严格递增")
                elif float(np.max(np.abs(diff - 1.0 / fps))) > step_tolerance:
                    violations.append(
                        f"ep {ep}: timestamp 步长偏离 1/fps 超过 {step_tolerance * 1e3:.1f}ms"
                        f"（最大 {float(np.max(np.abs(diff - 1.0 / fps))) * 1e3:.2f}ms）"
                    )

        for cam in view.video_features:
            loc = view.video_location(cam, ep)
            if loc is None:
                violations.append(f"ep {ep}/{cam}: 视频窗口缺失")
                continue
            path, from_ts, to_ts = loc
            span = float(to_ts) - float(from_ts)
            if abs(span - rows / fps) > frame_tolerance:
                violations.append(
                    f"ep {ep}/{cam}: 视频窗口跨度 {span * fps:.1f} 帧 ≠ data {rows} 帧"
                )
            if path not in duration_cache:
                duration_cache[path] = probe_video_frames(path) if path.is_file() else None
            frames = duration_cache[path]
            if not frames:
                violations.append(f"ep {ep}/{cam}: 视频文件无法探测: {path.name}")
            elif float(to_ts) > frames / fps + frame_tolerance:
                violations.append(
                    f"ep {ep}/{cam}: 窗口结束 {float(to_ts):.3f}s 超过文件时长 {frames / fps:.3f}s"
                )
    return violations


def vec(arr: np.ndarray) -> list[float]:
    return [round(float(x), 6) for x in np.asarray(arr).reshape(-1)]
