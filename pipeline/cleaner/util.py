"""清洗流程共用的文件系统与表操作工具。"""

from __future__ import annotations

import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa

from ..loader import load_dataset

LOG = logging.getLogger(__name__)


def assert_safe_output(src_root: Path, out_root: Path) -> None:
    src_root = src_root.resolve()
    out_root = out_root.resolve()
    if out_root == src_root or src_root in out_root.parents:
        raise ValueError(f"输出目录 {out_root} 位于源数据集 {src_root} 内部，拒绝写入（源数据集必须保持只读）")


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
    """保持原 arrow 类型替换列；列不存在时原样返回。"""
    idx = table.schema.get_field_index(name)
    if idx < 0:
        return table
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
    return {
        "ok": bool(ok and contiguous),
        "episodes": view.num_episodes,
        "frames": view.num_frames,
        "episode_indices_contiguous": contiguous,
    }


def vec(arr: np.ndarray) -> list[float]:
    return [round(float(x), 6) for x in np.asarray(arr).reshape(-1)]
