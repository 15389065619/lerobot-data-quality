"""清洗流程共用的文件系统与表操作工具。"""

from __future__ import annotations

import logging
import shutil
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
