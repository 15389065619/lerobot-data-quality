"""点云模块的共享工具函数。"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

LOG = logging.getLogger(__name__)


def to_matrix(cell: Any) -> np.ndarray:
    """把 parquet 单元格（list[list[float]] / ndarray 对象数组）统一成 (N, D) float32。"""
    if cell is None:
        return np.zeros((0, 0), dtype=np.float32)
    if isinstance(cell, np.ndarray) and cell.dtype != object:
        arr = np.asarray(cell, dtype=np.float32)
    elif isinstance(cell, np.ndarray) and cell.dtype == object:
        try:
            arr = np.asarray(cell.tolist(), dtype=np.float32)
        except (TypeError, ValueError):
            rows = [np.asarray(r, dtype=np.float32).reshape(-1) for r in cell]
            arr = np.stack(rows) if rows else np.zeros((0, 0), dtype=np.float32)
    else:
        try:
            arr = np.asarray(cell, dtype=np.float32)
        except (TypeError, ValueError):
            try:
                arr = np.asarray([np.asarray(r, dtype=np.float32).reshape(-1) for r in cell],
                                 dtype=np.float32)
            except (TypeError, ValueError):
                return np.zeros((0, 0), dtype=np.float32)
    if arr.ndim == 0:
        return np.zeros((0, 0), dtype=np.float32)
    if arr.ndim == 1:
        if arr.size % 3 == 0 and arr.size >= 3:
            arr = arr.reshape(-1, 3)
        else:
            arr = arr.reshape(-1, 1)
    if arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)
    return arr


def xyz(points: np.ndarray) -> np.ndarray:
    """取点云的前三列（xyz 坐标）。"""
    if points.ndim != 2 or points.shape[1] < 3:
        return np.zeros((0, 3), dtype=np.float32)
    return points[:, :3]


def to_nested(points: np.ndarray) -> list[list[float]]:
    """把 ndarray 转成 parquet 单元格需要的嵌套 list。"""
    return [[float(x) for x in row] for row in np.asarray(points, dtype=np.float32)]


def jsonable(obj: Any) -> Any:
    """把 numpy 标量/数组转成 JSON 原生类型（扫描缓存落盘用）。"""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [jsonable(v) for v in obj.tolist()]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def json_default(obj: Any) -> Any:
    """json.dump 的 default 回调：处理 numpy / Path 类型。"""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def vec_or_none(value: Any) -> np.ndarray | None:
    """把列表/None 转成 ndarray，不足 3 维返回 None。"""
    if value is None:
        return None
    arr = np.asarray(value, dtype=float)
    return arr if arr.size >= 3 else None


def read_info(root: Path) -> dict[str, Any]:
    """读取 meta/info.json。"""
    from .types import PointCloudFormatError
    path = root / "meta" / "info.json"
    if not path.is_file():
        raise PointCloudFormatError(f"缺少 meta/info.json: {root}")
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def discover_data_files(root: Path, info: dict[str, Any]) -> list[Path]:
    """枚举 data/*.parquet 文件。"""
    from ..loader import _discover_data_files as discover
    return discover(root, info)


def prepare_dir(root: Path, *, overwrite: bool) -> None:
    """准备输出目录：已存在时按 overwrite 决定是否清空。"""
    import shutil
    if root.exists():
        if not overwrite:
            raise FileExistsError(f"输出目录已存在: {root}")
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)


def copy_meta(src_root: Path, dst_root: Path) -> None:
    """复制 meta 目录到输出。"""
    import shutil
    meta_src = src_root / "meta"
    for name in ("info.json", "stats.json", "tasks.parquet", "tasks.jsonl", "subtasks.parquet"):
        src = meta_src / name
        if src.is_file():
            dst = dst_root / "meta" / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    ep_dir = meta_src / "episodes"
    if ep_dir.is_dir():
        for path in ep_dir.rglob("*.parquet"):
            dst = dst_root / "meta" / "episodes" / path.relative_to(ep_dir)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dst)


def copy_data_parquets(src_root: Path, dst_root: Path) -> None:
    """standalone 输出且无需修改时，原样复制 data 分片。"""
    import shutil
    for path in discover_data_files(src_root, read_info(src_root)):
        dst = dst_root / path.relative_to(src_root)
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dst)


# --- 终端表格对齐（中文按 2 宽） ---
def display_width(text: str) -> int:
    import unicodedata
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in str(text))


def ljust(text: str, width: int) -> str:
    return str(text) + " " * max(0, width - display_width(text))


def rjust(text: str, width: int) -> str:
    return " " * max(0, width - display_width(text)) + str(text)


def health_level(ratio: float) -> str:
    if ratio <= 0:
        return "clean"
    if ratio <= 0.05:
        return "minor"
    if ratio <= 0.25:
        return "moderate"
    return "severe"


def health_cn(level: str) -> str:
    return {"clean": "干净", "minor": "轻微", "moderate": "中等", "severe": "严重"}.get(level, level)
