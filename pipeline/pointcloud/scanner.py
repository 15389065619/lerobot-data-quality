"""点云扫描：串行/并行流式逐帧校验，带断点续传缓存。"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from ..config import cfg_get
from ..manifest import config_digest
from .context import PointCloudContext
from .types import FramesKey, FrameResult
from .utils import jsonable, to_matrix

LOG = logging.getLogger(__name__)


def _evaluate_frame(points, key, ctx):
    """Late-binding wrapper: lets tests monkeypatch pipeline.pointcloud.evaluate_frame."""
    from . import evaluate_frame as _eval
    return _eval(points, key, ctx)


def scan(
    files: list[Path], ctx: PointCloudContext, *, max_frames: int | None = None,
    cache: "ScanCache | None" = None,
) -> tuple[dict[FramesKey, FrameResult], dict[str, str], list[dict[str, Any]]]:
    """串行流式扫描。

    毒数据防冲击：单个 parquet 损坏（截断/schema 异常/缺列读失败）记录进
    corrupted 并跳过，不再中断整个点云阶段；断点续传：cache 命中且文件
    指纹未变时直接复用上次的逐帧结论。
    """
    results: dict[FramesKey, FrameResult] = {}
    skipped = scan_skipped(ctx)
    corrupted: list[dict[str, Any]] = []

    batch_size = max(1, int(cfg_get(ctx.config, "pointcloud.batch_size", 256)))
    for path in files:
        cached = cache.get(path) if cache else None
        if cached is not None:
            for res in cached:
                results[(res.episode_index, res.frame_index)] = res
            continue
        file_frames: list[FrameResult] = []
        try:
            pf = pq.ParquetFile(path)
            names = pf.schema_arrow.names
            cols = [c for c in ("episode_index", "frame_index", ctx.feature_key) if c in names]
            if ctx.feature_key not in cols:
                continue
            for batch in pf.iter_batches(batch_size=batch_size, columns=cols):
                data = batch.to_pydict()
                eps = data["episode_index"]
                fis = data["frame_index"]
                cells = data[ctx.feature_key]
                for ep, fi, cell in zip(eps, fis, cells):
                    key = (int(ep), int(fi))
                    res = _evaluate_frame(to_matrix(cell), key, ctx)
                    results[key] = res
                    file_frames.append(res)
                    if max_frames and len(results) >= max_frames:
                        LOG.info("--pointcloud_max_frames 生效：只校验前 %d 帧", max_frames)
                        return results, skipped, corrupted
        except Exception as exc:  # noqa: BLE001
            LOG.error("点云数据文件损坏，已跳过: %s -> %s", path.name, exc)
            corrupted.append({"file": str(path), "error": str(exc)})
            continue
        if cache:
            cache.put(path, file_frames)
    return results, skipped, corrupted


def scan_skipped(ctx: PointCloudContext) -> dict[str, str]:
    """扫描前即可确定的规则跳过原因（只看全局上下文，与具体数据无关）。"""
    skipped: dict[str, str] = {}
    if ctx.dim < 6:
        skipped["pc_color_invalid"] = "点云维度 < 6，没有颜色通道"
    if ctx.workspace_lower is None:
        skipped["pc_out_of_bounds"] = "无法推导工作空间包围盒（样本点不足）"
    if ctx.outlier_radius <= 1e-6:
        skipped["pc_outliers"] = "场景尺度≈0，无法推导离群半径"
    return skipped


class ScanCache:
    """点云逐帧扫描的断点续传缓存（写在 output_dir，不碰源数据集）。

    命中条件（全部满足才复用上次结论）：
      1. 配置指纹一致（阈值/严重度变了结论必须重算）；
      2. 同一特征列、同一解析出的自适应规则上下文（随数据分布变化）；
      3. 文件 stat 未变（路径 + 大小 + mtime_ns，廉价指纹，不读文件内容）。
    损坏修复、新增、改动的文件因指纹变化自动重扫；缓存损坏/写不进时
    静默降级为不缓存，绝不影响主流程。
    """

    VERSION = 1

    def __init__(self, path: Path, config: dict[str, Any], feature_key: str,
                 resolved: dict[str, Any]) -> None:
        self.path = path
        self._envelope = {
            "version": self.VERSION,
            "config_digest": config_digest(config),
            "feature_key": feature_key,
            "resolved": jsonable(resolved),
        }
        self._files: dict[str, dict[str, Any]] = {}
        self._dirty = False
        if not path.is_file():
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            LOG.warning("点云扫描缓存无法解析，按首次运行处理: %s", exc)
            return
        meta = {k: payload.get(k) for k in self._envelope}
        if meta == self._envelope:
            self._files = payload.get("files") or {}
            LOG.info("断点续传：载入点云扫描缓存（%d 个文件分片）", len(self._files))
        else:
            LOG.info("配置/特征列/自适应阈值已变化，点云扫描缓存失效，将全量重扫")

    @staticmethod
    def _key(path: Path) -> str:
        try:
            st = path.stat()
            sig = {"path": str(path), "size": st.st_size, "mtime_ns": st.st_mtime_ns}
        except OSError:
            sig = {"path": str(path), "exists": False}
        return hashlib.sha1(json.dumps(sig, sort_keys=True).encode()).hexdigest()[:16]

    def get(self, path: Path) -> list[FrameResult] | None:
        entry = self._files.get(self._key(path))
        if not entry:
            return None
        return [FrameResult.from_dict(d) for d in entry.get("frames") or []]

    def put(self, path: Path, frames: list[FrameResult]) -> None:
        self._files[self._key(path)] = {
            "file": str(path),
            "frames": [jsonable(r.to_dict()) for r in frames],
        }
        self._dirty = True

    def save(self) -> None:
        if not self._dirty and self.path.is_file():
            return
        payload = dict(self._envelope)
        payload["files"] = self._files
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(payload, ensure_ascii=False, default=str),
                                 encoding="utf-8")
            LOG.info("点云扫描缓存已保存: %s（%d 个文件分片）", self.path, len(self._files))
        except OSError as exc:
            LOG.warning("点云扫描缓存写入失败（不影响本次结果）: %s", exc)


# ------------------------------------------------------------------ 多进程扫描
_PC_CTX: PointCloudContext | None = None


def _init_pc_worker(ctx: PointCloudContext) -> None:
    """子进程 initializer：注入点云规则上下文（阈值 / 包围盒 / 颜色量程）。"""
    global _PC_CTX
    _PC_CTX = ctx


def _scan_row_group_worker(job: tuple[str, int]) -> dict[str, Any]:
    """worker：读 parquet 的一个 row group，逐帧跑 8 条规则。

    返回 {"file", "results", "corrupted"}：读取失败不再上抛拖垮进程池，
    而是作为 corrupteded 条目交回主进程记录（毒数据防冲击）。
    纯函数契约：只读输入文件、只经返回值交回结论，不写盘、不写共享状态；
    跨帧规则（pc_temporal_jump）由主进程在结果归并后统一执行。
    """
    path, rg_index = job
    ctx = _PC_CTX
    if ctx is None:
        raise RuntimeError("点云 worker 上下文未初始化（initializer 未执行）")
    try:
        pf = pq.ParquetFile(path)
        names = pf.schema_arrow.names
        cols = [c for c in ("episode_index", "frame_index", ctx.feature_key) if c in names]
        if ctx.feature_key not in cols:
            return {"file": path, "results": [], "corrupted": None}
        table = pf.read_row_group(rg_index, columns=cols)
    except Exception as exc:  # noqa: BLE001
        return {"file": path, "results": [],
                "corrupted": {"file": path, "row_group": rg_index, "error": str(exc)}}
    data = table.to_pydict()
    out = [
        _evaluate_frame(to_matrix(cell), (int(ep), int(fi)), ctx)
        for ep, fi, cell in zip(data["episode_index"], data["frame_index"], data[ctx.feature_key])
    ]
    return {"file": path, "results": out, "corrupted": None}


def scan_parallel(
    files: list[Path], ctx: PointCloudContext, *, max_frames: int | None, workers: int,
    cache: "ScanCache | None" = None,
) -> tuple[dict[FramesKey, FrameResult], dict[str, str], list[dict[str, Any]]]:
    """按 (parquet 文件, row group) 分片并发扫描，主进程按 (episode, frame) 归并。

    与串行 scan 的唯一语义差异在 --pointcloud_max_frames：串行是按文件序
    提前截断，并发是扫描全量后按 (episode_index, frame_index) 排序截取，
    两者各自确定，但截断窗口可能不同（max_frames 只是调试开关，不影响全量结果）。
    """
    from ..parallel import run_pool

    skipped = scan_skipped(ctx)
    corrupted: list[dict[str, Any]] = []
    results: dict[FramesKey, FrameResult] = {}
    per_file: dict[str, list[FrameResult]] = {}

    jobs: list[tuple[str, int]] = []
    for path in files:
        cached = cache.get(path) if cache else None
        if cached is not None:
            for res in cached:
                results[(res.episode_index, res.frame_index)] = res
            continue
        try:
            pf = pq.ParquetFile(path)
        except Exception as exc:  # noqa: BLE001
            LOG.error("点云数据文件损坏，已跳过：跳过: %s -> %s", path.name, exc)
            corrupted.append({"file": str(path), "error": str(exc)})
            continue
        if ctx.feature_key not in pf.schema_arrow.names:
            continue
        per_file[str(path)] = []
        for rg in range(pf.metadata.num_row_groups):
            jobs.append((str(path), rg))

    for chunk in run_pool(
        jobs, _scan_row_group_worker, ctx, workers,
        desc="点云帧质检", unit="RG", initializer=_init_pc_worker,
    ):
        if chunk["corrupted"]:
            corrupted.append(chunk["corrupted"])
        for res in chunk["results"]:
            results[(res.episode_index, res.frame_index)] = res
            per_file.setdefault(chunk["file"], []).append(res)

    results = {k: results[k] for k in sorted(results)}

    if cache:
        broken = {c["file"] for c in corrupted}
        for file_key, frames in per_file.items():
            if frames and file_key not in broken:
                cache.put(Path(file_key), frames)

    if max_frames and len(results) > max_frames:
        LOG.info("--pointcloud_max_frames 生效：并发模式扫描全量后按 (episode, frame) 截取前 %d 帧",
                 max_frames)
        keys = sorted(results)[:max_frames]
        results = {k: results[k] for k in keys}
    return results, skipped, corrupted
