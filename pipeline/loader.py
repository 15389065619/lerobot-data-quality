"""LeRobot v3 数据集加载器（只读）。

职责边界：只负责「把磁盘上的数据集读成内存里的结构」，不做任何质量校验、
不做任何写操作。校验逻辑一律放在 validator 包。

支持的布局：
  * v3.0（默认）：data/chunk-000/file-000.parquet，多 episode 共用一个 parquet 文件
                  meta/episodes/chunk-000/file-000.parquet
                  meta/tasks.parquet
                  videos/<camera_key>/chunk-000/file-000.mp4
  * v2.1（兼容）：data/chunk-000/episode_000000.parquet
                  meta/episodes.jsonl / meta/tasks.jsonl
                  videos/chunk-000/<camera_key>/episode_000000.mp4

对外主入口：load_dataset(root, max_episodes=None)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .schema import audit_schema

LOG = logging.getLogger(__name__)

# v3.0 路径模板（info.json 里没写时使用）
DEFAULT_DATA_PATH = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
DEFAULT_VIDEO_PATH = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
DEFAULT_EPISODES_PATH = "meta/episodes/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"

# 逐帧表里预期的核心列
CORE_FRAME_COLUMNS = ("episode_index", "frame_index", "timestamp", "index")

# 质检关注的语义字段（缺失时由 validator 报 missing_task / 跳过对应检查）
STATE_KEY = "observation.state"
ACTION_KEY = "action"
TASK_INDEX_KEY = "task_index"

#: 内部字段：记录每行来自哪个 parquet 文件，供 cleaner 按原文件重写
SOURCE_COL = "__source_file"

#: 二维特征（点云 / embedding）达到该元素数即视为「重型列」，默认不读进内存
HEAVY_FEATURE_MIN_VALUES = 1000


class DatasetFormatError(RuntimeError):
    """目录结构不符合 LeRobot 数据集约定。"""


@dataclass
class LeRobotDatasetView:
    """数据集的内存视图。字段全部是「已读出来的数据」，不含任何判断结论。"""

    root: Path
    info: dict[str, Any]
    stats: dict[str, Any]
    features: dict[str, Any]
    tasks: dict[int, str]
    episodes: pd.DataFrame
    frames: pd.DataFrame
    video_index: dict[str, dict[tuple[int, int], Path]] = field(default_factory=dict)
    codebase_version: str = "unknown"
    fps: float = 30.0
    #: --max_episodes 生效时记录被截断到的数量，便于报告里如实标注
    truncated_to: int | None = None
    #: data/*.parquet 列表，流式模式按文件分片读取时使用
    data_files: list[Path] = field(default_factory=list)
    #: 检测到但未读入内存的二维重型特征（如点云），由 pointcloud 阶段按需流式读取
    heavy_features: list[str] = field(default_factory=list)
    #: 读取失败被跳过的损坏数据文件（毒数据防冲击），元素为 {"file": 路径, "error": 原因}
    corrupted_files: list[dict[str, str]] = field(default_factory=list)
    #: data episode_index -> meta episode_index。两侧编号空间不一致时由加载器建立，
    #: 为空表示两侧同名（绝大多数数据集）。
    meta_id_map: dict[int, int] = field(default_factory=dict)
    #: 编号对齐诊断：mode(native/remapped/ambiguous/unavailable) + 校验证据，
    #: 报告据此说明「能否定位」，规则层据此区分「异常」与「不可评估」。
    id_reconciliation: dict[str, Any] = field(default_factory=dict)
    #: data/*.parquet 实际列名（只读 schema 得到），用于 Schema 契约审计
    data_columns: set[str] = field(default_factory=set)
    #: Schema 契约违约项，见 pipeline/schema.py；空列表表示布局合规
    schema_contract: list[dict[str, Any]] = field(default_factory=list)

    # ------------------------------------------------------------------ 概览
    @property
    def num_episodes(self) -> int:
        return int(len(self.episodes))

    # ---------------------------------------------------- 编号空间与 meta 定位
    @property
    def locator_available(self) -> bool:
        """能否可靠地把 episode 映射到 meta 行（决定视频/任务能否定位）。

        mode == "ambiguous" 表示两侧编号空间不一致且无法可靠对齐——此时定位必然
        失败，属于「不可评估」，不能当作数据异常（否则会误判整批数据不可用）。
        """
        return str(self.id_reconciliation.get("mode", "native")) != "ambiguous"

    def meta_episode_index(self, episode_index: int) -> int:
        return int(self.meta_id_map.get(int(episode_index), int(episode_index)))

    def meta_row(self, episode_index: int) -> pd.Series | None:
        """取该 episode 对应的 meta 行；编号对不上或 meta 缺失时返回 None。"""
        if self.episodes.empty or "episode_index" not in self.episodes.columns:
            return None
        rows = self.episodes[self.episodes["episode_index"] == self.meta_episode_index(episode_index)]
        return rows.iloc[0] if len(rows) else None

    @property
    def num_frames(self) -> int:
        """总帧数；流式模式（frames 未加载）下由 meta 的 episode length 求和。"""
        if len(self.frames):
            return int(len(self.frames))
        return int(self.frames_by_episode().sum())

    @property
    def episode_indices(self) -> list[int]:
        """数据侧的 episode 编号（与 parquet 里 episode_index 列同一空间）。

        两侧编号不一致时以 data 为准：valider 按数据流遍历，下游一律用这套编号。
        """
        if self.meta_id_map:
            return sorted(self.meta_id_map)
        if len(self.frames) and "episode_index" in self.frames.columns:
            return sorted(int(x) for x in self.frames["episode_index"].unique().tolist())
        return [int(x) for x in self.episodes["episode_index"].tolist()]

    @property
    def column_names(self) -> list[str]:
        """可用列名；frames 未加载时按 info.json 的特征 + 核心列推断。"""
        if len(self.frames):
            return [c for c in self.frames.columns if c != SOURCE_COL]
        return list(dict.fromkeys([*self.features.keys(), *CORE_FRAME_COLUMNS]))

    def has_column(self, name: str) -> bool:
        return name in self.frames.columns if len(self.frames) else name in self.column_names

    def frames_by_episode(self) -> pd.Series:
        """episode_index（数据侧）-> 帧数；frames 未加载时取 meta 的 length 列。"""
        if len(self.frames):
            if "episode_index" not in self.frames.columns:
                # 非标准字段名（episode 标识被改名/缺失）：无法按 episode 聚合
                return pd.Series(dtype="int64")
            return self.frames.groupby("episode_index").size()
        if len(self.episodes) and "length" in self.episodes.columns:
            lengths = pd.to_numeric(self.episodes["length"], errors="coerce").fillna(0).astype(int)
            indices = pd.to_numeric(self.episodes["episode_index"], errors="coerce").astype(int)
            series = pd.Series(lengths.to_numpy(), index=indices.to_numpy(), dtype="int64")
            if self.meta_id_map:
                # meta 侧编号要换算回数据侧，否则与 frames/校验结果对不上
                inv = {int(m): int(d) for d, m in self.meta_id_map.items()}
                series = pd.Series(
                    [int(v) for k, v in series.items() if int(k) in inv],
                    index=[inv[int(k)] for k in series.index if int(k) in inv],
                    dtype="int64",
                ).sort_index()
            return series
        return pd.Series(dtype="int64")

    def frame_count_of(self, episode_index: int) -> int:
        """单个 episode 的帧数；优先 meta 的 length，避免为拿长度去读整段帧数据。"""
        if len(self.frames):
            if "episode_index" not in self.frames.columns:
                return 0
            return int((self.frames["episode_index"] == episode_index).sum())
        row = self.meta_row(episode_index)
        if row is not None and "length" in self.episodes.columns:
            try:
                return int(row["length"])
            except (TypeError, ValueError):
                return 0
        return 0

    @property
    def data_features(self) -> list[str]:
        """数值型（存在于 parquet 中的）特征名，按 info.json 顺序。"""
        out: list[str] = []
        for key, spec in self.features.items():
            dtype = str(spec.get("dtype", ""))
            if dtype.startswith(("float", "int", "uint", "bool")):
                if self.has_column(key):
                    out.append(key)
        return out

    @property
    def video_features(self) -> list[str]:
        """真正的视频特征（mp4）。图像特征逐帧嵌在 parquet 里，不能按视频文件定位。"""
        return [
            key
            for key, spec in self.features.items()
            if str(spec.get("dtype", "")).lower() == "video"
        ]

    @property
    def image_features(self) -> list[str]:
        """内嵌图像特征（dtype=image），随 data/*.parquet 一起存储，无独立视频文件。"""
        return [
            key
            for key, spec in self.features.items()
            if str(spec.get("dtype", "")).lower() == "image"
        ]

    @property
    def joint_names(self) -> list[str]:
        spec = self.features.get(STATE_KEY, {})
        names = spec.get("names") or []
        if isinstance(names, dict):
            # 两种写法都要兼容：{0: "j0", ...}（按索引）与 {"joints": ["j0", ...]}（分组）
            vals = list(names.values())
            if all(isinstance(v, (list, tuple)) for v in vals):
                names = [str(x) for v in vals for x in v]
            else:
                try:
                    names = [names[k] for k in sorted(names, key=lambda s: int(s))]
                except (TypeError, ValueError):
                    names = [str(v) for v in vals]
        dim = _feature_dim(spec)
        if names and len(names) == dim:
            return [str(n) for n in names]
        return [f"dim_{i}" for i in range(dim)]

    @property
    def schema(self) -> dict[str, str]:
        """字段 schema：列名 -> "dtype shape"，直接从 parquet 实际类型 + info.json 描述合成。

        frames 未加载（流式）时退化为 info.json 描述，保证报告字段不空。
        """
        out: dict[str, str] = {}
        if not len(self.frames):
            for key, spec in self.features.items():
                dtype = str(spec.get("dtype", ""))
                if dtype.lower() in {"video", "image"}:
                    out[key] = f"video {spec.get('shape')}"
                else:
                    dim = _feature_dim(spec)
                    out[key] = dtype + (f" [{dim}]" if dim else "")
            return out
        for col, dtype in self.frames.dtypes.items():
            if col == SOURCE_COL:
                continue
            out[col] = str(dtype)
        for key, spec in self.features.items():
            if key in out:
                dim = _feature_dim(spec)
                out[key] = f"{out[key]}" + (f" [{dim}]" if dim else "")
            elif str(spec.get("dtype", "")).lower() in {"video", "image"}:
                shape = spec.get("shape")
                out[key] = f"video {shape}"
        return out

    # ------------------------------------------------------------- episode 级
    def episode_frame(self, episode_index: int) -> pd.DataFrame:
        """取某个 episode 的所有帧，按 frame_index 升序。

        frames 未加载（流式）时按需从 parquet 读该 episode，内存只驻留一段。
        """
        if len(self.frames):
            if "episode_index" not in self.frames.columns:
                return pd.DataFrame()
            sub = self.frames[self.frames["episode_index"] == episode_index]
        else:
            sub = self._read_episode(episode_index)
        if "frame_index" in sub.columns:
            sub = sub.sort_values("frame_index", kind="stable")
        return sub

    def data_file_of(self, episode_index: int) -> Path | None:
        """定位 episode 所在的数据文件；meta 没记录时靠单文件/扫描兜底。"""
        row = self.meta_row(episode_index)
        if row is not None:
            chunk, file_idx = row.get("data/chunk_index"), row.get("data/file_index")
            if pd.notna(chunk) and pd.notna(file_idx):
                tmpl = self.info.get("data_path") or DEFAULT_DATA_PATH
                try:
                    rel = tmpl.format(
                        chunk_index=int(chunk),
                        file_index=int(file_idx),
                        episode_index=int(episode_index),
                    )
                except (KeyError, IndexError, ValueError, TypeError):
                    rel = None
                if rel:
                    path = self.root / rel
                    if path.is_file():
                        return path
        if len(self.data_files) == 1:
            return self.data_files[0]
        return None

    def _read_episode(self, episode_index: int) -> pd.DataFrame:
        """流式模式下读单个 episode：优先 meta 指向的文件，否则逐文件过滤。"""
        target = self.data_file_of(episode_index)
        candidates = [target] if target is not None else list(self.data_files)
        for path in candidates:
            try:
                table = pq.read_table(path, filters=[("episode_index", "=", int(episode_index))])
            except Exception:  # noqa: BLE001 - 不支持 filter 时退回全量读
                try:
                    table = pq.read_table(path)
                except Exception as exc:  # noqa: BLE001
                    LOG.warning("读取 %s 失败: %s", path.name, exc)
                    continue
            df = table.to_pandas()
            if "episode_index" in df.columns:
                df = df[df["episode_index"] == episode_index]
            if not df.empty:
                return df.reset_index(drop=True)
        return pd.DataFrame()

    def task_of(self, episode_index: int) -> str:
        """episode 的任务描述；拿不到时返回空串（由 validator 判定是否异常）。"""
        row = self.meta_row(episode_index)
        if row is not None and "tasks" in self.episodes.columns:
            val = row.get("tasks")
            if isinstance(val, str):
                if val.strip():
                    return val.strip()
            elif isinstance(val, (list, tuple)) or hasattr(val, "__len__"):
                # parquet 的 list 列可能读成 ndarray/其它序列，不能只判断 list/tuple
                try:
                    if len(val):
                        first = val[0]
                        if isinstance(first, str) and first.strip():
                            return first.strip()
                except (TypeError, IndexError):
                    pass
        if TASK_INDEX_KEY in self.frames.columns and "episode_index" in self.frames.columns:
            sub = self.frames.loc[self.frames["episode_index"] == episode_index, TASK_INDEX_KEY]
            if len(sub):
                try:
                    ti = int(sub.iloc[0])
                except (TypeError, ValueError):
                    return ""
                return str(self.tasks.get(ti, "")).strip()
        return ""

    def video_file(self, camera: str, episode_index: int) -> Path | None:
        """定位某个 episode 在某相机下的视频文件。"""
        loc = self.video_location(camera, episode_index)
        return loc[0] if loc else None

    def video_location(
        self, camera: str, episode_index: int, *, require_exists: bool = True
    ) -> tuple[Path, float, float] | None:
        """返回 (视频文件, from_timestamp, to_timestamp)；无定位信息时 from/to 为 (-1, -1)。

        ``require_exists=False`` 时只要能从 meta/模板推出路径就返回，即使文件不存在。
        质检需要这个区分：路径推得出来但文件不在 = 视频缺失（数据缺陷），
        路径根本推不出来 = 不可评估（定位信息缺失）。默认 True 保持清洗器行为不变。
        """
        row = self.meta_row(episode_index)
        ck, fk = f"videos/{camera}/chunk_index", f"videos/{camera}/file_index"
        ft, tt = f"videos/{camera}/from_timestamp", f"videos/{camera}/to_timestamp"
        if row is not None:
            if ck in self.episodes.columns and fk in self.episodes.columns and pd.notna(row.get(ck)):
                path = self._video_path_from_template(
                    camera, int(row[ck]), int(row[fk]), require_exists=require_exists
                )
                if path is not None:
                    from_ts = float(row[ft]) if ft in self.episodes.columns and pd.notna(row.get(ft)) else -1.0
                    to_ts = float(row[tt]) if tt in self.episodes.columns and pd.notna(row.get(tt)) else -1.0
                    return path, from_ts, to_ts
        # v2.1：优先按 info.json 的 video_path 模板定位
        # （如 videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4）
        if str(self.codebase_version).startswith("v2") and self.info.get("video_path"):
            chunks_size = int(self.info.get("chunks_size") or 1000) or 1000
            episode_chunk = episode_index // chunks_size
            fields = {
                "video_key": camera, "camera": camera,
                "episode_index": episode_index, "episode_chunk": episode_chunk,
                "chunk_index": episode_chunk, "file_index": episode_index,
            }
            try:
                rel = str(self.info["video_path"]).format(**fields)
            except (KeyError, IndexError, ValueError):
                rel = None
            if rel:
                path = self.root / rel
                if path.is_file() or not require_exists:
                    return path, 0.0, -1.0
        # v2.1：文件名里带 episode 序号，视频按 chunk 分目录存放
        # （videos/chunk-000/<camera>/episode_000000.mp4，也兼容 videos/<camera>/）
        # 目录名常省略 observation.images. 前缀，因此短名也要试
        vroot = self.root / "videos"
        short = camera.split("observation.images.")[-1]
        names = [camera, short] if short != camera else [camera]
        cam_dirs = [vroot / name for name in names]
        cam_dirs += [p for name in names for p in sorted(vroot.glob(f"chunk-*/{name}"))]
        for cam_dir in cam_dirs:
            if cam_dir.is_dir():
                cands = sorted(cam_dir.rglob(f"*{episode_index:06d}.mp4"))
                if cands:
                    return cands[0], 0.0, -1.0
        # 兜底：该相机只有一个文件（常见于单 episode 导出）
        files = self.video_index.get(camera, {})
        if len(files) == 1:
            return next(iter(files.values())), 0.0, -1.0
        return None

    def _video_path_from_template(
        self, camera: str, chunk: int, file: int, *, require_exists: bool = True
    ) -> Path | None:
        idx = self.video_index.get(camera) or {}
        if (chunk, file) in idx:
            return idx[(chunk, file)]
        tmpl = self.info.get("video_path") or DEFAULT_VIDEO_PATH
        try:
            rel = tmpl.format(video_key=camera, chunk_index=chunk, file_index=file)
        except (KeyError, IndexError, ValueError):
            return None
        path = self.root / rel
        return path if path.is_file() or not require_exists else None

    # ----------------------------------------------------------------- 摘要
    def summary(self) -> dict[str, Any]:
        per_ep = self.frames_by_episode()
        columns = self.column_names
        return {
            "root": str(self.root),
            "codebase_version": self.codebase_version,
            "fps": self.fps,
            "robot_type": self.info.get("robot_type"),
            "num_episodes": self.num_episodes,
            "num_frames": self.num_frames,
            "total_tasks": len(self.tasks),
            "episode_indices": self.episode_indices,
            "truncated_to": self.truncated_to,
            "frames_per_episode": {
                "min": int(per_ep.min()) if len(per_ep) else 0,
                "max": int(per_ep.max()) if len(per_ep) else 0,
                "mean": round(float(per_ep.mean()), 2) if len(per_ep) else 0.0,
            },
            "columns": columns,
            "schema": self.schema,
            "state_key": STATE_KEY if self.has_column(STATE_KEY) else None,
            "action_key": ACTION_KEY if self.has_column(ACTION_KEY) else None,
            "num_state_dims": _feature_dim(self.features.get(STATE_KEY, {})),
            "num_action_dims": _feature_dim(self.features.get(ACTION_KEY, {})),
            "joint_names": self.joint_names,
            "video_features": {
                cam: [str(p.relative_to(self.root)) for p in files.values()]
                for cam, files in self.video_index.items()
            },
            "episode_meta_columns": [c for c in self.episodes.columns],
            "has_stats": bool(self.stats),
            "pointcloud_features": list(self.heavy_features),
        }


# --------------------------------------------------------------------- 加载
def load_dataset(
    dataset_path: str | Path,
    max_episodes: int | None = None,
    *,
    load_frames: bool = True,
    include_heavy: bool = False,
) -> LeRobotDatasetView:
    """加载本地 LeRobot 数据集，返回只读视图。

    Args:
        dataset_path: 数据集根目录（含 meta/info.json）。
        max_episodes: 只加载前 N 个 episode（按 episode_index 升序），None 表示全量。
        load_frames: False 时只读元信息，不读 parquet 帧数据。
        include_heavy: True 时把点云等二维重型特征也读进内存。默认 False：
            512×6 的点云乘以 2 万帧会占用 GB 级内存，且主质检流程用不到，
            点云清洗阶段改用流式逐帧读取。
    """
    root = Path(dataset_path).expanduser().resolve()
    if not root.is_dir():
        raise DatasetFormatError(f"数据集目录不存在: {root}")

    info = _read_info(root)
    stats = _read_json(root / "meta" / "stats.json")
    features = info.get("features") or {}
    fps = float(info.get("fps") or 30.0)
    version = str(info.get("codebase_version") or "unknown")

    tasks = _read_tasks(root)
    data_files = _discover_data_files(root, info)
    if load_frames and not data_files:
        raise DatasetFormatError(f"未在 {root} 下找到任何 data/*.parquet")

    episodes_meta = _read_episodes_meta(root, info)
    video_index = _discover_videos(root)

    heavy = heavy_feature_keys(features) if not include_heavy else []
    if heavy:
        LOG.info("检测到 %d 个重型二维特征，默认不载入内存（点云阶段按需流式读取）: %s",
                 len(heavy), ", ".join(heavy))

    frames = pd.DataFrame()
    truncated_to = None
    keep: set[int] | None = None

    # ---------------------------------------------- episode 编号空间对齐
    # behavior1k 这类数据集：meta/episodes 用全局 ID（420010, 420020, …），
    # 而 data/*.parquet 里的 episode_index 是 0..N-1。不先对齐，所有 meta 定位
    # （视频窗口、任务、data 文件）都会落空，进而把整批数据误判为不可用。
    data_summary = _data_episode_summary(frames if load_frames else None, data_files)
    meta_id_map, reconciliation = _reconcile_episode_ids(episodes_meta, data_summary, fps)

    # Schema 契约审计：只读 schema 与两列极值，不额外加载整表
    data_columns: set[str] = set()
    if data_files:
        try:
            data_columns = set(pq.read_schema(data_files[0]).names)
        except Exception:  # noqa: BLE001
            pass
    if reconciliation.get("mode") == "remapped":
        LOG.warning(
            "data 与 meta 的 episode 编号空间不同，已按顺序建立映射并通过证据校验：%s",
            {k: reconciliation[k] for k in ("data_ids", "meta_ids", "overlap",
                                            "length_agreement", "video_agreement")
             if k in reconciliation},
        )
    elif reconciliation.get("mode") == "ambiguous":
        LOG.warning("episode 编号无法可靠对齐，依赖 meta 的定位检查将标记为「不可评估」：%s",
                    reconciliation.get("reason"))

    if max_episodes is not None and max_episodes > 0:
        if meta_id_map:
            all_eps = [int(x) for x in data_summary["episode_index"].tolist()]
        else:
            all_eps = _episode_order(episodes_meta, data_files)
        keep = set(all_eps[:max_episodes])
        truncated_to = len(keep)
        LOG.info("--max_episodes 生效：只加载前 %d 个 episode", truncated_to)

    if load_frames:
        corrupted: list[dict[str, str]] = []
        frames = _read_frames(data_files, keep, exclude=tuple(heavy), corrupted=corrupted)
        if episodes_meta is None or episodes_meta.empty:
            LOG.warning("meta/episodes 缺失，回退为从帧数据推导 episode 元信息")
            episodes_meta = _episodes_from_frames(frames)

    if keep is not None and episodes_meta is not None and not episodes_meta.empty:
        wanted_meta = {int(meta_id_map.get(k, k)) for k in keep}
        episodes_meta = episodes_meta[episodes_meta["episode_index"].isin(wanted_meta)].reset_index(drop=True)
        if meta_id_map:
            meta_id_map = {k: v for k, v in meta_id_map.items() if k in keep}

    if episodes_meta is None:
        episodes_meta = pd.DataFrame(columns=["episode_index", "length"])

    view = LeRobotDatasetView(
        root=root,
        info=info,
        stats=stats,
        features=features,
        tasks=tasks,
        episodes=episodes_meta,
        frames=frames,
        video_index=video_index,
        codebase_version=version,
        fps=fps,
        truncated_to=truncated_to,
        data_files=data_files,
        heavy_features=heavy,
        corrupted_files=corrupted if load_frames else [],
        meta_id_map=meta_id_map,
        id_reconciliation=reconciliation,
        data_columns=data_columns,
    )
    # info.total_frames 必须与 data 的**实际行数**对账。流式模式（--streaming）下
    # view.num_frames 是 meta 里 length 之和，用它校验等于用 meta 自证，抓不到
    # data 被截断而 meta/info 内部一致的情况。data_summary 始终来自逐文件实际读取，
    # 行数取它才成立；仅在无数据文件时退回 num_frames。
    data_rows_actual = (
        int(data_summary["rows"].sum())
        if data_summary is not None and len(data_summary) and "rows" in data_summary.columns
        else view.num_frames
    )
    view.schema_contract = audit_schema(
        features=features,
        splits=info.get("splits"),
        total_episodes=info.get("total_episodes") or view.num_episodes,
        data_columns=data_columns,
        data_summary=data_summary,
        info_total_episodes=info.get("total_episodes"),
        info_total_frames=info.get("total_frames"),
        meta_episode_rows=len(view.episodes),
        data_rows=data_rows_actual,
        codebase_version=str(version),
        truncated=bool(view.truncated_to),
    )
    for item in view.schema_contract:
        LOG.warning("Schema 违约 [%s] %s：%s", item["severity"], item["column"], item["rule"])
    # 统一 info 中的 total_episodes / total_frames：以实际读到的为准（元数据可能已失真）
    LOG.info(
        "已加载 %d 个 episode / %d 帧（info.json 记录 %s episodes / %s frames），%d 个相机",
        view.num_episodes,
        view.num_frames,
        info.get("total_episodes"),
        info.get("total_frames"),
        len(video_index),
    )
    return view


# ------------------------------------------------------------------ 读取细节
def _read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except json.JSONDecodeError as exc:
        raise DatasetFormatError(f"{path} 不是合法 JSON: {exc}") from exc


def _read_info(root: Path) -> dict:
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise DatasetFormatError(f"缺少 meta/info.json，不是合法的 LeRobot 数据集: {root}")
    info = _read_json(info_path)
    if not info:
        raise DatasetFormatError(f"meta/info.json 为空: {info_path}")
    return info


def _read_tasks(root: Path) -> dict[int, str]:
    """task_index -> 任务文本。v3 用 tasks.parquet，v2.1 用 tasks.jsonl。"""
    pq_path = root / "meta" / "tasks.parquet"
    if pq_path.is_file():
        try:
            df = pq.read_table(pq_path).to_pandas()
        except Exception as exc:  # noqa: BLE001 - 损坏的元数据不应中断加载
            LOG.warning("读取 tasks.parquet 失败: %s", exc)
            df = pd.DataFrame()
        if not df.empty:
            tasks: dict[int, str] = {}
            if "task_index" in df.columns and "task" in df.columns:
                for ti, task in zip(df["task_index"], df["task"]):
                    tasks[int(ti)] = str(task)
            elif df.index.dtype.kind in "iu":
                tasks = {int(i): str(row.iloc[0]) for i, row in df.iterrows()}
            elif "task_index" in df.columns and len(df.columns) == 1:
                # 任务文本存放在索引里的变体（如 nsgzfei/w01_v3）：
                # index="Put the sweet potato into the white basket."、列只保留 task_index
                tasks = {int(ti): str(text).strip() for text, ti in df["task_index"].items()}
            if tasks:
                return tasks

    jsonl = root / "meta" / "tasks.jsonl"
    if jsonl.is_file():
        tasks = {}
        with jsonl.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "task_index" in obj:
                    tasks[int(obj["task_index"])] = str(obj.get("task", ""))
        return tasks

    LOG.warning("未找到 meta/tasks.parquet 或 tasks.jsonl，任务描述将无法解析")
    return {}


def _discover_data_files(root: Path, info: dict) -> list[Path]:
    """按 info.json 的 data_path 模板枚举，失败则 glob 兜底。"""
    tmpl = info.get("data_path") or DEFAULT_DATA_PATH
    found: list[Path] = []
    chunks = int(info.get("chunks_size") or 1000)
    files_per_chunk = 9999
    for ci in range(chunks):
        for fi in range(files_per_chunk):
            try:
                rel = tmpl.format(chunk_index=ci, file_index=fi, episode_index=fi)
            except (KeyError, IndexError, ValueError):
                break
            path = root / rel
            if path.is_file():
                found.append(path)
            elif fi > 0 and not _has_next(root, tmpl, ci, fi):
                break
    if not found:
        found = sorted((root / "data").rglob("*.parquet")) if (root / "data").is_dir() else []
    return found


def _has_next(root: Path, tmpl: str, chunk: int, file: int) -> bool:
    """探一下同 chunk 的下一个文件是否存在（避免盲目 glob 全盘）。"""
    try:
        rel = tmpl.format(chunk_index=chunk, file_index=file, episode_index=file)
    except (KeyError, IndexError, ValueError):
        return False
    return (root / rel).is_file()


def _discover_videos(root: Path) -> dict[str, dict[tuple[int, int], Path]]:
    """camera_key -> {(chunk, file): path}。兼容 v3 和 v2.1 两种命名。"""
    index: dict[str, dict[tuple[int, int], Path]] = {}
    vroot = root / "videos"
    if not vroot.is_dir():
        return index
    for path in sorted(vroot.rglob("*.mp4")):
        rel = path.relative_to(vroot)
        parts = rel.parts
        # v3: <camera>/chunk-000/file-000.mp4
        if len(parts) >= 3 and parts[1].startswith("chunk-"):
            cam = parts[0]
            try:
                chunk = int(parts[1].split("-")[1])
                file = int(path.stem.split("-")[1])
            except (IndexError, ValueError):
                continue
            index.setdefault(cam, {})[(chunk, file)] = path
        # v2.1: chunk-000/<camera>/episode_000000.mp4
        elif parts[0].startswith("chunk-") and len(parts) >= 3:
            cam = parts[1]
            try:
                chunk = int(parts[0].split("-")[1])
                ep = int(path.stem.split("_")[1])
            except (IndexError, ValueError):
                continue
            index.setdefault(cam, {})[(chunk, ep)] = path
    return index


def _read_episodes_meta(root: Path, info: dict) -> pd.DataFrame | None:
    """episode 级元数据（length / tasks / from_index / 视频区间）。"""
    ep_root = root / "meta" / "episodes"
    if ep_root.is_dir():
        files = sorted(ep_root.rglob("*.parquet"))
        if files:
            frames = []
            for f in files:
                try:
                    frames.append(pq.read_table(f).to_pandas())
                except Exception as exc:  # noqa: BLE001
                    LOG.warning("读取 episode 元数据 %s 失败: %s", f, exc)
            if frames:
                df = pd.concat(frames, ignore_index=True)
                df = df.drop_duplicates(subset=["episode_index"]).sort_values("episode_index")
                return df.reset_index(drop=True)

    jsonl = root / "meta" / "episodes.jsonl"
    if jsonl.is_file():
        rows = []
        with jsonl.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        if rows:
            return pd.DataFrame(rows).sort_values("episode_index").reset_index(drop=True)
    return None


def _read_frames(
    files: list[Path], keep: set[int] | None, exclude: tuple[str, ...] = (),
    corrupted: list[dict[str, str]] | None = None,
) -> pd.DataFrame:
    """把 data/*.parquet 读成一张长表，记录来源文件便于清洗阶段原地重写。

    exclude 用于跳过点云/embedding 这类二维重型列：不读出来才不会占用 GB 级内存。

    毒数据防冲击：传入 corrupted 列表时，单个文件损坏（schema 读不出 / parquet
    截断 / 缺列）不再中断整个加载，而是记录 {"file", "error"} 后跳过；
    全部文件都不可读时仍抛 DatasetFormatError（没有数据就没有质检的意义）。
    """
    parts: list[pd.DataFrame] = []
    for path in files:
        try:
            cols = _columns_excluding(path, exclude)
            df: pd.DataFrame | None = None
            if keep is not None:
                try:
                    table = pq.read_table(
                        path, columns=cols, filters=[("episode_index", "in", sorted(keep))]
                    )
                    df = table.to_pandas()
                except Exception as exc:  # noqa: BLE001 - 不支持 filter 时退回全量读
                    LOG.debug("按 episode 过滤读取 %s 失败，改为全量读取: %s", path.name, exc)
                    df = None
            if df is None:
                df = pq.read_table(path, columns=cols).to_pandas()
        except Exception as exc:  # noqa: BLE001
            if corrupted is None:
                raise DatasetFormatError(f"读取 parquet 失败: {path} -> {exc}") from exc
            LOG.error("数据文件损坏，已跳过（其中 episode 将标记 episode_read_error）: %s -> %s",
                      path.name, exc)
            corrupted.append({"file": str(path), "error": str(exc)})
            continue
        if df.empty:
            continue
        df[SOURCE_COL] = str(path)
        parts.append(df)

    if not parts:
        return pd.DataFrame()

    out = pd.concat(parts, ignore_index=True)
    if keep is not None and "episode_index" in out.columns:
        out = out[out["episode_index"].isin(keep)]
    sort_cols = [c for c in ("episode_index", "frame_index") if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols, kind="stable")
    out = out.reset_index(drop=True)
    for col in ("episode_index", "frame_index", "task_index", "index"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _episodes_from_frames(frames: pd.DataFrame) -> pd.DataFrame:
    if frames.empty or "episode_index" not in frames.columns:
        return pd.DataFrame(columns=["episode_index", "length"])
    grp = frames.groupby("episode_index")
    df = pd.DataFrame({"length": grp.size()}).reset_index()
    df["dataset_from_index"] = grp["index"].min().values if "index" in frames.columns else 0
    df["dataset_to_index"] = df["dataset_from_index"] + df["length"]
    return df


def _episode_order(episodes_meta: pd.DataFrame | None, data_files: list[Path]) -> list[int]:
    if episodes_meta is not None and not episodes_meta.empty and "episode_index" in episodes_meta.columns:
        return sorted(int(x) for x in episodes_meta["episode_index"].tolist())
    # 没有 episode 元数据时，从 parquet 里只读 episode_index 一列拿顺序
    seen: set[int] = set()
    for path in data_files:
        try:
            col = pq.read_table(path, columns=["episode_index"]).column("episode_index").to_pylist()
        except Exception:  # noqa: BLE001
            continue
        seen.update(int(x) for x in col if x is not None)
    return sorted(seen)


# ------------------------------------------------------------ episode 编号对齐
def _data_episode_summary(
    frames: pd.DataFrame | None, data_files: list[Path]
) -> pd.DataFrame:
    """数据侧每个 episode 的行数与首次出现顺序。

    有帧表时直接分组（零额外 IO）；流式模式下只读 episode_index 一列扫描，
    单列 IO 相对整表很小（128 万行约 10MB），是判定「能否安全映射」的必要代价。
    """
    if frames is not None and len(frames) and "episode_index" in frames.columns:
        agg = {"rows": ("episode_index", "size")}
        has_index = "index" in frames.columns
        if has_index:
            agg["idx_min"] = ("index", "min")
            agg["idx_max"] = ("index", "max")
            agg["idx_nunique"] = ("index", "nunique")
        out = frames.groupby("episode_index", sort=False).agg(**agg).reset_index()
        out["episode_index"] = out["episode_index"].astype("int64")
        return out
    if not data_files:
        return pd.DataFrame(columns=["episode_index", "rows"])

    # 顺带取 index 的每集极值：这是判定「index 是否还是全局帧号」的唯一证据，
    # 与 episode_index 同一趟读取，多一列的 IO 成本可忽略。
    has_index = False
    try:
        has_index = "index" in pq.read_schema(data_files[0]).names
    except Exception:  # noqa: BLE001
        pass

    order: list[int] = []
    counts: dict[int, int] = {}
    lows: dict[int, int] = {}
    highs: dict[int, int] = {}
    # 每集 index 的唯一值个数（按贡献文件分别记）：只判「跨度和行数相等」会漏掉
    # 「重复与断号互相抵消」——[0,1,1,3] 4 行、跨度 4，却有两个 1、缺一个 2。
    # 唯一值个数是唯一能戳穿它的廉价证据，与极值同一趟算出。
    uniques: dict[int, list[int]] = {}
    cols = ["episode_index", "index"] if has_index else ["episode_index"]
    for path in data_files:
        try:
            table = pq.read_table(path, columns=cols)
            col = table.column("episode_index").to_numpy()
        except Exception as exc:  # noqa: BLE001 - 损坏文件交由 corrupted_files 记录
            LOG.debug("编号对齐扫描跳过 %s: %s", path.name, exc)
            continue
        idx_col = table.column("index").to_numpy() if has_index else None
        for value in np.unique(col):
            ep = int(value)
            mask = col == value
            if ep not in counts:
                order.append(ep)
                counts[ep] = 0
            counts[ep] += int(mask.sum())
            if idx_col is not None:
                sub = idx_col[mask]
                lo, hi = int(sub.min()), int(sub.max())
                lows[ep] = min(lows.get(ep, lo), lo)
                highs[ep] = max(highs.get(ep, hi), hi)
                uniques.setdefault(ep, []).append(int(np.unique(sub).size))
    data = {"episode_index": order, "rows": [counts[e] for e in order]}
    if has_index:
        data["idx_min"] = [lows[e] for e in order]
        data["idx_max"] = [highs[e] for e in order]
        # 跨多个文件的 episode 无法在这里合并去重（避免为几百 MB 数据建全量集合），
        # 记 NaN 表示「未知」，由 schema 侧跳过唯一性判据而不是猜一个值。
        data["idx_nunique"] = [
            float(uniques[e][0]) if len(uniques.get(e) or ()) == 1 else float("nan")
            for e in order
        ]
    out = pd.DataFrame(data)
    for name in out.columns:
        out[name] = out[name].astype("float64" if name == "idx_nunique" else "int64")
    return out


def _reconcile_episode_ids(
    episodes_meta: pd.DataFrame | None,
    data_summary: pd.DataFrame | None,
    fps: float,
    *,
    min_agreement: float = 0.99,
) -> tuple[dict[int, int], dict[str, Any]]:
    """建立 data episode_index -> meta episode_index 的映射，并给出可核查的证据。

    只在两侧编号集合不一致、但能通过「顺序 + 行数 + 视频窗口」三重校验证明是同一批
    episode 时才映射；任何一项证据不足都退回 ambiguous，禁止猜测——猜错的映射会把
    视频窗口、任务全部错配，比不映射更危险。

    Returns:
        (映射表, 诊断信息)。映射表为空表示两侧同名或无法映射。
    """
    if episodes_meta is None or episodes_meta.empty or "episode_index" not in episodes_meta.columns:
        return {}, {"mode": "unavailable", "reason": "缺少 meta/episodes，无处对齐"}
    if data_summary is None or data_summary.empty:
        return {}, {"mode": "unavailable", "reason": "缺少数据侧 episode 信息，无处对齐"}

    data_ids = [int(x) for x in data_summary["episode_index"].tolist()]
    meta_ids = sorted(int(x) for x in episodes_meta["episode_index"].tolist())
    data_set, meta_set = set(data_ids), set(meta_ids)
    diag: dict[str, Any] = {
        "data_ids": len(data_set),
        "meta_ids": len(meta_set),
        "overlap": len(data_set & meta_set),
        "data_id_range": [min(data_ids), max(data_ids)],
        "meta_id_range": [min(meta_ids), max(meta_ids)],
        "data_order_sorted": data_ids == sorted(data_ids),
    }

    if data_set == meta_set:
        diag.update(mode="native", note="两侧 episode 编号一致，未做映射")
        return {}, diag

    if len(data_ids) != len(meta_ids):
        diag.update(
            mode="ambiguous",
            reason=f"episode 数量不一致（data {len(data_ids)} / meta {len(meta_ids)}），"
                   "无法建立一一映射，需人工确认后再质检",
        )
        return {}, diag

    # ---- 顺序假设验证：同名位置上的行数、视频窗口必须自洽
    rows_arr = data_summary["rows"].to_numpy(dtype="int64")
    lengths = pd.to_numeric(episodes_meta.sort_values("episode_index")["length"], errors="coerce")
    length_arr = lengths.fillna(-1).to_numpy(dtype="int64") if "length" in episodes_meta.columns else None

    length_ok = int((rows_arr == length_arr).sum()) if length_arr is not None else 0
    diag["length_agreement"] = round(length_ok / len(data_ids), 4) if length_arr is not None else None

    video_ok, video_total = 0, 0
    for col_to in episodes_meta.columns:
        if not col_to.startswith("videos/") or not col_to.endswith("/to_timestamp"):
            continue
        col_from = col_to[: -len("/to_timestamp")] + "/from_timestamp"
        if col_from not in episodes_meta.columns:
            continue
        window = (
            episodes_meta.sort_values("episode_index")[col_to].to_numpy(dtype=float)
            - episodes_meta.sort_values("episode_index")[col_from].to_numpy(dtype=float)
        )
        frames_in_window = np.rint(window * max(fps, 1e-6)).astype("int64")
        video_ok += int((frames_in_window == rows_arr).sum())
        video_total += len(frames_in_window)
    diag["video_agreement"] = round(video_ok / video_total, 4) if video_total else None

    evidence = [
        ratio for ratio in (diag.get("length_agreement"), diag.get("video_agreement"))
        if ratio is not None
    ]
    agreement = min(evidence) if evidence else 0.0
    diag["agreement"] = round(float(agreement), 4)
    diag["meta_id_step"] = int(np.median(np.diff(meta_ids))) if len(meta_ids) > 1 else None
    diag["meta_id_gaps"] = int((np.diff(meta_ids) != (diag["meta_id_step"] or 0)).sum()) if len(meta_ids) > 1 else 0

    if not evidence or agreement < min_agreement:
        diag.update(
            mode="ambiguous",
            reason="两侧编号不同且顺序映射的证据不足"
                   f"（行数一致率 {diag.get('length_agreement')}, 视频窗口一致率 {diag.get('video_agreement')}），"
                   "不做猜测映射",
        )
        return {}, diag

    mapping = {d: m for d, m in zip(data_ids, meta_ids)}
    diag.update(
        mode="remapped",
        note="两侧编号空间不同（meta 为全局 ID），已按顺序映射并通过行数/视频窗口校验",
        sample=[
            {"data": d, "meta": m, "rows": int(r), "meta_length": int(length_arr[i]) if length_arr is not None else None}
            for i, (d, m, r) in enumerate(zip(data_ids, meta_ids, rows_arr))
            if i < 3
        ],
    )
    return mapping, diag


def heavy_feature_keys(features: dict[str, Any]) -> list[str]:
    """识别二维重型特征（点云 / embedding）：shape=[N, D] 且 N*D 达到阈值。

    这类列用 pandas 读出来是「每行一个 ndarray」的对象列，Python 对象开销可达
    原始数据的数十倍；主质检流程不需要它们，因此默认不载入。
    """
    out: list[str] = []
    for key, spec in (features or {}).items():
        shape = spec.get("shape")
        if not isinstance(shape, (list, tuple)) or len(shape) != 2:
            continue
        try:
            values = int(shape[0]) * int(shape[1])
        except (TypeError, ValueError):
            continue
        if values >= HEAVY_FEATURE_MIN_VALUES:
            out.append(key)
    return out


def _columns_excluding(path: Path, exclude: tuple[str, ...]) -> list[str] | None:
    """parquet 实际列名去掉 exclude；无 exclude 时返回 None 表示读取全部列。"""
    if not exclude:
        return None
    try:
        names = pq.ParquetFile(path).schema_arrow.names
    except Exception:  # noqa: BLE001 - schema 读不出来就退回全量
        return None
    cols = [n for n in names if n not in set(exclude)]
    return cols or None


def _feature_dim(spec: dict[str, Any]) -> int:
    """从 info.json 的 feature 描述里推断该特征的维度。"""
    shape = spec.get("shape")
    if isinstance(shape, (list, tuple)) and shape:
        if len(shape) == 1:
            return int(shape[0])
        # 图像/视频 shape 是 [C, H, W]，不是向量维度
        return 0
    if isinstance(shape, int):
        return int(shape)
    names = spec.get("names")
    if isinstance(names, (list, tuple)):
        return len(names)
    return 0


def iter_episode_slices(frames: pd.DataFrame) -> Iterator[tuple[int, pd.DataFrame]]:
    """按 episode_index 迭代，避免在 validator/visualizer 里重复写 groupby。"""
    if frames.empty or "episode_index" not in frames.columns:
        return
    for ep, sub in frames.groupby("episode_index", sort=True):
        yield int(ep), sub.sort_values("frame_index", kind="stable") if "frame_index" in sub.columns else sub


def iter_frame_chunks(
    dataset: LeRobotDatasetView,
    columns: list[str] | None = None,
    episodes: Iterator[int] | list[int] | set[int] | None = None,
) -> Iterator[pd.DataFrame]:
    """按 data/*.parquet 分片顺序逐块产出帧数据（流式模式使用）。

    只读当前文件需要的列，内存峰值约等于单个 parquet 文件；episodes 非空时
    借助 pyarrow 的 row-group filter 只取这些 episode 的行。
    """
    wanted: list[str] | None = None
    if columns is not None:
        available = set(dataset.column_names)
        wanted = [c for c in columns if c in available]
    keep = None if episodes is None else {int(e) for e in episodes}

    for path in dataset.data_files:
        file_wanted = wanted
        if wanted is not None:
            # 部分 v3 数据集（如 behavior1k）缺少 frame_index 等列；按文件真实 schema 过滤
            try:
                names = set(pq.ParquetFile(path).schema_arrow.names)
                file_wanted = [c for c in wanted if c in names]
            except Exception:  # noqa: BLE001 - schema 读不出来时让 read_table 报错
                pass
        try:
            if keep is not None:
                try:
                    table = pq.read_table(path, columns=file_wanted, filters=[("episode_index", "in", sorted(keep))])
                except Exception:  # noqa: BLE001 - 老版本 pyarrow 不支持 filter
                    table = pq.read_table(path, columns=file_wanted)
            else:
                table = pq.read_table(path, columns=file_wanted)
        except Exception as exc:  # noqa: BLE001
            raise DatasetFormatError(f"读取 parquet 失败: {path} -> {exc}") from exc
        df = table.to_pandas()
        if df.empty:
            continue
        if keep is not None and "episode_index" in df.columns:
            df = df[df["episode_index"].isin(keep)]
            if df.empty:
                continue
        df[SOURCE_COL] = str(path)
        yield df
