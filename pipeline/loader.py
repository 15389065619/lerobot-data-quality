"""LeRobot v3 数据集加载器（只读）。

职责边界：只负责「把磁盘上的数据集读成内存里的结构」，不做任何质量校验、
不做任何写操作。校验逻辑一律放在 validator.py。

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

import pandas as pd
import pyarrow.parquet as pq

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

_SOURCE_COL = "__source_file"


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

    # ------------------------------------------------------------------ 概览
    @property
    def num_episodes(self) -> int:
        return int(len(self.episodes))

    @property
    def num_frames(self) -> int:
        return int(len(self.frames))

    @property
    def episode_indices(self) -> list[int]:
        return [int(x) for x in self.episodes["episode_index"].tolist()]

    @property
    def data_features(self) -> list[str]:
        """数值型（存在于 parquet 中的）特征名，按 info.json 顺序。"""
        out: list[str] = []
        for key, spec in self.features.items():
            dtype = str(spec.get("dtype", ""))
            if dtype.startswith(("float", "int", "uint", "bool")):
                if key in self.frames.columns:
                    out.append(key)
        return out

    @property
    def video_features(self) -> list[str]:
        return [
            key
            for key, spec in self.features.items()
            if str(spec.get("dtype", "")).lower() in {"video", "image"}
        ]

    @property
    def joint_names(self) -> list[str]:
        spec = self.features.get(STATE_KEY, {})
        names = spec.get("names") or []
        if isinstance(names, dict):  # 少数导出工具会写成 {0: "j0", ...}
            names = [names[k] for k in sorted(names, key=lambda s: int(s))]
        dim = _feature_dim(spec)
        if names and len(names) == dim:
            return [str(n) for n in names]
        return [f"dim_{i}" for i in range(dim)]

    @property
    def schema(self) -> dict[str, str]:
        """字段 schema：列名 -> "dtype shape"，直接从 parquet 实际类型 + info.json 描述合成。"""
        out: dict[str, str] = {}
        for col, dtype in self.frames.dtypes.items():
            if col == _SOURCE_COL:
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
        """取某个 episode 的所有帧，按 frame_index 升序。"""
        sub = self.frames[self.frames["episode_index"] == episode_index]
        if "frame_index" in sub.columns:
            sub = sub.sort_values("frame_index", kind="stable")
        return sub

    def task_of(self, episode_index: int) -> str:
        """episode 的任务描述；拿不到时返回空串（由 validator 判定是否异常）。"""
        if "tasks" in self.episodes.columns:
            row = self.episodes.loc[self.episodes["episode_index"] == episode_index, "tasks"]
            if len(row):
                val = row.iloc[0]
                if isinstance(val, (list, tuple)) and len(val):
                    return str(val[0]).strip()
                if isinstance(val, str) and val.strip():
                    return val.strip()
        if TASK_INDEX_KEY in self.frames.columns:
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

    def video_location(self, camera: str, episode_index: int) -> tuple[Path, float, float] | None:
        """返回 (视频文件, from_timestamp, to_timestamp)；无定位信息时 from/to 为 (-1, -1)。"""
        rows = self.episodes[self.episodes["episode_index"] == episode_index]
        ck, fk = f"videos/{camera}/chunk_index", f"videos/{camera}/file_index"
        ft, tt = f"videos/{camera}/from_timestamp", f"videos/{camera}/to_timestamp"
        if len(rows):
            row = rows.iloc[0]
            if ck in rows.columns and fk in rows.columns and pd.notna(row.get(ck)):
                path = self._video_path_from_template(camera, int(row[ck]), int(row[fk]))
                if path is not None:
                    from_ts = float(row[ft]) if ft in rows.columns and pd.notna(row.get(ft)) else -1.0
                    to_ts = float(row[tt]) if tt in rows.columns and pd.notna(row.get(tt)) else -1.0
                    return path, from_ts, to_ts
        # v2.1：文件名里带 episode 序号
        cam_dir = self.root / "videos" / camera
        if cam_dir.is_dir():
            cands = sorted(cam_dir.rglob(f"*{episode_index:06d}.mp4"))
            if cands:
                return cands[0], 0.0, -1.0
        # 兜底：该相机只有一个文件（常见于单 episode 导出）
        files = self.video_index.get(camera, {})
        if len(files) == 1:
            return next(iter(files.values())), 0.0, -1.0
        return None

    def _video_path_from_template(self, camera: str, chunk: int, file: int) -> Path | None:
        idx = self.video_index.get(camera) or {}
        if (chunk, file) in idx:
            return idx[(chunk, file)]
        tmpl = self.info.get("video_path") or DEFAULT_VIDEO_PATH
        try:
            rel = tmpl.format(video_key=camera, chunk_index=chunk, file_index=file)
        except (KeyError, IndexError, ValueError):
            return None
        path = self.root / rel
        return path if path.is_file() else None

    # ----------------------------------------------------------------- 摘要
    def summary(self) -> dict[str, Any]:
        per_ep = self.frames.groupby("episode_index").size() if self.num_frames else pd.Series(dtype=int)
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
            "columns": [c for c in self.frames.columns if c != _SOURCE_COL],
            "schema": self.schema,
            "state_key": STATE_KEY if STATE_KEY in self.frames.columns else None,
            "action_key": ACTION_KEY if ACTION_KEY in self.frames.columns else None,
            "num_state_dims": _feature_dim(self.features.get(STATE_KEY, {})),
            "num_action_dims": _feature_dim(self.features.get(ACTION_KEY, {})),
            "joint_names": self.joint_names,
            "video_features": {
                cam: [str(p.relative_to(self.root)) for p in files.values()]
                for cam, files in self.video_index.items()
            },
            "episode_meta_columns": [c for c in self.episodes.columns],
            "has_stats": bool(self.stats),
        }


# --------------------------------------------------------------------- 加载
def load_dataset(
    dataset_path: str | Path,
    max_episodes: int | None = None,
    *,
    load_frames: bool = True,
) -> LeRobotDatasetView:
    """加载本地 LeRobot 数据集，返回只读视图。

    Args:
        dataset_path: 数据集根目录（含 meta/info.json）。
        max_episodes: 只加载前 N 个 episode（按 episode_index 升序），None 表示全量。
        load_frames: False 时只读元信息，不读 parquet 帧数据。
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

    frames = pd.DataFrame()
    truncated_to = None
    if load_frames:
        keep = None
        if max_episodes is not None and max_episodes > 0:
            all_eps = _episode_order(episodes_meta, data_files)
            keep = set(all_eps[:max_episodes])
            truncated_to = len(keep)
            LOG.info("--max_episodes 生效：只加载前 %d 个 episode", truncated_to)
        frames = _read_frames(data_files, keep)
        if episodes_meta is None or episodes_meta.empty:
            LOG.warning("meta/episodes 缺失，回退为从帧数据推导 episode 元信息")
            episodes_meta = _episodes_from_frames(frames)
        if keep is not None:
            episodes_meta = episodes_meta[episodes_meta["episode_index"].isin(keep)].reset_index(drop=True)

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
    )
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


def _read_frames(files: list[Path], keep: set[int] | None) -> pd.DataFrame:
    """把 data/*.parquet 读成一张长表，记录来源文件便于清洗阶段原地重写。"""
    parts: list[pd.DataFrame] = []
    for path in files:
        df: pd.DataFrame | None = None
        if keep is not None:
            try:
                table = pq.read_table(path, filters=[("episode_index", "in", sorted(keep))])
                df = table.to_pandas()
            except Exception as exc:  # noqa: BLE001 - 不支持 filter 时退回全量读
                LOG.debug("按 episode 过滤读取 %s 失败，改为全量读取: %s", path.name, exc)
                df = None
        if df is None:
            try:
                df = pq.read_table(path).to_pandas()
            except Exception as exc:  # noqa: BLE001
                raise DatasetFormatError(f"读取 parquet 失败: {path} -> {exc}") from exc
        if df.empty:
            continue
        df[_SOURCE_COL] = str(path)
        parts.append(df)

    if not parts:
        return pd.DataFrame()

    out = pd.concat(parts, ignore_index=True)
    if keep is not None:
        out = out[out["episode_index"].isin(keep)]
    out = out.sort_values(["episode_index", "frame_index"], kind="stable").reset_index(drop=True)
    for col in ("episode_index", "frame_index", "task_index", "index"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _episodes_from_frames(frames: pd.DataFrame) -> pd.DataFrame:
    if frames.empty:
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
    if frames.empty:
        return
    for ep, sub in frames.groupby("episode_index", sort=True):
        yield int(ep), sub.sort_values("frame_index", kind="stable") if "frame_index" in sub.columns else sub
