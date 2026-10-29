"""增量处理的状态文件：记录每个 episode 的指纹与上次结论。

设计原则：
1. **不写源数据集**：manifest 落在 output_dir，源数据始终保持只读。
2. **指纹廉价**：只 stat 源文件 + 读取 meta/episodes 的结构字段，不做全量哈希，
   否则「判断有没有变化」本身的成本就接近重跑。
3. **配置参与指纹**：阈值/严重度变了，旧结论必须失效。
4. **按 episode 粒度**：只重跑新增或变更的 episode；未变的结论直接复用。
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from .loader import DEFAULT_DATA_PATH, LeRobotDatasetView
from .validator import EpisodeResult

LOG = logging.getLogger(__name__)

MANIFEST_FILENAME = "quality_manifest.json"
MANIFEST_SCHEMA_VERSION = 1


def config_digest(config: Mapping[str, Any]) -> str:
    """配置指纹：阈值或严重度变化时让旧结论失效。"""
    raw = json.dumps(config, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def cleaning_signature(decisions: Mapping[int, Any], signatures: Mapping[int, str]) -> str:
    """清洗计划指纹：处置 + 每个 episode 的内容指纹都相同才允许复用产出。"""
    plan = [
        {
            "episode_index": int(ep),
            "disposition": str(getattr(d, "disposition", d)),
            "repair": getattr(d, "repair", None),
            "trim": getattr(d, "trim", None),
            "signature": signatures.get(int(ep)),
        }
        for ep, d in sorted(decisions.items())
    ]
    raw = json.dumps(plan, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


@dataclass
class QualityManifest:
    path: Path
    payload: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------- 读写
    @classmethod
    def load(cls, path: str | Path) -> "QualityManifest":
        p = Path(path)
        if not p.is_file():
            return cls(p, {})
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            LOG.warning("manifest 无法解析，将按首次运行处理: %s", exc)
            return cls(p, {})
        if not isinstance(payload, dict) or payload.get("schema_version") != MANIFEST_SCHEMA_VERSION:
            LOG.warning("manifest 版本不兼容，将按首次运行处理: %s", p)
            return cls(p, {})
        return cls(p, payload)

    def save(self) -> None:
        self.payload["schema_version"] = MANIFEST_SCHEMA_VERSION
        self.payload["updated_at"] = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as fh:
            json.dump(self.payload, fh, ensure_ascii=False, indent=2, default=str)
        LOG.info("增量状态已保存: %s", self.path)

    # ------------------------------------------------------------- 状态查询
    @property
    def has_results(self) -> bool:
        return bool(self.payload.get("episodes"))

    def matches_root(self, dataset: LeRobotDatasetView) -> bool:
        return str(self.payload.get("dataset_root")) == str(dataset.root.resolve())

    def resolved_thresholds(self) -> dict[str, Any]:
        return self.payload.get("resolved_thresholds") or {}

    def cached_results(self) -> dict[int, EpisodeResult]:
        out: dict[int, EpisodeResult] = {}
        for raw_ep, entry in (self.payload.get("episodes") or {}).items():
            try:
                ep = int(raw_ep)
            except (TypeError, ValueError):
                continue
            data = (entry or {}).get("result")
            if isinstance(data, dict):
                out[ep] = EpisodeResult.from_dict(data)
        return out

    def cached_decisions(self) -> dict[int, dict[str, Any]]:
        out: dict[int, dict[str, Any]] = {}
        for raw_ep, entry in (self.payload.get("episodes") or {}).items():
            try:
                ep = int(raw_ep)
            except (TypeError, ValueError):
                continue
            decision = (entry or {}).get("decision")
            if isinstance(decision, dict):
                out[ep] = decision
        return out

    # ------------------------------------------------------------- 指纹
    def episode_signatures(self, dataset: LeRobotDatasetView) -> dict[int, str]:
        """每个 episode 的廉价指纹；只依赖 meta 与文件 stat，不读帧数据。"""
        stat_cache: dict[str, dict[str, Any]] = {}
        return {
            ep: _episode_signature(dataset, ep, stat_cache)
            for ep in dataset.episode_indices
        }

    def unchanged_episodes(self, dataset: LeRobotDatasetView, config: Mapping[str, Any]) -> set[int]:
        """可复用旧结论的 episode；配置或数据集根变化时返回空集。"""
        if not self.has_results or not self.matches_root(dataset):
            return set()
        if self.payload.get("config_digest") != config_digest(config):
            LOG.info("配置已变化，增量缓存失效，将全量重检")
            return set()
        known = self.payload.get("episodes") or {}
        current = self.episode_signatures(dataset)
        unchanged: set[int] = set()
        for ep, sig in current.items():
            entry = known.get(str(ep))
            if entry and entry.get("signature") == sig and isinstance(entry.get("result"), dict):
                unchanged.add(ep)
        return unchanged

    def changed_episodes(self, dataset: LeRobotDatasetView, config: Mapping[str, Any]) -> set[int]:
        unchanged = self.unchanged_episodes(dataset, config)
        return {ep for ep in dataset.episode_indices if ep not in unchanged}

    # ------------------------------------------------------------- 写入
    def store_results(
        self,
        dataset: LeRobotDatasetView,
        config: Mapping[str, Any],
        validation: Any,
        decisions: Mapping[int, Any],
        signatures: Mapping[int, str],
    ) -> None:
        self.payload["dataset_root"] = str(dataset.root.resolve())
        self.payload["config_digest"] = config_digest(config)
        self.payload["resolved_thresholds"] = validation.resolved_thresholds
        self.payload["episodes"] = {
            str(ep): {
                "signature": signatures.get(ep),
                "result": res.to_dict(),
                "decision": decisions[ep].to_dict() if ep in decisions else None,
            }
            for ep, res in sorted(validation.results.items())
        }

    def store_cleaning(self, signature: str, result: dict[str, Any]) -> None:
        self.payload["cleaning"] = {"signature": signature, "result": result}

    def reusable_cleaning(self, signature: str) -> dict[str, Any] | None:
        cleaning = self.payload.get("cleaning") or {}
        if cleaning.get("signature") != signature:
            return None
        result = cleaning.get("result")
        if not isinstance(result, dict):
            return None
        output = result.get("output_path")
        if not output or not Path(str(output)).is_dir():
            return None
        verification = result.get("verification") or {}
        if verification and verification.get("ok") is False:
            return None
        return result


# ------------------------------------------------------------------ 指纹细节
def _episode_signature(
    dataset: LeRobotDatasetView,
    episode_index: int,
    stat_cache: dict[str, dict[str, Any]] | None = None,
) -> str:
    cache = stat_cache if stat_cache is not None else {}
    rows = dataset.episodes[dataset.episodes["episode_index"] == episode_index]
    row = rows.iloc[0] if len(rows) else pd.Series(dtype=object)
    parts: dict[str, Any] = {
        "episode_index": int(episode_index),
        "length": _scalar(row.get("length")),
        "from_index": _scalar(row.get("dataset_from_index")),
        "to_index": _scalar(row.get("dataset_to_index")),
        "data_file": _stat(_data_file(dataset, row), cache),
    }
    for cam in dataset.video_features:
        loc = dataset.video_location(cam, episode_index)
        parts[f"video:{cam}:file"] = _stat(loc[0] if loc else None, cache)
        if loc is not None:
            parts[f"video:{cam}:window"] = [round(float(loc[1]), 6), round(float(loc[2]), 6)]
    raw = json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _data_file(dataset: LeRobotDatasetView, row: pd.Series) -> Path | None:
    tmpl = dataset.info.get("data_path") or DEFAULT_DATA_PATH
    chunk = row.get("data/chunk_index")
    file_idx = row.get("data/file_index")
    if chunk is None or file_idx is None or pd.isna(chunk) or pd.isna(file_idx):
        return None
    try:
        rel = tmpl.format(
            chunk_index=int(chunk),
            file_index=int(file_idx),
            episode_index=int(row.get("episode_index", 0)),
        )
    except (KeyError, IndexError, ValueError, TypeError):
        return None
    return dataset.root / rel


def _stat(path: Path | None, cache: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    if path is None:
        return {"path": None, "exists": False}
    key = str(path)
    if cache is not None and key in cache:
        return cache[key]
    try:
        st = path.stat()
    except OSError:
        result = {"path": key, "exists": False}
    else:
        # mtime 用纳秒：同一秒内重写也能识别
        result = {"path": key, "exists": True, "size": st.st_size, "mtime_ns": st.st_mtime_ns}
    if cache is not None:
        cache[key] = result
    return result


def _scalar(value: Any) -> Any:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, (list, tuple)):
        return list(value)
    return value.item() if hasattr(value, "item") else value
