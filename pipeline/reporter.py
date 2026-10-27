"""质检报告生成：quality_report.json + 终端简洁版。

报告要能同时服务两类读者：
- 机器/下游流水线：结构化 JSON（含处置建议、阈值快照、数据集指纹，可 diff 历史报告）
- 人：终端摘要 + 复核队列 CSV（谁该看哪几条、建议怎么处理）

真实业务里报告还不止「哪个 episode 脏」，更重要的是**归因**：
按采集员/日期/任务的通过率，才能推动上游修采集流程。
"""

from __future__ import annotations

import hashlib
import json
import logging
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import cfg_get, severity_of, snapshot as config_snapshot
from .loader import ACTION_KEY, STATE_KEY, LeRobotDatasetView
from .policy import Decision, summarize as summarize_decisions
from .validator import RULE_MAP, RULE_ORDER, ValidationResult

LOG = logging.getLogger(__name__)

REPORT_SCHEMA_VERSION = "1.1"
REPORT_FILENAME = "quality_report.json"

#: episodes 元数据里这些列是结构字段，不做归因
_NON_ATTRIBUTION_COLUMNS = {
    "episode_index", "length", "tasks", "dataset_from_index", "dataset_to_index",
    "data/chunk_index", "data/file_index", "meta/episode_id",
}


# ------------------------------------------------------------------- 构建
def build_report(
    dataset: LeRobotDatasetView,
    validation: ValidationResult,
    decisions: dict[int, Decision] | None = None,
    *,
    dataset_path: str | Path | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    decisions = decisions or {}
    total = validation.total_episodes
    valid = validation.valid_episodes
    invalid = total - valid
    frames_by_ep = dataset.frames.groupby("episode_index").size()
    total_frames = int(dataset.num_frames)

    dropped = [ep for ep, d in decisions.items() if d.disposition == "drop"]
    repaired = [ep for ep, d in decisions.items() if d.disposition == "repair"]
    drop_frames = int(frames_by_ep[frames_by_ep.index.isin(dropped)].sum()) if dropped else 0
    repair_frames = int(frames_by_ep[frames_by_ep.index.isin(repaired)].sum()) if repaired else 0
    # 修复里的「裁帧」会真实减少帧数，剩余帧数必须扣掉，否则与清洗产物对不上
    trim_frames = sum(
        int(d.trim.get("head_frames", 0)) + int(d.trim.get("tail_frames", 0))
        for d in decisions.values()
        if d.disposition == "repair" and d.trim
    )

    freq = validation.reason_frequency()
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "generated_at": generated_at or datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "dataset": {
            "path": str(dataset_path or dataset.root),
            "name": Path(str(dataset_path or dataset.root)).name,
            "codebase_version": dataset.codebase_version,
            "fps": dataset.fps,
            "robot_type": dataset.info.get("robot_type"),
            "total_episodes": total,
            "total_frames": total_frames,
            "max_episodes_applied": dataset.truncated_to,
            "video_features": dataset.video_features,
            "state_dims": len(dataset.joint_names),
            "fingerprint": dataset_fingerprint(dataset),
        },
        "config": config_snapshot(validation.config),
        "resolved_thresholds": validation.resolved_thresholds,
        "summary": {
            "total_episodes": total,
            "valid_episodes": valid,
            "invalid_episodes": invalid,
            "valid_ratio": _ratio(valid, total),
            "severity": validation.severity_summary(),
            "disposition": summarize_decisions(decisions)["counts"] if decisions else {},
            "repair_breakdown": summarize_decisions(decisions)["repair_breakdown"] if decisions else {},
            "overridden_by_human": summarize_decisions(decisions)["overridden"] if decisions else 0,
            "invalid_frames": int(frames_by_ep[frames_by_ep.index.isin(validation.invalid_indices)].sum())
            if invalid else 0,
            "drop_frames": drop_frames,
            "trim_frames": trim_frames,
            "repair_frames": repair_frames,
            "invalid_frame_ratio": _ratio(drop_frames, total_frames, digits=4),
            "remaining_episodes": total - len(dropped),
            "remaining_frames": total_frames - drop_frames - trim_frames,
            "health_level": _health_level(_ratio(invalid, total)),
            "has_video_probe": validation.probe_available,
        },
        "checks": [
            {
                "code": code,
                "label": RULE_MAP[code].label,
                "description": RULE_MAP[code].description,
                "severity": severity_of(validation.config, code),
                "repair": RULE_MAP[code].repair,
                "affected_episodes": freq.get(code, 0),
                "affected_ratio": _ratio(freq.get(code, 0), total),
                "status": "skipped" if code in validation.skipped_checks else "enabled",
            }
            for code in RULE_ORDER
        ],
        "reason_frequency": {code: freq.get(code, 0) for code in RULE_ORDER},
        "field_issue_stats": _field_issue_stats(dataset, validation),
        "dataset_issues": [issue.to_dict() for issue in validation.dataset_issues],
        "invalid_episode_details": [_episode_payload(r, validation, decisions) for r in validation.invalid_episodes],
        "attribution": attribute_by_metadata(dataset, validation, decisions),
        "skipped_checks": validation.skipped_checks,
        "cleaning": None,
        "artifacts": {},
        "recommendations": _recommendations(validation, decisions, drop_frames, total_frames),
    }
    LOG.info("报告已构建：%s", {k: report["summary"][k] for k in
                               ("total_episodes", "valid_episodes", "invalid_episodes", "health_level")})
    return report


def _episode_payload(res, validation: ValidationResult, decisions: dict[int, Decision]) -> dict[str, Any]:
    d = decisions.get(res.episode_index)
    return {
        "episode_index": res.episode_index,
        "frame_count": res.frame_count,
        "reasons": list(res.reasons),
        "reason_labels": [RULE_MAP[c].label if c in RULE_MAP else c for c in res.reasons],
        "severity": validation.severity_of(res),
        "disposition": d.disposition if d else None,
        "recommended_action": d.recommended_action if d else None,
        "details": res.details,
    }


def write_report(report: dict[str, Any], output_dir: str | Path) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / REPORT_FILENAME
    with path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2, default=_json_default)
    LOG.info("quality_report.json 已写入 %s", path)
    return path


def attach_cleaning_result(report_path: str | Path, cleaning: dict[str, Any]) -> None:
    """把清洗结果回填进已生成的报告。"""
    path = Path(report_path)
    if not path.is_file():
        LOG.warning("报告不存在，跳过清洗结果回填: %s", path)
        return
    with path.open("r", encoding="utf-8") as fh:
        report = json.load(fh)
    report["cleaning"] = cleaning
    summary = report.setdefault("summary", {})
    kept = cleaning.get("kept_episodes")
    if isinstance(kept, int):
        summary["after_cleaning_episodes"] = kept
    with path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2, default=_json_default)
    LOG.info("清洗结果已回填到 %s", path)


def attach_artifacts(report_path: str | Path, artifacts: dict[str, str]) -> None:
    path = Path(report_path)
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8") as fh:
        report = json.load(fh)
    report.setdefault("artifacts", {}).update(artifacts)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2, default=_json_default)


# ------------------------------------------------------------------- 终端版
def print_summary(report: dict[str, Any], *, invalid_limit: int = 12) -> None:
    ds = report["dataset"]
    sm = report["summary"]
    sev = sm.get("severity", {})
    disp = sm.get("disposition", {})
    line = "=" * 76

    print(line)
    print("LeRobot 数据质检报告")
    print(line)
    print(f"数据集      : {ds['path']}")
    print(f"格式版本    : {ds['codebase_version']}    FPS: {ds['fps']:g}    机器人: {ds['robot_type']}")
    print(f"Episode 总数: {sm['total_episodes']}    总帧数: {ds['total_frames']}")
    if ds.get("max_episodes_applied"):
        print(f"注意        : 已按 --max_episodes 截断为前 {ds['max_episodes_applied']} 个 episode")
    print(f"数据指纹    : {ds.get('fingerprint', '-')}")
    print("-" * 76)
    print(f"正常 {sm['valid_episodes']} ({sm['valid_ratio']:.1%})    "
          f"异常 {sm['invalid_episodes']}    健康度: {_health_cn(sm['health_level'])}")
    print(f"严重度      : error {sev.get('error', 0)} / warning {sev.get('warning', 0)} / info {sev.get('info', 0)}")
    if disp:
        print(f"处置建议    : 保留 {disp.get('keep', 0)} / 修复 {disp.get('repair', 0)} / "
              f"复核 {disp.get('review', 0)} / 剔除 {disp.get('drop', 0)}")
        extra = f"，裁帧 {sm['trim_frames']} 帧" if sm.get("trim_frames") else ""
        print(f"预计保留    : {sm['total_episodes'] - disp.get('drop', 0)} 个 episode, "
              f"{sm['remaining_frames']} 帧（丢弃 {sm['drop_frames']} 帧{extra}）")
    if sm.get("overridden_by_human"):
        print(f"人工复核    : 已覆盖 {sm['overridden_by_human']} 条自动判定")
    print("-" * 76)
    print(_ljust("检查项", 34) + _rjust("级别", 8) + _rjust("episodes", 10) + _rjust("占比", 9))
    for item in report["checks"]:
        if item["affected_episodes"] == 0 and item["status"] == "enabled":
            continue
        label = _truncate(item["label"], 32)
        if item["status"] == "skipped":
            print(_ljust(label, 34) + _rjust("跳过", 8) + _rjust("-", 10) + _rjust("-", 9))
        else:
            print(_ljust(label, 34) + _rjust(item["severity"], 8)
                  + _rjust(str(item["affected_episodes"]), 10) + _rjust(f"{item['affected_ratio']:.1%}", 9))
    print("-" * 76)

    details = report["invalid_episode_details"]
    if details:
        print(f"异常 episode 明细（最多显示 {invalid_limit} 条）：")
        for item in details[:invalid_limit]:
            action = item.get("recommended_action") or "-"
            print(f"  ep {item['episode_index']:>4}  {_ljust(item['severity'], 8)} "
                  f"frames={_ljust(item['frame_count'], 6)} {','.join(item['reasons']):<40} -> {action}")
        if len(details) > invalid_limit:
            print(f"  ... 其余 {len(details) - invalid_limit} 条见 quality_report.json / review_queue.csv")
    else:
        print("未发现异常 episode，数据可直接用于训练。")

    issues = report.get("dataset_issues") or []
    if issues:
        print("-" * 76)
        print("数据集级问题（不是单个 episode 的问题）：")
        for issue in issues:
            print(f"  - [{issue['code']}] {issue['label']}"
                  f"{'  (可修复: ' + issue['repair'] + ')' if issue.get('repair') else ''}")

    attribution = report.get("attribution") or []
    if attribution:
        print("-" * 76)
        worst = attribution[0]
        print(f"归因分析（按 {worst['dimension']}）：")
        for group in worst["groups"][:5]:
            print(f"  {_ljust(str(group['value']), 14)} episodes={_ljust(group['episodes'], 6)} "
                  f"异常率 {group['invalid_ratio']:.1%}")

    recs = report.get("recommendations") or []
    if recs:
        print("-" * 76)
        print("建议：")
        for i, rec in enumerate(recs, 1):
            print(f"  {i}. {rec}")

    skipped = report.get("skipped_checks") or {}
    if skipped:
        print("-" * 76)
        print("跳过的检查（报告中已标注，结论不含该项）：")
        for code, why in skipped.items():
            print(f"  - {code}: {why}")
    print(line)


# ------------------------------------------------------------------- 归因分析
def attribute_by_metadata(
    dataset: LeRobotDatasetView,
    validation: ValidationResult,
    decisions: dict[int, Decision],
) -> list[dict[str, Any]]:
    """按 episodes 元数据里的低基数维度（采集员/日期/批次）聚合异常率。

    真实业务的价值就在这一步：告诉上游「bob 采的那批有问题」，而不是「ep37 脏了」。
    """
    if dataset.episodes.empty:
        return []
    results: list[dict[str, Any]] = []
    for col in dataset.episodes.columns:
        if col in _NON_ATTRIBUTION_COLUMNS or col.startswith("videos/") or col.startswith("data/"):
            continue
        series = dataset.episodes[col]
        if series.dtype.kind not in "OUS" or not len(series):
            continue
        try:
            values = series.astype(str)
        except (TypeError, ValueError):
            continue
        nunique = values.nunique()
        if nunique < 2 or nunique > 20:
            continue
        groups = []
        for value, rows in dataset.episodes.groupby(values):
            eps = [int(e) for e in rows["episode_index"]]
            invalids = [e for e in eps if e in validation.results and validation.results[e].reasons]
            errors = [e for e in invalids if validation.severity_of(validation.results[e]) == "error"]
            groups.append({
                "value": str(value),
                "episodes": len(eps),
                "invalid": len(invalids),
                "error": len(errors),
                "invalid_ratio": _ratio(len(invalids), len(eps)),
            })
        groups.sort(key=lambda g: (-g["invalid_ratio"], -g["invalid"]))
        if len(groups) >= 2:
            results.append({"dimension": col, "groups": groups})
    results.sort(key=lambda d: -max((g["invalid_ratio"] for g in d["groups"]), default=0))
    return results


def dataset_fingerprint(dataset: LeRobotDatasetView) -> str:
    """数据集指纹：内容变了指纹就变，保证报告可追溯到具体那一版数据。"""
    h = hashlib.sha1()
    info_path = dataset.root / "meta" / "info.json"
    if info_path.is_file():
        h.update(info_path.read_bytes())
    h.update(f"{dataset.num_episodes}|{dataset.num_frames}|{dataset.codebase_version}".encode())
    for path in sorted((dataset.root / "data").rglob("*.parquet"))[:20]:
        h.update(f"{path.name}:{path.stat().st_size}".encode())
    return h.hexdigest()[:16]


# ------------------------------------------------------------------- 内部
def _field_issue_stats(dataset: LeRobotDatasetView, validation: ValidationResult) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    for key in (STATE_KEY, ACTION_KEY):
        if key not in dataset.frames.columns:
            continue
        try:
            arr = np.asarray(dataset.frames[key].to_list(), dtype=float)
        except (TypeError, ValueError):
            stats[key] = {"error": "无法解析为数值矩阵"}
            continue
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        finite = arr[np.isfinite(arr)]
        stats[key] = {
            "dims": int(arr.shape[1]) if arr.ndim == 2 else 0,
            "nan_count": int(np.isnan(arr).sum()),
            "inf_count": int(np.isinf(arr).sum()),
            "min": round(float(finite.min()), 6) if finite.size else None,
            "max": round(float(finite.max()), 6) if finite.size else None,
            "mean": round(float(finite.mean()), 6) if finite.size else None,
        }
    jump_episodes = [r.episode_index for r in validation.results.values() if "action_jump" in r.reasons]
    if jump_episodes:
        stats.setdefault(ACTION_KEY, {})["jump_episodes"] = len(jump_episodes)
    return stats


def _recommendations(
    validation: ValidationResult, decisions: dict[int, Decision], drop_frames: int, total_frames: int
) -> list[str]:
    freq = validation.reason_frequency()
    total = max(validation.total_episodes, 1)
    recs: list[str] = []
    drops = [d for d in decisions.values() if d.disposition == "drop"]
    repairs = [d for d in decisions.values() if d.disposition == "repair"]
    reviews = [d for d in decisions.values() if d.disposition == "review"]

    if not validation.invalid_episodes and not validation.dataset_issues:
        return ["未发现结构性问题：建议直接进入训练，并抽查 1-2 个 episode 的可视化回放。"]

    if repairs:
        kinds = {}
        for d in repairs:
            kinds[d.repair or "unknown"] = kinds.get(d.repair or "unknown", 0) + 1
        recs.append(
            f"{len(repairs)} 个 episode 可自动修复（"
            + "、".join(f"{k}:{v}" for k, v in kinds.items())
            + "）：优先修复而非删除，可避免误丢有效数据。"
        )
    if reviews:
        names = "、".join(f"ep{d.episode_index}" for d in reviews[:5])
        recs.append(
            f"{len(reviews)} 个 episode 需人工复核（{names}{'...' if len(reviews) > 5 else ''}）："
            "这些多是 action 跳变等「可能是真实动作」的问题，建议回放对应视频片段后再定。"
        )
    if freq.get("nan_state") or freq.get("nan_action"):
        recs.append(
            f"NaN/Inf 涉及 {freq.get('nan_state', 0)} 个 state / {freq.get('nan_action', 0)} 个 action episode："
            "优先排查采集端序列化与控制环丢包；中段坏帧无法插值，应剔除。"
        )
    if freq.get("blank_video"):
        recs.append(
            f"{freq['blank_video']} 个 episode 画面黑屏/纯色：典型原因是镜头遮挡或相机未推流，"
            "建议在采集端加实时预览与掉流告警，比事后清洗更省钱。"
        )
    if freq.get("duplicate_episode"):
        recs.append(
            f"{freq['duplicate_episode']} 个 episode 动作轨迹完全重复：数据集合并/重跑时容易重复入库，"
            "建议入库前按内容指纹去重。"
        )
    if freq.get("static_episode"):
        recs.append(
            f"{freq['static_episode']} 个 episode 几乎无动作：机器人在空转，属无效采集，建议剔除并排查操作流程。"
        )
    if freq.get("frame_mismatch"):
        recs.append(
            f"视频与 state 帧数不一致涉及 {freq['frame_mismatch']} 个 episode：多为采集中断导致 mp4 未收尾，"
            "训练按帧对齐会错位，建议剔除后重录。"
        )
    for issue in validation.dataset_issues:
        if issue.code == "stats_stale":
            recs.append("meta/stats.json 与实测数据不符：归一化会整体偏移，清洗后用重算的 stats 覆盖即可。")
        elif issue.code == "meta_index_inconsistent":
            recs.append("全局 index 不连续：会影响 DataLoader 采样与分片，清洗时已按真实行数重建。")

    ratio = len(validation.invalid_episodes) / total
    if ratio > 0.3:
        recs.append(f"异常率 {ratio:.0%} 偏高，属系统性问题：建议先修采集流程再清洗，否则剩余数据不足以支撑训练。")
    elif drop_frames and total_frames:
        recs.append(
            f"预计丢弃 {drop_frames} 帧（占 {drop_frames / total_frames:.1%}），"
            "清洗后 stats.json 会自动重算，避免归一化统计被脏数据污染。"
        )
    return recs


def _health_level(ratio: float) -> str:
    if ratio <= 0:
        return "clean"
    if ratio <= 0.05:
        return "minor"
    if ratio <= 0.25:
        return "moderate"
    return "severe"


def _health_cn(level: str) -> str:
    return {
        "clean": "干净",
        "minor": "轻微",
        "moderate": "中等（需清洗并复盘采集）",
        "severe": "严重（先修采集流程）",
    }.get(level, level)


def _ratio(num: float, den: float, digits: int = 4) -> float:
    if not den:
        return 0.0
    return round(float(num) / float(den), digits)


# --- 终端表格对齐：中文按 2 个显示宽度计算，否则列会错位 ---
def _display_width(text: str) -> int:
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in str(text))


def _ljust(text: str, width: int) -> str:
    return str(text) + " " * max(0, width - _display_width(text))


def _rjust(text: str, width: int) -> str:
    return " " * max(0, width - _display_width(text)) + str(text)


def _truncate(text: str, width: int) -> str:
    out, used = "", 0
    for ch in str(text):
        w = 2 if unicodedata.east_asian_width(ch) in "WF" else 1
        if used + w > width:
            break
        out += ch
        used += w
    return out


def _json_default(obj: Any):  # noqa: ANN401
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)
