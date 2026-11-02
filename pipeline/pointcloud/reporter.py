"""点云报告：build_report / write_report / print_summary。"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config import severity_of, snapshot as config_snapshot
from .types import (
    REPORT_FILENAME,
    RULE_MAP,
    RULE_ORDER,
    FramesKey,
    PointCloudCleaningResult,
    PointCloudDecision,
    PointCloudValidation,
)
from .utils import display_width, health_cn, health_level, json_default, ljust, rjust

LOG = logging.getLogger(__name__)


def build_report(
    validation: PointCloudValidation,
    decisions: dict[FramesKey, PointCloudDecision],
    cleaning: PointCloudCleaningResult | None = None,
    *,
    dataset_path: str | Path | None = None,
) -> dict[str, Any]:
    total = max(validation.total_frames, 1)
    counts = {d: sum(1 for x in decisions.values() if x.disposition == d)
              for d in ("keep", "repair", "review", "drop")}
    freq = validation.reason_frequency()
    sev = validation.severity_summary()
    invalid = len(validation.invalid_frames)
    repair_actions: dict[str, int] = {}
    for d in decisions.values():
        if d.disposition == "repair":
            for action in d.repair_actions:
                repair_actions[action] = repair_actions.get(action, 0) + 1

    report: dict[str, Any] = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "dataset": {
            "path": str(dataset_path or validation.dataset_root),
            "feature_key": validation.feature_key,
            "dim": validation.dim,
            "expected_points": validation.expected_points,
            "total_frames": validation.total_frames,
        },
        "config": config_snapshot(validation.config),
        "resolved": validation.resolved,
        "corrupted_files": validation.corrupted_files,
        "summary": {
            "total_frames": validation.total_frames,
            "valid_frames": validation.valid_frames,
            "invalid_frames": invalid,
            "valid_ratio": round(validation.valid_frames / total, 4),
            "severity": sev,
            "disposition": counts,
            "repair_actions": repair_actions,
            "health_level": health_level(invalid / total),
            "remaining_frames": validation.total_frames - counts["drop"],
        },
        "checks": [
            {
                "code": code,
                "label": RULE_MAP[code].label,
                "description": RULE_MAP[code].description,
                "severity": severity_of(validation.config, code),
                "repair": RULE_MAP[code].repair,
                "affected_frames": freq.get(code, 0),
                "affected_ratio": round(freq.get(code, 0) / total, 4),
                "status": "skipped" if code in validation.skipped else "enabled",
            }
            for code in RULE_ORDER
        ],
        "reason_frequency": freq,
        "episodes": validation.episode_summary(),
        "invalid_frame_details": [r.to_dict() for r in validation.invalid_frames],
        "dataset_issues": validation.dataset_issues,
        "skipped_checks": validation.skipped,
        "cleaning": cleaning.to_dict() if cleaning else None,
        "artifacts": {},
        "recommendations": _recommendations(validation, decisions, cleaning),
    }
    return report


def write_report(report: dict[str, Any], output_dir: str | Path) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / REPORT_FILENAME
    with path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2, default=json_default)
    LOG.info("pointcloud_report.json 已写入 %s", path)
    return path


def print_summary(report: dict[str, Any], *, limit: int = 10) -> None:
    ds = report["dataset"]
    sm = report["summary"]
    line = "=" * 76
    print(line)
    print("3D 点云质检报告")
    print(line)
    print(f"数据集      : {ds['path']}")
    print(f"点云特征    : {ds['feature_key']}  ({ds.get('expected_points') or '变长'} 点 × {ds['dim']} 维)")
    print(f"帧数        : {sm['total_frames']}    正常 {sm['valid_frames']} ({sm['valid_ratio']:.1%})    "
          f"异常 {sm['invalid_frames']}    健康度: {health_cn(sm['health_level'])}")
    print(f"严重度      : error {sm['severity'].get('error', 0)} / warning {sm['severity'].get('warning', 0)} / "
          f"info {sm['severity'].get('info', 0)}")
    print(f"处置        : 保留 {sm['disposition']['keep']} / 修复 {sm['disposition']['repair']} / "
          f"复核 {sm['disposition']['review']} / 剔除 {sm['disposition']['drop']}")
    ws = report.get("resolved") or {}
    if ws.get("workspace_lower"):
        lo, hi = ws["workspace_lower"], ws["workspace_upper"]
        print(f"工作空间    : [{lo[0]:.2f},{lo[1]:.2f},{lo[2]:.2f}] ~ [{hi[0]:.2f},{hi[1]:.2f},{hi[2]:.2f}]"
              f"（{ws.get('workspace_mode')}）")
    print(f"离群半径    : {ws.get('outlier_radius')} m（1-NN 中位数 {ws.get('outlier_nn_median')} m × "
          f"{ws.get('outlier_radius_k')}）    场景尺度: {ws.get('scene_extent')} m")
    print("-" * 76)
    print(ljust("检查项", 30) + rjust("级别", 8) + rjust("帧数", 8) + rjust("占比", 9))
    for item in report["checks"]:
        if item["status"] == "enabled" and item["affected_frames"] == 0:
            continue
        print(ljust(item["label"][:28], 30) + rjust(item["severity"], 8)
              + rjust("跳过" if item["status"] == "skipped" else str(item["affected_frames"]), 8)
              + rjust("-" if item["status"] == "skipped" else f"{item['affected_ratio']:.1%}", 9))
    details = report.get("invalid_frame_details") or []
    if details:
        print("-" * 76)
        print(f"异常帧明细（最多 {limit} 条，全量见 pointcloud_report.json）：")
        for item in details[:limit]:
            print(f"  ep {item['episode_index']:>3}  frame {item['frame_index']:>4}  "
                  f"points={item['point_count']:>4}  {','.join(item['reasons'])}")
    recs = report.get("recommendations") or []
    if recs:
        print("-" * 76)
        print("建议：")
        for i, rec in enumerate(recs, 1):
            print(f"  {i}. {rec}")
    print(line)


def _recommendations(
    validation: PointCloudValidation,
    decisions: dict[FramesKey, PointCloudDecision],
    cleaning: PointCloudCleaningResult | None,
) -> list[str]:
    freq = validation.reason_frequency()
    recs: list[str] = []
    if not validation.invalid_frames:
        recs.append("点云质量良好：未发现 NaN/离群/重复/越界问题，可直接用于训练。")
        return recs
    if freq.get("pc_non_finite"):
        recs.append(f"{freq['pc_non_finite']} 帧含 NaN/Inf：优先排查深度反投影与采样器序列化。")
    if freq.get("pc_outliers"):
        recs.append(f"{freq['pc_outliers']} 帧离群点超阈值：多为反光/透明物体穿透，已做半径滤波修复。")
    if freq.get("pc_duplicate"):
        recs.append(f"{freq['pc_duplicate']} 帧重复点超阈值：检查上游是否对点云做了重复采样。")
    if freq.get("pc_out_of_bounds"):
        recs.append(f"{freq['pc_out_of_bounds']} 帧越出工作空间：优先核对相机外参与机器人基座标系。")
    if freq.get("pc_temporal_jump"):
        recs.append(f"{freq['pc_temporal_jump']} 帧质心突变：已进入复核队列，回放相邻帧确认是真实运动还是跟踪跳变。")
    if freq.get("pc_empty_or_sparse"):
        recs.append(f"{freq['pc_empty_or_sparse']} 帧有效点过少：属不可修复采集事故，已用同 episode 最近好帧占位。")
    if cleaning and cleaning.warnings:
        recs.append(f"清洗产生 {len(cleaning.warnings)} 条告警，详见 pointcloud_cleaning_notes.json。")
    recs.append("点云修复保持固定点数契约：过滤坏点后用抖动重采样补点，不改变 parquet 行数。")
    return recs
