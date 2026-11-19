"""步态/光捕批量 QC 报告：目录级扫描 → 逐 episode 全规则 → JSON 留痕。

用法：
    python -m lerobot_data_quality.gait_unit.report \
        --source data/chingmu_motiondecode/samples \
        --out reports/gait/motiondecode_report.json [--max-files 5]

分类（按表头自动识别，不猜扩展名）：
  - 含 root_pos_x(m) 列 → MotionDecode 重定向轨迹（Tier1 通道规则 +
    G1 限位 + FK 空间规则）
  - 含 marker_id 列 → FZMotion marker 点集（点级规则）
  - 其余跳过并留痕

产物：单 JSON（每 episode 的 findings + 汇总）。规则无处置权——
findings 只是证据，处置由 policy 层/人决定。
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from lerobot_data_quality.gait_unit.adapters import (
    load_fzmotion_points,
    load_motiondecode_csv,
)
from lerobot_data_quality.gait_unit.ir import validate_clip
from lerobot_data_quality.gait_unit.kinematics import fk_motion_clip, load_skeleton
from lerobot_data_quality.gait_unit.limits import load_g1_limits
from lerobot_data_quality.gait_unit.motion import validate_motion
from lerobot_data_quality.gait_unit.profiles import profile_params, resolve_profile
from lerobot_data_quality.gait_unit.qc_rules.channel_rules import (
    rule_constant_channel,
    rule_mad_jump,
    rule_nan_runs,
    rule_range_check,
)
from lerobot_data_quality.gait_unit.qc_rules.point_rules import (
    rule_marker_dropout,
    rule_marker_jump,
)
from lerobot_data_quality.gait_unit.qc_rules.spatial_rules import (
    compute_foot_contact,
    rule_flight_phase,
    rule_foot_skating,
    rule_ground_penetration,
)

LOG = logging.getLogger(__name__)

DEFAULT_URDF = Path("data/external/g1_description/g1_29dof.urdf")
DEFAULT_LIMITS = Path("config/g1_29dof_limits.json")
DEFAULT_FEET = ("left_ankle_roll_joint", "right_ankle_roll_joint")


def _classify(path: Path) -> str:
    """读表头第一行分类。文件打不开/空 → unknown。"""
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            header = fh.readline()
    except OSError:
        return "unknown"
    if "root_pos_x(m)" in header:
        return "motiondecode"
    if "marker_id" in header:
        return "fzmotion_points"
    return "unknown"


def _report_motion(path: Path, urdf: Path | None, limits: dict | None) -> dict:
    entry: dict = {"file": str(path), "kind": "motiondecode"}
    clip = load_motiondecode_csv(path)
    entry["n_frames"], entry["n_channels"] = clip.n_frames, clip.n_channels
    issues = validate_motion(clip)
    entry["structural_issues"] = issues
    profile = resolve_profile(path)
    params = profile_params(profile)
    entry["profile"] = profile

    findings: dict[str, list] = {}
    findings["mad_jump"] = rule_mad_jump(clip, k=params["mad_jump_k"],
                                         skip_channel_patterns=
                                         params["mad_jump_skip_channel_patterns"])
    findings["nan_runs"] = rule_nan_runs(clip)
    findings["constant_channel"] = rule_constant_channel(clip)
    findings["range_check"] = rule_range_check(clip, limits)

    if urdf is not None and urdf.is_file():
        spec = load_skeleton(urdf)
        gait = fk_motion_clip(clip, spec)
        issues_fk = validate_clip(gait)
        entry["structural_issues"] += issues_fk
        findings["foot_skating"] = rule_foot_skating(gait, DEFAULT_FEET)
        findings["flight_phase"] = rule_flight_phase(gait, DEFAULT_FEET)
        contact = compute_foot_contact(gait, DEFAULT_FEET)
        entry["contact_ratio"] = {name: round(float(c.mean()), 4)
                                  for name, c in contact["contact"].items()}
    else:
        findings["spatial"] = [{"rule": "spatial", "skipped": True,
                                "reason": f"URDF 不存在: {urdf}"}]
    entry["findings"] = findings
    entry["n_findings"] = {k: len(v) for k, v in findings.items()}
    return entry


def _report_points(path: Path) -> dict:
    entry: dict = {"file": str(path), "kind": "fzmotion_points"}
    clip = load_fzmotion_points(path)
    entry["n_frames"] = clip.n_frames
    entry["n_markers"] = clip.n_joints
    entry["structural_issues"] = validate_clip(clip)
    findings = {"marker_dropout": rule_marker_dropout(clip),
                "marker_jump": rule_marker_jump(clip)}
    entry["findings"] = findings
    entry["n_findings"] = {k: len(v) for k, v in findings.items()}
    return entry


def run_gait_report(source_dir: str | Path, out_path: str | Path,
                    urdf: Path | None = None, limits_path: Path | None = None,
                    max_files: int | None = None) -> dict:
    """目录级批量 QC。返回汇总 dict 并写出 JSON。"""
    source_dir = Path(source_dir)
    if not source_dir.is_dir():
        raise FileNotFoundError(f"数据目录不存在: {source_dir}")
    files = sorted(source_dir.rglob("*.csv"))
    if max_files:
        files = files[:max_files]

    limits = None
    if limits_path is not None and Path(limits_path).is_file():
        limits = load_g1_limits(limits_path)
    urdf_path = Path(urdf) if urdf is not None else None

    episodes: list[dict] = []
    skipped: list[dict] = []
    for path in files:
        kind = _classify(path)
        try:
            if kind == "motiondecode":
                episodes.append(_report_motion(path, urdf_path, limits))
            elif kind == "fzmotion_points":
                episodes.append(_report_points(path))
            else:
                skipped.append({"file": str(path), "kind": kind})
        except Exception as exc:                      # 单文件失败不拖垮整批
            episodes.append({"file": str(path), "kind": kind,
                             "error": repr(exc)[:500]})

    summary: dict[str, int] = {}
    for ep in episodes:
        for rule, n in ep.get("n_findings", {}).items():
            summary[rule] = summary.get(rule, 0) + int(n)

    report = {
        "source_dir": str(source_dir),
        "n_files_scanned": len(files),
        "n_episodes": len(episodes),
        "n_skipped": len(skipped),
        "summary_findings_by_rule": summary,
        "episodes": episodes,
        "skipped": skipped,
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    LOG.info("报告写出: %s（%d episode / %d skipped）",
             out_path, len(episodes), len(skipped))
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description="光捕/步态批量 QC 报告")
    ap.add_argument("--source", required=True, help="数据目录（递归扫描 CSV）")
    ap.add_argument("--out", required=True, help="报告 JSON 输出路径")
    ap.add_argument("--urdf", default=str(DEFAULT_URDF), help="G1 URDF 路径")
    ap.add_argument("--limits", default=str(DEFAULT_LIMITS), help="G1 限位表 JSON")
    ap.add_argument("--max-files", type=int, default=None)
    args = ap.parse_args()
    report = run_gait_report(args.source, args.out,
                             urdf=Path(args.urdf) if args.urdf else None,
                             limits_path=Path(args.limits) if args.limits else None,
                             max_files=args.max_files)
    print(json.dumps(report["summary_findings_by_rule"], ensure_ascii=False, indent=2))
    print(f"报告: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
