"""批量报告验收：目录分类 / 双路径规则 / 汇总留痕 / 真实样本冒烟。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from lerobot_data_quality.gait_unit.report import run_gait_report

REAL_URDF = Path("data/external/g1_description/g1_29dof.urdf")
REAL_LIMITS = Path("config/g1_29dof_limits.json")
REAL_SAMPLE_DIR = Path("data/chingmu_motiondecode/samples")


def _motion_csv_text(n_frames: int = 240) -> str:
    """完整 36 通道合成 MotionDecode CSV（root 固定高度微起伏，角度小幅摆动）。

    dof 列从限位表自动生成——29 个通道缺一个，FK 预检就会拒绝（这是契约）。
    """
    limits = json.loads((Path("config/g1_29dof_limits.json")).read_text(encoding="utf-8"))
    dof_cols = [f"dof_{name}(rad)" for name in limits["joints"]]
    cols = (["root_pos_x(m)", "root_pos_y(m)", "root_pos_z(m)",
             "root_rot_w", "root_rot_x", "root_rot_y", "root_rot_z"] + dof_cols)
    lines = [",".join(cols)]
    for f in range(n_frames):
        vals = [0.0, 0.0, 0.90 + 0.01 * np.sin(f * 0.1),
                1.0, 0.0, 0.0, 0.0]
        for i, c in enumerate(dof_cols):
            vals.append(0.1 * np.sin(f * 0.05 + i))
        lines.append(",".join(f"{v:.6f}" for v in vals))
    return "\n".join(lines) + "\n"


def _marker_csv_text() -> str:
    lines = ["timestamp,marker_id,x,y,z"]
    for f in range(100):
        for m in range(3):
            lines.append(f"{f / 120.0:.6f},{m},{0.1 * m + f * 0.001:.6f},0.90,0.00")
    return "\n".join(lines) + "\n"


def _build_tree(tmp_path: Path) -> Path:
    src = tmp_path / "samples"
    (src / "1.1.Basic_Movement_Category" / "sub").mkdir(parents=True)
    (src / "1.1.1.High_Dynamic_Movement").mkdir(parents=True)
    (src / "1.1.Basic_Movement_Category" / "sub" / "ep_default.csv").write_text(
        _motion_csv_text(), encoding="utf-8")
    (src / "1.1.1.High_Dynamic_Movement" / "ep_hd.csv").write_text(
        _motion_csv_text(), encoding="utf-8")
    (src / "marker_ep.csv").write_text(_marker_csv_text(), encoding="utf-8")
    (src / "unknown.csv").write_text("foo,bar\n1,2\n", encoding="utf-8")
    return src


def test_report_classification_and_profiles(tmp_path):
    src = _build_tree(tmp_path)
    out = tmp_path / "report.json"
    report = run_gait_report(src, out, urdf=REAL_URDF, limits_path=REAL_LIMITS)

    kinds = [e["kind"] for e in report["episodes"]]
    assert kinds.count("motiondecode") == 2
    assert kinds.count("fzmotion_points") == 1
    assert report["n_skipped"] == 1          # unknown.csv
    profiles = {Path(e["file"]).name: e["profile"] for e in report["episodes"]
                if "profile" in e}
    assert profiles["ep_default.csv"] == "default"
    assert profiles["ep_hd.csv"] == "high_dynamic"
    # 汇总是各 episode findings 的合计
    assert isinstance(report["summary_findings_by_rule"], dict)
    assert out.exists()


def test_report_rules_run_on_motion(tmp_path):
    src = _build_tree(tmp_path)
    report = run_gait_report(src, tmp_path / "r2.json", urdf=REAL_URDF,
                             limits_path=REAL_LIMITS)
    ep = next(e for e in report["episodes"] if e["file"].endswith("ep_default.csv"))
    # 空间规则跑过（FK 成功）：足端接触占比存在、腾空规则有键
    assert "contact_ratio" in ep
    for rule in ("mad_jump", "nan_runs", "constant_channel", "range_check",
                 "foot_skating", "flight_phase"):
        assert rule in ep["findings"]
    # 合成数据：所有通道在限位内、无 NaN → 这些规则零 findings
    assert ep["findings"]["range_check"] == []
    assert ep["findings"]["nan_runs"] == []


@pytest.mark.skipif(not REAL_SAMPLE_DIR.exists(), reason="真实样本库不在本机")
def test_report_real_samples_smoke(tmp_path):
    report = run_gait_report(REAL_SAMPLE_DIR, tmp_path / "real.json",
                             urdf=REAL_URDF, limits_path=REAL_LIMITS,
                             max_files=3)
    motion = [e for e in report["episodes"] if e["kind"] == "motiondecode"]
    assert motion, "真实样本库里应有 motiondecode 轨迹"
    for e in motion:
        assert e["n_frames"] > 600
        assert "contact_ratio" in e            # FK 跑通 → 空间层有输出
