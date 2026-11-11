"""纯光捕场景：无视频特征 + 120Hz，不崩溃、不误报、能落报告。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from main import main
from pipeline import policy as policy_mod
from pipeline.config import load_config
from pipeline.loader import load_dataset
from pipeline.reporter import build_report
from pipeline.validator import validate_dataset

from conftest import JOINTS, build_dataset, smooth_episode

FPS = 120


def _mocap_episode(n: int = 240, seed: int = 0, jitter: float = 0.02) -> dict[str, np.ndarray]:
    """120Hz 光捕轨迹：平滑正弦 + 每帧微小抖动（不应被当成 action_jump）。"""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / FPS
    state = np.stack(
        [np.sin(2 * np.pi * 0.5 * t + k) for k in range(JOINTS)], axis=1
    ).astype(np.float32)
    action = state + rng.normal(0, jitter, size=state.shape).astype(np.float32)
    return {"state": state, "action": action.astype(np.float32)}


def _mocap_dataset(tmp_path: Path) -> Path:
    episodes = {ep: _mocap_episode(seed=ep) for ep in range(8)}
    # 第 7 集埋一个真实单帧尖峰：高帧率下仍应被检出（阈值缩放不是关掉检测）
    episodes[7]["action"][120, 0] += 5.0
    return build_dataset(tmp_path / "mocap", episodes, fps=FPS)


def _assert_no_video_skips(validation) -> None:
    assert "没有视频特征" in validation.skipped_checks["frame_mismatch"]
    assert "没有视频特征" in validation.skipped_checks["blank_video"]
    for res in validation.results.values():
        assert "frame_mismatch" not in res.reasons
        assert "blank_video" not in res.reasons


def test_mocap_no_video_120hz_scales_thresholds(tmp_path):
    root = _mocap_dataset(tmp_path)
    dataset = load_dataset(root)
    config = load_config(robot_type="test_robot")

    validation = validate_dataset(dataset, config)
    _assert_no_video_skips(validation)

    jump = validation.resolved_thresholds["action_jump"]
    assert jump["fps"] == 120.0
    assert jump["abs_floor_effective"] == 0.25          # 1.0 × 30/120
    assert jump["spike_ratio"] == 3.0
    # 时间戳容差同样按 1/fps 缩放，而不是固定 30Hz 的秒数
    assert validation.resolved_thresholds["timestamp_step_tolerance"] == round(0.25 / 120, 6)

    # 120Hz 的微小抖动不报警；真实尖峰仍被抓住
    assert "action_jump" not in validation.results[0].reasons
    assert "action_jump" in validation.results[7].reasons

    report = build_report(dataset, validation, policy_mod.decide(validation), dataset_path=root)
    assert report["summary"]["total_episodes"] == 8
    assert report["skipped_checks"]["frame_mismatch"]
    json.dumps(report, ensure_ascii=False)  # 报告必须可序列化落地


def test_mocap_no_video_120hz_streaming_path(tmp_path):
    root = _mocap_dataset(tmp_path)
    dataset = load_dataset(root)
    config = load_config(robot_type="test_robot")

    validation = validate_dataset(dataset, config, streaming=True)
    _assert_no_video_skips(validation)
    assert validation.resolved_thresholds["action_jump"]["fps"] == 120.0
    assert "action_jump" in validation.results[7].reasons

    report = build_report(dataset, validation, policy_mod.decide(validation), dataset_path=root)
    assert report["summary"]["invalid_episodes"] >= 1


def test_mocap_missing_action_column_does_not_crash(tmp_path):
    """字段名不标准（缺 action 列）时降级为跳过 action_jump，而不是 KeyError。"""
    root = build_dataset(tmp_path / "no_action", {ep: smooth_episode(40, seed=ep) for ep in range(4)})
    data = root / "data" / "chunk-000" / "file-000.parquet"
    pq.write_table(pq.read_table(data).drop(["action"]), data)
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["features"].pop("action")
    info_path.write_text(json.dumps(info), encoding="utf-8")

    dataset = load_dataset(root)
    validation = validate_dataset(dataset, load_config(robot_type="test_robot"))

    assert validation.skipped_checks.get("action_jump")
    assert all("action_jump" not in res.reasons for res in validation.results.values())
    report = build_report(dataset, validation, policy_mod.decide(validation), dataset_path=root)
    assert report["summary"]["total_episodes"] == 4
    json.dumps(report, ensure_ascii=False)


def test_fps_scaled_frame_bounds_and_windows():
    """帧数类配置按 30Hz 标定：120Hz 下 workload 要放大 4 倍，不能直接套用。"""
    from pipeline.validator.context import _resolve_frame_bounds, scale_frames

    config = load_config(robot_type="test_robot")
    base = _resolve_frame_bounds(np.array([], dtype=float), config, 30.0)
    high = _resolve_frame_bounds(np.array([], dtype=float), config, 120.0)

    assert (base["min"], base["max"]) == (8, 20000)
    assert (high["min"], high["max"]) == (32, 80000)
    assert scale_frames(3, config, 120.0) == 12
    # fps 未知时按 ref_fps 处理，保持原行为
    assert scale_frames(3, config, 0) == 3


def test_manual_frame_bounds_are_not_fps_scaled():
    """用户显式给的 min/max 是业务绝对值，不做帧率折算。"""
    from pipeline.validator.context import _resolve_frame_bounds

    config = load_config(robot_type="test_robot")
    config["defaults"]["frame_count"].update({"min": 100, "max": 500})
    bounds = _resolve_frame_bounds(np.array([], dtype=float), config, 120.0)
    assert (bounds["min"], bounds["max"]) == (100, 500)


def test_mocap_cli_writes_report(tmp_path):
    root = _mocap_dataset(tmp_path)
    out = tmp_path / "out"

    code = main([
        "--dataset_path", str(root),
        "--output_dir", str(out),
        "--streaming", "--skip-clean", "--no-cli", "--skip-plots",
    ])
    assert code == 0

    report = json.loads((out / "quality_report.json").read_text(encoding="utf-8"))
    assert report["dataset"]["fps"] == 120
    assert report["summary"]["total_episodes"] == 8
    assert "没有视频特征" in report["skipped_checks"]["frame_mismatch"]
    assert report["resolved_thresholds"]["action_jump"]["abs_floor_effective"] == 0.25
