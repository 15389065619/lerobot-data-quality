"""质检规则的单测：每条规则都要有「命中」和「不误报」两侧的用例。"""

from __future__ import annotations

import json

import numpy as np
import pytest

from pipeline.config import load_config
from pipeline.loader import load_dataset
from pipeline.validator import validate_dataset

from conftest import JOINTS, build_dataset, smooth_episode


def _validate(path, **overrides):
    dataset = load_dataset(path)
    return dataset, validate_dataset(dataset, load_config(robot_type="test_robot", overrides=overrides or None))


def test_clean_dataset_produces_no_findings(normal_dataset):
    dataset, result = _validate(normal_dataset)
    assert dataset.num_episodes == 6
    assert result.valid_episodes == 6
    assert result.invalid_episodes == []
    assert result.dataset_issues == []


def test_nan_state_in_middle_is_not_trimmable(tmp_path):
    ep = smooth_episode(40, seed=1)
    ep["state"][15:18, 0] = np.nan
    path = build_dataset(tmp_path / "ds", {0: ep, 1: smooth_episode(40, seed=2)})
    _, result = _validate(path)

    res = result.results[0]
    assert "nan_state" in res.reasons
    detail = res.details["nan_state"]
    assert detail["boundary_only"] is False
    assert detail["trimmable"] is False
    assert detail["nan_count"] == 3
    assert res.trim_plan() is None


def test_boundary_nan_is_trimmable(tmp_path):
    ep = smooth_episode(40, seed=3)
    ep["state"][:3, 1] = np.nan
    path = build_dataset(tmp_path / "ds", {0: ep, 1: smooth_episode(40, seed=4)})
    _, result = _validate(path)

    detail = result.results[0].details["nan_state"]
    assert detail["boundary_only"] is True
    assert detail["trimmable"] is True
    assert detail["head_frames"] == 3
    assert result.results[0].trim_plan() == {"head_frames": 3, "tail_frames": 0}


def test_action_jump_uses_adaptive_threshold(tmp_path):
    """正常抖动不应误报，单帧尖峰必须命中。"""
    normal = smooth_episode(40, seed=5)
    spike = smooth_episode(40, seed=6)
    spike["action"][20, 0] += 12.0

    path = build_dataset(tmp_path / "ds", {0: normal, 1: spike})
    _, result = _validate(path)

    assert "action_jump" not in result.results[0].reasons, "正常轨迹被误报为跳变"
    assert "action_jump" in result.results[1].reasons
    detail = result.results[1].details["action_jump"]
    assert detail["max_delta"] > 10
    assert detail["worst_dim"] == 0
    assert len(detail["thresholds_used"]) == 3


def _step_hold_episode(n_frames: int, seed: int, jump_at: int | None = None,
                       jump: float = 0.0) -> dict:
    """度制机器人（±100°）+ 步进保持 action：每 5 帧更新一次，单步约 2°。"""
    rng = np.random.default_rng(seed)
    base = rng.uniform(-90, 90, size=(1, 3)).astype(np.float32)
    action = np.repeat(base, n_frames, axis=0)          # 每帧初值相同
    step = (np.arange(n_frames) // 5)[:, None]           # 每 5 帧跳 2°，其余帧保持不变
    action += step * 2.0 * np.array([1, -1, 1], dtype=np.float32)
    if jump_at is not None:
        action[jump_at:] += jump
    return {"state": action.copy(), "action": action}


def test_step_hold_degree_action_not_flagged(tmp_path):
    """步进保持 + 度制 action（相邻帧大量完全相同）：不应整库误报。"""
    episodes = {ep: _step_hold_episode(120, seed=ep) for ep in range(20)}
    path = build_dataset(tmp_path / "ds", episodes)
    _, result = _validate(path)

    flagged = [r.episode_index for r in result.results.values() if "action_jump" in r.reasons]
    assert flagged == [], f"正常步进保持数据被误报: {flagged}"
    assert result.resolved_thresholds["action_jump"]["basis"] == "episode_max"


def test_step_hold_degree_action_outlier_detected(tmp_path):
    """同样的数据里注入一集 40° 突跳，只有该集应被标记。"""
    episodes = {ep: _step_hold_episode(120, seed=ep) for ep in range(20)}
    episodes[7] = _step_hold_episode(120, seed=7, jump_at=60, jump=40.0)
    path = build_dataset(tmp_path / "ds", episodes)
    _, result = _validate(path)

    flagged = [r.episode_index for r in result.results.values() if "action_jump" in r.reasons]
    assert flagged == [7]


def _bump_episode(seed: int, kind: str = "normal", n: int = 50) -> dict:
    """带连续缓动的轨迹：kind=fast 用正弦包络制造「快速但连续」的峰值，kind=spike 制造单帧突跳。"""
    action = np.zeros((n, 3), dtype=np.float32)
    action[:, 0] = np.arange(n) * 0.1
    if kind == "fast":
        # sin² 包络：两端速度自然收敛到 0，模拟真实的「加速—匀速—减速」快速摆动
        action[10:31, 0] += 30.0 * np.sin(np.pi * np.arange(21) / 20) ** 2
    elif kind == "spike":
        action[25, 0] += 3.0
    return {"state": action.copy(), "action": action}


def test_action_jump_ignores_smooth_fast_motion(tmp_path):
    """峰值超阈值但邻近帧同样大的快速连续运动，不应判为跳变；单帧尖峰仍要命中。"""
    episodes = {ep: _bump_episode(ep) for ep in range(20)}
    episodes[20] = _bump_episode(20, "fast")
    episodes[21] = _bump_episode(21, "spike")
    path = build_dataset(tmp_path / "ds", episodes)
    _, result = _validate(path)

    assert "action_jump" not in result.results[20].reasons, "连续快速运动被误报为跳变"
    assert "action_jump" in result.results[21].reasons
    detail = result.results[21].details["action_jump"]
    assert detail["spike_ratio"] == 3.0
    assert detail["max_delta"] > 2.5


def _quat_episode(seed: int, flip: bool = False, spike: bool = False, n: int = 50) -> dict:
    """JOINTS 个连续维模拟 root 四元数分量，翻转时整体取反（q≡-q）。"""
    t = np.arange(n)
    action = np.zeros((n, JOINTS), dtype=np.float32)
    for d, amp in enumerate([0.9, 0.7, 0.5][:JOINTS]):
        action[:, d] = amp * np.sin(0.1 * t + d)
    if flip:
        action[25:, :] *= -1
    if spike:
        action[25, 0] += 2.0
    return {"state": action.copy(), "action": action}


def test_action_jump_ignores_quaternion_sign_flip(tmp_path):
    """四元数整体取反（q≡-q）不应判为跳变；单维突跳仍要命中。"""
    episodes = {ep: _quat_episode(ep) for ep in range(20)}
    episodes[20] = _quat_episode(20, flip=True)
    episodes[21] = _quat_episode(21, spike=True)
    path = build_dataset(tmp_path / "ds", episodes)
    _, result = _validate(path)

    assert "action_jump" not in result.results[20].reasons, "四元数符号翻转被误报为跳变"
    assert "action_jump" in result.results[21].reasons


def test_action_jump_thresholds_scale_with_fps():
    """abs_floor 以 30Hz 标定，120Hz 数据集折算为 1/4，避免规则在光捕数据上失效或过松。"""
    from pipeline.config import load_config
    from pipeline.validator.context import _thresholds_from_arrays

    cfg = load_config(robot_type="test_robot")
    arrays = [np.zeros((50, 3), dtype=np.float32) for _ in range(3)]
    _, meta30 = _thresholds_from_arrays(arrays, cfg, fps=30.0)
    _, meta120 = _thresholds_from_arrays(arrays, cfg, fps=120.0)
    assert meta30["abs_floor_effective"] == 1.0
    assert meta120["abs_floor_effective"] == 0.25
    assert meta120["fps"] == 120.0


def test_thresholds_tolerate_mixed_action_dims():
    """跨 episode 维度不一致不应抛 ValueError（否则整次质检零报告）。"""
    from pipeline.validator.context import _thresholds_from_arrays

    cfg = {"defaults": {"action_jump": {}}}
    arrays = [np.zeros((10, 6)), np.zeros((10, 7)), np.ones((10, 6))]
    thresholds, meta = _thresholds_from_arrays(arrays, cfg)

    assert len(thresholds) == 6
    assert meta["dim_mismatch_episodes"] == 1


def test_frame_count_outlier_detected(tmp_path):
    episodes = {ep: smooth_episode(40, seed=ep) for ep in range(5)}
    episodes[5] = {"state": np.zeros((3, 3), dtype=np.float32),
                   "action": np.zeros((3, 3), dtype=np.float32)}
    path = build_dataset(tmp_path / "ds", episodes)
    dataset, result = _validate(path)

    bounds = result.resolved_thresholds["frame_count"]
    assert bounds["mode"] == "auto"
    assert bounds["min"] >= 8  # 自适应下限受 floor 约束

    res = result.results[5]
    assert "frame_count_abnormal" in res.reasons
    assert res.details["frame_count_abnormal"]["direction"] == "too_few"
    assert "frame_count_abnormal" not in result.results[0].reasons


def test_frame_bounds_are_median_plus_mad_and_hand_checkable():
    """手工可验算的阈值：10 集帧数 median=100、MAD=1 → 下界 95、上界 112。

    sigma = 1.4826·MAD = 1.4826；下界 100-3·sigma = 95.55（floor→95），
    上界 100+8·sigma = 111.86（ceil→112）。纯函数直接测，不跑整条流水线。
    """
    from pipeline.validator.context import _resolve_frame_bounds

    counts = np.array([100, 102, 98, 101, 99, 100, 103, 97, 100, 100], dtype=float)
    bounds = _resolve_frame_bounds(counts, load_config(robot_type="test_robot"))

    assert bounds["median"] == 100.0
    assert bounds["mad"] == 1.0
    assert bounds["min"] == 95
    assert bounds["max"] == 112

    # 插入一个 500 帧的离群集：中位数/MAD 不变，区间不应被单点拉宽
    widened = np.append(counts, 500.0)
    assert _resolve_frame_bounds(widened, load_config(robot_type="test_robot"))["min"] == 95
    assert _resolve_frame_bounds(widened, load_config(robot_type="test_robot"))["max"] == 112


def test_frame_count_outlier_only_flags_the_outlier(tmp_path):
    """同一组手工数据跑真规则：只有 500 帧那集被判 too_many，其余 10 集不误报。"""
    counts = [100, 102, 98, 101, 99, 100, 103, 97, 100, 100, 500]
    episodes = {ep: smooth_episode(n, seed=ep) for ep, n in enumerate(counts)}
    path = build_dataset(tmp_path / "ds", episodes)
    _, result = _validate(path)

    flagged = [r.episode_index for r in result.results.values()
               if "frame_count_abnormal" in r.reasons]
    assert flagged == [10]

    detail = result.results[10].details["frame_count_abnormal"]
    assert detail["direction"] == "too_many"
    assert detail["min_allowed"] == 95
    assert detail["max_allowed"] == 112


def test_timestamp_regression_and_step_deviation(tmp_path):
    bad = smooth_episode(40, seed=7)
    ts = np.arange(40, dtype=np.float32) / 30
    ts[20], ts[21] = ts[21], ts[20] - 0.5

    jitter = smooth_episode(40, seed=8)
    ts_jitter = np.arange(40, dtype=np.float32) / 30
    ts_jitter[20:] += 0.5

    path = build_dataset(tmp_path / "ds", {
        0: {**bad, "timestamp": ts},
        1: {**jitter, "timestamp": ts_jitter},
    })
    _, result = _validate(path)

    assert "timestamp_not_monotonic" in result.results[0].reasons
    # 步长突变只是 info，不应污染 timestamp_not_monotonic
    assert "timestamp_not_monotonic" not in result.results[1].reasons
    assert "timestamp_step_deviation" in result.results[1].reasons
    assert result.severity_of(result.results[1]) == "info"


def test_missing_task(tmp_path):
    ep = smooth_episode(40, seed=9)
    path = build_dataset(tmp_path / "ds", {0: {**ep, "task_index": -1}, 1: smooth_episode(40, seed=10)})
    _, result = _validate(path)
    assert "missing_task" in result.results[0].reasons
    assert "missing_task" not in result.results[1].reasons


def test_static_episode(tmp_path):
    still = {"state": np.full((40, 3), 0.5, dtype=np.float32) + np.random.default_rng(0).normal(0, 1e-8, (40, 3)),
             "action": np.full((40, 3), 0.5, dtype=np.float32)}
    path = build_dataset(tmp_path / "ds", {0: still, 1: smooth_episode(40, seed=11)})
    _, result = _validate(path)
    assert "static_episode" in result.results[0].reasons
    assert "static_episode" not in result.results[1].reasons


def test_duplicate_episode(tmp_path):
    original = smooth_episode(40, seed=12)
    copy = {"state": original["state"].copy(), "action": original["action"].copy()}
    path = build_dataset(tmp_path / "ds", {0: original, 1: copy, 2: smooth_episode(40, seed=13)})
    _, result = _validate(path)

    assert "duplicate_episode" not in result.results[0].reasons, "首条不应被判为重复"
    assert "duplicate_episode" in result.results[1].reasons
    assert result.results[1].details["duplicate_episode"]["duplicate_of"] == 0
    assert "duplicate_episode" not in result.results[2].reasons


def test_stale_stats_detected(tmp_path):
    episodes = {ep: smooth_episode(40, seed=ep) for ep in range(4)}
    path = build_dataset(tmp_path / "ds", episodes, stale_stats=True)
    _, result = _validate(path)

    codes = [i.code for i in result.dataset_issues]
    assert "stats_stale" in codes
    issue = next(i for i in result.dataset_issues if i.code == "stats_stale")
    assert issue.repair == "stats"
    assert issue.detail["max_relative_deviation"] > issue.detail["tolerance"]


def test_fresh_stats_not_flagged(normal_dataset):
    _, result = _validate(normal_dataset)
    assert all(i.code != "stats_stale" for i in result.dataset_issues)


def test_near_zero_mean_does_not_amplify_small_drift():
    """回归（POSTMORTEM 案例 2）：分母取 |mean| 时，近零均值维度会把微小漂移放大成假阳性。

    旧式：|0.02 - 0.0001| / |0.0001| ≈ 199 倍 → 远超 5% 容忍，报 stats_stale，但绝对漂移只有 0.02。
    现式：分母取该维自身量级 max(|mean|, std) = 1.0 → 0.02 < 0.05 → 不报。
    """
    from pipeline.validator.context import _stats_staleness_issue

    stats = {"observation.state": {"mean": [1e-4] * JOINTS, "std": [1.0] * JOINTS}}
    assert _stats_staleness_issue(
        stats, np.full(JOINTS, 0.02), load_config(robot_type="test_robot")
    ) is None


def test_real_stats_drift_is_reported_at_true_magnitude():
    """同一公式不能把真实过期一起放过：0.2 的漂移相对 std=1.0 必须报，且幅度是 0.2 量级。"""
    from pipeline.validator.context import _stats_staleness_issue

    stats = {"observation.state": {"mean": [1e-4] * JOINTS, "std": [1.0] * JOINTS}}
    issue = _stats_staleness_issue(
        stats, np.full(JOINTS, 0.2), load_config(robot_type="test_robot")
    )
    assert issue is not None and issue.code == "stats_stale"
    # 旧式会给出 2000 量级，这里要求是可解释的「绝对漂移 / 该维尺度」
    assert 0.19 <= issue.detail["max_relative_deviation"] <= 0.21


def test_fixed_mode_overrides_adaptive_threshold(tmp_path):
    episodes = {ep: smooth_episode(40, seed=ep) for ep in range(4)}
    episodes[4] = smooth_episode(12, seed=99)
    path = build_dataset(tmp_path / "ds", episodes)
    dataset = load_dataset(path)

    fixed = load_config(overrides={"frame_count": {"mode": "fixed", "floor": 20, "ceiling": 100}})
    result = validate_dataset(dataset, fixed)
    assert result.resolved_thresholds["frame_count"]["mode"] == "fixed"
    assert result.resolved_thresholds["frame_count"]["min"] == 20
    assert "frame_count_abnormal" in result.results[4].reasons


def test_contract_keys_present(normal_dataset):
    """报告消费方依赖的字段契约不能变。"""
    _, result = _validate(normal_dataset)
    payload = result.to_dict()
    assert {"total_episodes", "valid_episodes", "invalid_episodes"} <= set(payload)
    assert payload["total_episodes"] == 6
    for item in payload["invalid_episodes"]:
        assert {"episode_index", "reasons"} <= set(item)


def test_skip_reason_recorded_when_no_video(normal_dataset):
    """无视频特征时必须如实记录跳过的检查，不能假装检查过。"""
    _, result = _validate(normal_dataset)
    assert "frame_mismatch" in result.skipped_checks


def test_v21_video_in_chunk_dir_is_located(tmp_path):
    """v2.1 标准布局 videos/chunk-000/<camera>/episode_XXXXXX.mp4 必须能定位。"""
    root = tmp_path / "ds21"
    (root / "meta").mkdir(parents=True)
    video = root / "videos" / "chunk-000" / "observation.images.top" / "episode_000000.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"\x00\x00\x00\x18ftypmp42")

    info = {
        "codebase_version": "v2.1",
        "robot_type": "test_robot",
        "total_episodes": 1,
        "total_frames": 4,
        "fps": 30,
        "chunks_size": 1000,
        "data_path": "data/chunk-{chunk_index:03d}/episode_{episode_index:06d}.parquet",
        "features": {
            "observation.state": {"dtype": "float32", "shape": [2]},
            "action": {"dtype": "float32", "shape": [2]},
            "observation.images.top": {"dtype": "video", "shape": [240, 320, 3]},
        },
    }
    (root / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")
    (root / "meta" / "episodes.jsonl").write_text(
        json.dumps({"episode_index": 0, "tasks": ["pick"], "length": 4}) + "\n", encoding="utf-8"
    )

    dataset = load_dataset(root, load_frames=False)
    loc = dataset.video_location("observation.images.top", 0)
    assert loc is not None
    assert loc[0] == video


def test_v21_video_path_template_with_short_dir(tmp_path):
    """v2.1 常见写法：info.json 给 video_path 模板，目录名省略 observation.images. 前缀。"""
    root = tmp_path / "ds21_tmpl"
    (root / "meta").mkdir(parents=True)
    video = root / "videos" / "chunk-000" / "egocentric" / "episode_000005.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"\x00\x00\x00\x18ftypmp42")

    info = {
        "codebase_version": "v2.1",
        "robot_type": "test_robot",
        "total_episodes": 6,
        "total_frames": 24,
        "fps": 30,
        "chunks_size": 1000,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/egocentric/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.state": {"dtype": "float32", "shape": [2]},
            "action": {"dtype": "float32", "shape": [2]},
            "observation.images.egocentric": {"dtype": "video", "shape": [240, 320, 3]},
        },
    }
    (root / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")
    (root / "meta" / "episodes.jsonl").write_text(
        json.dumps({"episode_index": 5, "tasks": ["walk"], "length": 4}) + "\n", encoding="utf-8"
    )

    dataset = load_dataset(root, load_frames=False)
    loc = dataset.video_location("observation.images.egocentric", 5)
    assert loc is not None
    assert loc[0] == video


def test_meta_episode_index_mismatch_is_reported(tmp_path):
    """behavior1k 实测：meta 用全局 ID、data 用 0..N-1 时，必须报数据集级
    meta_episode_mismatch，而不是只得到一堆 missing_task/video_unlocatable。"""
    import pyarrow.parquet as pq

    path = build_dataset(tmp_path / "ds", {0: smooth_episode(40, seed=11),
                                           1: smooth_episode(40, seed=12)})
    meta_path = path / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    tbl = pq.read_table(meta_path)
    tbl = tbl.set_column(
        tbl.schema.get_field_index("episode_index"), "episode_index",
        [[420010, 420020]],
    )
    pq.write_table(tbl, meta_path)

    _, result = _validate(path)
    codes = {i.code for i in result.dataset_issues}
    assert "meta_episode_mismatch" in codes
    issue = next(i for i in result.dataset_issues if i.code == "meta_episode_mismatch")
    assert issue.detail["overlap"] == 0
    assert issue.detail["data_episodes"] == 2
    assert issue.detail["meta_episodes"] == 2


def test_missing_frame_index_is_tolerated(tmp_path):
    """部分 v3 数据集（behavior1k）没有 frame_index 列：加载与质检都不能崩。"""
    import pyarrow.parquet as pq

    path = build_dataset(tmp_path / "ds", {0: smooth_episode(40, seed=21),
                                           1: smooth_episode(40, seed=22)})
    data_path = path / "data" / "chunk-000" / "file-000.parquet"
    tbl = pq.read_table(data_path)
    tbl = tbl.drop(["frame_index"])
    pq.write_table(tbl, data_path)

    dataset, result = _validate(path)
    assert "frame_index" not in dataset.frames.columns
    assert result.valid_episodes + len(result.invalid_episodes) == 2
    assert "meta_episode_mismatch" not in {i.code for i in result.dataset_issues}


def test_image_features_are_not_treated_as_video(tmp_path):
    """dtype=image 的特征逐帧嵌在 parquet 里，没有 mp4 文件。

    以前 video_features 把 image 也算进去，video_location 必然定位失败，
    整批 episode 报 frame_mismatch（libero_10_image 实测 379/379 误报）。
    """
    path = build_dataset(tmp_path / "ds", {0: smooth_episode(20, seed=31),
                                           1: smooth_episode(20, seed=32)})
    info_path = path / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["features"]["observation.images.image"] = {"dtype": "image", "shape": [128, 128, 3]}
    info_path.write_text(json.dumps(info), encoding="utf-8")

    dataset, result = _validate(path)
    assert dataset.video_features == []
    assert dataset.image_features == ["observation.images.image"]
    assert "没有视频特征" in result.skipped_checks["frame_mismatch"]
    assert result.valid_episodes == 2
    assert all("frame_mismatch" not in r.reasons for r in result.results.values())


def test_streaming_stats_skipped_when_sampling_partial(tmp_path):
    """流式只采样 N 个 episode 时，子集均值不能拿来判定全量 stats 过期。

    libero_10_image（379 集，采样 200）实测偏差 2.5e-08 却报 stats_stale。
    """
    path = build_dataset(tmp_path / "ds", {0: smooth_episode(20, seed=41),
                                           1: smooth_episode(20, seed=42)})
    dataset = load_dataset(path)
    config = load_config(robot_type="test_robot", overrides={"stale_stats": {"sample_episodes": 1}})
    result = validate_dataset(dataset, config, streaming=True)

    assert all(i.code != "stats_stale" for i in result.dataset_issues)
    assert any("不可比" in why for why in result.skipped_checks.values())
