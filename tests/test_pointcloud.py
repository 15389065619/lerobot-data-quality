"""3D 点云清洗规则单测。

覆盖两件事：
1. 八条规则各自能被触发，处置结果符合业务约定（repair / drop / review）；
2. 修复后仍然满足 LeRobot 的固定点数契约（N 点、D 维、行数不变、extension 元数据保留）。

合成数据而不是依赖真实数据集：单测要能在几秒内跑完，真实数据集验证放在
`data/gym_xarm_pointcloud` 的端到端跑批里。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from pipeline.config import load_config
from pipeline.pointcloud import (
    PointCloudFormatError,
    clean_pointclouds,
    decide,
    detect_pointcloud_feature,
    validate_pointcloud_dataset,
)

POINTS = 256   # 256×6 超过 loader 的「重型特征」阈值（1000 元素），点云阶段才会识别
DIM = 6
FEATURE = "observation.environment_state"


# --------------------------------------------------------------------- 造数
def _grid_cloud(seed: int = 0, *, n: int = POINTS, jitter: float = 0.004) -> np.ndarray:
    """8×8 平面上的均匀点云（带轻微抖动），远看点云在半径 2.5×1-NN 下没有离群点。"""
    rng = np.random.default_rng(seed)
    side = int(np.sqrt(n))
    xs, ys = np.meshgrid(np.arange(side), np.arange(side))
    xyz = np.stack([xs.ravel(), ys.ravel(), np.zeros(n)], axis=1).astype(np.float32) * 0.05
    xyz += rng.normal(0, jitter, size=xyz.shape).astype(np.float32)
    colors = rng.uniform(0.2, 0.9, size=(n, 3)).astype(np.float32)
    return np.concatenate([xyz, colors], axis=1)


def _write_pc_dataset(root: Path, frames: list[np.ndarray], *, fps: int = 30) -> Path:
    """写一个最小 LeRobot v3 数据集，只含点云 + 索引列。"""
    (root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)

    n_frames = len(frames)
    ep = np.zeros(n_frames, dtype=np.int64)
    fi = np.arange(n_frames, dtype=np.int64)
    # 与真实 LeRobot v3 数据集保持一致：内层是可变长 list，不是 fixed_size_list
    field = pa.field(
        FEATURE,
        pa.list_(pa.list_(pa.float32())),
        metadata={
            b"ARROW:extension:name": b"datasets.features.features.Array2DExtensionType",
            b"ARROW:extension:metadata": json.dumps([[POINTS, DIM], "float32"]).encode(),
        },
    )
    table = pa.table({
        FEATURE: pa.array([f.astype(np.float32).tolist() for f in frames], type=field.type),
        "timestamp": pa.array(np.arange(n_frames, dtype=np.float32) / fps),
        "frame_index": pa.array(fi),
        "episode_index": pa.array(ep),
        "index": pa.array(fi),
        "task_index": pa.array(np.zeros(n_frames, dtype=np.int64)),
    })
    schema = table.schema.set(table.schema.get_field_index(FEATURE), field)
    pq.write_table(table.cast(schema), root / "data" / "chunk-000" / "file-000.parquet")

    pq.write_table(pa.Table.from_pylist([{
        "episode_index": 0, "length": n_frames, "tasks": ["pick cube"],
        "dataset_from_index": 0, "dataset_to_index": n_frames,
    }]), root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    pq.write_table(pa.table({"task_index": pa.array([0], type=pa.int64()),
                             "task": pa.array(["pick cube"])}), root / "meta" / "tasks.parquet")

    info = {
        "codebase_version": "v3.0",
        "robot_type": "pc_test_robot",
        "total_episodes": 1,
        "total_frames": n_frames,
        "total_tasks": 1,
        "chunks_size": 1000,
        "fps": fps,
        "splits": {"train": "0:1"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "features": {
            FEATURE: {"dtype": "float32", "shape": [POINTS, DIM]},
            "timestamp": {"dtype": "float32", "shape": [1]},
            "frame_index": {"dtype": "int64", "shape": [1]},
            "episode_index": {"dtype": "int64", "shape": [1]},
            "index": {"dtype": "int64", "shape": [1]},
            "task_index": {"dtype": "int64", "shape": [1]},
        },
    }
    (root / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")
    (root / "meta" / "stats.json").write_text(json.dumps({
        FEATURE: {"count": n_frames, "mean": [0.0] * DIM, "std": [1.0] * DIM,
                  "min": [-1.0] * DIM, "max": [1.0] * DIM}
    }), encoding="utf-8")
    return root


def _mixed_frames() -> list[np.ndarray]:
    """每类坏帧各一帧，其余为正常帧（正常帧要多，否则坏帧会污染自适应阈值）。"""
    good = [_grid_cloud(seed=i) for i in range(6)]
    more_good = [_grid_cloud(seed=100 + i) for i in range(20)]
    nan = _grid_cloud(seed=10)
    nan[0, :3] = np.nan                          # 1 个 NaN（0.4%），触发 NaN 规则
    sparse = _grid_cloud(seed=11)[:8]            # 8 < min_points(16)
    degenerate = np.repeat(_grid_cloud(seed=12)[:1], POINTS, axis=0)
    duplicate = _grid_cloud(seed=13)
    duplicate[1::4] = duplicate[0::4]            # 25% 复制点：超过重复阈值，但仍在可修复比例内
    outlier = _grid_cloud(seed=14)
    outlier[:40, :3] = np.random.default_rng(140).normal(8.0, 0.5, size=(40, 3))
    oob = _grid_cloud(seed=15)
    oob[:16, :3] += 10.0                         # 远超出自动包围盒（16/256 = 6.25%）
    color = _grid_cloud(seed=16)
    color[:20, 3:6] = -5.0                       # 负颜色 → 越界
    jump = _grid_cloud(seed=17)
    jump[:, :3] += 2.0                           # 质心整体平移 → 时序跳变
    return good + more_good + [nan, sparse, degenerate, duplicate, outlier, oob, color, jump]


@pytest.fixture
def mixed_dataset(tmp_path: Path) -> Path:
    return _write_pc_dataset(tmp_path / "pc_mixed", _mixed_frames())


@pytest.fixture
def dirty_dataset(tmp_path: Path) -> Path:
    """只含可修复的脏帧，用于端到端清洗验证。"""
    frames = [_grid_cloud(seed=i) for i in range(8)]
    nan = _grid_cloud(seed=20)
    nan[0, 0] = np.inf
    outlier = _grid_cloud(seed=21)
    outlier[0, :3] = [9.0, 9.0, 9.0]
    frames[2] = nan
    frames[5] = outlier
    return _write_pc_dataset(tmp_path / "pc_dirty", frames)


# --------------------------------------------------------------------- 用例
def test_detect_feature():
    info_features = {FEATURE: {"dtype": "float32", "shape": [POINTS, DIM]},
                     "observation.state": {"dtype": "float32", "shape": [3]}}
    assert detect_pointcloud_feature(info_features) == FEATURE
    assert detect_pointcloud_feature({"observation.state": {"shape": [3]}}) is None


def test_no_pointcloud_feature_raises(tmp_path: Path):
    root = tmp_path / "plain"
    _write_pc_dataset(root, [_grid_cloud(0)])
    with pytest.raises(PointCloudFormatError):
        validate_pointcloud_dataset(root, load_config(), feature_key="not.exists")


def test_all_rules_hit(mixed_dataset: Path):
    cfg = load_config()
    validation = validate_pointcloud_dataset(mixed_dataset, cfg)
    freq = {k: v for k, v in validation.reason_frequency().items() if v}
    for code in ("pc_non_finite", "pc_empty_or_sparse", "pc_degenerate", "pc_duplicate",
                 "pc_outliers", "pc_out_of_bounds", "pc_color_invalid", "pc_temporal_jump"):
        assert freq.get(code, 0) >= 1, f"{code} 未被触发: {freq}"


def test_clean_frames_pass(mixed_dataset: Path):
    validation = validate_pointcloud_dataset(mixed_dataset, load_config())
    # 前 6 帧是干净点云，不应有任何规则命中
    for fi in range(6):
        assert validation.results[(0, fi)].reasons == []


def test_dispositions(mixed_dataset: Path):
    validation = validate_pointcloud_dataset(mixed_dataset, load_config())
    decisions = decide(validation)
    by_frame = {k[1]: d for k, d in decisions.items()}
    # 26 正常帧后依次是 8 类坏帧
    base = 26
    assert by_frame[base].disposition == "repair"       # NaN 可过滤补点
    assert by_frame[base + 1].disposition == "drop"     # 有效点过少
    assert by_frame[base + 2].disposition == "drop"     # 几何退化
    assert by_frame[base + 3].disposition == "repair"   # 重复点
    assert by_frame[base + 4].disposition == "repair"   # 离群点
    assert by_frame[base + 5].disposition == "repair"   # 越界
    assert by_frame[base + 6].disposition == "repair"   # 颜色越界
    assert by_frame[base + 7].disposition == "review"   # 时序跳变：不改数据，等人审


def test_clean_in_place(dirty_dataset: Path, tmp_path: Path):
    cfg = load_config()
    validation = validate_pointcloud_dataset(dirty_dataset, cfg)
    decisions = decide(validation)
    assert validation.invalid_frames, "注入的脏帧应被检出"

    out_dir = tmp_path / "out"
    result = clean_pointclouds(dirty_dataset, validation, decisions, out_dir,
                               target_root=dirty_dataset)
    assert result.success
    assert result.verification.get("ok"), result.verification
    assert result.removed_points > 0 and result.added_points > 0

    # 行数与固定点数契约必须保持
    pf = pq.ParquetFile(dirty_dataset / "data" / "chunk-000" / "file-000.parquet")
    assert pf.metadata.num_rows == validation.total_frames
    field = pf.schema_arrow.field(pf.schema_arrow.get_field_index(FEATURE))
    assert field.metadata.get(b"ARROW:extension:name")
    # 注意：Table.to_pylist() 返回行字典，必须取列再转 list
    cells = pq.read_table(dirty_dataset / "data" / "chunk-000" / "file-000.parquet",
                          columns=[FEATURE]).column(FEATURE).to_pylist()
    assert all(len(cell) == POINTS for cell in cells)

    # 修复后的点云不应再有 NaN/Inf，也不应留下明显离群点
    arr = np.asarray([np.asarray(cell, dtype=np.float32) for cell in cells])
    assert np.isfinite(arr).all()
    repaired = np.asarray(cells[2], dtype=np.float32)
    assert len(repaired) == POINTS
    assert not np.allclose(repaired[:, :3], [9.0, 9.0, 9.0])


def test_standalone_output_copies_data(dirty_dataset: Path, tmp_path: Path):
    cfg = load_config()
    validation = validate_pointcloud_dataset(dirty_dataset, cfg)
    decisions = decide(validation)
    result = clean_pointclouds(dirty_dataset, validation, decisions, tmp_path / "out")
    out_root = Path(result.output_root)
    assert (out_root / "meta" / "info.json").is_file()
    assert list((out_root / "data").rglob("*.parquet")), "独立输出必须包含 data 分片"
    assert (out_root / "meta" / "pointcloud_cleaning_notes.json").is_file()


def test_noop_standalone_copies_data(tmp_path: Path):
    root = _write_pc_dataset(tmp_path / "pc_clean", [_grid_cloud(seed=i) for i in range(5)])
    validation = validate_pointcloud_dataset(root, load_config())
    decisions = decide(validation)
    assert all(d.disposition == "keep" for d in decisions.values())
    result = clean_pointclouds(root, validation, decisions, tmp_path / "out")
    out_root = Path(result.output_root)
    assert result.method == "standalone"
    assert list((out_root / "data").rglob("*.parquet"))
