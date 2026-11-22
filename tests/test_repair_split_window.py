"""结构损坏修复（拆分 / 收窗）与后验复验的单元测试。

真实场景：``so101_bomb_pick_place_v1`` 有 22 集在 meta 里各占两行（两段 take），
7 集视频窗口比数据长；这些都不该整集丢弃。
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from conftest import build_dataset, smooth_episode

from pipeline.cleaner.util import _verify_structure
from pipeline.config import load_config
from pipeline.loader import load_dataset
from pipeline.policy import decide
from pipeline.validator.types import EpisodeResult, ValidationResult

CAM = "observation.images.top"
FPS = 30


class _StubDataset:
    """只提供 policy 拆分/收窗所需的接口。"""

    fps = FPS
    video_features = [CAM]

    def __init__(self, segments: pd.DataFrame, frame_index: dict[int, np.ndarray]):
        self._segments = segments
        self._frame_index = frame_index
        self.frames = pd.DataFrame({"episode_index": [0]})  # 非空即可

    def episode_segments(self, episode_index: int) -> pd.DataFrame:
        return self._segments[self._segments["episode_index"] == episode_index]

    def episode_frame(self, episode_index: int) -> pd.DataFrame:
        return pd.DataFrame({"frame_index": self._frame_index.get(episode_index, [])})


def _meta_rows(rows: list[dict]) -> pd.DataFrame:
    out = []
    for row in rows:
        base = {"episode_index": row["episode_index"], "length": row["length"], "tasks": ["pick"]}
        base[f"videos/{CAM}/chunk_index"] = 0
        base[f"videos/{CAM}/file_index"] = 0
        base[f"videos/{CAM}/from_timestamp"] = row["from"]
        base[f"videos/{CAM}/to_timestamp"] = row["to"]
        out.append(base)
    return pd.DataFrame(out)


def _validation(res: EpisodeResult) -> ValidationResult:
    return ValidationResult(
        total_episodes=1,
        results={res.episode_index: res},
        config=load_config(robot_type="test_robot"),
        fps=FPS,
        dataset_root="",
    )


def test_split_repair_uses_meta_segments():
    spliced = np.concatenate([np.arange(15), np.arange(15)])
    dataset = _StubDataset(
        _meta_rows([
            {"episode_index": 1, "length": 15, "from": 0.0, "to": 0.5},
            {"episode_index": 1, "length": 15, "from": 10.0, "to": 10.5},
        ]),
        {1: spliced},
    )
    res = EpisodeResult(1, 30, reasons=[
        "timestamp_not_monotonic", "frame_index_gap", "meta_index_inconsistent",
        "video_span_mismatch", "frame_mismatch",
    ])
    decision = decide(_validation(res), dataset=dataset)[1]

    assert decision.disposition == "repair"
    assert decision.repair == "split"
    assert [seg["length"] for seg in decision.split_plan] == [15, 15]
    assert decision.split_plan[1]["videos"][CAM]["from"] == 10.0
    assert "拆分" in decision.recommended_action


def test_split_not_used_when_lengths_do_not_match_rows():
    dataset = _StubDataset(
        _meta_rows([
            {"episode_index": 1, "length": 10, "from": 0.0, "to": 0.3},
            {"episode_index": 1, "length": 10, "from": 5.0, "to": 5.3},
        ]),
        {1: np.arange(30)},
    )
    res = EpisodeResult(1, 30, reasons=["timestamp_not_monotonic", "meta_index_inconsistent"])
    decision = decide(_validation(res), dataset=dataset)[1]

    assert decision.repair != "split"


def test_split_respects_physical_row_order():
    """物理段顺序与 meta 行序相反时，必须按 data 行序切片、按长度配对窗口。

    真实数据里同一集的两段可能落在不同 parquet 文件，行序由文件名决定；
    若按 meta 行序切，会把后段的行拼进前段，留下一个未拆开的动作跳变。
    """
    dataset = _StubDataset(
        _meta_rows([
            {"episode_index": 1, "length": 20, "from": 0.0, "to": 20 / FPS},
            {"episode_index": 1, "length": 10, "from": 50.0, "to": 60 / FPS},
        ]),
        {1: np.concatenate([np.arange(10), np.arange(20)])},  # 物理上 10 帧的段在前
    )
    res = EpisodeResult(1, 30, reasons=[
        "timestamp_not_monotonic", "frame_index_gap", "meta_index_inconsistent",
    ])
    decision = decide(_validation(res), dataset=dataset)[1]

    assert decision.repair == "split"
    assert [seg["length"] for seg in decision.split_plan] == [10, 20]
    assert decision.split_plan[0]["videos"][CAM]["from"] == 50.0
    assert decision.split_plan[1]["videos"][CAM]["from"] == 0.0


def test_window_repair_narrows_to_real_frame_count():
    dataset = _StubDataset(
        _meta_rows([{"episode_index": 5, "length": 20, "from": 0.0, "to": 50 / FPS}]),
        {5: np.arange(20)},
    )
    res = EpisodeResult(5, 20, reasons=["video_span_mismatch", "frame_mismatch"])
    decision = decide(_validation(res), dataset=dataset)[5]

    assert decision.disposition == "repair"
    assert decision.repair == "window"
    assert abs(decision.window_plan[CAM]["to"] - 20 / FPS) < 1e-9
    assert "收窄" in decision.recommended_action


def test_soft_rules_never_auto_drop():
    res = EpisodeResult(2, 30, reasons=["state_jump"])
    decision = decide(_validation(res), on_review="drop")[2]
    assert decision.disposition == "review"

    res_info = EpisodeResult(3, 30, reasons=["timestamp_step_deviation"])
    decision = decide(_validation(res_info), on_review="drop")[3]
    assert decision.disposition == "keep"


def _add_video_feature(root, windows: dict[int, tuple[float, float]]) -> None:
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["features"][CAM] = {"dtype": "video", "shape": [8, 8, 3]}
    info_path.write_text(json.dumps(info), encoding="utf-8")

    ep_path = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    df = pq.read_table(ep_path).to_pandas()
    for field in ("chunk_index", "file_index"):
        df[f"videos/{CAM}/{field}"] = 0
    df[f"videos/{CAM}/from_timestamp"] = [windows[int(e)][0] for e in df["episode_index"]]
    df[f"videos/{CAM}/to_timestamp"] = [windows[int(e)][1] for e in df["episode_index"]]
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), ep_path)

    video = root / "videos" / CAM / "chunk-000" / "file-000.mp4"
    video.parent.mkdir(parents=True, exist_ok=True)
    video.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 64)


def _stub_probe(monkeypatch, frames: int = 10_000) -> None:
    import pipeline.validator.probes as probes

    monkeypatch.setattr(probes, "probe_video_frames", lambda path, precise=False: frames)


def test_verification_catches_broken_video_window(tmp_path, monkeypatch):
    root = build_dataset(tmp_path / "ds", {
        0: smooth_episode(20, seed=1),
        1: smooth_episode(25, seed=2),
    })
    _add_video_feature(root, {0: (0.0, 20 / FPS), 1: (20 / FPS, 45 / FPS)})
    _stub_probe(monkeypatch)

    view = load_dataset(root)
    assert _verify_structure(view) == []

    ep_path = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    df = pq.read_table(ep_path).to_pandas()
    df.loc[df["episode_index"] == 1, f"videos/{CAM}/to_timestamp"] += 0.1
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), ep_path)

    broken = load_dataset(root)
    violations = _verify_structure(broken)
    assert any("窗口跨度" in item and "ep 1" in item for item in violations), violations


def _splice_episode(root, ep: int, first: int, second: int) -> None:
    """把 ep 改造成两段 take 按行拼接：frame_index/timestamp 在中点回到 0。"""
    path = root / "data" / "chunk-000" / "file-000.parquet"
    table = pq.read_table(path).to_pandas()
    mask = table["episode_index"] == ep
    n = int(mask.sum())
    assert n == first + second
    table["frame_index"] = table["frame_index"].astype(np.int64)
    table["timestamp"] = table["timestamp"].astype(np.float64)
    table.loc[mask, "frame_index"] = np.concatenate([np.arange(first), np.arange(second)])
    table.loc[mask, "timestamp"] = np.concatenate([
        np.arange(first) / FPS, np.arange(second) / FPS,
    ])
    pq.write_table(pa.Table.from_pandas(table, preserve_index=False), path)


def test_clean_dataset_splits_episode_and_keeps_all_frames(tmp_path):
    root = build_dataset(tmp_path / "ds", {
        0: smooth_episode(30, seed=1),
        1: smooth_episode(30, seed=2),
        2: smooth_episode(30, seed=3),
    })
    _splice_episode(root, ep=1, first=15, second=15)

    # meta 里 ep1 换成两行，每段各带自己的 length
    ep_path = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    meta = pq.read_table(ep_path).to_pandas()
    row = meta[meta["episode_index"] == 1].iloc[0].copy()
    first = row.copy()
    first["length"] = 15
    second = row.copy()
    second["length"] = 15
    meta = meta[meta["episode_index"] != 1]
    meta = pd.concat([meta, pd.DataFrame([first, second])], ignore_index=True)
    meta = meta.sort_values("episode_index", kind="stable").reset_index(drop=True)
    pq.write_table(pa.Table.from_pandas(meta, preserve_index=False), ep_path)

    from pipeline.cleaner import clean_dataset
    from pipeline.validator import validate_dataset

    dataset = load_dataset(root)
    validation = validate_dataset(dataset, load_config(robot_type="test_robot"))
    decisions = decide(validation, dataset=dataset)
    assert decisions[1].repair == "split"

    result = clean_dataset(dataset, decisions, tmp_path / "out", prefer_cli=False)
    cleaned = load_dataset(result.output_root)

    assert cleaned.num_episodes == 4
    assert cleaned.num_frames == 90
    lengths = [int(cleaned.meta_row(ep)["length"]) for ep in cleaned.episode_indices]
    assert lengths == [30, 15, 15, 30]
    for ep in cleaned.episode_indices:
        frame = cleaned.episode_frame(ep)
        assert np.array_equal(frame["frame_index"].to_numpy(), np.arange(len(frame)))
        ts = frame["timestamp"].to_numpy(dtype=float)
        assert np.allclose(ts, np.arange(len(frame)) / FPS, atol=1e-5)
    assert result.verification.get("ok") is True, result.verification


def test_clean_handles_null_episode_index(tmp_path):
    """损坏数据里的 NULL 编号应被跳过，不能让 int(None) 打崩整次清洗。"""
    root = build_dataset(tmp_path / "ds", {
        0: smooth_episode(20, seed=1),
        1: smooth_episode(25, seed=2),
    })
    data_path = root / "data" / "chunk-000" / "file-000.parquet"
    table = pq.read_table(data_path).to_pandas()
    table["episode_index"] = table["episode_index"].astype("float64")
    table.loc[table.index[3], "episode_index"] = np.nan
    pq.write_table(pa.Table.from_pandas(table, preserve_index=False), data_path)

    from pipeline.cleaner import clean_dataset

    result = clean_dataset(load_dataset(root), [0], tmp_path / "out", prefer_cli=False)
    assert result.verification.get("ok") is True, result.verification
    assert result.verification.get("episodes") == 1
    assert result.verification.get("frames") == 25


def test_clean_works_with_streaming_view(tmp_path):
    """load_frames=False 时没有 __source_file 列，必须回退 data_files 而非产出空集。"""
    root = build_dataset(tmp_path / "ds", {
        0: smooth_episode(20, seed=1),
        1: smooth_episode(25, seed=2),
    })
    from pipeline.cleaner import clean_dataset

    view = load_dataset(root, load_frames=False)
    assert view.frames.empty
    result = clean_dataset(view, [0], tmp_path / "out", prefer_cli=False)
    assert result.verification.get("episodes") == 1
    assert result.verification.get("frames") == 25
    assert result.verification.get("ok") is True, result.verification


def test_verification_tolerates_capture_jitter(tmp_path):
    """未修复集保留原始采集时间戳，±2ms 抖动在 LeRobot 容差内，不应判违规。"""
    root = build_dataset(tmp_path / "ds", {
        0: smooth_episode(20, seed=1),
        1: smooth_episode(25, seed=2),
    })
    data_path = root / "data" / "chunk-000" / "file-000.parquet"
    table = pq.read_table(data_path).to_pandas()
    table["timestamp"] = table["timestamp"].astype("float64")
    mask = table["episode_index"] == 1
    jitter = np.random.default_rng(7).uniform(-0.002, 0.002, int(mask.sum()))
    table.loc[mask, "timestamp"] += jitter
    pq.write_table(pa.Table.from_pandas(table, preserve_index=False), data_path)

    from pipeline.cleaner import clean_dataset

    result = clean_dataset(load_dataset(root), [0], tmp_path / "out", prefer_cli=False)
    assert result.verification.get("structure_violations") == []
    assert result.verification.get("ok") is True, result.verification


def test_hard_verification_failure_does_not_publish(tmp_path, monkeypatch):
    """结构性硬失败必须中止发布，不能覆盖上一版产物。"""
    import pipeline.cleaner as cleaner

    root = build_dataset(tmp_path / "ds", {
        0: smooth_episode(20, seed=1),
        1: smooth_episode(25, seed=2),
    })
    out_root = tmp_path / "out" / "cleaned_dataset"
    out_root.mkdir(parents=True)
    marker = out_root / "keep_me.txt"
    marker.write_text("previous", encoding="utf-8")

    monkeypatch.setattr(cleaner, "verify", lambda *a, **k: {
        "ok": False, "episodes": 0, "frames": 0,
        "episode_indices_contiguous": True, "structure_violations": [],
    })
    with pytest.raises(RuntimeError, match="中止发布"):
        cleaner.clean_dataset(load_dataset(root), [0], tmp_path / "out",
                              out_root=out_root, prefer_cli=False)
    assert marker.read_text(encoding="utf-8") == "previous"


def test_soft_violation_still_publishes(tmp_path, monkeypatch):
    """只有软违规（结构细节 / 规则复检）时仍按原行为发布并告警。"""
    import pipeline.cleaner as cleaner

    root = build_dataset(tmp_path / "ds", {
        0: smooth_episode(20, seed=1),
        1: smooth_episode(25, seed=2),
    })
    monkeypatch.setattr(cleaner, "verify", lambda *a, **k: {
        "ok": False, "episodes": 1, "frames": 25,
        "episode_indices_contiguous": True,
        "structure_violations": ["ep 0: 窗口跨度偏差"],
    })
    result = cleaner.clean_dataset(load_dataset(root), [0], tmp_path / "out",
                                   prefer_cli=False)
    assert result.verification.get("ok") is False
    assert any("校验未通过" in w for w in result.warnings)


def test_split_across_two_parquet_files(tmp_path):
    """两段落在两个 parquet、且 meta 行序与物理段序相反时，仍按物理行序拆分。

    这是对两遍扫描行序的回归测试：第一遍建立的 (文件, 行号) 索引必须按
    文件名顺序拼接，否则 10 帧段和 20 帧段会接错，拆出来的动作就是乱的。
    """
    root = build_dataset(tmp_path / "ds", {
        0: smooth_episode(30, seed=1),
        1: smooth_episode(30, seed=2),
        2: smooth_episode(30, seed=3),
    })
    # ep1 改成两段 take：物理前段 10 帧、后段 20 帧，frame_index 各从 0 起
    data_path = root / "data" / "chunk-000" / "file-000.parquet"
    table = pq.read_table(data_path).to_pandas()
    table["frame_index"] = table["frame_index"].astype(np.int64)
    table["timestamp"] = table["timestamp"].astype(np.float64)
    spliced = np.concatenate([np.arange(10), np.arange(20)])
    table.loc[table.index[30:60], "frame_index"] = spliced
    table.loc[table.index[30:60], "timestamp"] = spliced / FPS
    # 前段 + ep0 落在 file-000，后段 + ep2 落在 file-001
    pq.write_table(pa.Table.from_pandas(table.iloc[:40].reset_index(drop=True),
                                        preserve_index=False), data_path)
    pq.write_table(pa.Table.from_pandas(table.iloc[40:].reset_index(drop=True),
                                        preserve_index=False),
                   root / "data" / "chunk-000" / "file-001.parquet")

    # meta 两行顺序与物理段相反：先写 20 帧段，再写 10 帧段
    ep_path = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    meta = pq.read_table(ep_path).to_pandas()
    row = meta[meta["episode_index"] == 1].iloc[0].copy()
    a, b = row.copy(), row.copy()
    a["length"], b["length"] = 20, 10
    meta = meta[meta["episode_index"] != 1]
    meta = pd.concat([meta, pd.DataFrame([a, b])], ignore_index=True)
    meta = meta.sort_values("episode_index", kind="stable").reset_index(drop=True)
    pq.write_table(pa.Table.from_pandas(meta, preserve_index=False), ep_path)

    from pipeline.cleaner import clean_dataset
    from pipeline.validator import validate_dataset

    dataset = load_dataset(root)
    decisions = decide(validate_dataset(dataset, load_config(robot_type="test_robot")),
                       dataset=dataset)
    assert decisions[1].repair == "split", decisions[1]
    assert [s["length"] for s in decisions[1].split_plan] == [10, 20]

    result = clean_dataset(dataset, decisions, tmp_path / "out", prefer_cli=False)
    cleaned = load_dataset(result.output_root)
    lengths = [int(cleaned.meta_row(ep)["length"]) for ep in cleaned.episode_indices]
    assert lengths == [30, 10, 20, 30], lengths
    assert cleaned.num_frames == 90
    for ep in cleaned.episode_indices:
        frame = cleaned.episode_frame(ep)
        assert np.array_equal(frame["frame_index"].to_numpy(), np.arange(len(frame)))
    assert result.verification.get("ok") is True, result.verification


def test_rewrite_tolerates_schema_difference_across_files(tmp_path):
    """不同导出批次的 parquet 列集合不一致时，permissive 合并不能让清洗崩掉。"""
    root = build_dataset(tmp_path / "ds", {
        0: smooth_episode(20, seed=1),
        1: smooth_episode(25, seed=2),
        2: smooth_episode(20, seed=3),
    })
    data_path = root / "data" / "chunk-000" / "file-000.parquet"
    table = pq.read_table(data_path).to_pandas()
    first = table.iloc[:45].reset_index(drop=True)   # ep0 + ep1
    second = table.iloc[45:].reset_index(drop=True)  # ep2
    second["extra_column"] = 1.0
    pq.write_table(pa.Table.from_pandas(first, preserve_index=False), data_path)
    pq.write_table(pa.Table.from_pandas(second, preserve_index=False),
                   root / "data" / "chunk-000" / "file-001.parquet")

    from pipeline.cleaner import clean_dataset

    result = clean_dataset(load_dataset(root), [0], tmp_path / "out", prefer_cli=False)
    assert result.verification.get("episodes") == 2
    assert result.verification.get("frames") == 45



def test_manual_clean_skips_unwritable_plan_items(tmp_path):
    """读取失败 / meta 谎报长度的 plan 项要降级为「剔除 + 告警」并重编号，
    不能让 _rebuild_videos / rewrite_episodes_meta 对缺失的 data_meta[new]
    KeyError 使整次清洗 abort（OCR P1-7）。

    场景：caller 直接给了 kept（清洗器也接受外部传入的保留清单），
    ep0 的 split 段声明长度 999 远超实际 10 行。
    """
    from pipeline.cleaner.manual import manual_clean

    root = build_dataset(tmp_path / "ds", {
        0: smooth_episode(10, seed=1), 1: smooth_episode(10, offset=1.0, seed=2),
    })
    dataset = load_dataset(root)
    split_plan = {0: [{"length": 999, "videos": {}}, {"length": 5, "videos": {}}]}

    result, total_frames = manual_clean(
        dataset, kept=[0, 1], out_root=tmp_path / "out", overwrite=True,
        trim_plan={}, rebuild_ts=set(), repairs={}, split_plan=split_plan,
    )
    assert result.success
    assert any("未能写入产物" in w for w in result.warnings)

    # 段起点是累计声明长度，seg0 谎报 999 时两段都写不出 → ep0 整体剔除，
    # ep1 重编号为 0；产物可读且编号连续（rewrite_info 依赖该假设）
    out = load_dataset(tmp_path / "out")
    assert [int(e) for e in out.episode_indices] == [0]
    assert [int(n) for n in out.episodes["length"]] == [10]
    assert total_frames == 10
    assert result.episode_index_map == {1: 0}
