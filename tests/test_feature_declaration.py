"""info.json 缺少相机声明、但视频数据仍在的回归测试。

真实场景：Pico 上传链路在写 ``meta/info.json`` 时丢掉了某路相机特征，
``videos/`` 目录与 ``meta/episodes`` 的视频时间窗却完整保留。
读侧只认 ``features``，该相机因此静默消失：所有帧级规则照常通过，
报告里也不会出现任何提示，训练时等于少了一路视角。
"""

from __future__ import annotations

from conftest import build_dataset, smooth_episode

from pipeline.config import load_config
from pipeline.loader import load_dataset
from pipeline.validator import validate_dataset


def _episodes(n: int = 3) -> dict[int, dict]:
    return {ep: smooth_episode(40 + ep * 5, offset=ep * 0.7, seed=ep) for ep in range(n)}


def _codes(result) -> set[str]:
    return {issue.code for issue in result.dataset_issues}


def test_undeclared_camera_with_video_dir_is_reported(tmp_path):
    root = build_dataset(tmp_path / "ds", _episodes())
    cam_dir = root / "videos" / "observation.images.wrist" / "chunk-000"
    cam_dir.mkdir(parents=True)
    (cam_dir / "file-000.mp4").write_bytes(b"\x00" * 16)  # 检测只看目录归属，不解析内容

    result = validate_dataset(load_dataset(root), load_config(robot_type="test_robot"))

    assert "feature_missing_declaration" in _codes(result)
    issue = next(i for i in result.dataset_issues if i.code == "feature_missing_declaration")
    assert issue.detail["cameras"] == ["observation.images.wrist"]
    assert issue.detail["evidence"] == ["videos/"]


def test_v2x_chunk_layout_uses_camera_as_feature(tmp_path):
    """v2.x 的 videos/chunk-000/<camera>/ 不能被误认成一路叫 chunk-000 的相机。"""
    root = build_dataset(tmp_path / "ds", _episodes())
    cam_dir = root / "videos" / "chunk-000" / "observation.images.top"
    cam_dir.mkdir(parents=True)
    (cam_dir / "episode_000000.mp4").write_bytes(b"\x00")

    import json

    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["features"]["observation.images.top"] = {
        "dtype": "video",
        "shape": [96, 128, 3],
        "names": None,
        "info": None,
    }
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    result = validate_dataset(load_dataset(root), load_config(robot_type="test_robot"))
    assert "feature_missing_declaration" not in _codes(result)


def test_short_camera_dir_name_matches_full_feature_key(tmp_path):
    """v2.x 的 humanoid 数据集用短目录名：features 是 observation.images.egocentric，
    目录是 videos/chunk-000/egocentric，两者应视为同一路相机。"""
    root = build_dataset(tmp_path / "ds", _episodes())
    cam_dir = root / "videos" / "chunk-000" / "egocentric"
    cam_dir.mkdir(parents=True)
    (cam_dir / "episode_000000.mp4").write_bytes(b"\x00")

    import json

    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["features"]["observation.images.egocentric"] = {
        "dtype": "video",
        "shape": [96, 128, 3],
        "names": None,
        "info": None,
    }
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    result = validate_dataset(load_dataset(root), load_config(robot_type="test_robot"))
    assert "feature_missing_declaration" not in _codes(result)


def test_declared_camera_is_not_flagged(tmp_path):
    root = build_dataset(tmp_path / "ds", _episodes())
    (root / "videos" / "observation.images.top" / "chunk-000").mkdir(parents=True)
    (root / "videos" / "observation.images.top" / "chunk-000" / "file-000.mp4").write_bytes(b"\x00")

    # 把相机补进 features（v2.1 写法，dtype=video），模拟声明完整的数据集
    import json

    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["features"]["observation.images.top"] = {
        "dtype": "video",
        "shape": [96, 128, 3],
        "names": None,
        "info": None,
    }
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    result = validate_dataset(load_dataset(root), load_config(robot_type="test_robot"))
    assert "feature_missing_declaration" not in _codes(result)
