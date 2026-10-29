"""优先调用官方 `lerobot-edit-dataset` CLI 的删除路径。"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from ..loader import LeRobotDatasetView
from .types import CleaningResult
from .util import prepare_dir

LOG = logging.getLogger(__name__)

CLI_NAME = "lerobot-edit-dataset"


def try_cli(
    dataset: LeRobotDatasetView,
    bad: list[int],
    out_root: Path,
    *,
    overwrite: bool,
    kept: list[int],
    repairs: dict[str, list[int]],
) -> CleaningResult | None:
    if not bad:
        return None
    exe = os.environ.get("LEROBOT_EDIT_DATASET_CMD") or shutil.which(CLI_NAME)
    if not exe:
        LOG.debug("未找到 %s", CLI_NAME)
        return None

    try:
        help_proc = subprocess.run([exe, "--help"], capture_output=True, text=True, timeout=120, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        LOG.warning("探测 %s 失败: %s", CLI_NAME, exc)
        return None
    help_text = (help_proc.stdout or "") + (help_proc.stderr or "")
    if "delete_episodes" not in help_text:
        LOG.warning("当前 %s 不支持 delete_episodes（版本过旧），改用手写实现", CLI_NAME)
        return None

    prepare_dir(out_root, overwrite=overwrite)
    hf_home = out_root.parent / ".hf_lerobot_home"
    cmd = [
        exe,
        "--repo-id", f"local/{dataset.root.name}",
        "--root", str(dataset.root),
        "--operation.type", "delete_episodes",
        "--operation.episode_indices", json.dumps(bad),
        "--new_repo_id", f"local/{out_root.name}",
    ]
    env = {**os.environ, "HF_LEROBOT_HOME": str(hf_home)}
    LOG.info("调用 CLI: %s", " ".join(cmd))
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=3600, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        LOG.warning("CLI 执行异常: %s", exc)
        return None
    if proc.returncode != 0:
        LOG.warning("CLI 返回码 %s，stderr: %s", proc.returncode, (proc.stderr or "")[-800:])
        return None

    produced = _find_cli_output(hf_home, out_root.name)
    if produced is None:
        LOG.warning("CLI 执行成功但未在 %s 下找到产出数据集，回退到手写实现", hf_home)
        return None

    if produced.resolve() != out_root.resolve():
        for item in produced.iterdir():
            target = out_root / item.name
            if target.is_dir():
                shutil.copytree(item, target, dirs_exist_ok=True)
            else:
                shutil.copy2(item, target)

    return CleaningResult(
        success=True,
        method="lerobot-edit-dataset",
        output_root=out_root,
        removed_episodes=sorted(bad),
        repaired_episodes=[],
        repairs=repairs,
        kept_episodes=len(kept),
        episode_index_map={old: i for i, old in enumerate(kept)},
        cli_command=" ".join(cmd),
    )


def _find_cli_output(hf_home: Path, name: str) -> Path | None:
    for cand in hf_home.rglob("meta/info.json"):
        root = cand.parent.parent
        if root.name in {name, f"{name}_tmp"} or name in root.name:
            return root
    for cand in hf_home.rglob("meta/info.json"):
        return cand.parent.parent
    return None
