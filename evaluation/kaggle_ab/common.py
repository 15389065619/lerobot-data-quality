# Copyright (c) 2026 wu
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Kaggle A/B 清洗收益实验——第 1 步本地构建脚本的共享常量与工具。

脚本目录：evaluation/kaggle_ab/（不放数据）
产物目录：artifacts/exports/kaggle_ab/（三份数据集 + eval 索引 + 各步骤 manifest）

流程与顺序（顺序不能换，apply_stats_fixed 依赖转码后的视频像素）：
    1. make_eval_split.py    冻结 15 个 eval episode + 320 个评测窗口（seeded）
    2. build_datasets.py     A=so101_minrepair / B=so101_clean / eval=so101_eval
    3. transcode_h264.py     三份视频统一 h264（Kaggle 4 vCPU 软解 AV1 会卡）
    4. apply_stats_fixed.py  用 B 组训练集算一份 stats，覆盖 A/B/eval（loss 可比的前提）
    5. verify_datasets.py    结构校验 + validator 三连跑 + 判定

关键口径（全部 seeded、全部留痕，任何一步重跑结果不变）：
    - eval 集从清洗后保留的 121 集里抽 15 个，两臂训练集都剔除这 15 集，避免泄漏；
    - eval episode 双编号：cleaned 新编号（0..120，B/eval 用）↔ bomb 原编号（A 用），
      映射来自 outputs_ext_bomb/cleaned_dataset/meta/quality_cleaning_notes.json；
    - 三份数据共用同一份 stats.json（B 组训练集算的），归一化口径一致。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "artifacts" / "exports" / "kaggle_ab"

# 源数据
SRC_BOMB = REPO / "data" / "external" / "so101_bomb_pick_place_v1"      # 150 集 / 88782 帧（parquet 真值）
SRC_CLEANED = REPO / "artifacts" / "outputs_ext_bomb" / "cleaned_dataset"              # 121 集 / 63185 帧

# 产物
DS_MINREPAIR = OUT / "so101_minrepair"   # A 组：157 块级 episode（172 物理块 − 15 eval，含全部 29 个坏集，只修结构不修内容）
DS_CLEAN = OUT / "so101_clean"           # B 组：106 集（完整清洗产物再剔 eval）
DS_EVAL = OUT / "so101_eval"             # eval：15 集（来自清洗产物，两臂共用的固定评测集）
EVAL_EPISODES_JSON = OUT / "eval_episodes.json"
EVAL_WINDOWS_JSON = OUT / "eval_windows.json"

# 实验口径
EVAL_SEED = 0
N_EVAL_EPISODES = 15
N_WINDOWS = 320
HORIZON = 50        # SmolVLA action chunk（与 Kaggle 训练 policy 配置一致，Session 1 核对）
WARMUP = 8          # 窗口避开每集开头帧
IMAGE_STRIDE = 5    # 图像统计抽帧步长（B 组 ~11k 帧/相机）

VALIDATOR_OUT = REPO / "artifacts" / "outputs_kaggle_ab"


def load_json(path: Path):
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def dump_json(path: Path, obj) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    return path


def ensure_repo_on_path() -> None:
    """让 kaggle_ab/ 下的脚本能 import pipeline.*（脚本不在 repo 根运行也成立）。"""
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))


def load_cleaned_index_map() -> dict[int, int]:
    """bomb 原编号 -> cleaned 新编号（来自上次清洗留痕）。"""
    notes = load_json(SRC_CLEANED / "meta" / "quality_cleaning_notes.json")
    raw = notes["episode_index_map"]
    mapping = {int(k): int(v) for k, v in raw.items()}
    vals = sorted(mapping.values())
    if vals != list(range(len(vals))):
        raise ValueError("cleaned episode_index_map 不是 0..N-1 连续，脚本假设不成立")
    if len(mapping) != 121:
        raise ValueError(f"cleaned 留痕映射应有 121 个 episode，实际 {len(mapping)}")
    return mapping
