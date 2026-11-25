# Copyright (c) 2026 wu
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""第 1 步 a：冻结 eval 划分与评测窗口（全 seeded，重跑结果不变）。

1. 从 cleaned_dataset（121 集，编号 0..120）里 seed 固定抽 15 集作 eval，
   同时用清洗留痕映射回 bomb 原编号（A 组剔除用）；
2. 每集按集长分层抽窗口，共 320 个 (episode_index, start_frame)，
   范围限定 [WARMUP, L-HORIZON-1]，避开每集开头、保证 action chunk 越界不可用；
3. 两个索引文件写进 exports/kaggle_ab/，两臂训练/评测共用。

窗口的 episode_index 用 eval 数据集（重排后 0..14）的编号——构建脚本保证
「15 个 eval 集按原编号排序后重排为 0..14」，因此本脚本可以提前算窗口。
"""

from __future__ import annotations

import argparse
import random

import sys as _sys
from pathlib import Path as _P
_SCRIPT_DIR = str(_P(__file__).resolve().parent)
if _SCRIPT_DIR not in _sys.path:
    _sys.path.insert(0, _SCRIPT_DIR)

from common import (
    DS_EVAL,
    DS_MINREPAIR,
    DS_CLEAN,
    EVAL_EPISODES_JSON,
    EVAL_SEED,
    EVAL_WINDOWS_JSON,
    HORIZON,
    N_EVAL_EPISODES,
    N_WINDOWS,
    SRC_CLEANED,
    WARMUP,
    dump_json,
    ensure_repo_on_path,
    load_cleaned_index_map,
)

ensure_repo_on_path()
from pipeline.loader import load_dataset  # noqa: E402


def sample_windows(lengths: list[int]) -> list[dict]:
    """每集分层均匀抽窗口：范围等分成 n_k 段，每段内 uniform 取一个起点。

    分层保证窗口在整集铺开（起点间隔 >= 段长，远大于 HORIZON），统计上接近独立。
    320 = 15*21 + 5：每集 21 个，余 5 个按 seeded 顺序补给前 5 集。
    """
    rng_extra = random.Random(f"lerobot-ab:{EVAL_SEED}:extra")
    per_ep = [N_WINDOWS // len(lengths)] * len(lengths)
    for i in rng_extra.sample(range(len(lengths)), N_WINDOWS % len(lengths)):
        per_ep[i] += 1

    windows: list[dict] = []
    for new_idx, (length, n_k) in enumerate(zip(lengths, per_ep)):
        lo, hi = WARMUP, length - HORIZON - 1
        if hi < lo:
            raise ValueError(f"episode {new_idx} 长 {length} 帧，放不下一个窗口（warmup+horizon）")
        seg = (hi - lo + 1) / n_k
        rng = random.Random(f"lerobot-ab:{EVAL_SEED}:win:{new_idx}")
        starts = sorted(lo + int(rng.uniform(0, seg)) + k * int(seg) for k in range(n_k))
        for s in starts:
            windows.append({
                "episode_index": new_idx,
                "original_episode_index": None,  # 由 main() 补上
                "start_frame": s,
            })
    return windows


def main() -> int:
    parser = argparse.ArgumentParser(description="冻结 eval 划分与评测窗口")
    parser.parse_args()

    view = load_dataset(SRC_CLEANED)
    cleaned_eps = sorted(view.episode_indices)
    if len(cleaned_eps) != 121:
        raise SystemExit(f"[错误] cleaned_dataset 应有 121 集，实际 {len(cleaned_eps)}")

    old2new = load_cleaned_index_map()          # bomb 原编号 -> cleaned 新编号
    new2old = {v: k for k, v in old2new.items()}

    # seed 必须是 str/int（random.seed 对 str 走 sha512，跨进程确定）；
    # 不能用 tuple——tuple 走 hash()，str 的 hash 随 PYTHONHASHSEED 变化，不可复现
    rng = random.Random(f"lerobot-ab:{EVAL_SEED}:episodes")
    picked_new = sorted(rng.sample(cleaned_eps, N_EVAL_EPISODES))
    picked_old = sorted(new2old[e] for e in picked_new)

    lengths = [int(view.frames_by_episode()[e]) for e in picked_new]
    windows = sample_windows(lengths)
    for w in windows:  # eval 数据集内的重排编号 = picked_new 排序后的下标
        w["original_episode_index"] = picked_old[w["episode_index"]]

    eval_episodes = {
        "seed": EVAL_SEED,
        "source": str(SRC_CLEANED),
        "n_episodes": N_EVAL_EPISODES,
        "cleaned_episode_indices": picked_new,   # B 组 / eval 数据集的剔除与编号基准
        "bomb_episode_indices": picked_old,      # A 组（bomb 原编号）的剔除基准
        "episode_lengths": {str(n): l for n, l in zip(picked_new, lengths)},
    }
    eval_windows = {
        "seed": EVAL_SEED,
        "horizon": HORIZON,
        "warmup": WARMUP,
        "n_windows": len(windows),
        "note": "episode_index 为 so101_eval 数据集内重排编号(0..14)；start_frame 为集内帧号；"
                "窗口语义: obs@start_frame, action chunk = start_frame..start_frame+HORIZON-1",
        "windows": windows,
    }

    dump_json(EVAL_EPISODES_JSON, eval_episodes)
    dump_json(EVAL_WINDOWS_JSON, eval_windows)

    print(f"eval 集（cleaned 编号）: {picked_new}")
    print(f"eval 集（bomb 原编号）: {picked_old}")
    print(f"集长: min={min(lengths)} max={max(lengths)} mean={sum(lengths)/len(lengths):.0f}")
    counts = {}
    for w in windows:
        counts[w["episode_index"]] = counts.get(w["episode_index"], 0) + 1
    print(f"窗口: {len(windows)} 个，每集 {sorted(counts.values())}，范围 [{WARMUP}, L-{HORIZON}-1]")
    print(f"产物: {EVAL_EPISODES_JSON}")
    print(f"      {EVAL_WINDOWS_JSON}")
    print("预期: A=so101_minrepair 157 块 / 81229 帧（重复 id 拆成独立块，行数不变）")
    print("      B=so101_clean 106 集 / eval=so101_eval 15 集")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
