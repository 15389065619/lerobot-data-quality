# Copyright (c) 2026 wu
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""坏/好窗口可学习性评测（Kaggle GPU kernel / 本地 CPU 双端通用）。

对 A_s1 / B2_s1 两个 final checkpoint 在 80 个新窗口（40 bad + 40 good，
badgood_windows.json，窗口编号 = so101_minrepair 的 episode_index）上逐窗口
flow-matching loss。两臂同窗口同 batch 顺序同 noise seed（torch.manual_seed(bi)），
保证配对。输出 badgood_results.json（路径见 OUT）。

判读（预注册）：gap_g = mean(A-B2) 分组 g；DiD = gap_bad - gap_good
≈ -(A 在坏集上的训练拟合增益)。显著负 → 脏 (s,a) 对可学习，支持 repair-not-drop；
≈0 → 坏集逐样本拟合价值≈0。good 组两臂都训过源集，gap_good 为臂质量基线差。

路径与设备均可用环境变量覆盖（Kaggle 上不设即用默认值）：
  BADGOOD_ROOT  数据集根（含 so101_minrepair/）
  BADGOOD_ZIP   badgood 评测包（badgood_windows.json + ckpt/）
  BADGOOD_EVAL  解包目录
  BADGOOD_OUT   结果 JSON 输出路径
  BADGOOD_DEVICE  强制设备（默认自动：有 cuda 用 cuda，否则 cpu）
"""
import subprocess
import sys
import os

# BADGOOD_SKIP_PIP=1：环境已自备依赖时跳过安装（PyPI 的 0.7.0 只有 Linux 轮子，
# Windows 本地从 GitHub 源码装）
if not os.environ.get('BADGOOD_SKIP_PIP'):
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'lerobot[dataset,smolvla]==0.7.0'], check=True)

import json
import os
import shutil
import zipfile
from pathlib import Path

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

ROOT = os.environ.get('BADGOOD_ROOT', '/kaggle/input/so101-kaggle-ab-v1')
ZIP = os.environ.get('BADGOOD_ZIP', '/kaggle/input/so101-ab-badgood-eval/so101_badgood_eval.zip')
EVAL = os.environ.get('BADGOOD_EVAL', '/kaggle/working/badgood')
OUT = os.environ.get('BADGOOD_OUT', '/kaggle/working/badgood_results.json')
DEVICE = os.environ.get('BADGOOD_DEVICE') or ('cuda' if torch.cuda.is_available() else 'cpu')

# 解包评测包（zipfile 跨平台；已解包则跳过；zip 若带一层包装目录则上提）
if not os.path.isfile(os.path.join(EVAL, 'badgood_windows.json')):
    with zipfile.ZipFile(ZIP) as zf:
        zf.extractall(EVAL)
    if not os.path.isfile(os.path.join(EVAL, 'badgood_windows.json')):
        inner = next(d for d in Path(EVAL).iterdir() if d.is_dir())
        for p in inner.iterdir():
            shutil.move(str(p), os.path.join(EVAL, p.name))
        inner.rmdir()
print('DEVICE', DEVICE, flush=True)

H = 50
FPS = 30.0
WIN = json.load(open(EVAL + '/badgood_windows.json'))['windows']


def build_cache(tolerance_s):
    ds = LeRobotDataset(
        repo_id='happyday1223/so101_minrepair', root=ROOT + '/so101_minrepair',
        video_backend='pyav', tolerance_s=tolerance_s,
        delta_timestamps={'action': [i / FPS for i in range(H)]})
    epcol = [int(e) for e in ds.hf_dataset['episode_index']]
    starts = {}
    for i, e in enumerate(epcol):
        starts.setdefault(e, i)
    gidx = [starts[w['episode_index']] + w['start_frame'] for w in WIN]
    return [[ds[i]] for i in gidx]


try:
    CACHE = build_cache(0.01)
    TOL = 0.01
except Exception as exc:
    print('TOL_FALLBACK', type(exc).__name__, str(exc)[:200], flush=True)
    CACHE = build_cache(0.04)
    TOL = 0.04
print('CACHE_BUILT', len(CACHE), 'tolerance_s', TOL, flush=True)


def to_float(o):
    if isinstance(o, dict):
        for k in ('loss', 'total_loss', 'l'):
            if k in o:
                return float(o[k])
        raise KeyError(list(o.keys()))
    if isinstance(o, (tuple, list)):
        return to_float(o[0])
    if torch.is_tensor(o):
        return float(o.reshape(-1)[0]) if o.numel() == 1 else float(o.float().mean())
    return float(o)


def run_ckpt(name):
    ck = f'{EVAL}/ckpt/{name}'
    pol = SmolVLAPolicy.from_pretrained(ck)
    pol.eval()
    pol.to(DEVICE)
    if DEVICE == 'cpu':
        pol.float()  # CPU 上 fp16 大部分算子未实现，权重统一升 fp32
    pre, _post = make_pre_post_processors(pol.config, ck)
    out = []
    with torch.no_grad():
        for bi, items in enumerate(CACHE):
            batch = {}
            for k in items[0]:
                vals = [it[k] for it in items]
                batch[k] = torch.stack(vals) if torch.is_tensor(vals[0]) else vals
            r = pre(batch)
            batch = r[0] if isinstance(r, (tuple, list)) else r
            batch = {k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in batch.items()}
            torch.manual_seed(bi)
            o = pol(batch)
            out.append(to_float(o))
            if bi == 0:
                print('PROBE', name, type(o), list(o.keys()) if isinstance(o, dict) else '', flush=True)
    del pol
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()
    return out


res = {'tolerance_s': TOL, 'windows': WIN}
for name in ('arm_A_s1', 'arm_B2_s1'):
    res[name] = run_ckpt(name)
    print('DONE', name, len(res[name]), flush=True)

a = np.asarray(res['arm_A_s1'])
b = np.asarray(res['arm_B2_s1'])
groups = np.asarray([w['group'] for w in WIN])
summary = {}
for g in ('bad', 'good'):
    d = (a - b)[groups == g]
    summary[g] = dict(n=int(len(d)), A=float(a[groups == g].mean()), B=float(b[groups == g].mean()),
                      gap=float(d.mean()), se=float(d.std(ddof=1) / np.sqrt(len(d))))
did = summary['bad']['gap'] - summary['good']['gap']
summary['diff_in_diff'] = did
res['summary'] = summary
json.dump(res, open(OUT, 'w'), indent=1)
print('SUMMARY', json.dumps(summary), flush=True)