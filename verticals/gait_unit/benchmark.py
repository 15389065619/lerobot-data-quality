# Copyright (c) 2026 wu
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""步态质检基准 harness：真实底片 + 随机注入缺陷（地面真值）→ 规则查全/误报可测量。

公开入口：
  inject_random       随机注入四类缺陷之一（frame_drop/duplicate_stamp/
                      joint_limit/drift），返回 (clip, truth)——truth 是
                      逐条地面真值（类型/目标关节/帧区间/参数）
  benchmark_on_clip   单底片 × 一次注入 → 全规则 → 与真值匹配（单 k 版本）
  run_benchmark       一批真实底片 × trials 注入 → recall_by_type / precision /
                      按类别聚合；mad_ks 传多个 k 时一次注入、mad_jump 按 k
                      各评一轮（四类缺陷中只有 drift 的检出依赖 k）
  calibrate_bvh_categories  LAFAN1 全库逐类别的 mad_jump 本底报警率表
                      （纯解析无 FK，供 profiles 分档与跨域对照）

匹配定义（v1，刻意宽松并文档化）：
  - 真值条目 target = 受影响关节名（drift = 子树根关节，joints 列出全子树）；
    clip 级缺陷（frame_drop/duplicate_stamp）target = None
  - finding 命中 = target 匹配（相等，或 finding 目标 ∈ truth 的 joints 子树）
    且帧区间有交叠
  - precision = 命中的 finding 数 / 总 finding 数；recall = 被命中的真值数 / 真值数
  - 单条 finding 同时命中多个真值时只计一次 TP
  - 误报 = 未命中真值且底片基线解释不了的 finding（run_benchmark 内实现）

不做的事（v1 边界）：
  - 空间规则（接触/腾空）不进基准——四类注入缺陷不触碰接触语义；
  - marker 规则不进基准——LAFAN1 是骨骼级（marker 基准待有 marker 真数据）；
  - 多缺陷叠加注入留待 v2（单缺陷匹配语义干净）。
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from verticals.gait_unit.adapters import parse_bvh
from verticals.gait_unit.defects import (
    inject_drift,
    inject_duplicate_stamp,
    inject_frame_drop,
    inject_joint_limit,
)
from verticals.gait_unit.ir import GaitClip, validate_clip
from verticals.gait_unit.motion import MotionClip
from verticals.gait_unit.qc_rules.channel_rules import rule_mad_jump
from verticals.gait_unit.qc_rules.skeleton_rules import (
    rule_bone_length_drift,
    rule_joint_rotation_limit,
    rule_timestamp_duplicates,
    rule_timestamp_gaps,
)

__all__ = ["inject_random", "run_benchmark", "calibrate_bvh_categories"]

LOG = logging.getLogger(__name__)

_DEFECT_TYPES = ("frame_drop", "duplicate_stamp", "joint_limit", "drift")


def _descendants(j: int, parents: tuple[int, ...]) -> set[int]:
    """j 及其全部子孙的锚点/关节索引集合。"""
    out = {j}
    changed = True
    while changed:
        changed = False
        for i, p in enumerate(parents):
            if p in out and i not in out:
                out.add(i)
                changed = True
    return out


def inject_random(clip: GaitClip, rng: np.random.Generator) -> tuple[GaitClip, list[dict]]:
    """随机注入四类缺陷之一，返回 (注入后 clip, 地面真值列表)。"""
    dtype = _DEFECT_TYPES[rng.integers(0, len(_DEFECT_TYPES))]
    T = clip.n_frames
    movable = tuple(n for n in clip.joint_names if not n.endswith("_end"))
    truth: list[dict] = []

    if dtype == "frame_drop":
        start = int(rng.integers(200, max(201, T - 300)))
        n = int(rng.integers(2, 9))
        out, manifest = inject_frame_drop(clip, start=start, n=n)
        truth.append({"type": dtype, "target": None, "joints": None,
                      "frames": (start, start + n), "params": manifest})

    elif dtype == "duplicate_stamp":
        start = int(rng.integers(200, max(201, T - 300)))
        n = int(rng.integers(30, 120))
        out, manifest = inject_duplicate_stamp(clip, start=start, n=n)
        truth.append({"type": dtype, "target": None, "joints": None,
                      "frames": (start, start + n), "params": manifest})

    elif dtype == "joint_limit":
        name = movable[int(rng.integers(0, len(movable)))]
        extra = float(rng.integers(60, 121))
        out, manifest = inject_joint_limit(clip, joint=name, axis="X", extra_deg=extra)
        truth.append({"type": dtype, "target": name, "joints": {name},
                      "frames": (0, T), "params": manifest})

    else:  # drift
        j = int(rng.integers(1, len(clip.joint_names)))   # 非根
        name = clip.joint_names[j]
        chain = _descendants(j, clip.parents)
        chain_names = {clip.joint_names[i] for i in chain}
        rate = float(rng.uniform(0.02, 0.06))
        axis = "xyz"[int(rng.integers(0, 3))]
        start = int(rng.integers(int(0.3 * T), max(int(0.3 * T) + 1, int(0.7 * T))))
        out, manifest = inject_drift(clip, rate=rate, axis=axis,
                                     joints=tuple(chain_names), start=start)
        truth.append({"type": dtype, "target": name, "joints": chain_names,
                      "frames": (start, T), "params": manifest})

    return out, truth


def _angle_motion_clip(clip: GaitClip) -> MotionClip | None:
    """GaitClip 局部旋转角 → 通道级 MotionClip（mad_jump 的骨骼域入口）。

    角度语义与 ir.quat_angle 一致：2·acos(clamp(w))（向量化，逐帧×关节）。
    """
    if clip.rotations is None:
        return None
    angles = 2.0 * np.arccos(np.clip(clip.rotations[..., 0], -1.0, 1.0))
    return MotionClip(fps=clip.fps, channel_names=clip.joint_names,
                      channels=angles, source={"source": "angle_channels"})


def _normalize(clip: GaitClip, findings: list[list[dict]]) -> list[dict]:
    """各规则的 findings → 归一化列表 [{rule, target, runs}]。"""
    norm = []
    for group in findings:
        for f in group:
            rule = f.get("rule", "?")
            if rule in ("timestamp_gaps", "timestamp_duplicates"):
                target, runs = None, [tuple(f["frames"])]
            elif rule == "bone_length_drift":
                child = f["bone"].split("→")[-1]
                target, runs = child, [tuple(r) for r in f.get("runs", [])]
            elif rule == "joint_rotation_limit":
                target, runs = f.get("joint"), [tuple(r) for r in f.get("runs", [])]
            elif rule == "mad_jump":
                target, runs = f.get("channel"), [tuple(r) for r in f.get("runs", [])]
            else:
                target, runs = None, []
            norm.append({"rule": rule, "target": target, "runs": runs})
    return norm


def _overlap(runs: list[tuple[int, int]], frames: tuple[int, int]) -> bool:
    s, e = frames
    return any(max(s, rs) < min(e, re_) for rs, re_ in runs)


def _match(truth: list[dict], norm_findings: list[dict]) -> tuple[set[int], set[int]]:
    """返回 (被命中的真值下标, 命中任一真值的 finding 下标)。"""
    hit_truth: set[int] = set()
    hit_findings: set[int] = set()
    for fi, f in enumerate(norm_findings):
        for ti, t in enumerate(truth):
            target_ok = (
                (t["target"] is None and f["target"] is None)
                or (t["target"] is not None and f["target"] == t["target"])
                or (t["target"] is not None and f["target"] in (t.get("joints") or set()))
            )
            if not target_ok:
                continue
            frames = t["frames"]
            if f["runs"] and _overlap(f["runs"], frames):
                hit_truth.add(ti)
                hit_findings.add(fi)
                break
            if not f["runs"] and f["rule"] in ("timestamp_gaps", "timestamp_duplicates"):
                hit_truth.add(ti)
                hit_findings.add(fi)
                break
    return hit_truth, hit_findings


def benchmark_on_clip(base: GaitClip, rng: np.random.Generator,
                      mad_k: float = 6.0) -> dict:
    """单个底片 × 一次随机注入 → 全规则 → 与真值匹配（单 k 版本）。"""
    return _benchmark_on_clip_multi(base, rng, (mad_k,))[mad_k]


def _benchmark_on_clip_multi(base: GaitClip, rng: np.random.Generator,
                             mad_ks: tuple[float, ...]) -> dict[float, dict]:
    """一次注入、按多个 k 各评一轮 mad_jump（其余规则与 k 无关，只算一次）。

    返回 {k: episode dict}，各 k 的 findings_detail 相互独立
    （骨架部分相同，mad_jump 部分随 k 变化）。
    """
    injected, truth = inject_random(base, rng)
    issues = validate_clip(injected)
    skeleton_norm = _normalize(injected, [
        rule_timestamp_gaps(injected),
        rule_timestamp_duplicates(injected),
        rule_joint_rotation_limit(injected),
        rule_bone_length_drift(injected),
    ])
    angle_clip = _angle_motion_clip(injected)

    per_k: dict[float, dict] = {}
    injected_json = [{"type": t["type"], "target": t["target"],
                      "frames": list(t["frames"])} for t in truth]
    for k in mad_ks:
        norm = list(skeleton_norm)
        if angle_clip is not None:
            norm += _normalize(injected, [rule_mad_jump(angle_clip, k=k)])
        hit_truth, hit_findings = _match(truth, norm)
        per_k[k] = {
            "injected": injected_json,
            "detected": sorted(hit_truth),
            "n_findings": len(norm),
            "n_findings_tp": len(hit_findings),
            "tp_findings_indices": sorted(hit_findings),
            "structural_issues": issues,
            "findings_detail": norm,
        }
    return per_k


def _matches_baseline(f: dict, baseline: list[dict], tol: int = 10) -> bool:
    """finding 是否与底片自身的基线 findings 重合（同规则/同目标/帧距 ≤ tol）。

    真实底片自有的动态（步态拐点、蹬伸）也会被规则标记——这类 finding 不是
    注入缺陷的误报，是基线。基准只把"真值未命中且基线也解释不了"的
    finding 计为误报。
    """
    for b in baseline:
        if b["rule"] != f["rule"] or b["target"] != f["target"]:
            continue
        for rs, re_ in f["runs"]:
            for bs, be in b["runs"]:
                if abs(rs - bs) <= tol or abs(re_ - be) <= tol:
                    return True
    return False


def run_benchmark(files: list[Path], trials_per_file: int = 5, seed: int = 0,
                  scale: float = 0.01, up_axis: str = "y",
                  mad_ks: tuple[float, ...] = (6.0,)) -> dict:
    """对一批真实底片跑基准。返回汇总 dict（含 JSON 可序列化全部内容）。

    mad_ks 传多个 k 时一次注入、mad_jump 按 k 各评一轮——四类注入缺陷中
    只有 drift 的查全依赖 k（mad_jump 是它的检出规则），其余三类的 recall
    在各 k 下相同，仍按 k 重复记录以保持矩阵形状。

    误报定义：未命中任何注入真值 **且** 无法被底片基线 findings 解释的
    finding（底片自身动态的标记不算误报——真实数据永远有基线 findings）。
    """
    rng = np.random.default_rng(seed)
    mad_ks = tuple(float(k) for k in mad_ks)
    per_type = {k: defaultdict(lambda: {"truth": 0, "detected": 0}) for k in mad_ks}
    n_tp = {k: 0 for k in mad_ks}
    n_fp = {k: 0 for k in mad_ks}
    # 按类别聚合：类别 = 文件名前导字母段（与 calibrate_bvh_categories 同口径）
    per_cat = {k: defaultdict(lambda: defaultdict(lambda: {"truth": 0, "detected": 0}))
               for k in mad_ks}
    cat_fp = {k: defaultdict(int) for k in mad_ks}
    episodes = []

    for fi, path in enumerate(files):
        category = re.match(r"[A-Za-z]+", path.stem).group(0)
        base = parse_bvh(path, scale=scale, up_axis=up_axis)
        # 底片基线：干净底片上全规则的 findings（真实数据的自有动态）。
        # 骨架规则与 k 无关只算一次；mad_jump 基线按 k 各一份。
        skeleton_base = _normalize(base, [
            rule_timestamp_gaps(base),
            rule_timestamp_duplicates(base),
            rule_joint_rotation_limit(base),
            rule_bone_length_drift(base),
        ])
        angle_base = _angle_motion_clip(base)
        baseline_norm = {k: list(skeleton_base) for k in mad_ks}
        if angle_base is not None:
            for k in mad_ks:
                baseline_norm[k] += _normalize(base, [rule_mad_jump(angle_base, k=k)])

        for trial in range(trials_per_file):
            epk = _benchmark_on_clip_multi(base, rng, mad_ks)
            entry = {"file": str(path), "category": category, "trial": trial,
                     "injected": epk[mad_ks[0]]["injected"],
                     "structural_issues": epk[mad_ks[0]]["structural_issues"],
                     "per_k": {}}
            for k in mad_ks:
                ep = epk[k]
                tp_idx = set(ep["tp_findings_indices"])
                fp = [f for fbi, f in enumerate(ep["findings_detail"])
                      if fbi not in tp_idx
                      and not _matches_baseline(f, baseline_norm[k])]
                for ti, inj in enumerate(ep["injected"]):
                    per_type[k][inj["type"]]["truth"] += 1
                    per_type[k][inj["type"]]["detected"] += int(ti in ep["detected"])
                    per_cat[k][category][inj["type"]]["truth"] += 1
                    per_cat[k][category][inj["type"]]["detected"] += int(ti in ep["detected"])
                n_tp[k] += ep["n_findings_tp"]
                n_fp[k] += len(fp)
                cat_fp[k][category] += len(fp)
                seg = {"detected": ep["detected"], "n_findings": ep["n_findings"],
                       "n_findings_tp": ep["n_findings_tp"], "n_fp": len(fp),
                       "tp_findings_indices": ep["tp_findings_indices"]}
                if fp:  # 真误报罕见：只保留有内容的 FP 明细，控制 JSON 体积
                    seg["fp_detail"] = fp
                entry["per_k"][str(k)] = seg
            episodes.append(entry)
        LOG.info("基准 %d/%d: %s (%s)", fi + 1, len(files), path.name, category)

    def _recall_table(table: dict) -> dict:
        out = {}
        for dtype, d in table.items():
            recall = d["detected"] / d["truth"] if d["truth"] else None
            out[dtype] = {"truth": d["truth"], "detected": d["detected"],
                          "recall": round(recall, 4) if recall is not None else None}
        return out

    # precision 口径 = TP / (TP + FP)：分母**不含**底片基线 findings——
    # 它们既非 TP 也非 FP（基线机制已解释），混进分母会把 precision 误压低
    recall_by_type = {str(k): _recall_table(per_type[k]) for k in mad_ks}
    precision = {str(k): (round(n_tp[k] / (n_tp[k] + n_fp[k]), 4)
                          if (n_tp[k] + n_fp[k]) else None) for k in mad_ks}
    per_category = {
        str(k): {cat: _recall_table(per_cat[k][cat]) for cat in sorted(per_cat[k])}
        for k in mad_ks
    }
    fp_by_category = {str(k): dict(sorted(cat_fp[k].items())) for k in mad_ks}

    return {
        "n_files": len(files), "trials_per_file": trials_per_file, "seed": seed,
        "mad_ks": list(mad_ks), "scale": scale, "up_axis": up_axis,
        "n_trials": len(episodes),
        "precision": precision,
        "n_false_positives": {str(k): n_fp[k] for k in mad_ks},
        "recall_by_type": recall_by_type,
        "recall_by_category": per_category,
        "fp_by_category": fp_by_category,
        "episodes": episodes,
    }


def calibrate_bvh_categories(source_dir: str | Path, out_path: str | Path,
                             scale: float = 0.01, up_axis: str = "y",
                             ks: tuple[float, ...] = (6.0, 8.0, 10.0, 12.0),
                             max_files: int | None = None) -> dict:
    """对 LAFAN1 全库逐类别标定 MAD 跳变规则的误报基线（纯解析，无 FK）。

    类别 = 文件名前导字母段（walk1_subject5 → walk）。输出
    {category: {k: 每千帧×关节被标帧数}} —— 分档参数 k 的数据依据。
    """
    source_dir = Path(source_dir)
    files = sorted(source_dir.rglob("*.bvh"))
    if max_files:
        files = files[:max_files]
    rates: dict[str, dict[str, float]] = defaultdict(dict)
    frames_x_joints: dict[str, int] = defaultdict(int)
    parsed = 0
    for path in files:
        category = re.match(r"[A-Za-z]+", path.stem).group(0)
        clip = parse_bvh(path, scale=scale, up_axis=up_axis)
        angle_clip = _angle_motion_clip(clip)
        T, J = clip.n_frames, clip.n_joints
        frames_x_joints[category] += T * J
        for k in ks:
            findings = rule_mad_jump(angle_clip, k=k)
            flagged = sum(f["n_flagged"] for f in findings)
            rates[category][str(k)] = rates[category].get(str(k), 0.0) + flagged
        parsed += 1
        LOG.info("校准 %d/%d: %s (%s)", parsed, len(files), path.name, category)

    out = {"_provenance": {"source": str(source_dir), "n_files": parsed,
                           "scale": scale, "up_axis": up_axis,
                           "note": "每千帧×关节被标帧数 = flagged/(该类 T×J 合计)×1000"},
           "categories": {}}
    for c, per_k in rates.items():
        denom = max(frames_x_joints[c], 1) / 1000.0
        out["categories"][c] = {k: round(v / denom, 2) for k, v in per_k.items()}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(out, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    LOG.info("校准完成: %s（%d 文件）", out_path, parsed)
    return out
