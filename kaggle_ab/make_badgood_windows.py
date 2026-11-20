"""坏/好训练集窗口抽样（本地，无 torch 依赖）。

目的：量化「脏样本的可学习性」。冻结 eval 的 320 窗口全部来自清洗存活集，
无法做坏/好对比；本脚本从 A（so101_minrepair，157 块）里另抽两组窗口：
  bad  = 源 id 属于 29 个 drop 集的块（含 22 个双拷贝块）；
  good = 其余训练块（两臂都训过这些集）。
两组各 40 窗口，分层抽样口径与 make_eval_split.py 一致（WARMUP=8, H=50）。

判读（diff-in-diff，预注册）：gap = mean(A_loss - B2_loss)。
  gap_bad - gap_good ≈ -(A 在坏集上的训练拟合增益)。
  显著为负 → 脏 (s,a) 对是可学习信号，支持 repair-not-drop；
  ≈0 → 坏集训练无额外拟合，逐样本价值≈0（覆盖效应另计）。
"""
from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXPORT = ROOT / "exports" / "kaggle_ab"
HORIZON = 50
WARMUP = 8
N_PER_GROUP = 40
SEED = "lerobot-ab:badgood:v1"


def stratified_windows(units: dict[int, list[tuple[int, int]]], n: int, rng: random.Random) -> list[dict]:
    """units: unit_key -> [(block, length)]。bad 组 unit=源 id（保证每 id 覆盖），
    good 组 unit=块。配额 = 每 unit 至少 1（unit 数 <= n 时）+ 余量按长度比例最大余数法。
    起点在 unit 配额内分段铺开；双拷贝 unit 的每个窗口 seeded 选块。"""
    keys = sorted(units)
    weight = {k: sum(l for _, l in units[k]) for k in keys}
    total = sum(weight.values())
    alloc = {k: 1 for k in keys} if len(keys) <= n else {k: 0 for k in keys}
    rest = n - sum(alloc.values())
    quota = {k: rest * weight[k] / total for k in keys}
    for k in keys:
        alloc[k] += int(quota[k])
    left = rest - sum(int(quota[k]) for k in keys)
    for k in sorted(keys, key=lambda k: quota[k] - int(quota[k]), reverse=True)[:left]:
        alloc[k] += 1
    out = []
    for k in keys:
        n_k = alloc[k]
        if n_k == 0:
            continue
        for j in range(n_k):
            block, length = rng.choice(units[k])
            lo, hi = WARMUP, length - HORIZON - 1
            if hi < lo:
                raise ValueError(f"block {block} length {length} too short")
            seg = (hi - lo + 1) / n_k
            start = lo + int((j + rng.uniform(0, 1)) * seg)
            out.append({"episode_index": block, "start_frame": min(start, hi)})
    return out


def main() -> None:
    manifest = json.loads((EXPORT / "rebuild_minrepair_manifest.json").read_text(encoding="utf-8"))
    new_to_old = {int(k): v for k, v in manifest["new_to_old_episode"].items()}
    notes = json.loads(
        (ROOT / "outputs_ext_bomb" / "cleaned_dataset" / "meta" / "quality_cleaning_notes.json").read_text(encoding="utf-8")
    )
    bad_ids = set(notes["removed_episode_indices"])
    import pandas as pd

    ep_dir = EXPORT / "so101_minrepair" / "meta" / "episodes"
    df = pd.concat([pd.read_parquet(p) for p in sorted(ep_dir.rglob("*.parquet"))])
    lengths = {int(r.episode_index): int(r.length) for r in df.itertuples()}
    assert set(lengths) == set(new_to_old), "meta/manifest block set mismatch"
    bad_units: dict[int, list[tuple[int, int]]] = {}
    good_units: dict[int, list[tuple[int, int]]] = {}
    for k, length in lengths.items():
        old = new_to_old[k]
        if old in bad_ids:
            bad_units.setdefault(old, []).append((k, length))
        else:
            good_units[k] = [(k, length)]
    rng = random.Random(SEED)
    windows = []
    for group, units in (("bad", bad_units), ("good", good_units)):
        for w in stratified_windows(units, N_PER_GROUP, rng):
            w.update(group=group, source_id=new_to_old[w["episode_index"]], length=lengths[w["episode_index"]])
            windows.append(w)
    out = {
        "seed": SEED,
        "horizon": HORIZON,
        "warmup": WARMUP,
        "n_bad_blocks": sum(len(v) for v in bad_units.values()),
        "n_good_blocks": len(good_units),
        "bad_source_ids": sorted(bad_ids),
        "windows": windows,
    }
    dst = EXPORT / "badgood_windows.json"
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    nb = sum(w["group"] == "bad" for w in windows)
    print(f"blocks bad={out['n_bad_blocks']} good={out['n_good_blocks']}; windows bad={nb} good={len(windows) - nb}")
    print("wrote", dst)


if __name__ == "__main__":
    main()