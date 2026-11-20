"""对比清洗产物 v3 与 A 臂 so101_minrepair 的 episode 长度多重集。

两步：
1) 按 exports/kaggle_ab/eval_episodes.json 的 15 个源编号，把 eval 留出集
   从 v3 产物中排除（这一步只发生在对比脚本，不影响清洗策略）；
2) 比较剩余 episode 的 length 多重集是否与 minrepair 完全一致。

用法：python kaggle_ab/compare_v3_vs_minrepair.py
"""

from __future__ import annotations

import collections
import json
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
V3 = ROOT / "outputs_ext_bomb_v3"
MINREPAIR = ROOT / "exports" / "kaggle_ab" / "so101_minrepair"
EVAL = ROOT / "exports" / "kaggle_ab" / "eval_episodes.json"


def _episode_lengths(dataset: Path) -> dict[int, int]:
    files = sorted((dataset / "meta" / "episodes").rglob("*.parquet"))
    rows: dict[int, int] = {}
    for path in files:
        table = pq.read_table(path).to_pandas()
        for _, row in table.iterrows():
            rows[int(row["episode_index"])] = int(row["length"])
    return rows


def _multiset(lengths) -> collections.Counter:
    return collections.Counter(int(v) for v in lengths)


def main() -> int:
    report = json.loads((V3 / "quality_report.json").read_text(encoding="utf-8"))
    cleaning = report["cleaning"]
    repairs = cleaning["repairs"]
    split = set(int(e) for e in repairs.get("split", []))
    window = set(int(e) for e in repairs.get("window", []))
    removed = [int(e) for e in cleaning.get("removed_episode_indices", [])]

    v3 = _episode_lengths(V3 / "cleaned_dataset")
    minrepair = _episode_lengths(MINREPAIR)
    eval_bomb = [int(e) for e in json.loads(EVAL.read_text(encoding="utf-8"))["bomb_episode_indices"]]

    old_to_new = {int(k): int(v) for k, v in cleaning["episode_index_map"].items()}
    excluded = {old_to_new[e] for e in eval_bomb if e in old_to_new}
    remaining = {ep: length for ep, length in v3.items() if ep not in excluded}

    print("分型统计")
    print(f"  repair_window : {len(window):3d}  {sorted(window)}")
    print(f"  repair_split  : {len(split):3d}  {sorted(split)}")
    print(f"  hard drop     : {len(removed):3d}  {sorted(removed)}")
    print(f"  v3 总帧数     : {sum(v3.values())}（拯救 {report['summary']['repair_frames']} 帧）")

    print("\n两步对比")
    print(f"  v3               : {len(v3):3d} 集 / {sum(v3.values()):6d} 帧")
    print(f"  排除 eval 15 集   : {len(excluded):3d} 集 / {sum(v3[e] for e in excluded):6d} 帧")
    print(f"  剩余             : {len(remaining):3d} 集 / {sum(remaining.values()):6d} 帧")
    print(f"  minrepair        : {len(minrepair):3d} 集 / {sum(minrepair.values()):6d} 帧")

    left = _multiset(remaining.values())
    right = _multiset(minrepair.values())
    if left == right:
        print("\nPASS: 长度多重集与 minrepair 完全一致")
        return 0

    print("\nFAIL: 长度多重集不一致")
    only_left = left - right
    only_right = right - left
    if only_left:
        print("  仅 v3 有:", dict(sorted(only_left.items())))
    if only_right:
        print("  仅 minrepair 有:", dict(sorted(only_right.items())))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
