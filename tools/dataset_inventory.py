"""数据集盘点：逐个数据集根目录报「info 声明」与「实测」的 episode / 帧数。

docs/EVAL.md §2.3 的「17 个数据集 / 4108 episode」是实测口径（不是 info 声明求和），
本脚本用来复算那一格，避免审计范围只剩一个手工数字。

用法：
    python tools/dataset_inventory.py [--root data] [--include-demo]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[1]


def dataset_roots(bases: list[Path]) -> list[Path]:
    roots = []
    for base in bases:
        if not base.is_dir():
            continue
        for info in sorted(base.rglob("meta/info.json")):
            root = info.parent.parent
            if ".cache" in root.parts:
                continue
            roots.append(root)
    return roots


def _rows(path: Path) -> int:
    """只读 parquet footer，不加载数据。"""
    try:
        return pq.ParquetFile(path).metadata.num_rows
    except Exception:
        return 0


def summarize(root: Path) -> dict:
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    data_files = sorted((root / "data").rglob("*.parquet"))
    if "{episode_index" in info.get("data_path", ""):      # v2.1：一集一文件
        actual_ep = len(data_files)
    else:                                                   # v3.0：数 episodes 元数据
        ep_meta = sorted((root / "meta" / "episodes").rglob("*.parquet"))
        actual_ep = sum(_rows(p) for p in ep_meta)
    return {
        "path": str(root.relative_to(REPO)),
        "version": info.get("codebase_version"),
        "fps": info.get("fps"),
        "robot_type": info.get("robot_type"),
        "has_action": "action" in (info.get("features") or {}),
        "declared_episodes": info.get("total_episodes"),
        "actual_episodes": actual_ep,
        "declared_frames": info.get("total_frames"),
        "actual_frames": sum(_rows(p) for p in data_files),
        "counts_match": actual_ep == info.get("total_episodes"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", action="append", default=None, help="要扫描的目录，可重复")
    ap.add_argument("--include-demo", action="store_true", help="把合成 demo_dataset 也计入")
    args = ap.parse_args()

    bases = [Path(r).resolve() for r in (args.root or [REPO / "data"])]
    if args.include_demo:
        bases.append(REPO / "demo_dataset")

    rows = [summarize(root) for root in dataset_roots(bases)]
    rows.sort(key=lambda r: r["path"])
    for r in rows:
        flag = "" if r["counts_match"] else "   <-- 声明与实测不符"
        print(f"{r['path']:44s} {r['version'] or '?':5s} fps={str(r['fps']):>6s} "
              f"ep {r['declared_episodes']}->{r['actual_episodes']}  "
              f"frames {r['declared_frames']}->{r['actual_frames']}"
              f"{'  [no action]' if not r['has_action'] else ''}{flag}")

    print(f"\n数据集 {len(rows)} 个 / 实测 episode {sum(r['actual_episodes'] for r in rows)} "
          f"/ 实测帧 {sum(r['actual_frames'] for r in rows)} / "
          f"计数不符 {sum(1 for r in rows if not r['counts_match'])} 个")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
