"""A 组（so101_minrepair）专用构建器：块级最小修复。

为什么不用 clean_dataset：bomb 的坏集 81-102 在数据里**物理存在两份分歧拷贝**
（第一份 file-004/005，第二份 file-006 且帧数不同，meta/episodes 172 行 = 150 id
+ 22 个第二拷贝行）。同一 episode_index 对应两块分离数据，v3 meta 的连续
from/to 无法表达；clean_dataset 按 id 处理，会把两份拷贝的行混在一起。

最小修复的正确语义（只动结构，不动内容）：
  1. 把源数据按「连续同 id 行段」切成物理块（172 块，与 meta/episodes 行一一对应）；
  2. 剔除 15 个 eval 块 -> 157 块，按物理位置重编号为 0..156；
  3. 每块内容列（timestamp/state/action/视频窗口）原样保留；
  4. meta/episodes 每块一行：以匹配到的源 meta 行为基底（tasks 等透传），
     覆盖 episode_index/length/dataset_from_to/data 索引；视频窗口沿用源行
     （块与源行的匹配：优先 from_index+length 全等，退 from_index，再退 length）；
  5. 视频文件字节级复制，info.json 以实际行数为真值重建。

结果：A = 157 集 / 81229 帧（88782 行 - eval 7553 行，一行不丢、一行不增），
重复 id 的两份拷贝各自成为独立 episode（这是方案口径「保留全部坏数据 + 只修结构」
的唯一无歧义实现，帧数口径与过审方案一致，集数口径 135 -> 157 见 README）。
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import sys as _sys
_SCRIPT_DIR = str(Path(__file__).resolve().parent)
if _SCRIPT_DIR not in _sys.path:
    _sys.path.insert(0, _SCRIPT_DIR)

import pyarrow as pa
import pyarrow.parquet as pq

from common import (
    DS_MINREPAIR,
    EVAL_EPISODES_JSON,
    OUT,
    SRC_BOMB,
    dump_json,
    ensure_repo_on_path,
    load_json,
)

ensure_repo_on_path()

from export_clean_dataset import _verify_export  # noqa: E402
from pipeline.loader import DEFAULT_DATA_PATH, DEFAULT_VIDEO_PATH  # noqa: E402

FFPROBE = shutil.which("ffprobe")


def scan_source_runs():
    """按全局行序扫源数据，切成连续同 id 的物理块。

    返回 runs: [{old_id, segments: [(file_idx, local_start, local_end)], start_global,
                 length}]。跨文件续块（本数据不存在，防御性支持）自动合并。
    """
    files = sorted((SRC_BOMB / "data" / "chunk-000").glob("*.parquet"))
    runs: list[dict] = []
    global_row = 0
    for file_idx, f in enumerate(files):
        eps = pq.read_table(f, columns=["episode_index"]).column("episode_index").to_pylist()
        for local_i, e in enumerate(eps):
            e = int(e)
            if runs and runs[-1]["old_id"] == e:
                seg = runs[-1]["segments"][-1]
                if seg[0] == file_idx and seg[2] == local_i:   # 续块
                    runs[-1]["segments"][-1] = (file_idx, seg[1], local_i + 1)
                else:                                          # 同 id 分离块（间隔其它块后重现）
                    runs.append({"old_id": e, "segments": [(file_idx, local_i, local_i + 1)],
                                 "start_global": global_row, "length": 0})
                runs[-1]["length"] += 1
            else:
                runs.append({"old_id": e, "segments": [(file_idx, local_i, local_i + 1)],
                             "start_global": global_row, "length": 0})
                runs[-1]["length"] += 1  # 占位，下面统一累计
            global_row += 1
    # 修 length（按 segments 累计，避免上面续块分支的累计错误）
    for r in runs:
        r["length"] = sum(e - s for _, s, e in r["segments"])
    return runs, files


def match_meta_rows(runs: list[dict], meta_rows: list[dict]) -> dict[int, dict]:
    """每个 run 匹配一行源 meta。优先 (from_index, length) 全等 > from_index > length > 唯一候选。"""
    by_start = defaultdict(list)
    by_id = defaultdict(list)
    for i, row in enumerate(meta_rows):
        by_start[int(row["dataset_from_index"])].append(i)
        by_id[int(row["episode_index"])].append(i)

    used: set[int] = set()
    mapping: dict[int, dict] = {}
    stats = defaultdict(int)
    unmatched = []
    for run_idx, r in enumerate(runs):
        cand = None
        how = None
        exact = [i for i in by_start.get(r["start_global"], []) if i not in used
                 and int(meta_rows[i]["length"]) == r["length"]]
        by_from = [i for i in by_start.get(r["start_global"], []) if i not in used]
        by_len = [i for i in by_id.get(r["old_id"], []) if i not in used
                  and int(meta_rows[i]["length"]) == r["length"]]
        if exact:
            cand, how = exact[0], "from_index+length"
        elif by_from:
            cand, how = by_from[0], "from_index"
        elif len(by_len) == 1:
            cand, how = by_len[0], "length"
        elif len(by_id.get(r["old_id"], [])) == 1:
            cand, how = by_id[r["old_id"]][0], "only_candidate"
        if cand is None:
            unmatched.append((run_idx, r["old_id"], r["length"], r["start_global"]))
            continue
        used.add(cand)
        mapping[run_idx] = meta_rows[cand]
        stats[how] += 1
    if unmatched:
        sys.exit(f"[错误] {len(unmatched)} 个块匹配不到源 meta 行: {unmatched[:5]}")
    print(f"块-元数据匹配: {dict(stats)}（共 {len(runs)} 块）")
    return mapping


def build(out_root: Path, eval_bomb_idx: set[int]) -> dict:
    meta_rows = []
    for f in sorted((SRC_BOMB / "meta" / "episodes").rglob("*.parquet")):
        meta_rows.extend(pq.read_table(f).to_pylist())
    print(f"源 meta/episodes: {len(meta_rows)} 行")

    runs, data_files = scan_source_runs()
    print(f"源数据物理块: {len(runs)} 块 / {sum(r['length'] for r in runs)} 行 / {len(data_files)} 个文件")
    mapping = match_meta_rows(runs, meta_rows)

    kept_idx = [i for i, r in enumerate(runs) if r["old_id"] not in eval_bomb_idx]
    new_of_run = {run_i: new for new, run_i in enumerate(kept_idx)}
    print(f"保留 {len(kept_idx)} 块（剔除 {len(runs) - len(kept_idx)} 个 eval 块）-> 重编号 0..{len(kept_idx) - 1}")

    # --- data parquet：内容列不动，重写 episode_index/frame_index/index
    out_root.mkdir(parents=True, exist_ok=True)
    global_index = 0
    frame_counter = defaultdict(int)          # run_idx -> 已写帧数
    run_file: dict[int, int] = {}
    per_file_frames = defaultdict(int)
    for file_idx, f in enumerate(data_files):
        table = pq.read_table(f)
        eps = table.column("episode_index").to_pylist()
        keep_local = []
        run_of_local = {}
        for run_i, r in enumerate(runs):
            if run_i not in new_of_run:
                continue  # eval 块：整块剔除，不进产物
            for (seg_f, s, e) in r["segments"]:
                if seg_f == file_idx:
                    for local in range(s, e):
                        keep_local.append(local)
                        run_of_local[local] = run_i
        if not keep_local:
            continue
        sub = table.take(pa.array(keep_local, type=pa.int64()))
        n = sub.num_rows
        run_of_pos = [run_of_local[local] for local in keep_local]  # 位置 -> run（keep_local 升序）
        new_eps = [new_of_run[r] for r in run_of_pos]
        frames = []
        for r in run_of_pos:  # 单趟自增；跨文件的 run 计数器延续
            frames.append(frame_counter[r])
            frame_counter[r] += 1
        idx = list(range(global_index, global_index + n))
        global_index += n
        sub = set_col(sub, "episode_index", new_eps, pa.int64())
        sub = set_col(sub, "frame_index", frames, pa.int64())
        sub = set_col(sub, "index", idx, pa.int64())
        rel = DEFAULT_DATA_PATH.format(chunk_index=0, file_index=file_idx)
        out_path = out_root / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(sub, out_path)
        for run_i in set(run_of_pos):
            run_file[run_i] = file_idx
        per_file_frames[file_idx] += n

    total_frames = global_index
    total_eps = len(kept_idx)

    # --- meta/episodes：以匹配源行为基底，覆盖计算列
    src_meta_table = pa.concat_tables(
        [pq.read_table(f) for f in sorted((SRC_BOMB / "meta" / "episodes").rglob("*.parquet"))],
        promote_options="permissive",
    )
    schema = src_meta_table.schema
    out_rows: list[dict] = []
    cumulative = 0
    for new_id, run_i in enumerate(kept_idx):
        r = runs[run_i]
        base = dict(mapping[run_i])
        base["episode_index"] = new_id
        base["length"] = r["length"]
        base["dataset_from_index"] = cumulative
        base["dataset_to_index"] = cumulative + r["length"]
        base["data/chunk_index"] = 0
        base["data/file_index"] = run_file[run_i]
        cumulative += r["length"]
        out_rows.append(base)

    cols = {}
    for field in schema:
        vals = [row.get(field.name) for row in out_rows]
        cols[field.name] = pa.array(vals, type=field.type)
    meta_out = pa.table(cols, schema=schema)
    meta_path = out_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(meta_out, meta_path)

    # --- info.json：以实际为真值重建（codebase/模板沿用源）
    info = load_json(SRC_BOMB / "meta" / "info.json")
    info["total_episodes"] = total_eps
    info["total_frames"] = int(total_frames)
    info["splits"] = {"train": f"0:{total_eps}"}
    info["data_path"] = DEFAULT_DATA_PATH
    info["video_path"] = DEFAULT_VIDEO_PATH
    (out_root / "meta").mkdir(parents=True, exist_ok=True)
    (out_root / "meta" / "info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")

    # --- tasks
    src_tasks = SRC_BOMB / "meta" / "tasks.parquet"
    if src_tasks.is_file():
        shutil.copy2(src_tasks, out_root / "meta" / "tasks.parquet")

    # --- stats 占位：结构校验要求 stats.json 存在；正式统计由 apply_stats_fixed.py
    #     以 B 组训练集计算后字节级覆盖，这里的源值只保证结构完整
    src_stats = SRC_BOMB / "meta" / "stats.json"
    if src_stats.is_file():
        shutil.copy2(src_stats, out_root / "meta" / "stats.json")

    # --- 视频字节级复制（引用到的文件）
    referenced = set()
    for row in out_rows:
        for cam in [k for k, v in (info.get("features") or {}).items()
                    if (v or {}).get("dtype") == "video"]:
            rel = DEFAULT_VIDEO_PATH.format(video_key=cam,
                                            chunk_index=int(row[f"videos/{cam}/chunk_index"]),
                                            file_index=int(row[f"videos/{cam}/file_index"]))
            referenced.add(rel)
    copied = 0
    for rel in sorted(referenced):
        src, dst = SRC_BOMB / rel, out_root / rel
        if not src.is_file():
            sys.exit(f"[错误] 源视频缺失: {rel}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1

    new_to_old = {str(new_of_run[i]): runs[i]["old_id"] for i in kept_idx}
    return {
        "total_episodes": total_eps,
        "total_frames": int(total_frames),
        "source_blocks": len(runs),
        "dropped_eval_blocks": len(runs) - len(kept_idx),
        "copied_video_files": copied,
        "referenced_video_files": len(referenced),
        "new_to_old_episode": new_to_old,
        "per_file_frames": {str(k): v for k, v in sorted(per_file_frames.items())},
    }


def set_col(table: pa.Table, name: str, values, dtype) -> pa.Table:
    col = pa.array(values, type=dtype)
    if name in table.column_names:
        return table.set_column(table.schema.get_field_index(name), name, col)
    return table.append_column(pa.field(name, dtype), col)


def probe_report(out_root: Path) -> list[dict]:
    import pandas as pd

    meta = pd.read_parquet(sorted((out_root / "meta" / "episodes").rglob("*.parquet"))[0])
    info = load_json(out_root / "meta" / "info.json")
    rows = []
    for cam in [k for k, v in (info.get("features") or {}).items()
                if (v or {}).get("dtype") == "video"]:
        per_file = defaultdict(int)
        for _, r in meta.iterrows():
            rel = info["video_path"].format(video_key=cam,
                                            chunk_index=int(r[f"videos/{cam}/chunk_index"]),
                                            file_index=int(r[f"videos/{cam}/file_index"]))
            per_file[rel] += int(r["length"])
        for rel, expected in sorted(per_file.items()):
            proc = subprocess.run(
                [FFPROBE, "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=nb_frames", "-of", "json", str(out_root / rel)],
                capture_output=True, text=True, timeout=120, check=False)
            actual = None
            if proc.returncode == 0:
                try:
                    actual = int(json.loads(proc.stdout)["streams"][0]["nb_frames"])
                except (KeyError, IndexError, TypeError, ValueError):
                    pass
            rows.append({"camera": cam, "file": rel, "expected": expected,
                         "actual": actual,
                         "delta": None if actual is None else actual - expected})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="A 组块级最小修复构建")
    parser.add_argument("--fresh", action="store_true", help="先删除已有 A 组输出目录")
    args = parser.parse_args()

    split = load_json(EVAL_EPISODES_JSON)
    eval_bomb = set(split["bomb_episode_indices"])

    if args.fresh and DS_MINREPAIR.exists():
        shutil.rmtree(DS_MINREPAIR)
        print(f"已删除旧输出: {DS_MINREPAIR}")

    result = build(DS_MINREPAIR, eval_bomb)
    print(f"A 组: {result['total_episodes']} 集 / {result['total_frames']} 帧 / "
          f"视频复制 {result['copied_video_files']} 文件")

    ok, verification = _verify_export(DS_MINREPAIR)
    print(f"结构校验: {'通过' if ok else '未通过'}")
    for issue in verification.get("issues", [])[:10]:
        print(f"  [校验] {issue}")
    if not ok:
        sys.exit("[错误] A 组结构校验未通过")

    probe = probe_report(DS_MINREPAIR)
    delta = [r for r in probe if r["delta"] not in (0,)]
    print(f"内容层（预期差异，如实留痕）: {len(delta)}/{len(probe)} 个文件帧数与元数据不符")
    for r in delta[:12]:
        print(f"  {r['file']}: 元数据 {r['expected']} vs 实际 {r['actual']}（差 {r['delta']}）")

    result["structure"] = verification
    result["video_probe"] = probe
    dump_json(OUT / "rebuild_minrepair_manifest.json", result)
    print(f"manifest: {OUT / 'rebuild_minrepair_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
