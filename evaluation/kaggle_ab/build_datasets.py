"""第 1 步 b：构建 B/eval 两份数据集（复用 pipeline.cleaner.clean_dataset 的重排机器）。

A 组 so101_minrepair 由 rebuild_minrepair_blocks.py 单独构建——bomb 的坏集 81-102
在数据里物理存在两份分歧拷贝，同一 episode_index 对应两块分离数据，clean_dataset
按 id 处理会把两份拷贝混在一起，必须按物理块拆分重编号。

B 组 so101_clean（完整清洗臂）：源 = outputs_ext_bomb/cleaned_dataset（121 集），
    剔除 15 个 eval 集，剩 106 集。
eval so101_eval：源 = cleaned_dataset，只留 15 个 eval 集，重排为 0..14。

产出后各自跑 _verify_export 结构校验，写 build_manifest.json。
视频编码不在这里处理（transcode_h264.py 统一转码）。
"""

from __future__ import annotations

import argparse
import sys
import time

import sys as _sys
from pathlib import Path as _P
_SCRIPT_DIR = str(_P(__file__).resolve().parent)
if _SCRIPT_DIR not in _sys.path:
    _sys.path.insert(0, _SCRIPT_DIR)

from common import (
    EVAL_EPISODES_JSON,
    OUT,
    SRC_BOMB,
    SRC_CLEANED,
    dump_json,
    ensure_repo_on_path,
    load_json,
)

ensure_repo_on_path()

from export_clean_dataset import _verify_export  # noqa: E402
from pipeline.cleaner import clean_dataset  # noqa: E402
from pipeline.loader import load_dataset  # noqa: E402


def build(name: str, src: str, drop: list[int], out_root, keep_expect: int) -> dict:
    print(f"\n=== 构建 {name}: 源={src} 剔除 {len(drop)} 集 -> {out_root}")
    t0 = time.time()
    view = load_dataset(src)
    print(f"  源加载: {len(view.episode_indices)} 集 / {view.num_frames} 帧 / {view.fps}Hz")
    result = clean_dataset(
        view,
        sorted(drop),
        out_root.parent,
        prefer_cli=False,      # 官方 CLI 只能删不能重建 meta；内置实现正是要的重排机器
        overwrite=True,
        out_root=out_root,
        link_mode="copy",
    )
    elapsed = time.time() - t0

    ok, verification = _verify_export(out_root)
    manifest = {
        "name": name,
        "source": src,
        "dropped_source_episodes": sorted(drop),
        "kept_episodes": result.kept_episodes,
        "removed_episodes": result.removed_episodes,
        "warnings": result.warnings,
        "verification": verification,
        "elapsed_sec": round(elapsed, 1),
        "expected_kept": keep_expect,
    }
    status = "OK" if ok and result.kept_episodes == keep_expect else "FAIL"
    print(f"  结果: {result.kept_episodes} 集（预期 {keep_expect}）/{verification.get('frames')} 帧 "
          f"/ 校验={'通过' if ok else '未通过'} / {elapsed:.0f}s -> {status}")
    for w in result.warnings[:10]:
        print(f"  [警告] {w}")
    if verification.get("issues"):
        for msg in verification["issues"][:10]:
            print(f"  [校验] {msg}")
    if status == "FAIL":
        sys.exit(f"[错误] {name} 构建校验未通过")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="构建 B/eval 两份数据集（A 组用 rebuild_minrepair_blocks.py）")
    parser.add_argument("--only", choices=["clean_eval"], default="clean_eval",
                        help="占位参数，保留兼容；本脚本只构建 B/eval")
    parser.add_argument("--source", default=None,
                        help="覆盖 B/eval 源数据集（默认 outputs_ext_bomb/cleaned_dataset；"
                             "v2 重清洗产物传 outputs_ext_bomb_v2/cleaned_dataset）")
    parser.add_argument("--suffix", default="",
                        help="输出数据集名后缀（如 _v2 → so101_clean_v2）；留痕 manifest 同名加后缀，"
                             "不覆盖 v1 产物与 manifest")
    args = parser.parse_args()

    split = load_json(EVAL_EPISODES_JSON)
    eval_cleaned_idx = split["cleaned_episode_indices"]   # cleaned 新编号（B/eval 源）

    cleaned_view_eps = 121
    OUT.mkdir(parents=True, exist_ok=True)

    # eval 集从 cleaned_dataset 剔除后的全集（用于 eval 臂的 drop 列表）
    all_cleaned = list(range(cleaned_view_eps))
    eval_arm_drop = sorted(set(all_cleaned) - set(eval_cleaned_idx))

    src = args.source or str(SRC_CLEANED)
    ds_clean = OUT / f"so101_clean{args.suffix}"
    ds_eval = OUT / f"so101_eval{args.suffix}"
    manifests = [
        build("so101_clean", src, eval_cleaned_idx, ds_clean, 106),
        build("so101_eval", src, eval_arm_drop, ds_eval, 15),
    ]
    manifest_path = OUT / f"build_manifest{args.suffix}.json"
    dump_json(manifest_path, {
        "note": "B/eval 构建留痕（A 由 rebuild_minrepair_blocks.py 构建并留痕）；"
                f"suffix={args.suffix!r}，源={src}",
        "datasets": manifests,
    })
    print(f"\nmanifest: {manifest_path}")
    print("下一步: transcode_h264.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
