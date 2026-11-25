# Copyright (c) 2026 wu
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""把 v2（漂移修复后 cleaner 重建）的 B′/eval′ 打包成单个 zip，供 Kaggle Dataset 上传。

与 pack_upload.py 同构；v2 只含 B′ 与 eval′（A 沿用 v1 数据集，A@10k 从零训不需要 ckpt）。
"""
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(r"G:\Lerobot\artifacts\exports\kaggle_ab")
OUT = Path(r"G:\Lerobot\artifacts\exports\kaggle_upload_v2")
INCLUDE_DIRS = ["so101_clean_v2", "so101_eval_v2"]
INCLUDE_FILES = ["eval_episodes.json", "eval_windows.json"]
ZIP_NAME = "so101_ab_v2.zip"

METADATA = {
    "title": "SO101 Kaggle A/B v2",
    "id": "happyday1223/so101-kaggle-ab-v2",
    "licenses": [{"name": "CC0-1.0"}],
    "subtitle": "漂移修复版 cleaner 重建的 B′/eval′（零帧差、pts 归零）",
    "description": (
        "so101-kaggle-ab-v1 的修复后重跑产物。v1 的 B 组视频经分段重编码存在帧数漂移"
        "（±1..13 帧）与首帧 pts 偏移（33ms，导致 B@4000 FrameTimestampError）；cleaner "
        "修复（-frames:v 硬截断 + setpts=PTS-STARTPTS）后全量重清洗重建。\n\n"
        "## 内容\n\n| 目录 | 用途 | 集数 | 帧数 |\n|---|---|---|---|\n"
        "| `so101_clean_v2` | B′ 组：同 v1-B 的 106 集，零漂移 | 106 | 55,632 |\n"
        "| `so101_eval_v2` | eval′：同 v1-eval 的 15 集，窗口索引不变 | 15 | 7,553 |\n\n"
        "- `eval_episodes.json` / `eval_windows.json`：与 v1 逐字节相同的冻结划分\n"
        "- 验收：帧数差 0/34 文件、pts 34/34 首帧归零、validator 0 非 keep\n"
        "- stats 与 v1 三份数据字节一致（md5 4da81c51），归一化跨跑不变\n\n"
        "## 使用方式\n\n```bash\n"
        "mkdir -p /kaggle/working/data\n"
        "unzip -q /kaggle/input/so101-kaggle-ab-v2/so101_ab_v2.zip -d /kaggle/working/data\n"
        "```\n"
    ),
    "keywords": ["robotics", "lerobot", "smolvla", "data quality"],
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dst = OUT / ZIP_NAME
    if dst.exists():
        dst.unlink()

    entries = []
    for name in INCLUDE_DIRS:
        entries.extend(sorted(p for p in (ROOT / name).rglob("*") if p.is_file()))
    entries.extend(ROOT / name for name in INCLUDE_FILES)

    with zipfile.ZipFile(dst, "w", zipfile.ZIP_STORED, allowZip64=True) as zf:
        for path in entries:
            zf.write(path, path.relative_to(ROOT).as_posix())

    (OUT / "dataset-metadata.json").write_text(
        json.dumps(METADATA, ensure_ascii=False, indent=2), encoding="utf-8")

    digest = hashlib.md5()
    with dst.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)

    print(f"zip={dst}")
    print(f"entries={len(entries)}")
    print(f"size={dst.stat().st_size / 1024 ** 3:.3f} GB")
    print(f"md5={digest.hexdigest()}")


if __name__ == "__main__":
    main()
