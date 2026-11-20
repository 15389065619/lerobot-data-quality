"""把三份数据集与评测索引打包成单个 zip，供 Kaggle Dataset 上传。

kaggle CLI 的 `datasets create` 对子目录默认 `dir_mode=skip`，会整目录丢弃，
因此必须先在本地打包成单文件，再在笔记本里解压使用。
"""
import hashlib
import zipfile
from pathlib import Path

ROOT = Path(r"G:\Lerobot\exports\kaggle_ab")
OUT = Path(r"G:\Lerobot\exports\kaggle_upload")
INCLUDE_DIRS = ["so101_minrepair", "so101_clean", "so101_eval"]
INCLUDE_FILES = ["eval_episodes.json", "eval_windows.json"]
ZIP_NAME = "so101_ab_v1.zip"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dst = OUT / ZIP_NAME
    if dst.exists():
        dst.unlink()

    entries = []
    for name in INCLUDE_DIRS:
        entries.extend(sorted(p for p in (ROOT / name).rglob("*") if p.is_file()))
    entries.extend(ROOT / name for name in INCLUDE_FILES)

    # 数据本身已是压缩格式，用 ZIP_STORED 避免无意义的二次压缩耗时
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_STORED, allowZip64=True) as zf:
        for path in entries:
            zf.write(path, path.relative_to(ROOT).as_posix())

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
