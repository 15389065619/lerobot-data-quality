#!/usr/bin/env python3
"""为源码文件批量插入版权头（幂等、可预演）。

用法：
    python tools/add_license_headers.py --holder "wuliangzhong" --license apache --dry-run
    python tools/add_license_headers.py --holder "wuliangzhong" --license apache
    python tools/add_license_headers.py --holder "wuliangzhong" --license mit --include-init
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

TARGET_DIRS = ("pipeline", "adapters", "verticals", "evaluation", "ui")
TARGET_ROOT_FILES = ("main.py", "export_clean_dataset.py", "generate_report.py")

# 目录名命中即整棵跳过
SKIP_DIR_NAMES = {
    "tests", "test", "artifacts", "archive", "dist", "build", "outputs",
    "reports", "data", "demo_dataset", "annotation", "examples", "docs",
    "__pycache__", ".git", ".venv", "venv", "env", ".env",
    "site-packages", "node_modules", "egg-info",
}

APACHE = """\
# Copyright (c) {year} {holder}
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""

MIT = """\
# Copyright (c) {year} {holder}
# Licensed under the MIT License.
# See the LICENSE file at the project root for the full text.
"""

DETECT_MARKERS = ("Copyright (c)", "Licensed under the Apache License", "Licensed under the MIT License")


def iter_py_files(include_init: bool) -> list[Path]:
    found: list[Path] = []
    for name in TARGET_DIRS:
        root = REPO_ROOT / name
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.py")):
            rel_parts = path.relative_to(REPO_ROOT).parts
            if any(part.lower().endswith("egg-info") or part in SKIP_DIR_NAMES for part in rel_parts[:-1]):
                continue
            if path.name == "__init__.py" and not include_init:
                continue
            found.append(path)
    for name in TARGET_ROOT_FILES:
        path = REPO_ROOT / name
        if path.is_file():
            found.append(path)
    return found


def split_prefix(lines: list[str]) -> int:
    """返回应该在多少行之后插入版权头（shebang / coding 行之后）。"""
    index = 0
    if index < len(lines) and lines[index].startswith("#!"):
        index += 1
    if index < len(lines):
        lowered = lines[index].lower()
        if "coding" in lowered and lowered.lstrip().startswith("#"):
            index += 1
    return index


def has_header(lines: list[str]) -> bool:
    head = "".join(lines[:5])
    return any(marker in head for marker in DETECT_MARKERS)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--holder", required=True, help="版权持有者")
    parser.add_argument("--year", default="2026")
    parser.add_argument("--license", choices=["apache", "mit"], default="apache")
    parser.add_argument("--include-init", action="store_true", help="同时处理 __init__.py")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    template = APACHE if args.license == "apache" else MIT
    header = template.format(year=args.year, holder=args.holder).rstrip("\n") + "\n"

    changed: list[Path] = []
    skipped: list[Path] = []
    for path in iter_py_files(args.include_init):
        try:
            raw = path.read_bytes()
        except OSError as exc:
            print(f"! 读取失败 {path}: {exc}")
            continue
        has_bom = raw.startswith(b"\xef\xbb\xbf")
        text = raw.decode("utf-8-sig")
        newline = "\r\n" if "\r\n" in text else "\n"
        lines = text.splitlines(keepends=True)

        if has_header(lines):
            skipped.append(path)
            continue

        insert_at = split_prefix(lines)
        block = [line + newline for line in header.splitlines()] + [newline]
        lines[insert_at:insert_at] = block
        out = "".join(lines)
        if has_bom:
            out = "\ufeff" + out
        changed.append(path)
        if not args.dry_run:
            path.write_bytes(out.encode("utf-8"))

    print(f"目标文件 {len(changed) + len(skipped)} 个：已处理 {len(changed)}，已有版权头跳过 {len(skipped)}")
    if args.dry_run:
        print("（dry-run，未写入）")
    for path in changed[:20]:
        print("  +", path.relative_to(REPO_ROOT))
    if len(changed) > 20:
        print(f"  ... 其余 {len(changed) - 20} 个")
    return 0


if __name__ == "__main__":
    sys.exit(main())
