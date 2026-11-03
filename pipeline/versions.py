"""数据集版本管理（轻量 DVC）：内容寻址快照 + 差异对比 + 回滚。

设计要点：
- **内容寻址**：每个文件的 sha1 决定存储位置（``.lrdq/objects/<前两位>/<sha1>``），
  同一文件内容只存一份，多版本之间自动去重。
- **增量快照**：与父版本相比 mtime_ns + size 未变的文件直接沿用父版本哈希，不重读
  （百万级数据集上二次快照几乎不耗 IO）。
- **回滚是显式动作**：默认只输出「将要改哪些文件」的 plan，必须加 ``--apply`` 才写盘，
  且不会删除目标版本里不存在的文件（除非 ``--delete-extra``）。

用法：
    python -m pipeline.versions commit  --dataset_path data/gym_xarm --note "清洗前基线"
    python -m pipeline.versions log     --dataset_path data/gym_xarm
    python -m pipeline.versions diff    --dataset_path data/gym_xarm
    python -m pipeline.versions restore --dataset_path data/gym_xarm --version <id>          # 只看 plan
    python -m pipeline.versions restore --dataset_path data/gym_xarm --version <id> --apply  # 真回滚
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LOG = logging.getLogger(__name__)

STORE_DIR = ".lrdq"
IGNORE_DIRS = {STORE_DIR, "__pycache__", ".git", ".venv", "venv", ".pytest_cache"}
CHUNK = 1 << 20


def _sha1_file(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def _iter_files(root: Path, store: Path) -> list[Path]:
    files = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if any(part in IGNORE_DIRS for part in rel.parts):
            continue
        if path.resolve().is_relative_to(store.resolve()):
            continue
        files.append(path)
    return sorted(files, key=lambda p: p.relative_to(root).as_posix())


class DatasetVersionStore:
    """把某个数据集的每个变更记成不可变快照。"""

    def __init__(self, dataset_path: str | Path, store_dir: str | Path | None = None) -> None:
        self.root = Path(dataset_path).expanduser().resolve()
        if not self.root.is_dir():
            raise NotADirectoryError(f"数据集目录不存在：{self.root}")
        self.store = Path(store_dir).expanduser().resolve() if store_dir else self.root / STORE_DIR
        self.versions_dir = self.store / "versions"
        self.objects_dir = self.store / "objects"
        self.versions_dir.mkdir(parents=True, exist_ok=True)

    # ---------- 快照 ----------

    def _latest(self) -> dict[str, Any] | None:
        paths = sorted(self.versions_dir.glob("*.json"), reverse=True)
        if not paths:
            return None
        return json.loads(paths[0].read_text(encoding="utf-8"))

    def _object_path(self, sha: str) -> Path:
        return self.objects_dir / sha[:2] / sha

    def snapshot(self, note: str = "", params: dict[str, Any] | None = None) -> dict[str, Any]:
        parent = self._latest()
        parent_files = (parent or {}).get("files", {})
        files: dict[str, Any] = {}
        for path in _iter_files(self.root, self.store):
            rel = path.relative_to(self.root).as_posix()
            stat = path.stat()
            prev = parent_files.get(rel)
            if prev and prev["size"] == stat.st_size and prev["mtime_ns"] == stat.st_mtime_ns:
                sha = prev["sha1"]
            else:
                sha = _sha1_file(path)
                obj = self._object_path(sha)
                if not obj.exists():
                    obj.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, obj)
            files[rel] = {"sha1": sha, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}

        tree = hashlib.sha1()
        for rel in sorted(files):
            tree.update(f"{rel}:{files[rel]['sha1']}\n".encode())
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        version_id = f"{stamp}-{tree.hexdigest()[:8]}"
        record = {
            "version_id": version_id,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "note": note,
            "parent": parent["version_id"] if parent else None,
            "tree_digest": tree.hexdigest(),
            "params": params or {},
            "file_count": len(files),
            "total_bytes": sum(f["size"] for f in files.values()),
            "files": files,
        }
        (self.versions_dir / f"{version_id}.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        LOG.info("已创建版本 %s（%d 个文件，%.1f MB）", version_id, len(files),
                 record["total_bytes"] / 1e6)
        return record

    # ---------- 查询 ----------

    def _read(self, version_id: str | None) -> dict[str, Any]:
        if not version_id:
            latest = self._latest()
            if not latest:
                raise ValueError("还没有任何版本，请先执行 commit")
            return latest
        path = self.versions_dir / f"{version_id}.json"
        if not path.is_file():
            raise ValueError(f"版本不存在：{version_id}")
        return json.loads(path.read_text(encoding="utf-8"))

    def log(self, limit: int | None = None) -> list[dict[str, Any]]:
        items = []
        for path in sorted(self.versions_dir.glob("*.json"), reverse=True):
            data = json.loads(path.read_text(encoding="utf-8"))
            items.append({k: data.get(k) for k in
                          ("version_id", "created_at", "note", "parent", "tree_digest",
                           "file_count", "total_bytes")})
        return items[:limit] if limit else items

    def diff(self, base: str | None = None, target: str | None = None) -> dict[str, Any]:
        """target 默认最新版；base 默认 target 的父版本。"""
        tgt = self._read(target)
        base_id = base or tgt.get("parent")
        ba = self._read(base_id) if base_id else {"version_id": None, "files": {}}
        added = sorted(set(tgt["files"]) - set(ba["files"]))
        removed = sorted(set(ba["files"]) - set(tgt["files"]))
        modified = sorted(
            rel for rel in set(ba["files"]) & set(tgt["files"])
            if ba["files"][rel]["sha1"] != tgt["files"][rel]["sha1"]
        )
        return {
            "base": ba["version_id"],
            "target": tgt["version_id"],
            "added": added,
            "removed": removed,
            "modified": modified,
            "unchanged": len(set(ba["files"]) & set(tgt["files"])) - len(modified),
        }

    # ---------- 回滚 ----------

    def restore(self, version_id: str, *, apply: bool = False,
                delete_extra: bool = False) -> dict[str, Any]:
        record = self._read(version_id)
        current = {rel: _sha1_file(self.root / rel)
                   for rel in (p.relative_to(self.root).as_posix()
                               for p in _iter_files(self.root, self.store))
                   if (self.root / rel).is_file()}
        to_write = []
        for rel, meta in sorted(record["files"].items()):
            obj = self._object_path(meta["sha1"])
            if current.get(rel) == meta["sha1"]:
                continue
            to_write.append({"path": rel, "sha1": meta["sha1"],
                             "available": obj.is_file()})
        extra = sorted(set(current) - set(record["files"]))

        missing = [item["path"] for item in to_write if not item["available"]]
        result: dict[str, Any] = {
            "version": record["version_id"],
            "applied": False,
            "to_write": to_write,
            "extra_files": extra,
            "missing_objects": missing,
        }
        if not apply:
            return result
        if missing:
            raise RuntimeError(f"对象库缺少 {len(missing)} 个文件的内容，无法完整回滚：{missing[:5]}")
        for item in to_write:
            dest = self.root / item["path"]
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self._object_path(item["sha1"]), dest)
        if delete_extra:
            for rel in extra:
                (self.root / rel).unlink()
        result["applied"] = True
        LOG.info("已回滚到 %s：写入 %d 个文件%s", record["version_id"], len(to_write),
                 f"，删除 {len(extra)} 个多余文件" if delete_extra else "")
        return result


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description="数据集版本管理（快照 / 对比 / 回滚）")
    ap.add_argument("--dataset_path", required=True)
    ap.add_argument("--store_dir", default=None, help="版本库目录（默认 <dataset>/.lrdq）")
    sub = ap.add_subparsers(dest="command", required=True)

    p_commit = sub.add_parser("commit", help="创建快照")
    p_commit.add_argument("--note", default="")

    p_log = sub.add_parser("log", help="列出版本")
    p_log.add_argument("--limit", type=int, default=None)

    p_diff = sub.add_parser("diff", help="对比两个版本")
    p_diff.add_argument("--from", dest="base", default=None)
    p_diff.add_argument("--to", dest="target", default=None)

    p_restore = sub.add_parser("restore", help="回滚到指定版本")
    p_restore.add_argument("--version", required=True)
    p_restore.add_argument("--apply", action="store_true", help="真正写盘（默认只打印 plan）")
    p_restore.add_argument("--delete-extra", action="store_true", help="同时删除目标版本中不存在的文件")

    p_status = sub.add_parser("status", help="当前工作区相对最新版本的改动")
    p_status.description = "等同 diff（base=最新版本, target=工作区）"

    args = ap.parse_args(argv)
    store = DatasetVersionStore(args.dataset_path, args.store_dir)

    if args.command == "commit":
        record = store.snapshot(args.note)
        print(json.dumps({k: record[k] for k in ("version_id", "file_count", "total_bytes", "note")},
                         ensure_ascii=False, indent=2))
        return 0

    if args.command == "log":
        for item in store.log(args.limit):
            print(f"{item['version_id']}  {item['created_at']}  "
                  f"{item['file_count']} files  {item['total_bytes'] / 1e6:.1f} MB  {item['note']}")
        return 0

    if args.command == "diff":
        print(json.dumps(store.diff(args.base, args.target), ensure_ascii=False, indent=2))
        return 0

    if args.command == "status":
        latest = store._latest()
        if not latest:
            print("还没有任何版本，先执行 commit")
            return 1
        current = {p.relative_to(store.root).as_posix(): _sha1_file(p)
                   for p in _iter_files(store.root, store.store)}
        changed = [rel for rel, sha in current.items()
                   if latest["files"].get(rel, {}).get("sha1") != sha]
        removed = [rel for rel in latest["files"] if rel not in current]
        print(json.dumps({"compared_to": latest["version_id"], "changed": sorted(changed),
                          "removed": sorted(removed)}, ensure_ascii=False, indent=2))
        return 0

    result = store.restore(args.version, apply=args.apply, delete_extra=args.delete_extra)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not args.apply and result["to_write"]:
        print("\n这是 plan，未写盘。确认后加 --apply 执行回滚。", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
