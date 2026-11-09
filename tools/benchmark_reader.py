#!/usr/bin/env python3
"""单机 Parquet 读取方式对比：pandas / duckdb / 流式 iter_batches。

每种方式在独立子进程中运行，避免内存与分配器缓存互相污染；父进程负责
挑选最大的 N 个 data parquet、汇总耗时与峰值 RSS，并打印对比表。

    python tools/benchmark_reader.py --dataset data/stress --n-files 5
    python tools/benchmark_reader.py --files a.parquet b.parquet --repeat 2
    python tools/benchmark_reader.py --dataset data/stress --markdown-out docs/bench.md

内存口径：psutil 每 10ms 采样子进程 RSS，取峰值；同时给出扣除 import 基线后的增量。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

METHODS = ("pandas", "duckdb", "streaming")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def pick_files(dataset: Path, n: int) -> list[Path]:
    files = [p for p in dataset.rglob("*.parquet") if p.is_file()]
    if not files:
        raise SystemExit(f"{dataset} 下没有 parquet 文件")
    files.sort(key=lambda p: p.stat().st_size, reverse=True)
    return files[:n]


class _PeakRSS(threading.Thread):
    """后台采样当前进程 RSS 峰值。"""

    def __init__(self, interval: float = 0.01) -> None:
        super().__init__(daemon=True)
        self.interval = interval
        self.peak = 0
        self._stop_event = threading.Event()
        self._process = None

    def run(self) -> None:
        import psutil

        self._process = psutil.Process()
        self.peak = self._process.memory_info().rss
        while not self._stop_event.wait(self.interval):
            self.peak = max(self.peak, self._process.memory_info().rss)

    def stop(self) -> int:
        self._stop_event.set()
        self.join(timeout=1.0)
        return self.peak


def _read_pandas(files: list[Path]) -> int:
    import pandas as pd

    frame = pd.read_parquet([str(p) for p in files])  # 一次性拼接读入，模拟最直观的用法
    rows = len(frame)
    del frame
    return rows


def _read_duckdb(files: list[Path]) -> int:
    try:
        import duckdb
    except ImportError as exc:
        raise SystemExit("未安装 duckdb：E:\\python\\python.exe -m pip install duckdb") from exc

    relation = duckdb.read_parquet([str(p) for p in files])
    table = relation.fetch_arrow_table()
    rows = table.num_rows
    del table, relation
    return rows


def _read_streaming(files: list[Path], batch_size: int) -> int:
    import pyarrow.parquet as pq

    rows = 0
    for path in files:
        parquet_file = pq.ParquetFile(path)
        for batch in parquet_file.iter_batches(batch_size=batch_size):
            rows += batch.num_rows
    return rows


def _ensure_imports(method: str) -> None:
    """先完成 import，再开始测内存，避免把库本身的基线算进增量。"""
    if method == "pandas":
        import pandas  # noqa: F401
    elif method == "duckdb":
        import duckdb  # noqa: F401
    else:
        import pyarrow.parquet  # noqa: F401


def run_child(method: str, files: list[Path], batch_size: int, repeat: int) -> dict:
    _ensure_imports(method)
    baseline = _PeakRSS()
    baseline.start()
    time.sleep(0.05)
    baseline_rss = baseline.peak

    sampler = _PeakRSS()
    sampler.start()
    timings: list[float] = []
    rows = 0
    try:
        for _ in range(repeat):
            start = time.perf_counter()
            if method == "pandas":
                rows = _read_pandas(files)
            elif method == "duckdb":
                rows = _read_duckdb(files)
            elif method == "streaming":
                rows = _read_streaming(files, batch_size)
            else:
                raise SystemExit(f"未知方法 {method}")
            timings.append(time.perf_counter() - start)
    finally:
        peak_rss = sampler.stop()
        baseline.stop()

    return {
        "method": method,
        "rows": rows,
        "seconds": min(timings),
        "all_seconds": timings,
        "baseline_mb": baseline_rss / (1024 * 1024),
        "peak_mb": peak_rss / (1024 * 1024),
    }


def run_parent(args: argparse.Namespace) -> int:
    if args.files:
        files = [Path(p) for p in args.files]
        missing = [str(p) for p in files if not p.exists()]
        if missing:
            raise SystemExit(f"文件不存在：{', '.join(missing)}")
    else:
        dataset = Path(args.dataset)
        if not dataset.is_absolute():
            dataset = repo_root() / dataset
        files = pick_files(dataset, args.n_files)
    files = sorted(files, key=lambda p: p.stat().st_size, reverse=True)
    total_bytes = sum(p.stat().st_size for p in files)

    print(f"[bench] {len(files)} 个文件，压缩体积 {total_bytes / (1024 ** 2):.1f} MB")
    for path in files:
        print(f"        {path.name:<22} {path.stat().st_size / (1024 ** 2):>8.1f} MB")
    print(f"[bench] 流式 batch_size={args.batch_size}，repeat={args.repeat}\n")

    results: list[dict] = []
    for method in METHODS:
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--method",
            method,
            "--files-json",
            json.dumps([str(p) for p in files]),
            "--batch-size",
            str(args.batch_size),
            "--repeat",
            str(args.repeat),
            "--child-json",
        ]
        completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8")
        payload = None
        for line in reversed(completed.stdout.splitlines()):
            if line.strip().startswith("{"):
                try:
                    payload = json.loads(line)
                    break
                except json.JSONDecodeError:
                    continue
        if payload is None or completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip().splitlines()
            results.append({"method": method, "error": detail[-1] if detail else "无输出"})
        else:
            payload["bytes"] = total_bytes
            results.append(payload)

    print_table(results)
    if args.markdown_out:
        write_markdown(results, Path(args.markdown_out), len(files), total_bytes)
    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n[out] 原始结果已写出 {args.json_out}")
    return 0


def _row(result: dict, total_bytes: int) -> dict:
    if "error" in result:
        return {
            "method": result["method"],
            "seconds": "-",
            "peak": "-",
            "delta": "-",
            "rows": "-",
            "throughput": "-",
        }
    seconds = result["seconds"]
    return {
        "method": result["method"],
        "seconds": f"{seconds:.2f}",
        "peak": f"{result['peak_mb']:.0f}",
        "delta": f"{max(0.0, result['peak_mb'] - result['baseline_mb']):.0f}",
        "rows": f"{result['rows']:,}",
        "throughput": f"{total_bytes / (1024 ** 2) / seconds:.0f}" if seconds > 0 else "-",
    }


def print_table(results: list[dict]) -> None:
    total_bytes = next((r.get("bytes", 0) for r in results if "bytes" in r), 0)
    header = (
        f"{'method':<12} {'rows':>10} {'seconds':>8} {'peak_RSS(MB)':>13} "
        f"{'delta(MB)':>10} {'MB/s':>7}"
    )
    print(header)
    print("-" * len(header))
    for result in results:
        row = _row(result, total_bytes)
        print(
            f"{row['method']:<12} {row['rows']:>10} {row['seconds']:>8} "
            f"{row['peak']:>13} {row['delta']:>10} {row['throughput']:>7}"
        )
        if "error" in result:
            print(f"  └─ {result['error']}")
    print(
        "\npeak_RSS 含解释器与 pyarrow 基线；delta 为峰值减去 import 后基线，"
        "更接近数据集本身占用的内存。"
    )


def write_markdown(results: list[dict], out_path: Path, n_files: int, total_bytes: int) -> None:
    lines = [
        f"数据：{n_files} 个大 parquet 文件，压缩体积 {total_bytes / (1024 ** 2):.1f} MB。",
        "",
        "| 方式 | 行数 | 耗时 (s) | 峰值 RSS (MB) | 增量 RSS (MB) | 吞吐 (MB/s) |",
        "|---|---|---|---|---|---|",
    ]
    for result in results:
        row = _row(result, total_bytes)
        if "error" in result:
            lines.append(f"| {row['method']} | - | - | - | - | 失败：{result['error']} |")
        else:
            lines.append(
                f"| {row['method']} | {row['rows']} | {row['seconds']} | "
                f"{row['peak']} | {row['delta']} | {row['throughput']} |"
            )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n[out] Markdown 表格已写出 {out_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="pandas / duckdb / 流式读取的性能与内存对比")
    parser.add_argument("--dataset", default="data/stress", help="数据集根目录（默认 data/stress）")
    parser.add_argument("--n-files", type=int, default=5, help="取最大的 N 个 parquet（默认 5）")
    parser.add_argument("--files", nargs="*", default=None, help="显式指定文件，覆盖 --dataset")
    parser.add_argument("--batch-size", type=int, default=256, help="流式批大小（默认 256）")
    parser.add_argument("--repeat", type=int, default=1, help="每种方式重复次数，取最快一次")
    parser.add_argument("--markdown-out", default=None, help="导出 Markdown 对比表")
    parser.add_argument("--json-out", default=None, help="导出原始 JSON 结果")
    # 子进程内部参数
    parser.add_argument("--method", choices=METHODS, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--files-json", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--child-json", action="store_true", help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.method:
        files = [Path(p) for p in json.loads(args.files_json or "[]")]
        result = run_child(args.method, files, args.batch_size, max(1, args.repeat))
        print(json.dumps(result, ensure_ascii=False))
        return 0
    return run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
