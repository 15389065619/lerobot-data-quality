# Copyright (c) 2026 wuliangzhong
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""多进程并发执行器：任务分发、tqdm 进度条、结果归并。

worker 是纯函数、不写盘，所有落盘都在主进程完成；结果按完成顺序产出，
调用方负责按业务 key 归并排序，保证并发结果与串行一致（tests/test_parallel.py 验证）。
"""

from __future__ import annotations

import logging
import os
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from typing import Any, TypeVar

LOG = logging.getLogger(__name__)

T = TypeVar("T")   # 任务类型
R = TypeVar("R")   # 结果类型


def run_pool(
    jobs: Iterable[T],
    worker_fn: Callable[[T], R],
    shared: Any,
    workers: int,
    *,
    desc: str,
    unit: str,
    initializer: Callable[..., None] | None = None,
) -> Iterator[R]:
    """用 ProcessPoolExecutor 并发执行 worker_fn，按完成顺序产出结果。

    Args:
        jobs: 任务列表（每个元素会 pickle 给一个子进程）。
        worker_fn: 模块顶层的 worker 函数，签名为 worker(job) -> result。
        shared: 只读共享上下文，经 initializer 在每个子进程注入一次；None 表示无共享。
        workers: 进程数；调用方保证 >1 才走这里（串行路径保持原代码不变）。
        desc/unit: tqdm 进度条的描述与单位（如 "episode 质检" / "ep"）。
        initializer: 子进程初始化函数（接收 shared 作为唯一参数）。

    Yields:
        worker 的返回值，顺序为完成顺序，调用方负责按 key 归并。
    """
    from tqdm import tqdm  # 延迟导入：串行路径不引入 tqdm 开销

    job_list = list(jobs)
    if not job_list:
        return
    # 进程数不要超过任务数和物理核数，避免空转进程白付 spawn 成本
    workers = max(1, min(int(workers), len(job_list), os.cpu_count() or 1))
    initargs = () if shared is None else (shared,)

    LOG.info("并发执行：%d 个任务 / %d 个进程（%s）", len(job_list), workers, desc)
    # 有界提交：一次性 submit 全部任务会把所有 DataFrame pickle 进 feeder 线程，
    # 峰值内存随总集数增长；只让 2×workers 个任务同时进执行队列。
    pending = deque(job_list)
    in_flight: dict[Any, T] = {}
    window = workers * 2
    pool = ProcessPoolExecutor(
        max_workers=workers, initializer=initializer, initargs=initargs
    )
    bar = tqdm(total=len(job_list), desc=desc, unit=unit, dynamic_ncols=True)

    def _submit() -> None:
        while pending and len(in_flight) < window:
            job = pending.popleft()
            try:
                in_flight[pool.submit(worker_fn, job)] = job
            except BrokenProcessPool:
                # 池在提交瞬间破裂：任务放回队首，交给串行回退，否则它会静默丢失
                pending.appendleft(job)
                raise

    broken_jobs: list[T] = []
    try:
        _submit()
        while in_flight:
            done, _ = wait(list(in_flight), return_when=FIRST_COMPLETED)
            broken_jobs.clear()
            for fut in done:
                job = in_flight.pop(fut, None)
                try:
                    # worker 抛出的异常在这里原样上抛，由主流程按阶段统一处理
                    result = fut.result()
                except BrokenProcessPool:
                    # job 已被 pop、尚未交付：记下来并回串行补跑，否则丢任务
                    if job is not None:
                        broken_jobs.append(job)
                    raise
                yield result
                bar.update(1)
                _submit()
    except BrokenProcessPool as exc:
        # 子进程被 OOM/段错误杀死时，剩余任务回退主进程串行跑完，保住完整结论
        remaining = list(in_flight.values()) + list(pending) + broken_jobs
        LOG.error("进程池异常终止（worker 可能被系统杀死），剩余 %d 个任务回退串行: %s",
                  len(remaining), exc)
        in_flight.clear()
        pending.clear()
        if initializer is not None:
            initializer(*initargs)
        for job in remaining:
            yield worker_fn(job)
            bar.update(1)
    except KeyboardInterrupt:
        # 不等运行中的任务收尾，否则长 episode 上 Ctrl+C 表现为卡住
        for fut in in_flight:
            fut.cancel()
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    finally:
        bar.close()
        pool.shutdown(wait=False)
