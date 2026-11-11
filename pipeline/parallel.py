"""多进程并发执行器：任务分发、tqdm 进度条、结果归并。

并发安全架构（本模块的唯一职责就是守住这几条契约）：

1. **worker 是纯函数**：只读自己的输入，返回可 pickle 的结果；
   绝不写文件、绝不写共享变量。因此「多个进程同时写同一目录/同一 JSON」
   的冲突在设计上就不存在——worker 根本不写盘。
2. **主进程统一写入**：所有 JSON / CSV / parquet 落盘都发生在主进程。
   结果回收直接用 Future 的返回值（`wait(FIRST_COMPLETED)`），
   Future 队列本身就是线程安全的结果通道，不需要再引入
   `multiprocessing.Manager().Queue()` 这一层转发进程。
3. **共享上下文只分发一次**：配置、阈值、指纹表等只读大数据通过
   `initializer/initargs` 在每个子进程启动时注入一次（存到子进程全局变量），
   而不是随每个任务重复 pickle 传输。
4. **结果与顺序无关**：Future 完成顺序不确定，调用方必须把结果按业务 key
   归并进 dict、再排序输出，保证并发跑出的结果与串行逐位一致
   （tests/test_parallel.py 会强制验证这一点）。
5. **Windows spawn 兼容**：worker 函数必须是模块顶层函数（按限定名 pickle），
   所有参数必须可 pickle；入口已有 `if __name__ == "__main__"` 保护。
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
        worker 的返回值，顺序为完成顺序（不确定！）——调用方负责按 key 归并。
    """
    from tqdm import tqdm  # 延迟导入：串行路径不引入 tqdm 开销

    job_list = list(jobs)
    if not job_list:
        return
    # 进程数不要超过任务数和物理核数，避免空转进程白付 spawn 成本
    workers = max(1, min(int(workers), len(job_list), os.cpu_count() or 1))
    initargs = () if shared is None else (shared,)

    LOG.info("并发执行：%d 个任务 / %d 个进程（%s）", len(job_list), workers, desc)
    # 有界提交：只让 2×workers 个任务同时进入执行队列。一次性 submit 全部任务
    # 会把所有 episode 的 DataFrame 都 pickle 进 feeder 线程，峰值内存随总集数增长。
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
            in_flight[pool.submit(worker_fn, job)] = job

    try:
        _submit()
        while in_flight:
            done, _ = wait(list(in_flight), return_when=FIRST_COMPLETED)
            for fut in done:
                in_flight.pop(fut, None)
                # worker 抛出的异常会在这里原样上抛（附带子进程 traceback），
                # 由主流程的阶段级 try/except 统一处理，行为与串行一致
                yield fut.result()
                bar.update(1)
                _submit()
    except BrokenProcessPool as exc:
        # 子进程被 OOM/段错误直接杀死时，尚未完成的任务不会再返回结果。
        # 此时把剩余任务在主进程串行跑完，保住已完成与未完成 episode 的完整结论；
        # 若主进程也扛不住，异常会继续上抛，由调用方决定如何降级。
        remaining = list(in_flight.values()) + list(pending)
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
        # 不再等待运行中的任务收尾，否则长 episode 上 Ctrl+C 会表现为「卡住」
        for fut in in_flight:
            fut.cancel()
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    finally:
        bar.close()
        pool.shutdown(wait=False)
