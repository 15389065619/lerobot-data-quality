"""多进程并发执行器：任务分发、tqdm 进度条、结果归并。

并发安全架构（本模块的唯一职责就是守住这几条契约）：

1. **worker 是纯函数**：只读自己的输入，返回可 pickle 的结果；
   绝不写文件、绝不写共享变量。因此「多个进程同时写同一目录/同一 JSON」
   的冲突在设计上就不存在——worker 根本不写盘。
2. **主进程统一写入**：所有 JSON / CSV / parquet 落盘都发生在主进程。
   结果回收直接用 Future 的返回值（`as_completed`），
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
from collections.abc import Callable, Iterable, Iterator
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
    from concurrent.futures import ProcessPoolExecutor, as_completed

    from tqdm import tqdm  # 延迟导入：串行路径不引入 tqdm 开销

    job_list = list(jobs)
    if not job_list:
        return
    # 进程数不要超过任务数和物理核数，避免空转进程白付 spawn 成本
    workers = max(1, min(int(workers), len(job_list), os.cpu_count() or 1))
    initargs = () if shared is None else (shared,)

    LOG.info("并发执行：%d 个任务 / %d 个进程（%s）", len(job_list), workers, desc)
    with ProcessPoolExecutor(
        max_workers=workers, initializer=initializer, initargs=initargs
    ) as pool:
        futures = [pool.submit(worker_fn, job) for job in job_list]
        try:
            with tqdm(total=len(futures), desc=desc, unit=unit, dynamic_ncols=True) as bar:
                for fut in as_completed(futures):
                    # worker 抛出的异常会在这里原样上抛（附带子进程 traceback），
                    # 由主流程的阶段级 try/except 统一处理，行为与串行一致
                    yield fut.result()
                    bar.update(1)
        finally:
            # 异常中断时尽量取消还没开始的任务，退出上下文会等待运行中的任务收尾
            for fut in futures:
                fut.cancel()
