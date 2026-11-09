"""逐 episode 质检的多进程 worker（供 validate_dataset(workers>1) 使用）。

Windows spawn 兼容的三条硬约束：
1. worker / initializer 必须是本模块的顶层函数（按限定名 pickle 到子进程）；
2. 所有跨进程参数必须可 pickle（DataFrame / dict / list / 基元类型）；
3. 不在 worker 里做任何写盘与全局写入——结论全部以返回值交回主进程。

共享上下文（config / 阈值 / 全局指纹表 / 数据集元信息）由
`init_episode_worker` 在子进程启动时注入一次，存到模块全局 `_SHARED`；
每个任务只传 `(episode_index, 该 episode 的逐帧 DataFrame)`。
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

import pandas as pd

from .context import DatasetContext
from .types import EpisodeResult

LOG = logging.getLogger(__name__)

#: 子进程内的共享上下文（由 initializer 注入；主进程内始终为空）
_SHARED: dict[str, Any] = {}


class EpisodeDatasetStub:
    """非视频规则所需的最小 dataset 视图（可 pickle）。

    真实的 LeRobotDatasetView 含有整张逐帧表，不能也不需要在子进程间分发；
    规则函数实际只读取这几个字段：
      - ``joint_names``            （check_joint_limits 报关节名）
      - ``episodes``               （check_meta_consistency 读 episode_index/length 两列）
      - ``task_of()`` / ``tasks``  （check_task 判任务描述缺失）
      - ``frames.columns``         （check_task 判 task_index 列是否存在）
      - ``fps``                    （check_timestamp 的步长判定）
    以上全部在主进程预抽取，随共享上下文一次性分发。
    """

    def __init__(
        self,
        *,
        joint_names: list[str],
        episodes_meta: pd.DataFrame,
        task_labels: dict[int, str],
        has_task_index_col: bool,
        has_tasks_meta: bool,
        fps: float,
        locator_available: bool = True,
    ) -> None:
        self.joint_names = joint_names
        self._episodes_meta = episodes_meta
        self._task_labels = task_labels
        self._has_task_index_col = has_task_index_col
        self._has_tasks_meta = has_tasks_meta
        self.fps = fps
        # 编号空间能否对齐：对不上时定位类规则在 worker 里也要跳过，否则并行与
        # 串行结论会不一致（串行跳过、worker 报 error）
        self.locator_available = locator_available

    @property
    def episodes(self) -> pd.DataFrame:
        return self._episodes_meta

    @property
    def tasks(self) -> dict[int, str]:
        # 规则里只用 tasks 的真假性（「meta/tasks 是否为空」），内容不重要
        return {0: "task"} if self._has_tasks_meta else {}

    @property
    def frames(self) -> Any:
        # 规则里只用 "task_index" in dataset.frames.columns
        cols = ["task_index"] if self._has_task_index_col else []
        return SimpleNamespace(columns=cols)

    def task_of(self, episode_index: int) -> str:
        return self._task_labels.get(int(episode_index), "")


def build_episode_shared(
    dataset: Any, ctx: DatasetContext, config: dict[str, Any]
) -> dict[str, Any]:
    """在主进程构建共享上下文（每个子进程只接收一份拷贝）。

    注意指纹表 ``ctx.fingerprints`` 在 DatasetContext.build 阶段已按
    「全部 episode 升序首次出现」预计算完毕，worker 只读它，
    因此重复 episode 的判定与串行完全一致（与任务执行顺序无关）。
    """
    eps = dataset.episodes
    if "length" in eps.columns:
        # 附带视频时间窗列，让多模态对齐规则在并行模式下也能生效
        video_cols = [c for c in eps.columns
                      if str(c).startswith("videos/") and str(c).endswith(("/from_timestamp", "/to_timestamp"))]
        episodes_meta = eps[["episode_index", "length", *video_cols]].copy()
    else:
        # 没有 length 列时 check_meta_consistency 直接跳过；空表保持同样语义
        episodes_meta = pd.DataFrame()
    if len(episodes_meta) and getattr(dataset, "meta_id_map", None):
        # meta 用全局编号、data 用 0..N-1 时，worker 侧也统一到数据侧编号，
        # 保证 check_meta_consistency 的 join 与串行视图（view.meta_row）口径一致
        inv = {int(m): int(d) for d, m in dataset.meta_id_map.items()}
        episodes_meta = episodes_meta[episodes_meta["episode_index"].isin(inv)].copy()
        episodes_meta["episode_index"] = episodes_meta["episode_index"].map(inv).astype("int64")
        episodes_meta = episodes_meta.sort_values("episode_index").reset_index(drop=True)
    return {
        "config": config,
        "frame_bounds": ctx.frame_bounds,
        "action_thresholds": ctx.action_thresholds,
        "fingerprints": ctx.fingerprints,
        "stub": {
            "joint_names": list(dataset.joint_names),
            "episodes_meta": episodes_meta,
            "task_labels": {int(ep): dataset.task_of(int(ep)) for ep in dataset.episode_indices},
            "has_task_index_col": "task_index" in dataset.frames.columns,
            "has_tasks_meta": bool(dataset.tasks),
            "fps": float(dataset.fps),
            "locator_available": bool(getattr(dataset, "locator_available", True)),
        },
    }


def init_episode_worker(shared: dict[str, Any]) -> None:
    """子进程 initializer：注入共享上下文（每个子进程只调一次）。"""
    global _SHARED
    _SHARED = shared


def validate_episode_worker(
    job: tuple[int, pd.DataFrame]
) -> tuple[int, EpisodeResult, dict[str, str]]:
    """worker：跑单个 episode 的全部非视频规则，返回结论（不写盘）。

    Returns:
        (episode_index, EpisodeResult, skipped_delta)；skipped_delta 是
        worker 内新发现的「规则跳过原因」，由主进程合并展示。
    """
    from . import _validate_non_video  # 延迟导入：避免与 validator/__init__ 循环依赖

    ep_index, ep_df = job
    shared = _SHARED
    # DatasetContext 是纯数据 dataclass，按字段重组一份（issues/resolved 在
    # 逐 episode 规则里用不到，主进程的那份才会进最终报告）
    ctx = DatasetContext(
        frame_bounds=shared["frame_bounds"],
        action_thresholds=shared["action_thresholds"],
        fingerprints=shared["fingerprints"],
        issues=[],
        resolved={},
    )
    stub = EpisodeDatasetStub(**shared["stub"])
    skipped: dict[str, str] = {}
    try:
        res = _validate_non_video(ep_df, ep_index, stub, ctx, shared["config"], skipped)
    except Exception as exc:  # noqa: BLE001 - 毒数据防冲击：单 episode 失败不拖垮进程池
        res = EpisodeResult(episode_index=ep_index, frame_count=int(len(ep_df)))
        res.add("episode_read_error", error=str(exc))
    return ep_index, res, skipped
