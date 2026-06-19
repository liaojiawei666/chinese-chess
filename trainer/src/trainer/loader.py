"""SampleLoader：消费端按数据预算 + reuse 驱动的滑动窗口批次迭代器。

把「拉分片喂窗口、删已消费分片、维持 reuse、收尾收缩、何时结束」全收进一个迭代器：
训练侧只管 `for batch in loader: train_step(...)`，终止权完全在 loader。

reuse（每条样本平均被抽取次数）是与硬件无关的算法超参；窗口推进、收尾、终止都由
`total_samples` + `target_reuse` 决定，生产端只管产。
"""

from __future__ import annotations

import time
from typing import Iterator, Protocol

import numpy as np

from .replay import Batch, ReplayBuffer
from .shard_io import TrainSample


class ShardSourceLike(Protocol):
    """SampleLoader 依赖的分片源接口（ShardSource 即其本地实现）。"""

    def list_shard(self) -> list[str]: ...

    def read_shard(self, name: str) -> list[TrainSample]: ...

    def delete_shard(self, name: str) -> None: ...


class SampleLoader:
    """迭代产出训练 batch `(states, pis, zs)`；耗尽数据预算并收尾收缩后自然 StopIteration。

    - 正常段：维持 reuse 不变量 `steps*batch <= reuse*consumed`，不够就拉新分片喂窗口；
      未消化的分片留在盘上当 backlog（撞 datagen 的 max_pending_shards 时反压生产）。
    - 收尾段：消费满 total_samples 后不再拉新，每步把窗口从最旧端收缩 batch/reuse 条
      （右侧已无新样本，等价于继续右滑后截断），窗口缩到 < min_buffer_size 即结束。
      收缩期对最新样本的额外抽取，正好补回它们稳态期的尾部欠采。

    `idle_poll_limit`：连续多少次「数据饿等待」后停（None 表示长驻，与 datagen 并跑永不空停）。
    """

    def __init__(
        self,
        source: ShardSourceLike,
        *,
        total_samples: int,
        target_reuse: float,
        batch_size: int,
        buffer_capacity: int,
        min_buffer_size: int,
        idle_poll_limit: int | None = None,
        poll_interval_s: float = 1.0,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.source = source
        self.total_samples = total_samples
        self.target_reuse = target_reuse
        self.batch_size = batch_size
        self.min_buffer_size = min_buffer_size
        self.idle_poll_limit = idle_poll_limit
        self.poll_interval_s = poll_interval_s

        self.buffer = ReplayBuffer(buffer_capacity, rng=rng)
        # 进度计数（迭代中更新，供训练侧读取/打印）。
        self.steps = 0
        self.shards_consumed = 0
        self.samples_seen = 0

    def __iter__(self) -> Iterator[Batch]:
        batch = self.batch_size
        reuse = self.target_reuse
        min_buf = self.min_buffer_size
        total = self.total_samples
        # 收尾每步从最旧端淘汰多少（= 稳态每步窗口推进量 batch/reuse）。
        advance_per_step = max(1, round(batch / reuse))

        consumed = 0  # 累计拉入窗口的样本数（单调；= 数据预算进度）
        idle_polls = 0

        def wait_or_stop() -> bool:
            """数据饿/冷启动时等待；返回 True 表示应终止（达到 idle 上限）。"""
            nonlocal idle_polls
            idle_polls += 1
            if self.idle_poll_limit is not None and idle_polls >= self.idle_poll_limit:
                return True
            time.sleep(self.poll_interval_s)
            return False

        while True:
            within_budget = (self.steps + 1) * batch <= reuse * consumed
            production_done = consumed >= total
            # 需要更多数据：reuse 预算不够，或窗口还没到地板（冷启动）。
            need_data = not within_budget or len(self.buffer) < min_buf

            # 正常段：拉新分片喂窗口（够用即止，余量留盘上当 backlog 触发生产端反压）。
            if need_data and not production_done:
                available = self.source.list_shard()  # 未消费即未删；不用游标，避免乱序漏读
                if not available:
                    if wait_or_stop():
                        return
                    continue
                idle_polls = 0
                for name in available:
                    samples = self.source.read_shard(name)
                    self.buffer.add(samples)
                    self.source.delete_shard(name)
                    consumed += len(samples)
                    self.shards_consumed += 1
                    self.samples_seen += len(samples)
                    enough_budget = (self.steps + 1) * batch <= reuse * consumed
                    if consumed >= total or (enough_budget and len(self.buffer) >= min_buf):
                        break
                continue

            # 收尾段：预算耗尽且消费满 total → 收缩窗口（右侧无新样本，截断）。
            if not within_budget and production_done:
                self.buffer.evict_oldest(advance_per_step)

            # 窗口不足地板：收尾则结束（已收缩到底），否则数据饿等待。
            if len(self.buffer) < min_buf:
                if production_done:
                    return
                if wait_or_stop():
                    return
                continue

            # 产出一个训练 batch。
            idle_polls = 0
            self.steps += 1
            yield self.buffer.sample(batch)
