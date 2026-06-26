"""SampleLoader：消费端按数据预算 + reuse 驱动的滑动窗口批次迭代器。"""

from __future__ import annotations

import logging
import struct
import time
from typing import Iterator, Protocol

import numpy as np

from replay import Batch, ReplayBuffer
from shard_io import CompactSample

logger = logging.getLogger(__name__)


class ShardSourceLike(Protocol):
    def list_shard(self) -> list[str]: ...

    def read_shard(self, name: str) -> list[CompactSample]: ...

    def archive_shard(self, name: str) -> None: ...


class SampleLoader:
    """迭代产出训练 batch；耗尽数据预算并收尾收缩后自然 StopIteration。"""

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
        self.steps = 0
        self.shards_consumed = 0
        self.samples_seen = 0

    def __iter__(self) -> Iterator[Batch]:
        batch = self.batch_size
        reuse = self.target_reuse
        min_buf = self.min_buffer_size
        total = self.total_samples
        advance_per_step = max(1, round(batch / reuse))

        consumed = 0
        idle_polls = 0

        def wait_or_stop() -> bool:
            nonlocal idle_polls
            idle_polls += 1
            if self.idle_poll_limit is not None and idle_polls >= self.idle_poll_limit:
                return True
            # 降频：首次以及之后每 ~30s 提示一次，避免刷屏。
            log_every = max(1, round(30.0 / max(self.poll_interval_s, 1e-3)))
            if idle_polls == 1 or idle_polls % log_every == 0:
                logger.info("waiting for new shards... (idle poll #%d)", idle_polls)
            time.sleep(self.poll_interval_s)
            return False

        while True:
            within_budget = (self.steps + 1) * batch <= reuse * consumed
            production_done = consumed >= total
            need_data = not within_budget or len(self.buffer) < min_buf

            if need_data and not production_done:
                available = self.source.list_shard()
                if not available:
                    if wait_or_stop():
                        return
                    continue
                idle_polls = 0
                for name in available:
                    try:
                        samples = self.source.read_shard(name)
                    except (ValueError, OSError, struct.error) as e:
                        # 理论上 datagen 已用原子写规避半截文件；这里兜底，
                        # 避免坏/截断分片直接打挂训练，归档隔离后跳过。
                        logger.warning("skip bad shard %s: %s", name, e)
                        self.source.archive_shard(name)
                        continue
                    self.buffer.add(samples)
                    self.source.archive_shard(name)
                    consumed += len(samples)
                    self.shards_consumed += 1
                    self.samples_seen += len(samples)
                    logger.info(
                        "ingested %s (%d samples), buffer: %d/%d",
                        name, len(samples), len(self.buffer), self.buffer.capacity,
                    )
                    enough_budget = (self.steps + 1) * batch <= reuse * consumed
                    if consumed >= total or (enough_budget and len(self.buffer) >= min_buf):
                        break
                continue

            if not within_budget and production_done:
                self.buffer.evict_oldest(advance_per_step)
                logger.debug(
                    "tail phase: evicted %d oldest, buffer: %d/%d",
                    advance_per_step, len(self.buffer), self.buffer.capacity,
                )

            if len(self.buffer) < min_buf:
                if production_done:
                    return
                if wait_or_stop():
                    return
                continue

            idle_polls = 0
            self.steps += 1
            yield self.buffer.sample(batch)
