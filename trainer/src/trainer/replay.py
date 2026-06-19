from __future__ import annotations

from collections import deque
from typing import Iterable

import numpy as np

from .shard_io import TrainSample

# 一批训练数据：(states, pis, zs)，均为堆叠好的 numpy 数组，由 train 侧搬上 device。
Batch = tuple[np.ndarray, np.ndarray, np.ndarray]


class ReplayBuffer:
    """最近样本的滑动窗口（第一阶段纯内存）。

    用 deque(maxlen) 自动淘汰最旧样本；sample 均匀随机取一批，打散同局相邻样本的相关性。
    """

    def __init__(self, capacity: int, rng: np.random.Generator | None = None) -> None:
        self.buffer: deque[TrainSample] = deque(maxlen=capacity)
        self.rng = rng if rng is not None else np.random.default_rng()

    def __len__(self) -> int:
        return len(self.buffer)

    def add(self, samples: Iterable[TrainSample]) -> None:
        self.buffer.extend(samples)

    def evict_oldest(self, k: int) -> int:
        """从最旧端（左）淘汰至多 k 条，返回实际淘汰数。

        收尾收缩用：生产已停、右侧无新样本时，每个训练步靠它把窗口往右截断收缩，
        直到窗口小于 min_buffer_size 结束。
        """
        evicted = 0
        for _ in range(min(k, len(self.buffer))):
            self.buffer.popleft()
            evicted += 1
        return evicted

    def sample(self, batch_size: int) -> Batch:
        """均匀随机采样一批；buffer 不足 batch_size 时取全部。"""
        n = min(batch_size, len(self.buffer))
        indices = self.rng.choice(len(self.buffer), size=n, replace=False)
        chosen = [self.buffer[i] for i in indices]
        states = np.stack([s.state for s in chosen])
        pis = np.stack([s.pi for s in chosen])
        zs = np.array([s.z for s in chosen], dtype=np.float32)
        return states, pis, zs
