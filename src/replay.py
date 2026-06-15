from __future__ import annotations

from collections import deque
from typing import Iterable

import numpy as np

from selfplay import Sample

# 一批训练数据：(states, pis, zs)，均为堆叠好的 numpy 数组，由 train 侧搬上 device。
Batch = tuple[np.ndarray, np.ndarray, np.ndarray]


class ReplayBuffer:
    """最近样本的滑动窗口（第一阶段纯内存）。

    用 deque(maxlen) 自动淘汰最旧样本；sample 均匀随机取一批，打散同局相邻样本的相关性。
    """

    def __init__(self, capacity: int, rng: np.random.Generator | None = None) -> None:
        self.buffer: deque[Sample] = deque(maxlen=capacity)
        self.rng = rng if rng is not None else np.random.default_rng()

    def __len__(self) -> int:
        return len(self.buffer)

    def add(self, samples: Iterable[Sample]) -> None:
        self.buffer.extend(samples)

    def sample(self, batch_size: int) -> Batch:
        """均匀随机采样一批；buffer 不足 batch_size 时取全部。"""
        n = min(batch_size, len(self.buffer))
        indices = self.rng.choice(len(self.buffer), size=n, replace=False)
        chosen = [self.buffer[i] for i in indices]
        states = np.stack([s.state for s in chosen])
        pis = np.stack([s.pi for s in chosen])
        zs = np.array([s.z for s in chosen], dtype=np.float32)
        return states, pis, zs
