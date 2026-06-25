from __future__ import annotations

from collections import deque
from typing import Iterable

import numpy as np

from shard_io import CompactSample, decompress_batch

Batch = tuple[np.ndarray, np.ndarray, np.ndarray]


class ReplayBuffer:
    """压缩样本的滑动窗口。

    buffer 存 CompactSample（uint8 state + 稀疏 policy），
    sample() 采样后批量解压为 float32 tensor 送训练。
    """

    def __init__(self, capacity: int, rng: np.random.Generator | None = None) -> None:
        self.buffer: deque[CompactSample] = deque(maxlen=capacity)
        self.rng = rng if rng is not None else np.random.default_rng()

    @property
    def capacity(self) -> int:
        return self.buffer.maxlen or 0

    def __len__(self) -> int:
        return len(self.buffer)

    def add(self, samples: Iterable[CompactSample]) -> None:
        self.buffer.extend(samples)

    def evict_oldest(self, k: int) -> int:
        evicted = 0
        for _ in range(min(k, len(self.buffer))):
            self.buffer.popleft()
            evicted += 1
        return evicted

    def sample(self, batch_size: int) -> Batch:
        """均匀随机采样一批，批量解压为 (states, pis, zs)。"""
        n = min(batch_size, len(self.buffer))
        indices = self.rng.choice(len(self.buffer), size=n, replace=False)
        chosen = [self.buffer[i] for i in indices]
        return decompress_batch(chosen)
