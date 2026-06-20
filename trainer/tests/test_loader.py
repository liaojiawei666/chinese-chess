from __future__ import annotations

import numpy as np

from trainer.loader import SampleLoader
from trainer.shard_io import TrainSample


def _sample(v: float) -> TrainSample:
    return TrainSample(
        state=np.full((1,), v, dtype=np.float32),
        pi=np.full((1,), v, dtype=np.float32),
        z=v,
    )


class FakeSource:
    """内存分片源：每个分片是一组样本，记录归档顺序。"""

    def __init__(self, shards: dict[str, list[TrainSample]]) -> None:
        self.shards = dict(shards)
        self.archived: list[str] = []

    def list_shard(self) -> list[str]:
        return sorted(self.shards)

    def read_shard(self, name: str) -> list[TrainSample]:
        return self.shards[name]

    def archive_shard(self, name: str) -> None:
        self.shards.pop(name, None)
        self.archived.append(name)


def _shards(num: int, per: int) -> dict[str, list[TrainSample]]:
    return {
        f"shard_000000_w00_{i:06d}.st": [_sample(float(i)) for _ in range(per)]
        for i in range(num)
    }


def test_loader_consumes_budget_and_terminates():
    # 12 个分片 × 10 条 = 120 条；预算 100。
    source = FakeSource(_shards(num=12, per=10))
    loader = SampleLoader(
        source,
        total_samples=100,
        target_reuse=2.0,
        batch_size=10,
        buffer_capacity=10_000,
        min_buffer_size=10,
    )

    batches = list(loader)

    # 自然终止（不挂死），且产出了若干 batch。
    assert loader.steps == len(batches) > 0
    # 每个 batch 形状正确。
    for states, pis, zs in batches:
        assert states.shape == (10, 1)
        assert pis.shape == (10, 1)
        assert zs.shape == (10,)

    # 消费到预算即停拉新：约 100 条（最后一个分片可能让它略超）。
    assert loader.samples_seen >= 100
    assert loader.samples_seen <= 110
    # 只归档已消费的分片，不会把 12 个全归档。
    assert len(source.archived) == loader.shards_consumed <= 11

    # reuse=2、预算=100、batch=10 → 稳态约 20 步，叠加收尾收缩，步数应明显多于 20。
    assert loader.steps >= 20


def test_loader_idle_stop_when_no_data():
    source = FakeSource({})
    loader = SampleLoader(
        source,
        total_samples=100,
        target_reuse=2.0,
        batch_size=10,
        buffer_capacity=100,
        min_buffer_size=10,
        idle_poll_limit=2,
        poll_interval_s=0.0,
    )
    batches = list(loader)
    assert batches == []
    assert loader.steps == 0
