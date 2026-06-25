from __future__ import annotations

import numpy as np

from loader import SampleLoader
from shard_io import CompactSample

PIECES_SIZE = 8820


def _sample(seed: int) -> CompactSample:
    rng = np.random.default_rng(seed)
    return CompactSample(
        state_pieces=rng.integers(0, 2, size=PIECES_SIZE, dtype=np.uint8),
        no_capture_plies=np.uint8(0),
        policy_ids=np.array([0, 1, 2], dtype=np.uint16),
        policy_probs=np.array([0.5, 0.3, 0.2], dtype=np.float32),
        value=float(rng.choice([-1.0, 0.0, 1.0])),
    )


class FakeSource:
    def __init__(self, shards: dict[str, list[CompactSample]]) -> None:
        self.shards = dict(shards)
        self.archived: list[str] = []

    def list_shard(self) -> list[str]:
        return sorted(self.shards)

    def read_shard(self, name: str) -> list[CompactSample]:
        return self.shards[name]

    def archive_shard(self, name: str) -> None:
        self.shards.pop(name, None)
        self.archived.append(name)


def _shards(num: int, per: int) -> dict[str, list[CompactSample]]:
    idx = 0
    result = {}
    for i in range(num):
        result[f"shard_{i:06d}.bin"] = [_sample(idx + j) for j in range(per)]
        idx += per
    return result


def test_loader_consumes_budget_and_terminates():
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

    assert loader.steps == len(batches) > 0
    for states, pis, zs in batches:
        assert states.shape == (10, 99, 10, 9)
        assert pis.shape == (10, 8100)
        assert zs.shape == (10,)

    assert loader.samples_seen >= 100
    assert loader.samples_seen <= 110
    assert len(source.archived) == loader.shards_consumed <= 11
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
