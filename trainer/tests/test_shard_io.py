from __future__ import annotations

import numpy as np

from trainer.config import ACTION_SPACE_SIZE, BOARD_HEIGHT, BOARD_WIDTH, INPUT_CHANNELS
from trainer.shard_io import TrainSample, read_shard, write_shard


def _make_sample(seed: int) -> TrainSample:
    rng = np.random.default_rng(seed)
    state = np.zeros((INPUT_CHANNELS, BOARD_HEIGHT, BOARD_WIDTH), dtype=np.float32)
    # 98 个 0/1 平面：随机置位
    binary = rng.integers(0, 2, size=(98, BOARD_HEIGHT, BOARD_WIDTH)).astype(np.float32)
    state[:98] = binary
    # 末通道：未吃子比例（量化到 1/255 网格，保证 u8 往返无损）
    ratio = round(0.37 * 255) / 255.0
    state[98] = ratio

    pi = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
    idx = rng.choice(ACTION_SPACE_SIZE, size=5, replace=False)
    vals = rng.random(5).astype(np.float32)
    vals /= vals.sum()
    pi[idx] = vals
    return TrainSample(state=state, pi=pi, z=float(rng.choice([-1.0, 0.0, 1.0])))


def test_write_read_roundtrip():
    samples = [_make_sample(i) for i in range(3)]
    data = write_shard(samples)
    restored = read_shard(data)

    assert len(restored) == len(samples)
    for orig, got in zip(samples, restored):
        assert got.state.shape == (INPUT_CHANNELS, BOARD_HEIGHT, BOARD_WIDTH)
        assert got.state.dtype == np.float32
        # uint8 量化往返：0/1 平面无损，ratio 平面在 1/255 网格上无损
        np.testing.assert_allclose(got.state, orig.state, atol=1.0 / 255 / 2)
        # 稀疏 π 还原后与原稠密一致
        np.testing.assert_allclose(got.pi, orig.pi, atol=1e-6)
        assert got.pi.shape == (ACTION_SPACE_SIZE,)
        assert abs(got.z - orig.z) < 1e-9


def test_pi_rows_normalized_after_restore():
    samples = [_make_sample(7)]
    restored = read_shard(write_shard(samples))
    assert abs(float(restored[0].pi.sum()) - 1.0) < 1e-5
