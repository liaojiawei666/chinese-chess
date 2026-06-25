from __future__ import annotations

import numpy as np

from config import ACTION_SPACE_SIZE, NO_CAPTURE_DRAW_PLIES
from shard_io import (
    CompactSample,
    ShardSource,
    decode_shard,
    decompress_batch,
    encode_shard,
)

PIECES_SIZE = 8820


def _make_sample(seed: int) -> CompactSample:
    rng = np.random.default_rng(seed)
    state_pieces = rng.integers(0, 2, size=PIECES_SIZE, dtype=np.uint8)
    no_capture_plies = np.uint8(rng.integers(0, NO_CAPTURE_DRAW_PLIES + 1))

    count = rng.integers(3, 10)
    ids = rng.choice(ACTION_SPACE_SIZE, size=count, replace=False).astype(np.uint16)
    probs = rng.random(count).astype(np.float32)
    probs /= probs.sum()

    value = float(rng.choice([-1.0, 0.0, 1.0]))
    return CompactSample(
        state_pieces=state_pieces,
        no_capture_plies=no_capture_plies,
        policy_ids=ids,
        policy_probs=probs,
        value=value,
    )


def test_encode_decode_roundtrip():
    samples = [_make_sample(i) for i in range(5)]
    data = encode_shard(samples)
    restored = decode_shard(data)

    assert len(restored) == len(samples)
    for orig, got in zip(samples, restored):
        np.testing.assert_array_equal(got.state_pieces, orig.state_pieces)
        assert got.no_capture_plies == orig.no_capture_plies
        np.testing.assert_array_equal(got.policy_ids, orig.policy_ids)
        np.testing.assert_allclose(got.policy_probs, orig.policy_probs, atol=1e-7)
        assert abs(got.value - orig.value) < 1e-9


def test_decompress_batch():
    samples = [_make_sample(i) for i in range(3)]
    states, pis, zs = decompress_batch(samples)

    assert states.shape == (3, 99, 10, 9)
    assert states.dtype == np.float32
    assert pis.shape == (3, ACTION_SPACE_SIZE)
    assert zs.shape == (3,)

    for i, s in enumerate(samples):
        np.testing.assert_array_equal(
            states[i, :98],
            s.state_pieces.reshape(98, 10, 9).astype(np.float32),
        )
        expected_plies = float(s.no_capture_plies) / NO_CAPTURE_DRAW_PLIES
        np.testing.assert_allclose(states[i, 98], expected_plies)
        expected_pi = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
        expected_pi[s.policy_ids] = s.policy_probs
        np.testing.assert_allclose(pis[i], expected_pi, atol=1e-7)
        assert abs(zs[i] - s.value) < 1e-9


def test_policy_sum_one():
    s = _make_sample(7)
    _, pis, _ = decompress_batch([s])
    assert abs(float(pis[0].sum()) - 1.0) < 1e-5


def test_shard_source_write_list_read_archive(tmp_path):
    source = ShardSource(tmp_path)
    assert source.list_shard() == []

    source.write_shard("shard_000000.bin", [_make_sample(1)])
    source.write_shard("shard_000001.bin", [_make_sample(2), _make_sample(3)])
    (tmp_path / "shard_000002.bin.tmp").write_bytes(b"x")
    (tmp_path / "notes.txt").write_text("hi")

    assert source.list_shard() == ["shard_000000.bin", "shard_000001.bin"]
    restored = source.read_shard("shard_000001.bin")
    assert len(restored) == 2

    source.archive_shard("shard_000000.bin")
    assert source.list_shard() == ["shard_000001.bin"]
    assert (tmp_path / "archive" / "shard_000000.bin").exists()
