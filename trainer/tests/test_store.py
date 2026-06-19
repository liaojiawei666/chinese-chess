from __future__ import annotations

import json

from trainer.store import LocalModelStore, LocalSampleStore


def test_sample_store_roundtrip_and_listing(tmp_path):
    store = LocalSampleStore(tmp_path)
    assert store.list_shards() == []

    store.put_shard("shard_000000_w00_000000.st", b"a")
    store.put_shard("shard_000000_w01_000000.st", b"b")
    # 半成品 .tmp 与非分片文件不应被列出。
    (tmp_path / "shard_000000_w00_000001.st.tmp").write_bytes(b"x")
    (tmp_path / "notes.txt").write_text("hi")

    assert store.list_shards() == [
        "shard_000000_w00_000000.st",
        "shard_000000_w01_000000.st",
    ]
    assert store.list_shards(after="shard_000000_w00_000000.st") == [
        "shard_000000_w01_000000.st",
    ]
    assert store.get_shard("shard_000000_w00_000000.st") == b"a"

    store.delete_shard("shard_000000_w00_000000.st")
    assert store.list_shards() == ["shard_000000_w01_000000.st"]


def test_model_store_pointer_and_atomic(tmp_path):
    store = LocalModelStore(tmp_path, keep_recent=3, checkpoint_every=2000, keep_checkpoints=3)
    assert store.get_version() is None
    assert store.get_latest() is None

    store.put_model(40, b"weights-40")
    assert store.get_version() == 40
    version, data = store.get_latest()
    assert version == 40 and data == b"weights-40"

    pointer = json.loads((tmp_path / "latest.json").read_text())
    assert pointer["version"] == 40
    assert pointer["path"] == "model_000040.pt"


def test_retention_keeps_recent_checkpoints_and_latest(tmp_path):
    store = LocalModelStore(tmp_path, keep_recent=3, checkpoint_every=2000, keep_checkpoints=3)

    # 模拟训练每 20 步导出一次，跨过几个 checkpoint 边界。
    for step in range(20, 6020 + 1, 20):
        store.put_model(step, f"w{step}".encode())

    existing = sorted(
        int(p.stem.split("_")[1]) for p in tmp_path.glob("model_*.pt")
    )

    # 最近 3 个：5980, 6000, 6020
    for v in (5980, 6000, 6020):
        assert v in existing
    # checkpoint（2000 的倍数）最多保留 3 个：2000, 4000, 6000
    for v in (2000, 4000, 6000):
        assert v in existing
    # 既非最近也非 checkpoint 的早期版本应被淘汰
    assert 20 not in existing
    assert 3000 not in existing
    # latest 指向的版本必然在
    assert store.get_version() == 6020
