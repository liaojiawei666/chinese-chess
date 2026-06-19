from __future__ import annotations

import io
import json

import torch

from trainer.config import NetworkConfig
from trainer.model_io import ModelIO, scripted_bytes
from trainer.network import PolicyValueNet


def _tiny_net() -> PolicyValueNet:
    return PolicyValueNet(NetworkConfig(hidden_channels=8, residual_blocks=1))


def test_scripted_bytes_loadable_and_matches_eager():
    net = _tiny_net()
    net.eval()
    data = scripted_bytes(net)

    module = torch.jit.load(io.BytesIO(data))
    x = torch.zeros(1, net.config.input_channels, 10, 9)
    with torch.no_grad():
        logits_e, value_e = net(x)
        logits_s, value_s = module(x)
    torch.testing.assert_close(logits_s, logits_e)
    torch.testing.assert_close(value_s, value_e)


def test_save_writes_versioned_and_pointer(tmp_path):
    model_io = ModelIO(tmp_path)
    net = _tiny_net()
    model_io.save(40, net)

    assert model_io.latest_version() == 40
    assert (tmp_path / "model_000040.pt").exists()

    pointer = json.loads((tmp_path / "latest.json").read_text())
    assert pointer["version"] == 40 and pointer["path"] == "model_000040.pt"

    version, blob = model_io.load_latest()
    assert version == 40
    module = torch.jit.load(io.BytesIO(blob))
    out = module(torch.zeros(1, net.config.input_channels, 10, 9))
    assert out[0].shape[-1] == net.config.action_space_size


def test_put_model_pointer_and_load(tmp_path):
    model_io = ModelIO(tmp_path)
    assert model_io.latest_version() is None
    assert model_io.load_latest() is None

    model_io.put_model(40, b"weights-40")
    assert model_io.latest_version() == 40
    assert model_io.load_latest() == (40, b"weights-40")


def test_retention_keeps_recent_checkpoints_and_latest(tmp_path):
    model_io = ModelIO(tmp_path, keep_recent=3, checkpoint_every=2000, keep_checkpoints=3)

    # 模拟训练每 20 步导出一次，跨过几个 checkpoint 边界（用字节级 put_model，避免反复 jit.script）。
    for step in range(20, 6020 + 1, 20):
        model_io.put_model(step, f"w{step}".encode())

    existing = sorted(int(p.stem.split("_")[1]) for p in tmp_path.glob("model_*.pt"))

    # 最近 3 个
    for v in (5980, 6000, 6020):
        assert v in existing
    # checkpoint（2000 的倍数）最多保留 3 个
    for v in (2000, 4000, 6000):
        assert v in existing
    # 既非最近也非 checkpoint 的早期版本应被淘汰
    assert 20 not in existing
    assert 3000 not in existing
    assert model_io.latest_version() == 6020
