from __future__ import annotations

import io

import torch

from trainer.config import NetworkConfig
from trainer.exporter import export_model, scripted_bytes
from trainer.network import PolicyValueNet
from trainer.store import LocalModelStore


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


def test_export_model_writes_versioned_and_pointer(tmp_path):
    store = LocalModelStore(tmp_path)
    net = _tiny_net()
    export_model(store, net, 40)

    assert store.get_version() == 40
    assert (tmp_path / "model_000040.pt").exists()

    version, blob = store.get_latest()
    assert version == 40
    module = torch.jit.load(io.BytesIO(blob))
    out = module(torch.zeros(1, net.config.input_channels, 10, 9))
    assert out[0].shape[-1] == net.config.action_space_size
