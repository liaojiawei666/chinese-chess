from __future__ import annotations

import json

import numpy as np
import torch

from config import NetworkConfig
from model_io import ModelIO, onnx_bytes
from network import PolicyValueNet


def _tiny_net() -> PolicyValueNet:
    return PolicyValueNet(NetworkConfig(hidden_channels=8, residual_blocks=1))


def test_onnx_bytes_produces_valid_onnx():
    net = _tiny_net()
    net.eval()
    data = onnx_bytes(net)
    assert len(data) > 0
    assert data[:4] != b""


def test_save_writes_versioned_and_pointer(tmp_path):
    model_io = ModelIO(tmp_path)
    net = _tiny_net()
    model_io.save(1, net)

    assert model_io.latest_version() == 1
    assert (tmp_path / "model_gen_0001.onnx").exists()

    pointer = json.loads((tmp_path / "latest.json").read_text())
    assert pointer["generation"] == 1


def test_put_model_pointer_and_load(tmp_path):
    model_io = ModelIO(tmp_path)
    assert model_io.latest_version() is None
    assert model_io.load_latest() is None

    model_io.put_model(1, b"weights-1")
    assert model_io.latest_version() == 1
    gen, data = model_io.load_latest()
    assert gen == 1
    assert data == b"weights-1"


def test_retention_keeps_recent_and_latest(tmp_path):
    model_io = ModelIO(tmp_path, keep_recent=3)

    for gen in range(1, 11):
        model_io.put_model(gen, f"w{gen}".encode())

    existing = sorted(
        int(p.stem.split("_")[-1])
        for p in tmp_path.glob("model_gen_*.onnx")
    )

    assert 8 in existing
    assert 9 in existing
    assert 10 in existing
    assert 1 not in existing
    assert model_io.latest_version() == 10
