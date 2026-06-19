"""把 PolicyValueNet 导出成 TorchScript 字节流，经 ModelStore 版本化落盘。

datagen(Rust) 用 tch-rs 加载这些 model_{step}.pt 做前向，因此必须是可脱离 Python 的
TorchScript（优先 jit.script；个别脚本不友好写法退化为 jit.trace）。保留策略由
ModelStore.put_model 内部执行。导出格式见 shared/model-format.md。
"""

from __future__ import annotations

import io

import torch
from torch import nn

from .config import BOARD_HEIGHT, BOARD_WIDTH, INPUT_CHANNELS
from .store import ModelStore


def scripted_bytes(net: nn.Module, device: torch.device | str = "cpu") -> bytes:
    """把网络转成 TorchScript 并序列化成字节流。jit.script 失败时退化为 jit.trace。"""
    net.eval()
    try:
        module = torch.jit.script(net)
    except Exception:
        example = torch.zeros(
            1, INPUT_CHANNELS, BOARD_HEIGHT, BOARD_WIDTH, device=device
        )
        module = torch.jit.trace(net, example)

    buffer = io.BytesIO()
    torch.jit.save(module, buffer)
    return buffer.getvalue()


def export_model(
    model_store: ModelStore,
    net: nn.Module,
    step: int,
    device: torch.device | str = "cpu",
) -> None:
    """导出当前权重为 model_{step}.pt 并更新 latest.json（保留策略在 store 内执行）。"""
    was_training = net.training
    data = scripted_bytes(net, device)
    model_store.put_model(step, data)
    if was_training:
        net.train()
