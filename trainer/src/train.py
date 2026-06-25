from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from config import TrainConfig
from replay import Batch


def train_step(
    net: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: Batch,
    device: torch.device,
    config: TrainConfig,
) -> dict[str, float]:
    """单个训练步：前向 → 算 loss → 反向 → 裁剪 → 更新。返回各项 loss（float）。"""
    net.train()

    states, pis, zs = batch
    states_t = torch.from_numpy(states).to(device)
    pis_t = torch.from_numpy(pis).to(device)
    zs_t = torch.from_numpy(zs).to(device)

    policy_logits, value = net(states_t)
    value = value.squeeze(-1)

    value_loss = F.mse_loss(value, zs_t)
    log_probs = F.log_softmax(policy_logits, dim=1)
    policy_loss = -(pis_t * log_probs).sum(dim=1).mean()
    loss = value_loss + policy_loss

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(net.parameters(), config.grad_clip_norm)
    optimizer.step()

    return {
        "loss": float(loss.item()),
        "policy_loss": float(policy_loss.item()),
        "value_loss": float(value_loss.item()),
    }
