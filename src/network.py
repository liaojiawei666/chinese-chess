from __future__ import annotations

import torch
from torch import nn

from config import BOARD_HEIGHT, BOARD_WIDTH, NetworkConfig


class ResidualBlock(nn.Module):
    """标准残差块：两层 3x3 卷积 + BN，残差相加后再 ReLU。"""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return torch.relu(out + x)


class PolicyValueNet(nn.Module):
    """中国象棋 AlphaZero 风格双头网络。

    纯张量进出：输入 canonical 局面张量，输出原始 policy logits 和 value。
    不在网络内部做合法动作 mask / softmax，这些交给 Evaluator 处理，
    以便 train.py 直接复用本网络计算梯度。
    """

    def __init__(self, config: NetworkConfig | None = None) -> None:
        super().__init__()
        self.config = config or NetworkConfig()
        c = self.config

        self.stem = nn.Sequential(
            nn.Conv2d(c.input_channels, c.hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c.hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.residual_tower = nn.Sequential(
            *(ResidualBlock(c.hidden_channels) for _ in range(c.residual_blocks))
        )

        # policy head：1x1 卷积压缩通道后展平，线性映射到固定动作空间。
        self.policy_conv = nn.Sequential(
            nn.Conv2d(c.hidden_channels, c.policy_head_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(c.policy_head_channels),
            nn.ReLU(inplace=True),
        )
        self.policy_fc = nn.Linear(
            c.policy_head_channels * BOARD_HEIGHT * BOARD_WIDTH, c.action_space_size
        )

        # value head：1x1 卷积压缩通道后展平，两层全连接 + Tanh 输出 [-1, 1]。
        self.value_conv = nn.Sequential(
            nn.Conv2d(c.hidden_channels, c.value_head_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(c.value_head_channels),
            nn.ReLU(inplace=True),
        )
        self.value_fc = nn.Sequential(
            nn.Linear(c.value_head_channels * BOARD_HEIGHT * BOARD_WIDTH, c.value_fc_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(c.value_fc_hidden, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (B, input_channels, 10, 9) -> (policy_logits [B, A], value [B, 1])。"""
        features = self.residual_tower(self.stem(x))

        policy = self.policy_conv(features)
        policy_logits = self.policy_fc(policy.flatten(start_dim=1))

        value = self.value_conv(features)
        value = self.value_fc(value.flatten(start_dim=1))
        return policy_logits, value
