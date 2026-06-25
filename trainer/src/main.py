#!/usr/bin/env python3
"""训练主循环入口：消费 datagen 产的样本分片，训练 PolicyValueNet，版本化导出 ONNX 权重。

用法：
    cd trainer && uv run python src/main.py --config ../config.json
    cd trainer && uv run python src/main.py --config ../config.json --idle-poll-limit 3
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from config import load_config
from loop import run_training_loop
from model_io import ModelIO
from network import PolicyValueNet
from shard_io import ShardSource

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="配置文件路径")
    parser.add_argument("--idle-poll-limit", type=int, default=None)
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument("--log-interval", type=int, default=50)
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = parse_args()
    config = load_config(args.config)
    device = resolve_device(config.device)

    net = PolicyValueNet(config.network)
    optimizer = torch.optim.Adam(
        net.parameters(),
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
    )

    source = ShardSource(config.datagen.samples_dir)
    model_io = ModelIO(
        config.datagen.models_dir,
        keep_recent=3,
    )

    stats = run_training_loop(
        config,
        net,
        optimizer,
        source,
        model_io,
        device=device,
        idle_poll_limit=args.idle_poll_limit,
        poll_interval_s=args.poll_interval,
        log_interval=args.log_interval,
    )
    logger.info(
        "loop done: steps=%d shards=%d samples=%d last_loss=%.4f latest_model_version=%s",
        stats.steps, stats.shards_consumed, stats.samples_seen,
        stats.last_loss, model_io.latest_version(),
    )


if __name__ == "__main__":
    main()
