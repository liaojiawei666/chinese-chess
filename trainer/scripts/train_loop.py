#!/usr/bin/env python3
"""训练主循环入口：消费 datagen 产的样本分片，训练 PolicyValueNet，版本化导出权重。

读取档位（CHESS_PROFILE），按 DataGenConfig 的目录连接本地分片源 / 模型目录。终止由
total_samples + reuse 决定（与 datagen 并跑时长驻）；--idle-poll-limit 仅作数据饿早停（smoke）。

用法：
    CHESS_PROFILE=local python trainer/scripts/train_loop.py --idle-poll-limit 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from trainer.config import load_config  # noqa: E402
from trainer.loop import run_training_loop  # noqa: E402
from trainer.model_io import ModelIO  # noqa: E402
from trainer.network import PolicyValueNet  # noqa: E402
from trainer.shard_io import ShardSource  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--idle-poll-limit",
        type=int,
        default=None,
        help="连续多少次无新数据后停（smoke 用；省略则长驻）",
    )
    parser.add_argument("--poll-interval", type=float, default=1.0, help="空闲轮询间隔（秒）")
    parser.add_argument("--log-interval", type=int, default=50, help="每多少步打印一次训练日志")
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
    args = parse_args()
    config = load_config()
    device = resolve_device(config.device)

    net = PolicyValueNet(config.network)
    optimizer = torch.optim.Adam(
        net.parameters(),
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
    )

    source = ShardSource(config.datagen.samples_dir)
    model_io = ModelIO(
        config.datagen.model_dir,
        keep_recent=config.datagen.keep_recent_models,
        checkpoint_every=config.datagen.checkpoint_every,
        keep_checkpoints=config.datagen.keep_checkpoints,
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
    print(
        f"loop done: steps={stats.steps} shards={stats.shards_consumed} "
        f"samples={stats.samples_seen} last_loss={stats.last_loss:.4f} "
        f"latest_model_version={model_io.latest_version()}"
    )


if __name__ == "__main__":
    main()
