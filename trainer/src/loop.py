"""训练主循环：迭代 SampleLoader → train_step → 定期版本化导出权重。"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)

from config import Config
from loader import SampleLoader, ShardSourceLike
from model_io import ModelIO
from train import train_step


@dataclass
class LoopStats:
    steps: int = 0
    shards_consumed: int = 0
    samples_seen: int = 0
    last_loss: float = 0.0


def run_training_loop(
    config: Config,
    net: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    source: ShardSourceLike,
    model_io: ModelIO,
    *,
    device: torch.device | str = "cpu",
    idle_poll_limit: int | None = None,
    poll_interval_s: float = 1.0,
    export_initial: bool = True,
    log_interval: int = 50,
) -> LoopStats:
    device = torch.device(device) if isinstance(device, str) else device
    net.to(device)

    tc = config.train
    export_interval = tc.weight_sync_interval

    loader = SampleLoader(
        source,
        total_samples=config.total_samples,
        target_reuse=tc.target_reuse,
        batch_size=tc.batch_size,
        buffer_capacity=tc.buffer_capacity,
        min_buffer_size=tc.min_buffer_size,
        idle_poll_limit=idle_poll_limit,
        poll_interval_s=poll_interval_s,
    )

    if export_initial:
        model_io.save(0, net, device)
        logger.info("exported initial model gen=0")

    stats = LoopStats()
    log_interval = max(1, log_interval)
    t_last = time.perf_counter()
    trained_last = 0
    loss_sum = policy_sum = value_sum = 0.0

    for batch in loader:
        metrics = train_step(net, optimizer, batch, device, tc)
        stats.steps += 1
        stats.last_loss = metrics["loss"]
        loss_sum += metrics["loss"]
        policy_sum += metrics["policy_loss"]
        value_sum += metrics["value_loss"]

        if stats.steps % log_interval == 0:
            now = time.perf_counter()
            dt = max(now - t_last, 1e-6)
            trained = stats.steps * tc.batch_size
            throughput = (trained - trained_last) / dt
            steps_per_s = log_interval / dt
            logger.info(
                "[train] step %d | loss %.3f (p %.3f v %.3f)"
                " | trained %d | ingested %d/%d"
                " | %.0f samples/s, %.1f steps/s",
                stats.steps, loss_sum / log_interval,
                policy_sum / log_interval, value_sum / log_interval,
                trained, loader.samples_seen, config.total_samples,
                throughput, steps_per_s,
            )
            loss_sum = policy_sum = value_sum = 0.0
            t_last = now
            trained_last = trained

        if export_interval > 0 and stats.steps % export_interval == 0:
            model_io.save(stats.steps, net, device)
            logger.info(
                "model exported at step %d (gen=%d), buffer: %d/%d",
                stats.steps, stats.steps,
                len(loader.buffer), loader.buffer.capacity,
            )

    stats.shards_consumed = loader.shards_consumed
    stats.samples_seen = loader.samples_seen

    if export_interval <= 0 or stats.steps % export_interval != 0:
        model_io.save(stats.steps, net, device)
        logger.info("final model exported at step %d", stats.steps)

    logger.info(
        "training complete: %d steps, %d shards, %d samples ingested",
        stats.steps, stats.shards_consumed, stats.samples_seen,
    )
    return stats
