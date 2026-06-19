"""训练主循环：吃 datagen 产的分片 → ReplayBuffer 滑动窗口 → train_step → 定期导出权重。

取代旧 reference/pipeline.py 的「推理服务 + worker」内置自对弈那套：自对弈已外移到
datagen(Rust)，trainer 只负责消费分片、训练、版本化导出模型，两侧经磁盘 Store 解耦。
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from .config import Config
from .exporter import export_model
from .replay import ReplayBuffer
from .shard_io import read_shard
from .store import ModelStore, SampleStore
from .train import train_step


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
    sample_store: SampleStore,
    model_store: ModelStore,
    *,
    device: torch.device | str = "cpu",
    max_steps: int | None = None,
    idle_poll_limit: int | None = None,
    poll_interval_s: float = 1.0,
    export_initial: bool = True,
) -> LoopStats:
    """消费端按 reuse 预算驱动的滑动窗口训练循环。

    控制权全在消费端：reuse（每条样本平均被抽取次数）是与硬件无关的算法超参，
    窗口推进、收尾收缩、终止都由 `total_samples` + `target_reuse` 决定，生产端只管产。

    单一数据预算 `config.total_samples`：
      - 正常段：维持 reuse 不变量 `steps*batch <= reuse*consumed`，不够就拉新分片喂窗口；
        未消化的分片留在盘上当 backlog（撞 max_pending_shards 时反压生产）。
      - 收尾段：累计消费满 total_samples 后不再拉新，每步把窗口从最旧端收缩 batch/reuse 条
        （右侧已无新样本，等价于"继续右滑、截断"），窗口缩到 < min_buffer_size 即结束。
        收缩期对最新样本的额外抽取，正好补回它们稳态期的尾部欠采。

    `max_steps` / `idle_poll_limit` 仅作 smoke/安全早停：
      - `max_steps`：到步数即停（None 表示只按数据预算收口）。
      - `idle_poll_limit`：连续多少次「数据饿等待」后停（None 表示长驻，永不因空闲停）。
    """
    device = torch.device(device) if isinstance(device, str) else device
    net.to(device)

    tc = config.train
    batch = tc.batch_size
    reuse = tc.target_reuse
    min_buf = tc.min_buffer_size
    total_samples = config.total_samples
    # 收尾每步从最旧端淘汰多少（= 稳态每步窗口推进量 batch/reuse）。
    advance_per_step = max(1, round(batch / reuse))
    export_interval = config.datagen.model_export_interval

    buffer = ReplayBuffer(tc.buffer_capacity)
    stats = LoopStats()
    consumed = 0  # 累计拉入窗口的样本数（单调；= 数据预算进度）
    idle_polls = 0

    # 先导出一份初始权重（version=0），让 datagen 一启动就有模型可热加载。
    if export_initial:
        export_model(model_store, net, 0, device)

    def wait_or_stop() -> bool:
        """数据饿/冷启动时等待；返回 True 表示应终止（达到 idle 上限）。"""
        nonlocal idle_polls
        idle_polls += 1
        if idle_poll_limit is not None and idle_polls >= idle_poll_limit:
            return True
        time.sleep(poll_interval_s)
        return False

    while True:
        if max_steps is not None and stats.steps >= max_steps:
            break

        within_budget = (stats.steps + 1) * batch <= reuse * consumed
        production_done = consumed >= total_samples
        # 需要更多数据：reuse 预算不够，或窗口还没到地板（冷启动）。
        need_data = not within_budget or len(buffer) < min_buf

        # 正常段：拉新分片喂窗口（够用即止，余量留盘上当 backlog 触发生产端反压）。
        if need_data and not production_done:
            available = sample_store.list_shards()  # 未消费即未删；不用游标，避免乱序漏读
            if not available:
                if wait_or_stop():
                    break
                continue
            idle_polls = 0
            for name in available:
                samples = read_shard(sample_store.get_shard(name))
                buffer.add(samples)
                sample_store.delete_shard(name)
                consumed += len(samples)
                stats.shards_consumed += 1
                stats.samples_seen += len(samples)
                enough_budget = (stats.steps + 1) * batch <= reuse * consumed
                if consumed >= total_samples or (enough_budget and len(buffer) >= min_buf):
                    break
            continue

        # 收尾段：预算已耗尽且消费满 total_samples → 收缩窗口（右侧无新样本，截断）。
        if not within_budget and production_done:
            buffer.evict_oldest(advance_per_step)

        # 窗口不足地板：收尾则结束（已收缩到底），否则数据饿等待。
        if len(buffer) < min_buf:
            if production_done:
                break
            if wait_or_stop():
                break
            continue

        # 训练一步。
        idle_polls = 0
        batch_data = buffer.sample(batch)
        metrics = train_step(net, optimizer, batch_data, device, tc)
        stats.steps += 1
        stats.last_loss = metrics["loss"]
        if export_interval > 0 and stats.steps % export_interval == 0:
            export_model(model_store, net, stats.steps, device)

    # 收尾导出最终权重（除非刚好已在该步导出）。
    if export_interval <= 0 or stats.steps % export_interval != 0:
        export_model(model_store, net, stats.steps, device)
    return stats
