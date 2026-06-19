from __future__ import annotations

from queue import Empty, Full

import numpy as np
import torch
import torch.multiprocessing as mp

from ..config import Config, load_config
from .evaluator import resolve_device
from .inference_server import serve
from ..network import PolicyValueNet
from ..replay import ReplayBuffer
from .selfplay import worker_loop
from ..train import train_step


def _state_dict_to_numpy(net: torch.nn.Module) -> dict:
    """把权重转成 numpy 再跨进程传给推理服务。

    用 numpy 而非 torch 张量，是为了走标准 pickle、绕开 torch 的共享内存管理器
    （某些受限环境下 torch_shm_manager 不可用），跨平台更稳。
    """
    return {k: v.detach().cpu().numpy() for k, v in net.state_dict().items()}


def run(config: Config | None = None, seed: int = 0) -> PolicyValueNet:
    """异步自对弈 ↔ 训练流水线。

    架构：
    - 推理服务进程（serve）：独占网络做批量推理，被所有 worker 共享。
    - N 个自对弈 worker 进程：纯 CPU 跑规则引擎 + MCTS，叶子评估走推理服务。
    - 主进程（本函数）：兼训练器，吃 worker 产出的样本进 replay buffer、跑训练步，
      并周期性把最新权重推给推理服务。训练（主进程）与推理（服务进程）并发共享 GPU。

    终止条件：累计自对弈样本数达到 config.total_samples。返回训练后的网络。

    注：本文件是被 trainer/loop.py 取代的旧版「进程内推理服务 + worker」参考实现，
    新流水线（datagen 产数据 + loop.py 按 reuse 预算消费）已不再走这条路。
    """
    config = config or load_config()
    ctx = mp.get_context("spawn")
    device = resolve_device(config.device)
    torch.manual_seed(seed)

    net = PolicyValueNet(config.network).to(device)
    optimizer = torch.optim.SGD(
        net.parameters(),
        lr=config.train.learning_rate,
        momentum=0.9,
        weight_decay=config.train.weight_decay,
    )
    rng = np.random.default_rng(seed)
    buffer = ReplayBuffer(config.train.buffer_capacity, rng)

    n_workers = config.selfplay.num_workers
    request_q = ctx.Queue()
    response_qs = [ctx.Queue() for _ in range(n_workers)]
    sample_q = ctx.Queue()
    weight_q = ctx.Queue(maxsize=4)
    worker_stop = ctx.Event()  # 通知 worker 不再开新局
    server_stop = ctx.Event()  # 通知推理服务退出（worker 全部收口后才置位）
    ready = ctx.Event()        # 推理服务就绪信号

    server = ctx.Process(
        target=serve,
        args=(config, _state_dict_to_numpy(net), request_q, response_qs,
              weight_q, server_stop, ready),
        daemon=True,
    )
    server.start()
    ready.wait(timeout=120)

    workers = [
        ctx.Process(
            target=worker_loop,
            args=(i, config, seed + 1 + i, request_q, response_qs[i],
                  sample_q, worker_stop),
            daemon=True,
        )
        for i in range(n_workers)
    ]
    for w in workers:
        w.start()

    print(f"device={device} workers={n_workers} total_samples={config.total_samples}")
    samples_seen = 0
    steps = 0
    try:
        while samples_seen < config.total_samples:
            # 把当前可取的整局样本全部吃进 buffer（无则阻塞等一局）。
            drained = 0
            while True:
                try:
                    game_samples = sample_q.get_nowait()
                    buffer.add(game_samples)
                    samples_seen += len(game_samples)
                    drained += 1
                except Empty:
                    break
            if drained == 0:
                try:
                    game_samples = sample_q.get(timeout=2.0)
                    buffer.add(game_samples)
                    samples_seen += len(game_samples)
                except Empty:
                    continue

            if len(buffer) < config.train.min_buffer_size:
                continue

            last: dict[str, float] = {}
            for _ in range(config.train.steps_per_iteration):
                batch = buffer.sample(config.train.batch_size)
                last = train_step(net, optimizer, batch, device, config.train)
                steps += 1
                if steps % config.train.weight_sync_interval == 0:
                    try:
                        weight_q.put_nowait(_state_dict_to_numpy(net))
                    except Full:
                        pass  # 服务端还没消费完，跳过这次同步，下次再推
            print(
                f"samples={samples_seen} steps={steps} buffer={len(buffer)} "
                f"loss={last['loss']:.4f} policy={last['policy_loss']:.4f} "
                f"value={last['value_loss']:.4f}"
            )
    finally:
        worker_stop.set()
        for w in workers:
            w.join(timeout=30)
        server_stop.set()
        server.join(timeout=30)
        for p in (server, *workers):
            if p.is_alive():
                p.terminate()

    return net


if __name__ == "__main__":
    run()
