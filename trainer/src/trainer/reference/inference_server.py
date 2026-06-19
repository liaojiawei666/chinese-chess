from __future__ import annotations

import time
from queue import Empty

import numpy as np
import torch

from ..config import Config
from .evaluator import resolve_device
from ..network import PolicyValueNet


def _load_numpy_state_dict(net: PolicyValueNet, sd: dict) -> None:
    """把 numpy 形式的权重还原成张量后加载（与 pipeline 的传输格式对应）。"""
    net.load_state_dict({k: torch.from_numpy(v) for k, v in sd.items()})
    net.eval()


def serve(
    config: Config,
    init_state_dict: dict,
    request_q,
    response_qs: list,
    weight_q,
    stop_event,
    ready_event,
) -> None:
    """推理服务进程入口：独占一份网络在 device 上，聚合各 worker 的评估请求成 batch 前向。

    设计要点：
    - 跨 worker 凑批：每个 worker 同时只有 1 个在途请求，服务端把它们攒成一个 batch，
      把单局面 batch=1 的 GPU 浪费拉平为 batch≈num_workers，喂满显卡。
    - 权重热更新：训练进程每隔若干步把最新权重塞进 weight_q，这里每轮循环取最新的加载，
      让自对弈逐渐用上更强的网络（AlphaZero 的 actor 始终追最新参数）。
    - 关停：stop_event 由主进程在所有 worker 收口后才置位，保证服务在 worker 还需推理时不退。
    """
    device = resolve_device(config.device)
    net = PolicyValueNet(config.network).to(device)
    _load_numpy_state_dict(net, init_state_dict)

    max_batch = max(1, config.selfplay.eval_batch_size)
    timeout = config.selfplay.inference_timeout_ms / 1000.0
    ready_event.set()

    while not stop_event.is_set():
        # 1) 取最新权重（多条只保留最后一条）。
        latest_sd = None
        while True:
            try:
                latest_sd = weight_q.get_nowait()
            except Empty:
                break
        if latest_sd is not None:
            _load_numpy_state_dict(net, latest_sd)

        # 2) 凑一个 batch：先阻塞等首个请求（短超时以便周期性检查 stop_event），
        #    拿到后在 timeout 窗口内尽量多凑，直到攒满 max_batch。
        try:
            first = request_q.get(timeout=0.05)
        except Empty:
            continue
        batch = [first]
        deadline = time.monotonic() + timeout
        while len(batch) < max_batch:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                batch.append(request_q.get(timeout=remaining))
            except Empty:
                break

        # 3) 前向，把结果分发回各自的 response_q。
        states = np.stack([item[1] for item in batch])
        with torch.no_grad():
            logits_t, value_t = net(torch.from_numpy(states).to(device))
        logits = logits_t.cpu().numpy()
        values = value_t.squeeze(-1).cpu().numpy()
        for i, (worker_id, _) in enumerate(batch):
            response_qs[worker_id].put((logits[i], float(values[i])))
