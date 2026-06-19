"""样本分片的读/写（与 datagen store crate 同一 safetensors 布局）。

生产路径只用 `read_shard`（datagen 负责写）；`write_shard` 是纯 Python 参考写盘，
供差分测试与无 Rust 时的兜底。布局规范见 shared/shard-format.md。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from safetensors.numpy import load as st_load
from safetensors.numpy import save as st_save

from .config import ACTION_SPACE_SIZE, BOARD_HEIGHT, BOARD_WIDTH, INPUT_CHANNELS


@dataclass
class TrainSample:
    """训练用样本：state 已升回 f32，pi 已还原成稠密动作分布，z 为走棋方视角胜负。

    与 reference.selfplay.Sample 鸭子类型兼容（.state/.pi/.z），可直接喂 ReplayBuffer。
    """

    state: np.ndarray  # (INPUT_CHANNELS, 10, 9) float32
    pi: np.ndarray  # (ACTION_SPACE_SIZE,) float32
    z: float


def read_shard(data: bytes) -> list[TrainSample]:
    """把一个分片字节流解析成样本列表：state /255 升 f32、CSR π 还原成稠密。"""
    tensors = st_load(data)
    states = tensors["state"].astype(np.float32) / 255.0
    pi_ptr = tensors["pi_ptr"]
    pi_idx = tensors["pi_idx"]
    pi_val = tensors["pi_val"]
    z = tensors["z"]

    n = states.shape[0]
    samples: list[TrainSample] = []
    for i in range(n):
        pi = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
        start, end = int(pi_ptr[i]), int(pi_ptr[i + 1])
        pi[pi_idx[start:end]] = pi_val[start:end]
        samples.append(TrainSample(state=states[i], pi=pi, z=float(z[i])))
    return samples


def write_shard(samples: list[TrainSample]) -> bytes:
    """参考写盘：把样本编成与 datagen 一致的 safetensors 布局（state uint8 + CSR π + z）。"""
    n = len(samples)
    state = np.zeros((n, INPUT_CHANNELS, BOARD_HEIGHT, BOARD_WIDTH), dtype=np.uint8)
    pi_ptr = np.zeros(n + 1, dtype=np.int32)
    pi_idx_parts: list[np.ndarray] = []
    pi_val_parts: list[np.ndarray] = []
    z = np.zeros(n, dtype=np.float32)

    offset = 0
    for i, s in enumerate(samples):
        state[i] = np.rint(np.clip(s.state, 0.0, 1.0) * 255.0).astype(np.uint8)
        nz = np.nonzero(s.pi)[0].astype(np.int32)
        pi_idx_parts.append(nz)
        pi_val_parts.append(s.pi[nz].astype(np.float32))
        offset += nz.shape[0]
        pi_ptr[i + 1] = offset
        z[i] = s.z

    pi_idx = (
        np.concatenate(pi_idx_parts) if pi_idx_parts else np.zeros(0, dtype=np.int32)
    )
    pi_val = (
        np.concatenate(pi_val_parts) if pi_val_parts else np.zeros(0, dtype=np.float32)
    )

    return st_save(
        {
            "state": state,
            "pi_ptr": pi_ptr,
            "pi_idx": pi_idx,
            "pi_val": pi_val,
            "z": z,
        }
    )
