"""样本分片的编解码 + 目录存取（与 datagen 侧 store crate 同一 safetensors 布局）。

- 纯编解码：`decode_shard` / `encode_shard`（state uint8↔f32、稀疏 CSR π↔稠密）。
- `ShardSource`：绑定一个样本目录，暴露 `list_shard` / `read_shard` / `archive_shard`，
  正好是 `SampleLoader` 要注入的「分片源」。读完的分片移到 `consumed/` 子目录归档。
  `write_shard` 仅供差分测试 / 无 Rust 时兜底。

布局规范见 shared/shard-format.md。
"""

from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from safetensors.numpy import load as st_load
from safetensors.numpy import save as st_save

from .config import ACTION_SPACE_SIZE, BOARD_HEIGHT, BOARD_WIDTH, INPUT_CHANNELS

# 分片文件名：shard_{model_version:06d}_w{worker:02d}_{seq:06d}.st
_SHARD_RE = re.compile(r"^shard_\d{6}_w\d{2}_\d{6}\.st$")


@dataclass
class TrainSample:
    """训练用样本：state 已升回 f32，pi 已还原成稠密动作分布，z 为走棋方视角胜负。

    decode_shard 解析后产出，直接喂 ReplayBuffer。
    """

    state: np.ndarray  # (INPUT_CHANNELS, 10, 9) float32
    pi: np.ndarray  # (ACTION_SPACE_SIZE,) float32
    z: float


def decode_shard(data: bytes) -> list[TrainSample]:
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


def encode_shard(samples: list[TrainSample]) -> bytes:
    """参考编码：把样本编成与 datagen 一致的 safetensors 布局（state uint8 + CSR π + z）。"""
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

    pi_idx = np.concatenate(pi_idx_parts) if pi_idx_parts else np.zeros(0, dtype=np.int32)
    pi_val = np.concatenate(pi_val_parts) if pi_val_parts else np.zeros(0, dtype=np.float32)

    return st_save(
        {
            "state": state,
            "pi_ptr": pi_ptr,
            "pi_idx": pi_idx,
            "pi_val": pi_val,
            "z": z,
        }
    )


class ShardSource:
    """绑定一个本地样本目录的分片源。只识别 `.st`，忽略 `.st.tmp` 半成品。

    暴露 `list_shard` / `read_shard` / `archive_shard` 供 SampleLoader 注入；未来要换 OSS，
    只需另写一个同方法的类替换它，SampleLoader 不变。
    """

    def __init__(
        self,
        samples_dir: str | os.PathLike[str],
        archive_subdir: str = "consumed",
    ) -> None:
        self.root = Path(samples_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self._archive = self.root / archive_subdir
        self._archive.mkdir(parents=True, exist_ok=True)

    def list_shard(self) -> list[str]:
        """按文件名字典序（≈时间序）列出全部 `.st` 分片。"""
        return sorted(
            entry.name
            for entry in self.root.iterdir()
            if entry.is_file() and _SHARD_RE.match(entry.name)
        )

    def read_shard(self, name: str) -> list[TrainSample]:
        """读一个分片并解码成样本列表。"""
        return decode_shard((self.root / name).read_bytes())

    def archive_shard(self, name: str) -> None:
        """消费完毕后移到 consumed/ 子目录（替代直接删除，便于调试/回溯）。"""
        src = self.root / name
        if src.exists():
            import shutil
            shutil.move(str(src), str(self._archive / name))

    def write_shard(self, name: str, samples: list[TrainSample]) -> None:
        """参考写盘（测试/兜底）：编码后原子落盘。生产里分片由 datagen 写。"""
        atomic_write(self.root / name, encode_shard(samples))


def atomic_write(path: Path, data: bytes) -> None:
    """先写同目录临时文件并 fsync，再 rename 成目标（本地 rename 原子）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
