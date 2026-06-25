"""Shard 二进制格式的编解码 + 目录存取。

格式见 docs/pipeline-protocol.md §3。datagen（Rust）写入，trainer（Python）读取。
"""

from __future__ import annotations

import os
import re
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from config import ACTION_SPACE_SIZE, NO_CAPTURE_DRAW_PLIES

MAGIC = 0x4358_5347
PIECES_SIZE = 8820  # channels 0-97: 98 × 10 × 9

_SHARD_RE = re.compile(r"^shard_\d{6}\.bin$")


@dataclass
class CompactSample:
    """压缩存储的训练样本，直接进 replay buffer。

    解压为 float32 tensor 仅在采样 batch 时发生。
    """

    state_pieces: np.ndarray  # (8820,) uint8, binary 0/1
    no_capture_plies: np.uint8
    policy_ids: np.ndarray  # (count,) uint16
    policy_probs: np.ndarray  # (count,) float32
    value: float


def decode_shard(data: bytes) -> list[CompactSample]:
    """读取 CXSG 二进制 shard，返回压缩样本列表。"""
    magic, n = struct.unpack_from("<II", data, 0)
    if magic != MAGIC:
        raise ValueError(f"bad shard magic: 0x{magic:08X}, expected 0x{MAGIC:08X}")

    samples: list[CompactSample] = []
    off = 8
    for _ in range(n):
        state_pieces = np.frombuffer(data, dtype=np.uint8, count=PIECES_SIZE, offset=off).copy()
        off += PIECES_SIZE

        no_capture_plies = data[off]
        off += 1

        (count,) = struct.unpack_from("<H", data, off)
        off += 2

        policy_ids = np.frombuffer(data, dtype=np.uint16, count=count, offset=off).copy()
        off += count * 2

        policy_probs = np.frombuffer(data, dtype=np.float32, count=count, offset=off).copy()
        off += count * 4

        (value,) = struct.unpack_from("<f", data, off)
        off += 4

        samples.append(CompactSample(
            state_pieces=state_pieces,
            no_capture_plies=np.uint8(no_capture_plies),
            policy_ids=policy_ids,
            policy_probs=policy_probs,
            value=float(value),
        ))
    return samples


def encode_shard(samples: list[CompactSample]) -> bytes:
    """将压缩样本列表编码为 CXSG 二进制（测试 / 兜底用）。"""
    buf = bytearray()
    buf.extend(struct.pack("<II", MAGIC, len(samples)))

    for s in samples:
        buf.extend(s.state_pieces.tobytes())
        buf.append(int(s.no_capture_plies))

        count = len(s.policy_ids)
        buf.extend(struct.pack("<H", count))
        buf.extend(s.policy_ids.astype(np.uint16).tobytes())
        buf.extend(s.policy_probs.astype(np.float32).tobytes())

        buf.extend(struct.pack("<f", s.value))

    return bytes(buf)


def decompress_batch(
    samples: list[CompactSample],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """批量解压为训练用 float32 tensor。

    Returns: (states, pis, zs)
      states: (n, 99, 10, 9) float32
      pis:    (n, ACTION_SPACE_SIZE) float32
      zs:     (n,) float32
    """
    n = len(samples)
    states = np.zeros((n, 99, 10, 9), dtype=np.float32)
    pis = np.zeros((n, ACTION_SPACE_SIZE), dtype=np.float32)
    zs = np.empty(n, dtype=np.float32)

    for i, s in enumerate(samples):
        states[i, :98] = s.state_pieces.reshape(98, 10, 9).astype(np.float32)
        states[i, 98] = float(s.no_capture_plies) / NO_CAPTURE_DRAW_PLIES
        pis[i, s.policy_ids] = s.policy_probs
        zs[i] = s.value

    return states, pis, zs


class ShardSource:
    """绑定一个本地样本目录的分片源。"""

    def __init__(
        self,
        samples_dir: str | os.PathLike[str],
        archive_subdir: str = "archive",
    ) -> None:
        self.root = Path(samples_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self._archive = self.root / archive_subdir
        self._archive.mkdir(parents=True, exist_ok=True)

    def list_shard(self) -> list[str]:
        return sorted(
            entry.name
            for entry in self.root.iterdir()
            if entry.is_file() and _SHARD_RE.match(entry.name)
        )

    def read_shard(self, name: str) -> list[CompactSample]:
        return decode_shard((self.root / name).read_bytes())

    def archive_shard(self, name: str) -> None:
        src = self.root / name
        if src.exists():
            import shutil
            shutil.move(str(src), str(self._archive / name))

    def write_shard(self, name: str, samples: list[CompactSample]) -> None:
        atomic_write(self.root / name, encode_shard(samples))


def atomic_write(path: Path, data: bytes) -> None:
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
