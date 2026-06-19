"""样本分片与模型权重的读写抽象（对应 datagen 侧 store crate）。

把「样本分片 spool」和「版本化模型目录」的读写收敛到两个 Protocol 后面，第一阶段实现
本地目录版（LocalSampleStore / LocalModelStore）；未来要多机时再加 OSS 版同接口替换，
loop/exporter 等业务代码不动。

约定见 shared/shard-format.md 与 shared/model-format.md。
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

# 分片文件名：shard_{model_version:06d}_w{worker:02d}_{seq:06d}.st
_SHARD_RE = re.compile(r"^shard_\d{6}_w\d{2}_\d{6}\.st$")
# 模型文件名：model_{step:06d}.pt
_MODEL_RE = re.compile(r"^model_(\d{6})\.pt$")


class SampleStore(Protocol):
    """样本分片 spool 的读写接口。list 按名字典序（≈时间序）。"""

    def list_shards(self, after: str | None = None) -> list[str]: ...

    def get_shard(self, name: str) -> bytes: ...

    def put_shard(self, name: str, data: bytes) -> None: ...

    def delete_shard(self, name: str) -> None: ...


class ModelStore(Protocol):
    """版本化模型目录的读写接口。"""

    def put_model(self, version: int, data: bytes) -> None: ...

    def get_version(self) -> int | None: ...

    def get_latest(self) -> tuple[int, bytes] | None: ...


class LocalSampleStore:
    """本地目录实现的 SampleStore。只识别 `.st`，忽略 `.st.tmp` 半成品。"""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def list_shards(self, after: str | None = None) -> list[str]:
        names = sorted(
            entry.name
            for entry in self.root.iterdir()
            if entry.is_file() and _SHARD_RE.match(entry.name)
        )
        if after is not None:
            names = [n for n in names if n > after]
        return names

    def get_shard(self, name: str) -> bytes:
        return (self.root / name).read_bytes()

    def put_shard(self, name: str, data: bytes) -> None:
        _atomic_write(self.root / name, data)

    def delete_shard(self, name: str) -> None:
        (self.root / name).unlink(missing_ok=True)


class LocalModelStore:
    """本地目录实现的 ModelStore：版本化命名 + latest.json + 保留策略。"""

    def __init__(
        self,
        root: str | os.PathLike[str],
        keep_recent: int = 3,
        checkpoint_every: int = 2000,
        keep_checkpoints: int = 3,
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.keep_recent = keep_recent
        self.checkpoint_every = checkpoint_every
        self.keep_checkpoints = keep_checkpoints

    def put_model(self, version: int, data: bytes) -> None:
        """写 model_{version}.pt（不可变）→ 原子重写 latest.json → 执行保留策略。"""
        name = f"model_{version:06d}.pt"
        _atomic_write(self.root / name, data)

        pointer = {
            "version": version,
            "path": name,
            "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        _atomic_write(
            self.root / "latest.json",
            (json.dumps(pointer, ensure_ascii=False) + "\n").encode("utf-8"),
        )
        self._apply_retention(latest_version=version)

    def get_version(self) -> int | None:
        pointer = self._read_pointer()
        return None if pointer is None else int(pointer["version"])

    def get_latest(self) -> tuple[int, bytes] | None:
        pointer = self._read_pointer()
        if pointer is None:
            return None
        version = int(pointer["version"])
        data = (self.root / pointer["path"]).read_bytes()
        return version, data

    def _read_pointer(self) -> dict | None:
        path = self.root / "latest.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def _list_versions(self) -> list[int]:
        versions = []
        for entry in self.root.iterdir():
            m = _MODEL_RE.match(entry.name)
            if m:
                versions.append(int(m.group(1)))
        return sorted(versions)

    def _apply_retention(self, latest_version: int) -> None:
        """保留：最近 keep_recent 个 + 每 checkpoint_every 步的 checkpoint（最多 keep_checkpoints）
        + latest 指向版本，其余删除。两类计数独立，可重叠。"""
        versions = self._list_versions()
        keep: set[int] = set()

        keep.update(versions[-self.keep_recent :])

        if self.checkpoint_every > 0:
            checkpoints = [v for v in versions if v % self.checkpoint_every == 0]
            keep.update(checkpoints[-self.keep_checkpoints :])

        keep.add(latest_version)

        for v in versions:
            if v not in keep:
                (self.root / f"model_{v:06d}.pt").unlink(missing_ok=True)


def _atomic_write(path: Path, data: bytes) -> None:
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
