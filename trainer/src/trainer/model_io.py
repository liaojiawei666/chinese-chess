"""模型权重的 TorchScript 打包 + 版本化存取 + 保留策略（对应 datagen 侧 store crate）。

datagen(Rust) 用 tch-rs 加载 `model_{step}.pt`，故必须是可脱离 Python 的 TorchScript
（优先 jit.script，个别脚本不友好写法退化为 jit.trace）。目录布局与 latest.json 约定、
保留策略见 shared/model-format.md。
"""

from __future__ import annotations

import io
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import nn

from .config import BOARD_HEIGHT, BOARD_WIDTH, INPUT_CHANNELS
from .shard_io import atomic_write

# 模型文件名：model_{step:06d}.pt
_MODEL_RE = re.compile(r"^model_(\d{6})\.pt$")


def scripted_bytes(net: nn.Module, device: torch.device | str = "cpu") -> bytes:
    """把网络转成 TorchScript 字节流。

    始终在 CPU 上 script/trace，便于 Rust tch-rs 用 load_on_device 迁到 CUDA；
    若在 GPU 上直接导出，反序列化时可能因 CUDA 后端未初始化而失败。
    """
    was_training = net.training
    orig_device = next(net.parameters()).device
    net.eval()
    cpu_net = net.cpu()
    try:
        module = torch.jit.script(cpu_net)
    except Exception:
        example = torch.zeros(1, INPUT_CHANNELS, BOARD_HEIGHT, BOARD_WIDTH, device="cpu")
        module = torch.jit.trace(cpu_net, example)
    buffer = io.BytesIO()
    torch.jit.save(module, buffer)
    if was_training:
        net.train()
    net.to(orig_device)
    return buffer.getvalue()


class ModelIO:
    """绑定一个本地模型目录：版本化保存 TorchScript 权重 + latest.json 指针 + 保留策略。

    暴露 `save`（打包+落盘）/ `put_model`（字节级落盘）/ `latest_version` / `load_latest`。
    未来要换 OSS，另写一个同方法的类替换即可。
    """

    def __init__(
        self,
        model_dir: str | os.PathLike[str],
        keep_recent: int = 3,
        checkpoint_every: int = 2000,
        keep_checkpoints: int = 3,
    ) -> None:
        self.root = Path(model_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.keep_recent = keep_recent
        self.checkpoint_every = checkpoint_every
        self.keep_checkpoints = keep_checkpoints

    def save(self, step: int, net: nn.Module, device: torch.device | str = "cpu") -> None:
        """打包当前权重为 model_{step}.pt 并更新 latest.json（含保留策略）。"""
        # device 仅保留签名兼容；TorchScript 导出总在 CPU 上完成。
        _ = device
        self.put_model(step, scripted_bytes(net))

    def put_model(self, step: int, data: bytes) -> None:
        """字节级落盘：写 model_{step}.pt（不可变）→ 原子重写 latest.json → 执行保留策略。"""
        name = f"model_{step:06d}.pt"
        atomic_write(self.root / name, data)

        pointer = {
            "version": step,
            "path": name,
            "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        atomic_write(
            self.root / "latest.json",
            (json.dumps(pointer, ensure_ascii=False) + "\n").encode("utf-8"),
        )
        self._apply_retention(latest_version=step)

    def latest_version(self) -> int | None:
        pointer = self._read_pointer()
        return None if pointer is None else int(pointer["version"])

    def load_latest(self) -> tuple[int, bytes] | None:
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
