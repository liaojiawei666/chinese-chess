"""模型权重的 ONNX 导出 + 版本化存取 + 保留策略。

datagen(Rust) 通过 ort 加载 ONNX 模型。命名 model_gen_{generation:04d}.onnx，
latest.json 指向当前模型，格式见 docs/pipeline-protocol.md §2。
"""

from __future__ import annotations

import io
import json
import logging
import os
import re
from pathlib import Path

import torch
from torch import nn

from config import BOARD_HEIGHT, BOARD_WIDTH, INPUT_CHANNELS
from shard_io import atomic_write

logger = logging.getLogger(__name__)

_MODEL_RE = re.compile(r"^model_gen_(\d{4})\.onnx$")


def onnx_bytes(net: nn.Module) -> bytes:
    """将网络导出为 ONNX 字节流。始终在 CPU 上导出。"""
    was_training = net.training
    orig_device = next(net.parameters()).device
    net.eval()
    cpu_net = net.cpu()

    example = torch.zeros(1, INPUT_CHANNELS, BOARD_HEIGHT, BOARD_WIDTH)
    buffer = io.BytesIO()
    torch.onnx.export(
        cpu_net,
        (example,),
        buffer,
        input_names=["state"],
        output_names=["policy", "value"],
        dynamic_axes={"state": {0: "batch"}, "policy": {0: "batch"}, "value": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )

    if was_training:
        net.train()
    net.to(orig_device)
    return buffer.getvalue()


class ModelIO:
    """版本化 ONNX 模型管理：导出 + latest.json 指针 + 保留策略。"""

    def __init__(
        self,
        model_dir: str | os.PathLike[str],
        keep_recent: int = 3,
    ) -> None:
        self.root = Path(model_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.keep_recent = keep_recent

    def save(self, generation: int, net: nn.Module, device: torch.device | str = "cpu") -> None:
        _ = device
        self.put_model(generation, onnx_bytes(net))

    def put_model(self, generation: int, data: bytes) -> None:
        name = f"model_gen_{generation:04d}.onnx"
        model_path = self.root / name
        atomic_write(model_path, data)

        size_mb = len(data) / (1024 * 1024)
        logger.info(
            "exported model gen=%d -> %s (%.1f MB)",
            generation, model_path, size_mb,
        )

        pointer = {
            "generation": generation,
            "model_path": str(model_path),
        }
        atomic_write(
            self.root / "latest.json",
            (json.dumps(pointer, ensure_ascii=False) + "\n").encode("utf-8"),
        )
        self._apply_retention(latest_generation=generation)

    def latest_version(self) -> int | None:
        pointer = self._read_pointer()
        return None if pointer is None else int(pointer["generation"])

    def load_latest(self) -> tuple[int, bytes] | None:
        pointer = self._read_pointer()
        if pointer is None:
            return None
        generation = int(pointer["generation"])
        model_path = Path(pointer["model_path"])
        if not model_path.is_absolute():
            model_path = self.root / model_path
        data = model_path.read_bytes()
        return generation, data

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

    def _apply_retention(self, latest_generation: int) -> None:
        versions = self._list_versions()
        keep: set[int] = set()
        keep.update(versions[-self.keep_recent:])
        keep.add(latest_generation)

        removed = []
        for v in versions:
            if v not in keep:
                (self.root / f"model_gen_{v:04d}.onnx").unlink(missing_ok=True)
                removed.append(v)

        kept = sorted(keep & set(versions))
        if removed:
            logger.info("retention: kept %s, removed %s", kept, removed)
        else:
            logger.debug("retention: kept %s, nothing to remove", kept)
