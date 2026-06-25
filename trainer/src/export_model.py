#!/usr/bin/env python3
"""若 models_dir 尚无权重，导出随机初始化的 PolicyValueNet（ONNX）。

供 datagen 在缺少 latest.json 时 bootstrap。

用法：
    cd trainer && uv run python src/export_model.py --config ../config.json
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from config import load_config
from model_io import ModelIO
from network import PolicyValueNet

logger = logging.getLogger(__name__)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="配置文件路径")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    device = resolve_device(config.device)

    model_io = ModelIO(config.datagen.models_dir, keep_recent=3)

    if model_io.latest_version() is not None and not args.force:
        logger.info("已有模型 version=%s，跳过", model_io.latest_version())
        return

    net = PolicyValueNet(config.network)
    net.to(device)
    model_io.save(0, net, device)
    logger.info(
        "已导出随机初始权重 → %s/model_gen_0000.onnx (device=%s)",
        config.datagen.models_dir, device,
    )


if __name__ == "__main__":
    main()
