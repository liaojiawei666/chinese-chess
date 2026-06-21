#!/usr/bin/env python3
"""若 model_dir 尚无权重，导出随机初始化的 PolicyValueNet（TorchScript）。

供 datagen 在缺少 latest.json 时 bootstrap；与 train 启动时 export_initial 写出的
model_000000.pt 格式一致，Rust tch-rs 可直接加载。
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from trainer.config import default_config_path, load_config  # noqa: E402
from trainer.model_io import ModelIO  # noqa: E402
from trainer.network import PolicyValueNet  # noqa: E402

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
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="配置文件（默认 config/{CHESS_PROFILE}.json）",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="即使已有 latest.json 也重新导出 version=0",
    )
    args = parser.parse_args()

    config_path = args.config or default_config_path()
    config = load_config(config_path)
    device = resolve_device(config.device)

    model_io = ModelIO(
        config.datagen.model_dir,
        keep_recent=config.datagen.keep_recent_models,
        checkpoint_every=config.datagen.checkpoint_every,
        keep_checkpoints=config.datagen.keep_checkpoints,
    )

    if model_io.latest_version() is not None and not args.force:
        logger.info("已有模型 version=%s，跳过", model_io.latest_version())
        return

    net = PolicyValueNet(config.network)
    net.to(device)
    model_io.save(0, net, device)
    logger.info(
        "已导出随机初始权重 → %s/model_000000.pt（device=%s）",
        config.datagen.model_dir,
        device,
    )


if __name__ == "__main__":
    main()
