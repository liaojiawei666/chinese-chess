from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# 结构常量：与 Rust engine::constants 的镜像。
BOARD_WIDTH = 9
BOARD_HEIGHT = 10
SQUARE_COUNT = BOARD_WIDTH * BOARD_HEIGHT
ACTION_SPACE_SIZE = SQUARE_COUNT * SQUARE_COUNT
MAX_TOTAL_PLIES = 300
NO_CAPTURE_DRAW_PLIES = 100
N_HISTORY = 7
PLANES_PER_FRAME = 14
INPUT_CHANNELS = N_HISTORY * PLANES_PER_FRAME + 1  # 99


@dataclass(frozen=True)
class NetworkConfig:
    hidden_channels: int = 64
    residual_blocks: int = 4
    policy_head_channels: int = 32
    value_head_channels: int = 32
    value_fc_hidden: int = 128

    @property
    def input_channels(self) -> int:
        return INPUT_CHANNELS

    @property
    def action_space_size(self) -> int:
        return ACTION_SPACE_SIZE


@dataclass(frozen=True)
class MCTSConfig:
    n_simulations: int = 200
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    temperature_moves: int = 30


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    buffer_capacity: int = 10000
    min_buffer_size: int = 256
    target_reuse: float = 2.0
    weight_sync_interval: int = 50


@dataclass(frozen=True)
class DataGenConfig:
    samples_dir: str = "local/samples"
    models_dir: str = "local/models"
    num_parallel_games: int = 64
    eval_batch_size: int = 16
    inference_timeout_ms: float = 5.0
    shard_size: int = 4096


@dataclass(frozen=True)
class Config:
    device: str = "cpu"
    total_samples: int = 200_000
    network: NetworkConfig = field(default_factory=NetworkConfig)
    mcts: MCTSConfig = field(default_factory=MCTSConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    datagen: DataGenConfig = field(default_factory=DataGenConfig)


def _resolve_path(base_dir: Path, path: str) -> str:
    """相对路径以 config 文件所在目录为基准解析为绝对路径。"""
    p = Path(path)
    if p.is_absolute():
        return str(p)
    return str(base_dir / p)


def _from_dict(raw: dict[str, Any], base_dir: Path) -> Config:
    datagen_raw = dict(raw["datagen"])
    datagen_raw["samples_dir"] = _resolve_path(base_dir, datagen_raw["samples_dir"])
    datagen_raw["models_dir"] = _resolve_path(base_dir, datagen_raw["models_dir"])
    return Config(
        device=raw.get("device", "cpu"),
        total_samples=int(raw["total_samples"]),
        network=NetworkConfig(**raw["network"]),
        mcts=MCTSConfig(**raw["mcts"]),
        train=TrainConfig(**raw["train"]),
        datagen=DataGenConfig(**datagen_raw),
    )


def load_config(path: str | Path) -> Config:
    """从 JSON 配置文件加载。相对路径以 config 文件所在目录为基准解析。"""
    cfg_path = Path(path).resolve()
    base_dir = cfg_path.parent
    text = cfg_path.read_text(encoding="utf-8")
    return _from_dict(json.loads(text), base_dir)
