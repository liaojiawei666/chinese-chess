from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# 结构常量：与 Rust engine const 的镜像。Python 代码直接 import。
BOARD_WIDTH = 9
BOARD_HEIGHT = 10
SQUARE_COUNT = BOARD_WIDTH * BOARD_HEIGHT
ACTION_SPACE_SIZE = SQUARE_COUNT * SQUARE_COUNT
MAX_TOTAL_PLIES = 300
NO_CAPTURE_DRAW_PLIES = 100
N_HISTORY = 7
PLANES_PER_FRAME = 14
INPUT_CHANNELS = N_HISTORY * PLANES_PER_FRAME + 1  # 99

_REPO_ROOT = Path(__file__).resolve().parents[3]
PROFILE_ENV_VAR = "CHESS_PROFILE"
DEFAULT_PROFILE = "local"


def _detect_default_profile() -> str:
    """自动检测 GPU：有 CUDA 用 gpu 配置，否则 local。"""
    try:
        import torch
        if torch.cuda.is_available():
            return "gpu"
    except ImportError:
        pass
    return "local"


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
    steps_per_iteration: int = 20
    weight_sync_interval: int = 50


@dataclass(frozen=True)
class DataGenConfig:
    samples_dir: str = "data/samples"
    model_dir: str = "data/models"
    num_workers: int = 4
    games_per_worker: int = 1
    eval_batch_size: int = 16
    inference_timeout_ms: float = 5.0
    shard_games: int = 8
    max_pending_shards: int = 64
    model_export_interval: int = 100
    keep_recent_models: int = 3
    checkpoint_every: int = 500
    keep_checkpoints: int = 3


@dataclass(frozen=True)
class Config:
    device: str = "cpu"
    total_samples: int = 200_000
    profile: str = DEFAULT_PROFILE
    network: NetworkConfig = field(default_factory=NetworkConfig)
    mcts: MCTSConfig = field(default_factory=MCTSConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    datagen: DataGenConfig = field(default_factory=DataGenConfig)


def default_config_path(profile: str | None = None) -> Path:
    """默认配置文件路径：config/{profile}.json。

    优先级：显式传入 > 环境变量 CHESS_PROFILE > 自动检测 GPU。
    """
    p = profile or os.environ.get(PROFILE_ENV_VAR) or _detect_default_profile()
    return _REPO_ROOT / "config" / f"{p}.json"


def _from_dict(raw: dict[str, Any]) -> Config:
    return Config(
        device=raw["device"],
        total_samples=int(raw["total_samples"]),
        profile=raw.get("profile", DEFAULT_PROFILE),
        network=NetworkConfig(**raw["network"]),
        mcts=MCTSConfig(**raw["mcts"]),
        train=TrainConfig(**raw["train"]),
        datagen=DataGenConfig(**raw["datagen"]),
    )


def load_config(path: str | Path | None = None) -> Config:
    """从 JSON 配置文件加载（默认 config/{CHESS_PROFILE}.json）。"""
    if path is None:
        cfg_path = default_config_path()
    else:
        cfg_path = Path(path)
        if not cfg_path.is_absolute():
            cfg_path = cfg_path.resolve()
    text = cfg_path.read_text(encoding="utf-8")
    return _from_dict(json.loads(text))
