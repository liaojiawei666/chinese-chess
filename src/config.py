from __future__ import annotations

import os
from dataclasses import dataclass, field


# 本模块是全工程常量/超参的单一来源（single source of truth）。
# 它不依赖任何其他模块，处于依赖链最底层，供 engine / encoder / network / mcts 等导入，
# 避免同一常量在多个文件里各写一份导致不一致。


# 棋盘几何
# ----------------------------------------------------------------------------
# 中国象棋棋盘为 9 列 10 行，坐标范围分别是 x=0..8、y=0..9。
BOARD_WIDTH = 9
BOARD_HEIGHT = 10

# 棋盘总格数。动作编码会把源格和目标格都映射到 0..89。
SQUARE_COUNT = BOARD_WIDTH * BOARD_HEIGHT

# 动作空间大小：任意源格 * 任意目标格。非法动作后续由规则过滤。
ACTION_SPACE_SIZE = SQUARE_COUNT * SQUARE_COUNT


# 规则上限
# ----------------------------------------------------------------------------
# 单局最多半回合数，超过后按步数上限终局。
MAX_TOTAL_PLIES = 300

# 连续未吃子的半回合数上限，达到后按无吃子规则判和。
NO_CAPTURE_DRAW_PLIES = 100


# 局面编码
# ----------------------------------------------------------------------------
# 历史帧数：编码会堆叠最近 N_HISTORY 个局面。
N_HISTORY = 7

# 每一帧的通道数：己方 7 子种 + 对方 7 子种。
PLANES_PER_FRAME = 14

# 网络输入通道数：N_HISTORY 帧 * 每帧 14 通道 + 1 个未吃子通道。
INPUT_CHANNELS = N_HISTORY * PLANES_PER_FRAME + 1  # 99


# 网络结构超参
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class NetworkConfig:
    """CNN 双头网络的结构超参。默认值偏小，便于本地 CPU 验证正确性；
    云端训练时再按算法文档 Stage A/B 调大（如 hidden_channels=128, residual_blocks=6）。"""

    input_channels: int = INPUT_CHANNELS
    action_space_size: int = ACTION_SPACE_SIZE
    hidden_channels: int = 64
    residual_blocks: int = 4
    policy_head_channels: int = 32
    value_head_channels: int = 32
    value_fc_hidden: int = 128


# MCTS 搜索超参
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class MCTSConfig:
    """蒙特卡洛树搜索超参。默认值为 AlphaZero 量级、本地可跑；
    云端可调大 n_simulations 以提升棋力。"""

    # 每次 run 跑多少次模拟。
    n_simulations: int = 200

    # PUCT 探索系数：越大越偏向先验/探索，越小越偏向已有价值估计。
    c_puct: float = 1.5

    # 根节点 Dirichlet 噪声参数，鼓励自对弈在开局阶段探索不同走法。
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25


# 训练超参
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class TrainConfig:
    """train_step 的超参。batch_size 是纯数值，CPU/GPU 跑法一致，切设备不需改代码。"""

    # 每个训练步从 replay buffer 采样的样本数。
    batch_size: int = 256

    learning_rate: float = 1e-3

    # L2 正则，由优化器 weight_decay 实现。
    weight_decay: float = 1e-4

    # 梯度裁剪范数上限，防止偶发大梯度。
    grad_clip_norm: float = 1.0

    # replay buffer 容量（保留最近多少条样本）。
    buffer_capacity: int = 10000

    # buffer 样本数达到此阈值才开始训练，避免一开始拿太少数据过拟合。
    min_buffer_size: int = 256

    # 主循环每次「吃一批自对弈数据」后跑多少个训练步。
    steps_per_iteration: int = 20

    # 每多少个训练步把最新权重推给推理服务（让自对弈用上更新后的网络）。
    weight_sync_interval: int = 50


# 自对弈超参
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class SelfPlayConfig:
    """自对弈/数据生成的超参。"""

    # 前若干手用 τ=1 采样保多样性，之后切到 argmax 求最强。
    temperature_moves: int = 30

    # 推理服务单次前向的最大批量：跨 worker 聚合的叶子评估请求数上限。
    # 每个 worker 同一时刻只有 1 个在途请求，故有效批量 ≈ min(num_workers, eval_batch_size)。
    eval_batch_size: int = 16

    # 推理服务凑批的等待窗口（毫秒）：收到首个请求后最多再等这么久凑满 batch。
    inference_timeout_ms: float = 5.0

    # 并行自对弈进程数（绕开 GIL，把 CPU 规则引擎打满）。
    num_workers: int = 8


# 顶层配置与运行档位
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Config:
    """聚合全部子配置 + 运行设备。切换 CPU/GPU 只改 device 与各档位数值，不动业务代码。

    device 取字符串（"cpu" / "cuda" / "auto"），不在本模块解析成 torch.device——
    config 处于依赖链最底层、不依赖 torch；"auto" 由使用方（evaluator/train）解析。
    """

    device: str = "cpu"

    # 训练终止条件：自对弈累计产出多少局后停止（异步流水线按局数收口）。
    total_games: int = 100

    network: NetworkConfig = field(default_factory=NetworkConfig)
    mcts: MCTSConfig = field(default_factory=MCTSConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    selfplay: SelfPlayConfig = field(default_factory=SelfPlayConfig)


# 预设档位：本地 CPU 验证正确性 vs 云端 GPU 上规模。
# 切换方式：设环境变量 XQ_PROFILE=gpu（配合项目 .env，uv run 自动加载），无需改代码。
PROFILES: dict[str, Config] = {
    "local": Config(
        device="cpu",
        total_games=20,
        network=NetworkConfig(hidden_channels=64, residual_blocks=4),
        mcts=MCTSConfig(n_simulations=50),
        train=TrainConfig(
            batch_size=64,
            buffer_capacity=5000,
            min_buffer_size=128,
            steps_per_iteration=10,
            weight_sync_interval=20,
        ),
        selfplay=SelfPlayConfig(
            num_workers=2,
            eval_batch_size=2,
            inference_timeout_ms=5.0,
        ),
    ),
    # gpu 档位面向 RTX 3070 + i7-14700K + 32GB，目标约一天跑完一轮训练。
    "gpu": Config(
        device="cuda",
        total_games=8000,
        network=NetworkConfig(hidden_channels=128, residual_blocks=10),
        mcts=MCTSConfig(n_simulations=128),
        train=TrainConfig(
            batch_size=512,
            buffer_capacity=200000,
            min_buffer_size=10000,
            steps_per_iteration=150,
            weight_sync_interval=50,
        ),
        selfplay=SelfPlayConfig(
            num_workers=16,
            eval_batch_size=16,
            inference_timeout_ms=5.0,
            temperature_moves=20,
        ),
    ),
}


def load_config() -> Config:
    """按环境变量 XQ_PROFILE 选择运行档位（默认 local）。"""
    profile = os.environ.get("XQ_PROFILE", "local")
    if profile not in PROFILES:
        raise ValueError(
            f"未知 XQ_PROFILE={profile!r}，可选：{sorted(PROFILES)}"
        )
    return PROFILES[profile]
