from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field


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

    # 叶子并行宽度：一波收集多少个叶子后批量评估（virtual loss 防同波重复选）。
    # =1 为逐叶串行（与原始逐位可复现一致）；>1 用 GPU 批量摊薄推理往返延迟、提自对弈吞吐。
    collect_batch_size: int = 1


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

    # replay buffer 容量（= 滑动窗口宽度 = staleness 上限；不影响平均 reuse）。
    buffer_capacity: int = 10000

    # buffer 样本数达到此阈值才开始训练（冷启动地板）；收尾窗口收缩到该值以下则结束。
    min_buffer_size: int = 256

    # 样本复用率（replay ratio）：每条产出样本平均被训练抽取的次数。
    # 消费端据此控制滑动窗口推进（每步推进 batch/target_reuse 条新样本），
    # 把 reuse 钉成与硬件无关的算法超参（详见 docs/diagrams 流水线图）。
    target_reuse: float = 2.0

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


# 数据生成 / 交换契约超参
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class DataGenConfig:
    """Rust 自对弈数据生成器与 Python 训练器之间的交换契约参数。

    样本分片与模型都落在磁盘目录（或未来 OSS），两侧只通过这些路径与命名约定耦合。
    """

    # 样本分片目录：datagen 写 *.st，trainer 读后删。
    samples_dir: str = "data/samples"

    # 模型目录：trainer 写 model_{step:06d}.pt + latest.json，datagen 轮询热加载。
    model_dir: str = "data/models"

    # 导出的跨语言运行配置路径（export_run_config.py 生成，两侧都读）。
    run_config_path: str = "data/config/run-config.json"

    # 每个分片打包多少局自对弈。
    shard_games: int = 8

    # 背压阈值：未消费分片数超过它，datagen 暂停产出，防止训练落后时磁盘膨胀。
    max_pending_shards: int = 64

    # 每多少个训练步导出一次权重。
    model_export_interval: int = 100

    # 滚动保留最近多少个 model_*.pt。
    keep_recent_models: int = 3

    # 每多少步留 1 个长期 checkpoint（也是 arena 守护的对杀触发节奏）。
    # 须为 model_export_interval 的整数倍，否则该步不会被导出、checkpoint 取模筛不到。
    checkpoint_every: int = 500

    # 长期 checkpoint 最多保留数（满了淘汰最旧）。
    keep_checkpoints: int = 3

    # datagen 并行自对弈线程数（rayon）。
    num_workers: int = 4


# 顶层配置与运行档位
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Config:
    """聚合全部子配置 + 运行设备。切换 CPU/GPU 只改 device 与各档位数值，不动业务代码。

    device 取字符串（"cpu" / "cuda" / "auto"），不在本模块解析成 torch.device——
    config 处于依赖链最底层、不依赖 torch；"auto" 由使用方（evaluator/train）解析。
    """

    device: str = "cpu"

    # 数据预算（唯一收口旋钮）：累计产出/消费多少条样本后停止。
    # 生产端产到 total_samples(+并发 slack) 停；消费端消费满 total_samples 后进入收缩收尾。
    # reuse 固定时，总训练步 = target_reuse * total_samples / batch_size，自动推导、与硬件无关。
    total_samples: int = 200000

    network: NetworkConfig = field(default_factory=NetworkConfig)
    mcts: MCTSConfig = field(default_factory=MCTSConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    selfplay: SelfPlayConfig = field(default_factory=SelfPlayConfig)
    datagen: DataGenConfig = field(default_factory=DataGenConfig)


# 预设档位：本地 CPU 验证正确性 vs 云端 GPU 上规模。
# 切换方式：设环境变量 CHESS_PROFILE=gpu（配合项目 .env，uv run 自动加载），无需改代码。
PROFILES: dict[str, Config] = {
    "local": Config(
        device="cpu",
        total_samples=2000,
        network=NetworkConfig(hidden_channels=64, residual_blocks=4),
        mcts=MCTSConfig(n_simulations=50),
        train=TrainConfig(
            batch_size=64,
            buffer_capacity=5000,
            min_buffer_size=128,
            target_reuse=2.0,
            steps_per_iteration=10,
            weight_sync_interval=20,
        ),
        selfplay=SelfPlayConfig(
            num_workers=2,
            eval_batch_size=2,
            inference_timeout_ms=5.0,
        ),
        datagen=DataGenConfig(
            shard_games=4,
            max_pending_shards=32,
            model_export_interval=20,
            num_workers=2,
        ),
    ),
    # gpu 档位面向 RTX 3070 + i7-14700K + 32GB，目标约一天跑完一轮训练。
    "gpu": Config(
        device="cuda",
        total_samples=1_200_000,
        network=NetworkConfig(hidden_channels=128, residual_blocks=10),
        mcts=MCTSConfig(n_simulations=128, collect_batch_size=8),
        train=TrainConfig(
            batch_size=512,
            buffer_capacity=200000,
            min_buffer_size=10000,
            target_reuse=2.0,
            steps_per_iteration=150,
            weight_sync_interval=50,
        ),
        selfplay=SelfPlayConfig(
            num_workers=16,
            # 叶子并行后每 worker 一波可有 ~collect_batch_size 个在途请求，
            # 故有效批量上限随之增大；适当调大单次前向批量、缩短凑批等待窗口。
            eval_batch_size=64,
            inference_timeout_ms=2.0,
            temperature_moves=20,
        ),
        datagen=DataGenConfig(
            shard_games=4,
            max_pending_shards=256,
            model_export_interval=100,
            num_workers=20,
        ),
    ),
}

DEFAULT_PROFILE = "local"
PROFILE_ENV_VAR = "CHESS_PROFILE"


def load_config() -> Config:
    """按环境变量 CHESS_PROFILE 选择运行档位（默认 local）。"""
    profile = os.environ.get(PROFILE_ENV_VAR, DEFAULT_PROFILE)
    if profile not in PROFILES:
        raise ValueError(
            f"未知 {PROFILE_ENV_VAR}={profile!r}，可选：{sorted(PROFILES)}"
        )
    return PROFILES[profile]


def to_run_config(config: Config, profile: str) -> dict:
    """把 Config 序列化成跨语言运行配置 dict（写成 data/config/run-config.json）。

    既含 datagen（Rust）需要的结构常量（board/rules/encoding），也含可调超参
    （network/mcts/train/selfplay/datagen）。Rust 用 serde_json 反序列化，
    并以结构常量段对自身内置 const 做一致性断言，防止两侧漂移。
    """
    return {
        "profile": profile,
        "device": config.device,
        "total_samples": config.total_samples,
        "board": {
            "width": BOARD_WIDTH,
            "height": BOARD_HEIGHT,
            "square_count": SQUARE_COUNT,
            "action_space_size": ACTION_SPACE_SIZE,
        },
        "rules": {
            "max_total_plies": MAX_TOTAL_PLIES,
            "no_capture_draw_plies": NO_CAPTURE_DRAW_PLIES,
        },
        "encoding": {
            "n_history": N_HISTORY,
            "planes_per_frame": PLANES_PER_FRAME,
            "input_channels": INPUT_CHANNELS,
        },
        "network": asdict(config.network),
        "mcts": asdict(config.mcts),
        "train": asdict(config.train),
        "selfplay": asdict(config.selfplay),
        "datagen": asdict(config.datagen),
    }
