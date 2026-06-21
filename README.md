# chinese-chess

AlphaZero 风格的中国象棋 AI。单一 git 仓库的 polyglot monorepo：

```
config/      运行配置（入库）：local.json / gpu.json，Python 与 Rust 两侧共读
crates/      Rust workspace（Cargo.toml / Cargo.lock / rust-toolchain.toml 均在此）
  core/      Rust 核心库：规则引擎 / 编码 / MCTS / 推理 / 模型IO / 自对弈
  datagen/   Rust 二进制：多线程自对弈数据生成（批量 tch-rs 推理）
  arena/     Rust 二进制：两模型对杀评估
  play_gui/  Rust 二进制：人机对战 GUI（eframe）
trainer/     Python：神经网络训练 + 编排（PyTorch）
shared/      跨语言契约：run-config schema、样本分片格式、模型格式
data/        运行期产物（gitignore）：样本分片、导出模型
docs/        算法与设计文档
```

trainer 与 datagen 只通过磁盘文件解耦交换：datagen 写样本分片 + 轮询热加载模型，
trainer 读分片训练 + 周期性导出版本化权重。契约见 `shared/`。

## 统一入口（just）

装了 [just](https://github.com/casey/just) 后，常用流程都有现成命令：

```bash
just                 # 列出所有命令
just sync            # 同步 Python 依赖
just test            # Rust 单测 + Python 单测
just selfplay        # 跑 datagen 自对弈（默认 config/local.json）
just train           # 跑训练主循环（长驻，与 selfplay 并跑）
just smoke           # 端到端 smoke
just play-gui        # 人机对战 GUI
CHESS_PROFILE=gpu just selfplay   # 切档位（读 config/gpu.json）
```

日志：Rust 端 `RUST_LOG=info`；Python 端 `logging` stdlib。

## 配置

**`config/local.json` 与 `config/gpu.json` 是单一真相**，直接入库。Python 与 Rust 都读同一份 JSON；
board/rules/encoding 结构常量仍由 Rust `engine::constants` 定义，`RunConfig::verify_constants()` 启动时断言 JSON 未漂移。

```bash
# datagen：CLI 可覆盖 JSON 字段（cargo 需指定 manifest 或在 crates/ 下运行）
cargo --manifest-path crates/Cargo.toml run -p datagen -- \
  --config config/gpu.json --workers 8 --duration-secs 30

# trainer
cd trainer && uv run python scripts/train_loop.py --config ../config/gpu.json
```

档位切换：`CHESS_PROFILE=gpu`（just 默认读 `config/{profile}.json`）。

## 数据流

```
datagen (Rust, rayon 多线程)                 trainer (Python)
  规则引擎 + MCTS + tch-rs 前向                 读分片 → ReplayBuffer → 训练
        │ 写 data/samples/*.st  ───────────────▶        │
        ▲                                                ▼
        └──── 轮询 data/models/latest.json ◀──── 导出 model_{step}.pt
```

## trainer（Python）

```bash
cd trainer && uv sync
PYTHONPATH=src uv run python -m pytest tests -q
uv run python scripts/train_loop.py --config ../config/local.json
```

## core（Rust 核心库）

```bash
cargo --manifest-path crates/Cargo.toml test -p cc_core
cargo --manifest-path crates/Cargo.toml bench -p cc_core   # criterion：engine / encode / mcts
```

### libtorch

Rust 始终链接 trainer venv 里 Python `torch` wheel 自带的 libtorch。先 `just sync`，再：

```bash
PATH="trainer/.venv/bin:$PATH" LIBTORCH_USE_PYTORCH=1 \
  cargo --manifest-path crates/Cargo.toml build --release -p datagen
```

无 `latest.json` 时 datagen 自动退化为均匀评估器（smoke 可用）。

### arena / play_gui

```bash
CHESS_PROFILE=gpu just play-gui
just arena data/models/model_000100.pt data/models/model_000040.pt
just arena-daemon   # checkpoint 触发对杀守护
```

## 性能分析

流水线分两段：**datagen（Rust 自对弈产样本）** 和 **trainer（Python GPU 训练）**。按你想查的段选命令。

### 1. 自对弈瓶颈（Rust / CPU + GPU 推理）

datagen 启动时若 `data/models/` 无权重，会自动调用 Python 导出**随机初始**网络（与训练导出的格式相同），再 GPU 加载。

**端到端吞吐**（真实 datagen 配置，30 秒）：

```bash
CHESS_PROFILE=gpu just bench-datagen
# 日志里看「样本/s」；可调 workers / batch：
CHESS_PROFILE=gpu just bench-datagen -- --workers 8 --eval-batch-size 256
```

**各函数大致耗时**（MCTS 内部分解，单位 µs，文本直读）：

```bash
just bench-profile
```

输出示例：`legal_moves ~9µs`、`step(select) ~55%`、`evaluate ~40%` 等。

**单函数精确对比**（criterion 微基准，HTML 报告）：

```bash
just bench
# 报告：crates/target/criterion/
```

### 2. 训练瓶颈（Python / PyTorch）

训练循环已每 50 步打日志（吞吐、loss）：

```bash
CHESS_PROFILE=gpu just train
# 或：cd trainer && uv run python scripts/train_loop.py --config ../config/gpu.json
```

看 `[train] 步 N | ... | 吞吐 X 样本/s` 判断 GPU 训练是否跟上。

**GPU 是否在干活**（另开终端，与 selfplay/train 并行）：

```bash
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv -l 1
```

### 3. 推荐顺序

```bash
# 样本生成慢？
CHESS_PROFILE=gpu just bench-datagen    # 整体样本/s
just bench-profile                      # Rust 哪段 CPU 函数慢

# 训练慢或 GPU 空转？
CHESS_PROFILE=gpu just train            # 看训练吞吐日志 + nvidia-smi
just bench                              # 对比 engine/encode 单函数基线
```

## 跨语言契约

- `shared/run-config.schema.json`：运行配置 JSON Schema。
- `shared/shard-format.md`：样本分片格式。
- `shared/model-format.md`：版本化模型格式。
