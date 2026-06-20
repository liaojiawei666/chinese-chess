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

三层观测链：**火焰图找热点 → criterion 精确单函数 → datagen 端到端吞吐**。

### 1. 端到端吞吐基线

```bash
# 30 秒吞吐对比（可调 batch / workers）
just bench-datagen                    # 等价快捷命令
just bench-datagen -- --eval-batch-size 256
just bench-datagen -- --workers 20
```

### 2. CPU 火焰图

```bash
cargo install flamegraph   # 一次性
just profile             # macOS 需 sudo（dtrace）；Linux 用 perf
```

Windows：管理员 PowerShell 下手动运行。

macOS 无 sudo 可选 [samply](https://github.com/mstange/samply)：`samply record cargo --manifest-path crates/Cargo.toml run --release -p datagen -- --duration-secs 30`。

### 3. criterion 单函数基准

```bash
just bench
```

HTML 报告在 `crates/target/criterion/`。

### 4. GPU 利用率（外部工具）

Rust 代码不内嵌 CUDA profiling；并行采集：

```bash
# 终端 1
just selfplay-torch

# 终端 2（NVIDIA）
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used \
  --format=csv -l 1 > gpu_log.csv
```

### 5. 缓存 / 内存（可选，低优先级）

- Linux：`perf stat -e cache-misses,cache-references,instructions cargo --manifest-path crates/Cargo.toml run ...`
- 内存分配热点通常已在火焰图中可见（`alloc::` 调用栈）

## 跨语言契约

- `shared/run-config.schema.json`：运行配置 JSON Schema。
- `shared/shard-format.md`：样本分片格式。
- `shared/model-format.md`：版本化模型格式。
