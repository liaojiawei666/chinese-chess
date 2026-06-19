# chinese-chess

AlphaZero 风格的中国象棋 AI。单一 git 仓库的 polyglot monorepo：

```
trainer/   Python：神经网络训练 + 编排（PyTorch）
datagen/   Rust：多线程自对弈数据生成（规则引擎 + MCTS + tch-rs 推理）
shared/    跨语言契约：run-config schema、样本分片格式、模型格式
data/      运行期产物（gitignore）：run-config.<profile>.json、样本分片、导出模型
docs/      算法与设计文档
```

trainer 与 datagen 只通过磁盘文件解耦交换：datagen 写样本分片 + 轮询热加载模型，
trainer 读分片训练 + 周期性导出版本化权重。契约见 `shared/`。

## 统一入口（just）

装了 [just](https://github.com/casey/just) 后，常用流程都有现成命令：

```bash
just                 # 列出所有命令
just sync            # 同步 Python 依赖
just test            # Rust 差分/单测 + Python 单测
just selfplay        # 跑 datagen 自对弈（默认均匀先验评估器，无需 libtorch）
just train           # 跑训练主循环（长驻，与 selfplay 并跑）
just smoke           # 端到端 smoke：产分片 → 训练 → 导出 model_{step}.pt + latest.json
CHESS_PROFILE=gpu just build   # 切档位
```

## 数据流

```
datagen (Rust, rayon 多线程)                 trainer (Python)
  规则引擎 + MCTS + tch-rs 前向                 读分片 → ReplayBuffer → 训练
        │ 写 data/samples/*.st  ───────────────▶        │
        ▲                                                ▼
        └──── 轮询 data/models/latest.json ◀──── 导出 model_{step}.pt
```

## trainer（Python）

依赖用 [uv](https://github.com/astral-sh/uv) 管理（`trainer/pyproject.toml`）。

```bash
cd trainer
uv sync                       # 安装依赖到 .venv
# 运行测试
PYTHONPATH=src uv run python -m pytest tests -q
# 导出运行配置（datagen 与 trainer 都读这一份）
CHESS_PROFILE=local uv run python scripts/export_run_config.py
# 训练主循环：消费 data/samples 分片 → 训练 → 版本化导出 data/models/model_{step}.pt
CHESS_PROFILE=local uv run python scripts/train_loop.py
```

运行档位由环境变量 `CHESS_PROFILE`（`local` / `gpu`）切换，定义在 `trainer/src/trainer/config.py`。

trainer 侧模块职责：`store.py`（SampleStore/ModelStore 抽象 + 本地实现 + 模型保留策略）、
`shard_io.py`（分片读写：uint8→f32、稀疏 CSR↔稠密 π）、`exporter.py`（TorchScript 版本化导出）、
`loop.py`（发现分片 → ReplayBuffer 滑动窗口 → train_step → 定期导出）。

## datagen（Rust）

标准 cargo workspace。纯逻辑 crate（`engine` / `encoder` / `mcts`）零重依赖，可直接编译测试；
`inference` crate 的 tch-rs 前向在 `torch` 特性后启用，需要 libtorch。

```bash
cd datagen
cargo build                                   # 不含 torch 特性，无需 libtorch
cargo test                                    # 跑纯逻辑 + 差分测试
cargo run -p selfplay -- ../data/config/run-config.local.json
```

### libtorch（tch-rs，仅 `torch` 特性需要）

`inference` crate 在启用 `--features torch` 时链接 libtorch。需与导出 `model.pt` 的
PyTorch 版本兼容。两种方式：

1. 自动下载（最省心）：让 `tch`/`torch-sys` 自行下载匹配的 libtorch
   （首次编译较慢，需要网络）。
2. 手动指定：下载 libtorch 后设环境变量
   ```bash
   export LIBTORCH=/path/to/libtorch
   export DYLD_LIBRARY_PATH=$LIBTORCH/lib:$DYLD_LIBRARY_PATH   # macOS
   # Linux 用 LD_LIBRARY_PATH
   ```

启用前向推理构建：

```bash
cargo build -p inference --features torch
```

版本匹配很重要：`tch` crate 绑定特定 libtorch 版本，而 libtorch 又要能加载由
trainer 侧 torch 导出的 TorchScript `model.pt`。三者（导出用 torch、运行用 libtorch、
`tch` 版本）需对齐，否则 `model.pt` 可能加载失败或 ABI 不兼容。若刻意用不匹配的
libtorch 调试，可设 `LIBTORCH_BYPASS_VERSION_CHECK=1`（仅自担风险）。

tch 前向差分测试（可选）：先用 trainer 侧导出夹具，再跑带特性的测试。

```bash
python trainer/scripts/dump_torch_fixture.py        # 写 data/fixtures/torch_forward/{model.pt,expected.json}
cargo test -p inference --features torch torch_forward_matches_python
```

缺少夹具或 libtorch 时该测试自动跳过；纯掩码/softmax 对齐测试（无需 libtorch）
始终随 `cargo test -p inference` 运行。

## 跨语言契约

- `shared/run-config.schema.json`：运行配置 JSON Schema。`trainer/scripts/export_run_config.py`
  从各档位生成 `data/config/run-config.<profile>.json`（如 `run-config.local.json` /
  `run-config.gpu.json`），trainer 与 datagen 都读它。datagen 启动时用其中的结构常量对
  `engine` crate 的内置 const 做一致性断言，防止漂移。
- `shared/shard-format.md`：样本分片（uint8 state + 稀疏 CSR π + z）。
- `shared/model-format.md`：版本化 `model_{step}.pt` + `latest.json` + 保留策略。
