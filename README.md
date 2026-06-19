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
just test            # Rust 单测（规则/编码/搜索）+ Python 单测
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
cargo test                                    # 跑规则/编码/搜索单测
cargo run -p selfplay -- ../data/config/run-config.local.json
```

### libtorch（tch-rs，仅 `torch` 特性需要）

`inference` crate 在启用 `--features torch` 时链接 libtorch。**本仓库复用 trainer venv 里
那一份 libtorch**（即 Python `torch` wheel 自带的 `site-packages/torch/lib/`），不再单独
下载 `.libtorch/`——全项目只有一份 libtorch，Python 与 Rust 同版本，杜绝版本漂移。

机制：`tch` 用环境变量 `LIBTORCH_USE_PYTORCH=1` 经 venv 的 python 定位 libtorch。所以
**先 `just sync` 装好 Python torch**，Rust 才能编译/运行 torch 特性。`just build-torch` /
`just selfplay-torch` 已封装好下面两点：

- **编译期**：venv 的 python 需在 `PATH` 上（`torch-sys` 调它取库路径）。
- **运行期**：`DYLD_LIBRARY_PATH` 指向 venv 的 `torch/lib`（`LIBTORCH_USE_PYTORCH` 不写
  rpath；macOS 用 `DYLD_LIBRARY_PATH`，Linux 用 `LD_LIBRARY_PATH`）。这两个变量只在
  torch recipe 内联设置，不全局 export。

手动编译等价命令：

```bash
PATH="trainer/.venv/bin:$PATH" LIBTORCH_USE_PYTORCH=1 \
  cargo build --manifest-path datagen/Cargo.toml -p selfplay --features torch
```

版本对齐：`tch` 绑定特定 libtorch 版本，而 libtorch 又要能加载 trainer 侧 torch 导出的
TorchScript `model.pt`。复用 venv 的 libtorch 天然保证「导出 torch = 运行 libtorch」，只需
让 **`tch` 版本与 venv 的 torch 版本匹配**（当前：`torch 2.11` ↔ `tch 0.24`，均基于
libtorch 2.11）。升级 torch 时需同步升 `tch`。若刻意用不匹配的 libtorch 调试，可设
`LIBTORCH_BYPASS_VERSION_CHECK=1`（仅自担风险）。

`inference` 的纯掩码/softmax 单测（无需 libtorch）始终随 `cargo test -p inference` 运行；
真实 `model.pt` 前向（`torch` 特性）在 `infer` 里程碑接入。

### arena（棋力评估）

`arena` bin 让两个模型版本对杀，输出 A 相对 B 的胜率与 Elo，用来回答「训练有没有在变强」。
**纯指标**：不改 `latest.json`、不打断训练——训练持续产出冻结的 `model_*.pt`，arena 在旁边
load 两个快照对杀即可。

```bash
just arena data/models/model_000100.pt data/models/model_000040.pt
```

- **关噪声**（ε=0）下 MCTS 是确定性的，多样性来自**开局**：用均匀评估器 + 固定种子采样
  k 步、去重后**冻结**到 `data/arena/openings.json`（不依赖训练参数、可复现；删掉该文件即重生成）。
- 每个开局**红黑各打一局**（颜色互换抵消先手优势），τ=0 / ε=0 全程确定性，整场比赛零 RNG。
- 总局数 = 开局数 × 2；得分率 → Elo（`±2σ ≈ sqrt(p(1-p)/N)`，要分辨 ~35 Elo 约需 400 局）。
- 需门控时由外层据得分率（如 ≥55%）决定是否改写 `latest.json`，本工具不做这步。
- 不带 `--features torch` 时以均匀评估器跑 A=B 自检（得分率应 ≈0.5），仅验证管线。
- `--table data/arena/table.csv` 会**追加**一行战绩（首次写表头）；随训练多次跑 arena 即成
  趋势表，可直接看 Elo 走势。`just arena` 默认带上该参数。

## 监控日志

两端都会周期打印进度，直接看 stdout（或重定向到文件）即可：

- **数据生成端（`just selfplay` / `selfplay-torch`）**：每 5s 一行——累计棋局数、累计样本数、
  平均每局样本、瞬时与平均样本速率、对 `total_samples` 的进度百分比。例：
  `[datagen] 30s | 局 42 | 样本 5120 | 均 121.9/局 | 瞬时 360 样本/s（均 358/s）| 进度 5120/100000 (5.1%)`
- **训练端（`just train`）**：每 `--log-interval` 步（默认 50）一行——区间均 loss（总/policy/value）、
  累计训练样本（含 reuse = 步数 × batch）、从分片拉入的样本数 / `total_samples`、吞吐（样本/s、步/s）。例：
  `[train] 步 200 | loss 1.832（p 1.214 v 0.618）| 训练样本 25600 | 拉入 8200/100000 | 吞吐 1280 样本/s, 10.0 步/s`
- **棋力趋势**：用上面的 `arena --table` CSV 战绩表。

## 跨语言契约

- `shared/run-config.schema.json`：运行配置 JSON Schema。`trainer/scripts/export_run_config.py`
  从各档位生成 `data/config/run-config.<profile>.json`（如 `run-config.local.json` /
  `run-config.gpu.json`），trainer 与 datagen 都读它。datagen 启动时用其中的结构常量对
  `engine` crate 的内置 const 做一致性断言，防止漂移。
- `shared/shard-format.md`：样本分片（uint8 state + 稀疏 CSR π + z）。
- `shared/model-format.md`：版本化 `model_{step}.pt` + `latest.json` + 保留策略。
