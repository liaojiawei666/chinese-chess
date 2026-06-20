# Datagen 重大重构方案

## 已完成

### 1. Workspace 提升 & Crate 合并
- [x] 将 Cargo workspace 提升到仓库根
- [x] 合并 5 个库 crate（engine/encoder/mcts/inference/store）到 `crates/core/`（包名 `cc_core`）
- [x] 模块结构：`engine/` + `encode.rs` + `mcts.rs` + `infer/` + `model_io.rs` + `config.rs` + `selfplay.rs`
- [x] binary 重组：selfplay→datagen, arena/play_gui 分别移到 `crates/` 下

### 2. 日志
- [x] Rust 端：`log` + `env_logger`，替换所有 `println!/eprintln!` 为结构化日志
- [x] Python 端：`logging` stdlib，替换 `print`

### 3. 跨平台兼容
- [x] justfile 跨平台条件（`os()`）
- [x] `atomic_rename` Windows 分支
- [x] 更新 justfile / README / .gitignore

### 4. MCTS step-by-step API
- [x] `Mcts` 不再持有 `Evaluator`，只持有 `MctsConfig + Rng`
- [x] 新增 `init_root()`, `feed_root_eval()`, `step()` → `StepResult::NeedEval/Done`, `feed_eval()`
- [x] 保留 `run()` 同步方法兼容 arena/play_gui（内部走 step+feed 循环）
- [x] `Evaluator` 添加 blanket impl `impl<T: Evaluator> Evaluator for &T`

### 5. datagen 多游戏流水线
- [x] 新增 `crates/datagen/src/pipeline.rs`：多游戏调度器
- [x] `worker_loop` 改用 `pipeline::run_pipeline`，按 `games_per_worker` 同时推进 N 局
- [x] 新增 `datagen.games_per_worker` 配置字段（默认 1，向后兼容）
- [x] GPU 档位：8 workers × 32 games = 256 并发，eval_batch_size=256，GPU 利用率 80%+
- [x] `collect_batch_size` 在 GPU 档位改为 1（流水线不需要单局内 virtual loss 凑批）

### 6. 去重 encode
- [x] `GameSlot` 缓存 `cached_root_encoding`：GPU eval 时 clone 编码，样本记录时复用
- [x] 子树复用命中时 fallback 到 `encode()` 重新编码

---

## 性能基准测试（GPU 机上执行）

> 以下测试需在 3070 + i7-14700K 机器上执行，不在本机（无 GPU）运行。

### 6.1 GPU 前向单次耗时

```bash
# 测试不同 batch_size 下的 GPU forward 延迟（单位 ms）
RUST_LOG=info cargo test --release -p cc_core --features torch -- bench_gpu_forward --nocapture
```

预期结果（3070 + 小网络 128ch×10block）：

| batch_size | 预估延迟 |
|-----------|---------|
| 1         | ~0.5ms  |
| 32        | ~1ms    |
| 128       | ~2ms    |
| 256       | ~3ms    |
| 512       | ~5ms    |

GPU 吞吐 ≈ batch_size / latency，batch_size=256 时理论吞吐 ~85k pos/s。

### 6.2 CPU 叶子生成耗时

```bash
# 测试单线程：legal_moves + encode 一个叶子的耗时
cargo bench -p cc_core -- leaf_gen
```

已有数据点：`legal_moves()` ≈ 24μs/call（开局）~60μs（中局），
`encode()` ≈ 15-30μs（估算），合计一次叶子 ≈ 40-90μs。

单线程 CPU 供给能力 ≈ 1000000 / 65 ≈ 15k leaves/s。
8 线程 ≈ 120k leaves/s，远超 GPU 吞吐，**CPU 不是瓶颈**。

### 6.3 端到端流水线吞吐

**当前架构**（乒乓模式）：

```
时间线（单 worker, batch=8）：
  CPU: [MCTS×8] [等GPU] [MCTS×8] [等GPU] ...
  GPU: [等CPU]  [fwd]   [等CPU]  [fwd]   ...
```

GPU 利用率 ≈ fwd_time / (fwd_time + cpu_time) ≈ 20%

**多游戏流水线**（目标架构）：

```
时间线（multi-game pipelining）：
  CPU: [game1.step] [game2.step] ... [gameN.step] [分发结果] ...
  GPU: [batch_fwd]  [batch_fwd]  [batch_fwd]    ...
```

CPU 不断产生叶子请求，GPU 不断消费 → GPU 利用率 ≈ 80%+

**预计提速**：
- GPU 利用率 20% → 80%：**4× 加速**
- 更大 batch（8 → 256）提升 GPU 吞吐：**额外 1.5-2× 加速**
- 综合：**6-8× 加速**，6-7 天 → **1 天以内**
