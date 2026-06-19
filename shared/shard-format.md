# 样本分片格式（shard format）

datagen(Rust) 产出、trainer(Python) 消费的自对弈样本分片规范。两侧实现都以本文件为准。

## 文件命名

```
shard_{model_version:06d}_w{worker:02d}_{seq:06d}.st
```

- `model_version`：生成该分片所用的模型版本（训练步数），便于溯源 / 按版本过滤。
- `worker`：产出线程/进程号，保证多机多线程不撞名。
- `seq`：该 worker 的递增序号，使文件名字典序 ≈ 时间序。
- 扩展名 `.st` = [safetensors](https://github.com/huggingface/safetensors)。

写盘原子性：先写 `*.st.tmp`，`fsync` 后 `rename` 成 `*.st`（本地 rename 原子；对象存储直接 PUT 整对象即原子）。trainer 只识别 `.st`，忽略 `.tmp`。

## 张量布局

一个分片含若干局、合计 `N` 条样本。safetensors 里包含以下张量：

| key | dtype | shape | 说明 |
|---|---|---|---|
| `state`   | uint8   | `[N, 99, 10, 9]` | 局面 canonical 张量。98 个 0/1 平面 + 1 个未吃子平面（值 = round(ratio*255)，ratio∈[0,1]）。trainer 读时整体 `/255` 升 f32。 |
| `pi_ptr`  | int32   | `[N+1]`          | 稀疏策略 π 的 CSR 行偏移。第 i 条样本的非零项在 `[pi_ptr[i], pi_ptr[i+1])`。 |
| `pi_idx`  | int32   | `[nnz]`          | 非零项的 canonical action_id（0..8099）。 |
| `pi_val`  | float32 | `[nnz]`          | 非零项概率（MCTS 访问分布，τ=1 归一化，按行和为 1）。 |
| `z`       | float32 | `[N]`            | 终局回填：该样本走棋方视角的胜负 +1/0/-1。 |

其中 `nnz = pi_ptr[N]`。

## 还原（trainer 侧）

- `state`: `astype(float32) / 255.0`，得到 `(N,99,10,9)` 的 f32 张量。
- `pi`: 对第 i 条，构造 `(8100,)` 全 0 向量，在 `pi_idx[pi_ptr[i]:pi_ptr[i+1]]` 处填 `pi_val[...]`，得到稠密 π。
- `z`: 直接用。

> 设计动机：`state` 用 uint8 相比 f32 省 4×；π 用稀疏 CSR 把每条从 8100 维稠密降到几十个非零项。详见计划 `交换契约` 一节。
