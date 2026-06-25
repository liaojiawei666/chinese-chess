# Pipeline Protocol

datagen（Rust）与 training（Python）之间的文件系统协议。

---

## 1. 文件路径约定

```
local/
  models/
    model_gen_0001.onnx       ← training 导出
    model_gen_0002.onnx
    latest.json               ← training 每次导出后更新
  samples/
    shard_000000.bin          ← datagen 写入
    shard_000001.bin
    archive/                  ← training 读完后移入
      shard_000000.bin
config.json                   ← 双方共享，只读
```

- 模型命名：`model_gen_{generation:04d}.onnx`
- Shard 命名：`shard_{index:06d}.bin`

---

## 2. latest.json

training 每次导出新模型后**原子写入**（先写临时文件再 rename）。

```json
{
  "generation": 3,
  "model_path": "local/models/model_gen_0003.onnx"
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `generation` | int | 模型代数，从 1 递增 |
| `model_path` | string | 相对于项目根目录的模型路径 |

**datagen 行为**：
- 启动时轮询 `latest.json`（每 3 秒），等到文件出现后加载模型开始工作
- 运行中每完成一局检查 `generation` 是否变化，变化则热加载新模型

---

## 3. Shard 二进制格式

小端序。棋子平面用 uint8 二值，未吃子步数独立存储，policy 用稀疏格式。

```
[Header]  8 bytes
  magic:            u32 = 0x43585347 ("CXSG")
  num_samples:      u32

[Sample × num_samples]  变长
  state_pieces:     u8  × 8820  (channels 0-97, binary 0/1, 共 98 × 10 × 9)
  no_capture_plies: u8          (原始未吃子步数 0-100，读取时除以 NO_CAPTURE_DRAW_PLIES 还原)
  policy_count:     u16         (非零项数量，即有访问次数的走法数)
  policy_ids:       u16 × count (action_id，范围 [0, 8100))
  policy_probs:     f32 × count (归一化后的概率，sum = 1.0)
  value:            f32         (+1 胜 / -1 负 / 0 和，当前行棋方视角)
```

典型样本大小：`8820 + 1 + 2 + 50×6 + 4 = 9127 bytes`（约 9 KB，对比密集格式 68 KB）。

---

## 4. ONNX 模型 I/O

| 方向 | 名称 | shape | dtype | 说明 |
|------|------|-------|-------|------|
| input | `state` | `[batch, 99, 10, 9]` | float32 | 编码规则见 `docs/encode.mdc` |
| output[0] | `policy` | `[batch, 8100]` | float32 | raw logits，softmax 由 MCTS 做 |
| output[1] | `value` | `[batch, 1]` | float32 | tanh 输出，[-1, 1] |

---

## 5. 清理机制

### 模型

training 保留最新 **3 个** 模型文件，导出第 N 个时删除第 N-3 个。

### Shard

training 将 shard 加载进 replay buffer 后，移入 `local/samples/archive/`。archive 目录仅用于事后分析，可定期手动清理。

---

## 6. 模型导出与终止

```
datagen 持续产出 shard
        ↓
training 加载 shard → replay buffer
        ↓
每训练 weight_sync_interval 步 → 导出 ONNX → 更新 latest.json
        ↓
datagen 检测到新 generation → 热加载模型
```

| 参数 | config 路径 | 默认值 | 说明 |
|------|------------|--------|------|
| 总样本上限 | `total_samples` | 1200000 | 双方共享的终止条件 |
| 模型导出间隔 | `train.weight_sync_interval` | 50 | 每 50 个训练步导出一次 ONNX 并更新 latest.json |

### 终止

- datagen：累计写入样本数 >= `total_samples` 后停止产出并退出
- training：无新 shard 可读且 datagen 已退出后结束训练并退出
