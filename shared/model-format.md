# 模型交换格式（model format）

trainer(Python) 导出、datagen(Rust) 热加载的模型权重规范。

## 目录布局

```
data/models/
  model_000020.pt        # 文件名 = model_{step:06d}.pt，step = 训练步数，内容不可变
  model_000040.pt
  ...
  latest.json            # 指针，原子写
```

- `model_{step:06d}.pt`：TorchScript 序列化的 `PolicyValueNet`（`torch.jit.script(net).save(...)`），可脱离 Python 由 tch-rs 直接加载。前向签名：输入 `(N, 99, 10, 9)` f32，输出 `(policy_logits [N,8100], value [N,1])`。
- `latest.json`：当前最新可用模型的指针。

## latest.json

```json
{ "version": 40, "path": "model_000040.pt", "ts": "2026-06-16T07:23:05Z" }
```

- `version`：整数训练步数，单调递增。
- `path`：相对 `data/models/` 的文件名。
- `ts`：导出时刻 ISO-8601 UTC（仅供人看）。

写盘原子性：先写 `latest.json.tmp` 再 `rename`。datagen 每局开局只读 `latest.json`（几十字节）；`version` 比本地已加载的大才去读对应 `.pt` 并热加载。

## 保留策略（trainer 导出时执行）

- **滚动**：保留最近 `keep_recent_models`（默认 3）个 `model_*.pt`。
- **长期 checkpoint**：步数为 `checkpoint_every`（默认 2000）整数倍的模型额外作为 checkpoint 永久候选，checkpoint 最多保留 `keep_checkpoints`（默认 3）个，满了淘汰最旧。
- `latest.json` 指向的版本永不被回收。
- 两类计数相互独立：一个模型可同时属于「最近 N 个」和「checkpoint」。
