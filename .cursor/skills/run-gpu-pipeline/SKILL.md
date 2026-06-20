---
name: run-gpu-pipeline
description: >-
  在 Windows + PowerShell 上启动本项目的 GPU 全流水线：训练（train_loop）、自对弈数据生成
  （selfplay，torch 特性 GPU 推理）、arena 比赛评价三者并跑。当用户说要「在 GPU 上跑」「跑训练
  / 自对弈 / 比赛评价」「启动流水线」，或遇到 CUDA 后端报错（aten::* CUDA backend）、torch_cuda.dll
  未加载、device=cuda 不生效等问题时使用。
disable-model-invocation: true
---

# 在 Windows 上跑 GPU 流水线（训练 + 自对弈 + arena）

本项目 justfile 是 macOS/Linux 写法（`{{venv}}/bin`、`DYLD_LIBRARY_PATH`），**Windows 上不能用 `just`**。
用下面的 PowerShell 原生命令。三个进程必须**都从仓库根 `C:\project\chinese-chess` 跑**，因为
`trainer/src/trainer/config.py` 里的 `data/samples`、`data/models` 是相对路径，cwd 不一致就会各写各的目录、对不上。

## 一次性前置（环境只需做一次）

1. **同步 Python 依赖**（`pyproject.toml` 已按平台配置 torch 来源：Windows 使用 PyTorch cu126 index，
   macOS 使用默认 CPU/mac wheel；tch 0.24 绑定 libtorch 2.11，版本必须留在 2.11.x）：

```powershell
cd C:\project\chinese-chess\trainer
uv sync
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 期望：2.11.0+cu126 True   （cu124 索引没有 2.11，用 cu126）
```

2. **导出 GPU 档位配置**（给 Rust 端读）：

```powershell
cd C:\project\chinese-chess\trainer
$env:CHESS_PROFILE="gpu"; uv run python scripts/export_run_config.py
# 生成 data/config/run-config.gpu.json
```

3. **确认源码里有 torch_cuda.dll 预加载补丁**（Windows GPU 自对弈的关键，见下文「核心坑 3」）。
   检查 `datagen/crates/inference/src/torch_model.rs` 的 `TorchModel::load` 是否在 CUDA 设备时调用
   `ensure_cuda_loaded()`；没有就按「核心坑 3」补上。

## 启动流水线

每个 PowerShell 窗口先设环境（venv 的 `torch/lib` 必须在 PATH 上，运行时才找得到 CUDA DLL）：

```powershell
$venv="C:\project\chinese-chess\trainer\.venv"
$env:CHESS_PROFILE="gpu"; $env:LIBTORCH_USE_PYTORCH="1"
$env:PATH="$venv\Scripts;$venv\Lib\site-packages\torch\lib;$env:PATH"
cd C:\project\chinese-chess
```

启动顺序：训练先起（它会先导出 version=0 初始权重，自对弈才有模型可热加载）。

```powershell
# 1) 训练（GPU）
& "$venv\Scripts\python.exe" trainer\scripts\train_loop.py

# 2) 自对弈数据生成（GPU 推理）
datagen\target\release\selfplay.exe data\config\run-config.gpu.json

# 3) arena 比赛评价（默认 CPU，不抢卡；等 ≥2 个 checkpoint 后自动对杀）
& "$venv\Scripts\python.exe" trainer\scripts\arena_daemon.py --checkpoints-only
```

后台跑可用 `Start-Process ... -RedirectStandardOutput/-RedirectStandardError` 把日志落到 `data\*.log`。

## 验证健康

```powershell
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader   # 显存 >baseline 说明 torch_cuda 已加载
Get-Content data\selfplay.out.log -Tail 1   # 「局 N | 样本 M」在涨即正常
```

流水线推进：自对弈每满 `shard_games`（gpu=16）局写一个 `data\samples\*.st` → 训练攒够 `min_buffer_size`
（gpu=10000）样本开训、每 `checkpoint_every`（500）步存一个 checkpoint → arena 检测到 ≥2 个 checkpoint 后用
CPU 对杀，写 `data\arena\table.csv`。所以三者「同时启动」时，训练和 arena 会先空转等数据，属正常。

## 三个核心坑（务必知道）

**坑 1：venv 是 CPU 版 torch。** `import torch; torch.cuda.is_available()` 为 False 时，`device=cuda` 必崩。
按前置第 1 步 `uv sync`，确认 lockfile 里的 Windows torch 是 `2.11.0+cu126`。

**坑 2：torch 版本变了，`torch-sys` 不会自动重编。** cargo 不依赖 venv 里 torch 的内容变化。换过 torch 后必须手动清：

```powershell
Remove-Item -Recurse -Force datagen\target\release\build\torch-sys-*,datagen\target\release\build\tch-*
Remove-Item -Force datagen\target\release\deps\*torch_sys*,datagen\target\release\deps\*libtch*
# 然后带 env（venv\Scripts 在 PATH 最前，build 脚本跑裸 python.exe 取库路径）重编：
cargo build --manifest-path datagen/Cargo.toml --release -p selfplay --features torch
```

`torch-sys` build.rs 在 lib 目录里看到 `torch_cuda.dll` 才会 `link("torch_cuda")`，所以必须对着 cu126 的
`torch/lib` 重编。

**坑 3：Windows/MSVC 丢弃未引用的 torch_cuda import → CUDA 后端未注册。** 即使链接了 torch_cuda，因为 Rust
代码没引用它的符号，`torch_cuda.dll` 运行时不加载，前向报
`Could not run 'aten::empty_strided' with arguments from the 'CUDA' backend`。
修法是在用 CUDA 设备加载模型前显式 `LoadLibrary("torch_cuda.dll")`（一次即可），已加在
`datagen/crates/inference/src/torch_model.rs`：

```rust
#[cfg(target_os = "windows")]
fn ensure_cuda_loaded() {
    use std::sync::Once;
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        extern "system" { fn LoadLibraryW(name: *const u16) -> *mut std::ffi::c_void; }
        let wide: Vec<u16> = "torch_cuda.dll".encode_utf16().chain(std::iter::once(0)).collect();
        let h = unsafe { LoadLibraryW(wide.as_ptr()) };
        if h.is_null() { eprintln!("警告：加载 torch_cuda.dll 失败，CUDA 推理不可用"); }
    });
}
```

改完 `torch_model.rs` 后若 cargo 没重编（增量指纹偶尔失灵、只花 ~2s），手动删产物强制重编：

```powershell
Remove-Item -Force datagen\target\release\deps\*inference*,datagen\target\release\selfplay.* ,datagen\target\release\deps\selfplay*
cargo build --manifest-path datagen/Cargo.toml --release -p selfplay --features torch
```

## 停止

```powershell
taskkill /PID <训练pid> /T /F; taskkill /PID <自对弈pid> /T /F; taskkill /PID <arena pid> /T /F
```
