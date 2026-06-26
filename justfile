# Chinese Chess AlphaZero — Build & Run Recipes
# Usage: just <recipe> [config=config.json]

config := "config.json"

# Rust 端 onnxruntime 走 GPU 时，运行时需要能找到 CUDA/cuDNN。
# 复用 trainer venv 里 torch 自带的 nvidia-* 库，避免单独安装 CUDA Toolkit。
cuda_ld := justfile_directory() / "trainer/.venv/lib/python*/site-packages/nvidia/*/lib"

# ── Full Pipeline ──

# Start the complete training pipeline: export model → trainer + datagen
train:
    #!/usr/bin/env bash
    set -euo pipefail
    export LD_LIBRARY_PATH="$(echo {{cuda_ld}} | tr ' ' ':')${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    mkdir -p local/logs
    echo "=== Step 1: Export initial model (if needed) ==="
    ( cd trainer && uv run python src/export_model.py --config "../{{config}}" )
    echo "=== Step 2: Start trainer (background) -> local/logs/trainer.log ==="
    ( cd trainer && uv run python src/main.py --config "../{{config}}" ) > local/logs/trainer.log 2>&1 &
    TRAINER_PID=$!
    echo "=== Step 3: Start datagen (background) -> local/logs/datagen.log ==="
    cargo run --release -p datagen -- "{{config}}" > local/logs/datagen.log 2>&1 &
    DATAGEN_PID=$!
    echo "trainer PID=$TRAINER_PID -> local/logs/trainer.log"
    echo "datagen PID=$DATAGEN_PID -> local/logs/datagen.log"
    echo "查看日志: tail -f local/logs/datagen.log   |   tail -f local/logs/trainer.log"
    echo "Press Ctrl+C to stop both."
    trap "kill $TRAINER_PID $DATAGEN_PID 2>/dev/null; exit" INT TERM
    wait $TRAINER_PID $DATAGEN_PID

# ── Individual Components ──

# Run only the datagen (self-play data generation)
datagen:
    #!/usr/bin/env bash
    set -euo pipefail
    export LD_LIBRARY_PATH="$(echo {{cuda_ld}} | tr ' ' ':')${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    cargo run --release -p datagen -- "{{config}}"

# Run only the trainer (consumes shards, exports models)
trainer:
    cd trainer && uv run python src/main.py --config "../{{config}}"

# Export an initial random-weights ONNX model (bootstrap)
export-model:
    cd trainer && uv run python src/export_model.py --config "../{{config}}"

# Launch human-vs-AI GUI (reads latest model from models_dir)
play:
    #!/usr/bin/env bash
    set -euo pipefail
    export LD_LIBRARY_PATH="$(echo {{cuda_ld}} | tr ' ' ':')${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    # 下棋用 CPU 软件渲染（llvmpipe）即可，绕开 WSLg 默认 ZINK 找不到 Vulkan 设备的报错。
    export LIBGL_ALWAYS_SOFTWARE=1
    export GALLIUM_DRIVER=llvmpipe
    # 强制 winit 走 Wayland 后端；并去掉 DISPLAY，避免 arboard 起 X11 剪贴板线程，
    # 该线程在 WSLg 下会被 XWayland 断连(Broken pipe)从而拖垮整个事件循环。
    export WINIT_UNIX_BACKEND=wayland
    unset DISPLAY
    cargo run --release -p play_gui -- "{{config}}"

# ── Testing ──

# Run all tests (Rust + Python)
test:
    cargo test --workspace
    cd trainer && uv run pytest

# Run only Rust tests
test-rust:
    cargo test --workspace

# Run only Python tests
test-python:
    cd trainer && uv run pytest

# ── Build ──

# Build all Rust crates in release mode
build:
    cargo build --workspace --release

# Install Python dependencies
sync:
    cd trainer && uv sync --group dev

# ── Utilities ──

# Clean build artifacts
clean:
    cargo clean
    rm -rf trainer/.venv
