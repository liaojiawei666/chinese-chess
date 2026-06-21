# 统一入口（需要 just：https://github.com/casey/just）。
# 档位用环境变量 CHESS_PROFILE 切换（local / gpu），默认 local。

profile := env_var_or_default("CHESS_PROFILE", "local")
config := "config/" + profile + ".json"
manifest := "--manifest-path crates/Cargo.toml"

# ── libtorch 环境设置 ──
# 全项目唯一来源：trainer venv 里的 Python PyTorch 自带 libtorch。
# torch-sys build script 看到 LIBTORCH_USE_PYTORCH=1 就会调 python 找 PyTorch 的 lib 目录。

venv := justfile_directory() / "trainer" / ".venv"
run_log := justfile_directory() / "scripts" / "run-with-log.sh"

torch_lib_cmd := "'" + venv / "bin" / "python" + "' -c 'import torch,os;print(os.path.join(os.path.dirname(torch.__file__),\"lib\"))'"

# pip 安装的 nvidia-cu13 等（libnvrtc-builtins 等，GPU 前向 JIT 需要）
nvidia_lib_cmd := "'" + venv / "bin" / "python" + "' -c 'import glob,site; print(\":\".join(glob.glob(f\"{site.getsitepackages()[0]}/nvidia/*/lib\")))'"

default:
    @just --list

sync:
    cd trainer && uv sync

# ── 构建 ──

[unix]
build-torch:
    #!/usr/bin/env bash
    set -euo pipefail
    TORCH_LIB=$( {{torch_lib_cmd}} )
    NVIDIA_LIBS=$( {{nvidia_lib_cmd}} )
    PATH="{{venv}}/bin:$PATH" LIBTORCH_USE_PYTORCH=1 \
        DYLD_LIBRARY_PATH="$TORCH_LIB${NVIDIA_LIBS:+:$NVIDIA_LIBS}${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}" \
        LD_LIBRARY_PATH="$TORCH_LIB${NVIDIA_LIBS:+:$NVIDIA_LIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
        cargo build {{manifest}} --release

# ── 测试 ──

test: test-rust test-py

test-rust:
    cargo test {{manifest}}

test-py:
    cd trainer && uv run python -m pytest -q

fmt:
    cargo fmt {{manifest}}
    -cd trainer && uv run ruff format src tests scripts

# ── 基准测试 ──

# criterion 微基准（engine / encode / mcts）
[unix]
bench:
    #!/usr/bin/env bash
    set -euo pipefail
    TORCH_LIB=$( {{torch_lib_cmd}} )
    NVIDIA_LIBS=$( {{nvidia_lib_cmd}} )
    PATH="{{venv}}/bin:$PATH" LIBTORCH_USE_PYTORCH=1 \
        DYLD_LIBRARY_PATH="$TORCH_LIB${NVIDIA_LIBS:+:$NVIDIA_LIBS}${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}" \
        LD_LIBRARY_PATH="$TORCH_LIB${NVIDIA_LIBS:+:$NVIDIA_LIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
        cargo bench {{manifest}} -p cc_core

# MCTS 性能热点分解
[unix]
bench-profile:
    #!/usr/bin/env bash
    set -euo pipefail
    TORCH_LIB=$( {{torch_lib_cmd}} )
    NVIDIA_LIBS=$( {{nvidia_lib_cmd}} )
    PATH="{{venv}}/bin:$PATH" LIBTORCH_USE_PYTORCH=1 \
        DYLD_LIBRARY_PATH="$TORCH_LIB${NVIDIA_LIBS:+:$NVIDIA_LIBS}${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}" \
        LD_LIBRARY_PATH="$TORCH_LIB${NVIDIA_LIBS:+:$NVIDIA_LIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
        cargo run {{manifest}} --release -p cc_core --example profile_mcts

# 端到端吞吐基线（30 秒）
[unix]
bench-datagen:
    #!/usr/bin/env bash
    set -euo pipefail
    TORCH_LIB=$( {{torch_lib_cmd}} )
    NVIDIA_LIBS=$( {{nvidia_lib_cmd}} )
    {{run_log}} bench-datagen -- env \
        PATH="{{venv}}/bin:$PATH" LIBTORCH_USE_PYTORCH=1 \
        DYLD_LIBRARY_PATH="$TORCH_LIB${NVIDIA_LIBS:+:$NVIDIA_LIBS}${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}" \
        LD_LIBRARY_PATH="$TORCH_LIB${NVIDIA_LIBS:+:$NVIDIA_LIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
        RUST_LOG=info \
        cargo run {{manifest}} --release -p datagen -- --config {{config}} --duration-secs 90

# ── 自对弈 ──

[unix]
selfplay:
    #!/usr/bin/env bash
    set -euo pipefail
    TORCH_LIB=$( {{torch_lib_cmd}} )
    NVIDIA_LIBS=$( {{nvidia_lib_cmd}} )
    {{run_log}} selfplay -- env \
        PATH="{{venv}}/bin:$PATH" LIBTORCH_USE_PYTORCH=1 \
        DYLD_LIBRARY_PATH="$TORCH_LIB${NVIDIA_LIBS:+:$NVIDIA_LIBS}${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}" \
        LD_LIBRARY_PATH="$TORCH_LIB${NVIDIA_LIBS:+:$NVIDIA_LIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
        RUST_LOG=info \
        cargo run {{manifest}} --release -p datagen -- --config {{config}}

# ── Arena ──

[unix]
arena model_a model_b:
    #!/usr/bin/env bash
    set -euo pipefail
    TORCH_LIB=$( {{torch_lib_cmd}} )
    NVIDIA_LIBS=$( {{nvidia_lib_cmd}} )
    PATH="{{venv}}/bin:$PATH" LIBTORCH_USE_PYTORCH=1 \
        DYLD_LIBRARY_PATH="$TORCH_LIB${NVIDIA_LIBS:+:$NVIDIA_LIBS}${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}" \
        LD_LIBRARY_PATH="$TORCH_LIB${NVIDIA_LIBS:+:$NVIDIA_LIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
        RUST_LOG=info \
        cargo run {{manifest}} --release -p arena -- \
        --run-config {{config}} \
        --model-a {{model_a}} --model-b {{model_b}} \
        --report data/arena/report.json --table data/arena/table.csv

[unix]
arena-latest:
    #!/usr/bin/env bash
    set -euo pipefail
    TORCH_LIB=$( {{torch_lib_cmd}} )
    NVIDIA_LIBS=$( {{nvidia_lib_cmd}} )
    PATH="{{venv}}/bin:$PATH" LIBTORCH_USE_PYTORCH=1 \
        DYLD_LIBRARY_PATH="$TORCH_LIB${NVIDIA_LIBS:+:$NVIDIA_LIBS}${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}" \
        LD_LIBRARY_PATH="$TORCH_LIB${NVIDIA_LIBS:+:$NVIDIA_LIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
        RUST_LOG=info \
        cargo run {{manifest}} --release -p arena -- \
        --run-config {{config}} \
        --report data/arena/report.json --table data/arena/table.csv

# ── Play GUI ──

[unix]
play-gui:
    #!/usr/bin/env bash
    set -euo pipefail
    TORCH_LIB=$( {{torch_lib_cmd}} )
    NVIDIA_LIBS=$( {{nvidia_lib_cmd}} )
    PATH="{{venv}}/bin:$PATH" LIBTORCH_USE_PYTORCH=1 \
        DYLD_LIBRARY_PATH="$TORCH_LIB${NVIDIA_LIBS:+:$NVIDIA_LIBS}${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}" \
        LD_LIBRARY_PATH="$TORCH_LIB${NVIDIA_LIBS:+:$NVIDIA_LIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
        cargo run {{manifest}} --release -p play_gui -- --run-config {{config}}

# ── Arena 守护进程 + 训练 ──

arena-daemon:
    #!/usr/bin/env bash
    set -euo pipefail
    {{run_log}} arena-daemon -- bash -c \
        'cd trainer && uv run python scripts/arena_daemon.py --config ../{{config}} --checkpoints-only'

train:
    #!/usr/bin/env bash
    set -euo pipefail
    {{run_log}} train -- bash -c \
        'cd trainer && uv run python scripts/train_loop.py --config ../{{config}}'

# ── Smoke test（自对弈 + 训练全链路） ──

[unix]
smoke:
    #!/usr/bin/env bash
    set -euo pipefail
    TORCH_LIB=$( {{torch_lib_cmd}} )
    NVIDIA_LIBS=$( {{nvidia_lib_cmd}} )
    PATH="{{venv}}/bin:$PATH" LIBTORCH_USE_PYTORCH=1 \
        DYLD_LIBRARY_PATH="$TORCH_LIB${NVIDIA_LIBS:+:$NVIDIA_LIBS}${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}" \
        LD_LIBRARY_PATH="$TORCH_LIB${NVIDIA_LIBS:+:$NVIDIA_LIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
        RUST_LOG=info \
        cargo run {{manifest}} --release -p datagen -- --config {{config}}
    cd trainer && uv run python scripts/train_loop.py --config ../{{config}} --idle-poll-limit 3
