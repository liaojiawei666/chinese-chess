# 统一入口（需要 just：https://github.com/casey/just）。
# 档位用环境变量 CHESS_PROFILE 切换（local / gpu），默认 local。

profile := env_var_or_default("CHESS_PROFILE", "local")
config := "config/" + profile + ".json"
manifest := "--manifest-path crates/Cargo.toml"

# ── libtorch 环境设置 ──
# 全项目唯一来源：trainer venv 里的 Python PyTorch 自带 libtorch。
# torch-sys build script 看到 LIBTORCH_USE_PYTORCH=1 就会调 python 找 PyTorch 的 lib 目录。

venv := justfile_directory() / "trainer" / ".venv"

# 获取 torch/lib 路径的命令（跨平台）
torch_lib_cmd := if os() == "windows" {
    "python -c \"import torch,os;print(os.path.join(os.path.dirname(torch.__file__),'lib'))\""
} else {
    "'" + venv / "bin" / "python" + "' -c 'import torch,os;print(os.path.join(os.path.dirname(torch.__file__),\"lib\"))'"
}

default:
    @just --list

sync:
    cd trainer && uv sync

# ── 构建 ──

[unix]
build-torch:
    PATH="{{venv}}/bin:$PATH" LIBTORCH_USE_PYTORCH=1 cargo build {{manifest}} --release

[windows]
build-torch:
    @powershell -Command "$torch_lib = python -c 'import torch,os;print(os.path.join(os.path.dirname(torch.__file__),\"lib\"))'; $env:LIBTORCH_USE_PYTORCH='1'; $env:PATH=\"$torch_lib;$env:PATH\"; cargo build {{manifest}} --release"

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
    PATH="{{venv}}/bin:$PATH" LIBTORCH_USE_PYTORCH=1 \
        DYLD_LIBRARY_PATH="$TORCH_LIB${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}" \
        LD_LIBRARY_PATH="$TORCH_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
        cargo bench {{manifest}} -p cc_core

[windows]
bench:
    @powershell -Command "$torch_lib = python -c 'import torch,os;print(os.path.join(os.path.dirname(torch.__file__),\"lib\"))'; $env:LIBTORCH_USE_PYTORCH='1'; $env:PATH=\"$torch_lib;$env:PATH\"; cargo bench {{manifest}} -p cc_core"

# MCTS 性能热点分解
[unix]
bench-profile:
    #!/usr/bin/env bash
    set -euo pipefail
    TORCH_LIB=$( {{torch_lib_cmd}} )
    PATH="{{venv}}/bin:$PATH" LIBTORCH_USE_PYTORCH=1 \
        DYLD_LIBRARY_PATH="$TORCH_LIB${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}" \
        LD_LIBRARY_PATH="$TORCH_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
        cargo run {{manifest}} --release -p cc_core --example profile_mcts

[windows]
bench-profile:
    @powershell -Command "$torch_lib = python -c 'import torch,os;print(os.path.join(os.path.dirname(torch.__file__),\"lib\"))'; $env:LIBTORCH_USE_PYTORCH='1'; $env:PATH=\"$torch_lib;$env:PATH\"; cargo run {{manifest}} --release -p cc_core --example profile_mcts"

# 端到端吞吐基线（30 秒）
[unix]
bench-datagen:
    #!/usr/bin/env bash
    set -euo pipefail
    TORCH_LIB=$( {{torch_lib_cmd}} )
    PATH="{{venv}}/bin:$PATH" LIBTORCH_USE_PYTORCH=1 \
        DYLD_LIBRARY_PATH="$TORCH_LIB${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}" \
        LD_LIBRARY_PATH="$TORCH_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
        RUST_LOG=info \
        cargo run {{manifest}} --release -p datagen -- --config {{config}} --duration-secs 30

[windows]
bench-datagen:
    @powershell -Command "$torch_lib = python -c 'import torch,os;print(os.path.join(os.path.dirname(torch.__file__),\"lib\"))'; $env:LIBTORCH_USE_PYTORCH='1'; $env:PATH=\"$torch_lib;$env:PATH\"; $env:RUST_LOG='info'; cargo run {{manifest}} --release -p datagen -- --config {{config}} --duration-secs 30"

# ── CPU 火焰图 ──
# Linux: cargo-flamegraph + perf（cargo install flamegraph）
# macOS: samply（cargo install samply），无需 sudo
# Windows: 用 Visual Studio Profiler 或 cargo-xray

[linux]
profile-flame:
    #!/usr/bin/env bash
    set -euo pipefail
    TORCH_LIB=$( {{torch_lib_cmd}} )
    PATH="{{venv}}/bin:$PATH" LIBTORCH_USE_PYTORCH=1 \
        LD_LIBRARY_PATH="$TORCH_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
        cargo flamegraph {{manifest}} -p datagen -o flamegraph.svg -- --config {{config}} --duration-secs 30
    @echo "火焰图已生成：flamegraph.svg"

[macos]
profile-flame:
    #!/usr/bin/env bash
    set -euo pipefail
    TORCH_LIB=$( {{torch_lib_cmd}} )
    PATH="{{venv}}/bin:$PATH" LIBTORCH_USE_PYTORCH=1 \
        DYLD_LIBRARY_PATH="$TORCH_LIB${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}" \
        cargo build {{manifest}} --release -p cc_core --example profile_mcts
    BENCH_BIN=$(cargo build {{manifest}} --release -p cc_core --example profile_mcts --message-format=json 2>/dev/null \
        | jq -r 'select(.executable) | .executable' | head -1)
    if [ -z "$BENCH_BIN" ]; then
        BENCH_BIN=$(find crates/target/release/examples -name "profile_mcts" -type f -perm +111 2>/dev/null | head -1)
    fi
    DYLD_LIBRARY_PATH="$TORCH_LIB${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}" \
        samply record --save-only -o profile.json -- "$BENCH_BIN"
    @echo "Profile 已生成：profile.json（用 samply load profile.json 查看）"

# ── 自对弈 ──

[unix]
selfplay:
    #!/usr/bin/env bash
    set -euo pipefail
    TORCH_LIB=$( {{torch_lib_cmd}} )
    PATH="{{venv}}/bin:$PATH" LIBTORCH_USE_PYTORCH=1 \
        DYLD_LIBRARY_PATH="$TORCH_LIB${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}" \
        LD_LIBRARY_PATH="$TORCH_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
        RUST_LOG=info \
        cargo run {{manifest}} --release -p datagen -- --config {{config}}

[windows]
selfplay:
    @powershell -Command "$torch_lib = python -c 'import torch,os;print(os.path.join(os.path.dirname(torch.__file__),\"lib\"))'; $env:LIBTORCH_USE_PYTORCH='1'; $env:PATH=\"$torch_lib;$env:PATH\"; $env:RUST_LOG='info'; cargo run {{manifest}} --release -p datagen -- --config {{config}}"

# ── Arena ──

[unix]
arena model_a model_b:
    #!/usr/bin/env bash
    set -euo pipefail
    TORCH_LIB=$( {{torch_lib_cmd}} )
    PATH="{{venv}}/bin:$PATH" LIBTORCH_USE_PYTORCH=1 \
        DYLD_LIBRARY_PATH="$TORCH_LIB${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}" \
        LD_LIBRARY_PATH="$TORCH_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
        RUST_LOG=info \
        cargo run {{manifest}} --release -p arena -- \
        --run-config {{config}} \
        --model-a {{model_a}} --model-b {{model_b}} \
        --report data/arena/report.json --table data/arena/table.csv

[windows]
arena model_a model_b:
    @powershell -Command "$torch_lib = python -c 'import torch,os;print(os.path.join(os.path.dirname(torch.__file__),\"lib\"))'; $env:LIBTORCH_USE_PYTORCH='1'; $env:PATH=\"$torch_lib;$env:PATH\"; $env:RUST_LOG='info'; cargo run {{manifest}} --release -p arena -- --run-config {{config}} --model-a {{model_a}} --model-b {{model_b}} --report data/arena/report.json --table data/arena/table.csv"

[unix]
arena-latest:
    #!/usr/bin/env bash
    set -euo pipefail
    TORCH_LIB=$( {{torch_lib_cmd}} )
    PATH="{{venv}}/bin:$PATH" LIBTORCH_USE_PYTORCH=1 \
        DYLD_LIBRARY_PATH="$TORCH_LIB${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}" \
        LD_LIBRARY_PATH="$TORCH_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
        RUST_LOG=info \
        cargo run {{manifest}} --release -p arena -- \
        --run-config {{config}} \
        --report data/arena/report.json --table data/arena/table.csv

[windows]
arena-latest:
    @powershell -Command "$torch_lib = python -c 'import torch,os;print(os.path.join(os.path.dirname(torch.__file__),\"lib\"))'; $env:LIBTORCH_USE_PYTORCH='1'; $env:PATH=\"$torch_lib;$env:PATH\"; $env:RUST_LOG='info'; cargo run {{manifest}} --release -p arena -- --run-config {{config}} --report data/arena/report.json --table data/arena/table.csv"

# ── Play GUI ──

[unix]
play-gui:
    #!/usr/bin/env bash
    set -euo pipefail
    TORCH_LIB=$( {{torch_lib_cmd}} )
    PATH="{{venv}}/bin:$PATH" LIBTORCH_USE_PYTORCH=1 \
        DYLD_LIBRARY_PATH="$TORCH_LIB${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}" \
        LD_LIBRARY_PATH="$TORCH_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
        cargo run {{manifest}} --release -p play_gui -- --run-config {{config}}

[windows]
play-gui:
    @powershell -Command "$torch_lib = python -c 'import torch,os;print(os.path.join(os.path.dirname(torch.__file__),\"lib\"))'; $env:LIBTORCH_USE_PYTORCH='1'; $env:PATH=\"$torch_lib;$env:PATH\"; cargo run {{manifest}} --release -p play_gui -- --run-config {{config}}"

# ── Arena 守护进程 + 训练 ──

arena-daemon:
    cd trainer && uv run python scripts/arena_daemon.py --config ../{{config}} --checkpoints-only

train:
    cd trainer && uv run python scripts/train_loop.py --config ../{{config}}

# ── Smoke test（自对弈 + 训练全链路） ──

[unix]
smoke:
    #!/usr/bin/env bash
    set -euo pipefail
    TORCH_LIB=$( {{torch_lib_cmd}} )
    PATH="{{venv}}/bin:$PATH" LIBTORCH_USE_PYTORCH=1 \
        DYLD_LIBRARY_PATH="$TORCH_LIB${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}" \
        LD_LIBRARY_PATH="$TORCH_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
        RUST_LOG=info \
        cargo run {{manifest}} --release -p datagen -- --config {{config}}
    cd trainer && uv run python scripts/train_loop.py --config ../{{config}} --idle-poll-limit 3

[windows]
smoke:
    @powershell -Command "$torch_lib = python -c 'import torch,os;print(os.path.join(os.path.dirname(torch.__file__),\"lib\"))'; $env:LIBTORCH_USE_PYTORCH='1'; $env:PATH=\"$torch_lib;$env:PATH\"; $env:RUST_LOG='info'; cargo run {{manifest}} --release -p datagen -- --config {{config}}"
    cd trainer && uv run python scripts/train_loop.py --config ../{{config}} --idle-poll-limit 3
