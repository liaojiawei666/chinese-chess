# 统一入口（需要 just：https://github.com/casey/just）。
# 档位用环境变量 CHESS_PROFILE 切换（local / gpu），默认 local。

profile := env_var_or_default("CHESS_PROFILE", "local")

# torch 特性只使用 trainer venv 里的 Python torch 自带 libtorch（全项目唯一来源）。
# tch 用 LIBTORCH_USE_PYTORCH=1 经 venv 的 python 定位它；不再设置 LIBTORCH 或维护 .libtorch。
# 动态库搜索路径只在 torch 相关 recipe 内联设置、指向同一份 torch/lib，不全局 export。
venv := justfile_directory() / "trainer" / ".venv"

# 列出所有可用命令
default:
    @just --list

# 安装/同步 Python 依赖；torch 由 uv 按平台选择（macOS CPU/mac wheel，Windows cu126 CUDA wheel）。
sync:
    cd trainer && uv sync

# 导出所有档位的跨语言运行配置 data/config/run-config.<profile>.json
export-config:
    cd trainer && uv run python scripts/export_run_config.py

# 编译 datagen（不含 torch 特性，无需 libtorch）
build:
    cd datagen && cargo build --release

# 编译 datagen 的 selfplay（启用 torch 特性）。tch 经 venv 的 python 定位并链接唯一 libtorch，
# 故需先 `just sync`。编译期只需 venv 的 python 在 PATH 上（torch-sys 调它取库路径）。
build-torch:
    PATH="{{venv}}/bin:$PATH" LIBTORCH_USE_PYTORCH=1 cargo build --manifest-path datagen/Cargo.toml --release -p selfplay --features torch

# 跑全部测试：Rust 单测（规则/编码/搜索）+ Python 单测
test: test-rust test-py

test-rust:
    cd datagen && cargo test

test-py:
    cd trainer && uv run python -m pytest -q

# 格式化两侧代码
fmt:
    cd datagen && cargo fmt
    -cd trainer && uv run ruff format src tests scripts

# 跑自对弈数据生成（默认评估器为均匀先验，无需 libtorch）
selfplay: export-config
    cd datagen && cargo run --release -p selfplay -- ../data/config/run-config.{{profile}}.json

# 跑自对弈（torch 真实网络：跨 worker 批量推理 actor）。需先 `just sync`，且 model_dir 已有导出模型。
# 运行期动态库搜索路径指向 venv 的 torch/lib（同一份 libtorch；LIBTORCH_USE_PYTORCH 不写 rpath）。
# 从仓库根运行，使 config 里的相对 data/ 路径命中真实 data/models、data/samples。
selfplay-torch: export-config
    PATH="{{venv}}/bin:$PATH" LIBTORCH_USE_PYTORCH=1 DYLD_LIBRARY_PATH="$('{{venv}}/bin/python' -c 'import torch,os;print(os.path.join(os.path.dirname(torch.__file__),"lib"))')" cargo run --manifest-path datagen/Cargo.toml --release -p selfplay --features torch -- data/config/run-config.{{profile}}.json

# 两模型对杀评估，输出 A 相对 B 的胜率/Elo（纯指标，不改 latest.json、不打断训练）。
# 需先 `just sync` 且 model_a/model_b 为已导出的 .pt。开局首次生成后冻结在 data/arena/openings.json。
# 例：just arena data/models/model_000100.pt data/models/model_000040.pt
arena model_a model_b: export-config
    PATH="{{venv}}/bin:$PATH" LIBTORCH_USE_PYTORCH=1 DYLD_LIBRARY_PATH="$('{{venv}}/bin/python' -c 'import torch,os;print(os.path.join(os.path.dirname(torch.__file__),"lib"))')" cargo run --manifest-path datagen/Cargo.toml --release -p arena --features torch -- --run-config data/config/run-config.{{profile}}.json --model-a {{model_a}} --model-b {{model_b}} --report data/arena/report.json --table data/arena/table.csv

# 直接开打：自动取 data/models 里最近两版（A=最新 vs B=上一版），不用传参。
arena-latest: export-config
    PATH="{{venv}}/bin:$PATH" LIBTORCH_USE_PYTORCH=1 DYLD_LIBRARY_PATH="$('{{venv}}/bin/python' -c 'import torch,os;print(os.path.join(os.path.dirname(torch.__file__),"lib"))')" cargo run --manifest-path datagen/Cargo.toml --release -p arena --features torch -- --run-config data/config/run-config.{{profile}}.json --report data/arena/report.json --table data/arena/table.csv

# 启动人机对战 GUI（macOS/Linux 写法；Windows 见 README 的 PowerShell 命令）。
play-gui: export-config
    PATH="{{venv}}/bin:$PATH" LIBTORCH_USE_PYTORCH=1 DYLD_LIBRARY_PATH="$('{{venv}}/bin/python' -c 'import torch,os;print(os.path.join(os.path.dirname(torch.__file__),"lib"))')" LD_LIBRARY_PATH="$('{{venv}}/bin/python' -c 'import torch,os;print(os.path.join(os.path.dirname(torch.__file__),"lib"))')" cargo run --manifest-path datagen/Cargo.toml --release -p play_gui --features torch -- --run-config data/config/run-config.{{profile}}.json

# arena 守护：轮询 model_dir，每出现新 checkpoint 就低优先级触发对杀（与训练解耦、不阻塞）。
# checkpoint 间隔大（默认 2000 步）、Elo 信号干净、省 GPU。需先 `just sync`。
arena-daemon: export-config
    cd trainer && CHESS_PROFILE={{profile}} uv run python scripts/arena_daemon.py --checkpoints-only

# 跑训练主循环（长驻；与 selfplay 并跑）
train: export-config
    cd trainer && CHESS_PROFILE={{profile}} uv run python scripts/train_loop.py

# 端到端 smoke：先产一批分片，再让 trainer 消费训练并导出模型
smoke: export-config
    cd datagen && cargo run --release -p selfplay -- ../data/config/run-config.{{profile}}.json
    cd trainer && CHESS_PROFILE={{profile}} uv run python scripts/train_loop.py --idle-poll-limit 3
