# 统一入口（需要 just：https://github.com/casey/just）。
# 档位用环境变量 CHESS_PROFILE 切换（local / gpu），默认 local。

profile := env_var_or_default("CHESS_PROFILE", "local")

# libtorch 安装路径（torch 特性需要）。默认用仓库内手动下载的 .libtorch；
# 可用环境变量 LIBTORCH 覆盖（如系统装的 / venv 里的，但版本须匹配 tch，见 README）。
# 注意：DYLD_LIBRARY_PATH 只在 torch 相关 recipe 内联设置，不全局 export，
# 以免污染 trainer 的 Python torch（版本不同会加载错动态库）。
libtorch := env_var_or_default("LIBTORCH", justfile_directory() / ".libtorch" / "libtorch")

# 列出所有可用命令
default:
    @just --list

# 安装/同步 Python 依赖
sync:
    cd trainer && uv sync

# 导出所有档位的跨语言运行配置 data/config/run-config.<profile>.json
export-config:
    cd trainer && uv run python scripts/export_run_config.py

# 编译 datagen（不含 torch 特性，无需 libtorch）
build:
    cd datagen && cargo build --release

# 编译 datagen 的 selfplay（启用 torch 特性，需 libtorch；从仓库根跑以对齐相对路径）
build-torch:
    LIBTORCH="{{libtorch}}" DYLD_LIBRARY_PATH="{{libtorch}}/lib" cargo build --manifest-path datagen/Cargo.toml --release -p selfplay --features torch

# 跑全部测试：Rust 差分/单测 + Python 单测
test: test-rust test-py

test-rust:
    cd datagen && cargo test

test-py:
    cd trainer && uv run python -m pytest -q

# 格式化两侧代码
fmt:
    cd datagen && cargo fmt
    -cd trainer && uv run ruff format src tests scripts

# 重新生成差分测试夹具（修改 reference/* 后用）
fixtures:
    cd trainer && PYTHONPATH=src uv run python scripts/dump_fixtures.py

# 跑自对弈数据生成（默认评估器为均匀先验，无需 libtorch）
selfplay: export-config
    cd datagen && cargo run --release -p selfplay -- ../data/config/run-config.{{profile}}.json

# 跑自对弈（torch 真实网络：跨 worker 批量推理 actor）。需 libtorch，且 model_dir 已有导出模型。
# 从仓库根运行，使 config 里的相对 data/ 路径命中真实 data/models、data/samples。
selfplay-torch: export-config
    LIBTORCH="{{libtorch}}" DYLD_LIBRARY_PATH="{{libtorch}}/lib" cargo run --manifest-path datagen/Cargo.toml --release -p selfplay --features torch -- data/config/run-config.{{profile}}.json

# 跑训练主循环（长驻；与 selfplay 并跑）
train: export-config
    cd trainer && CHESS_PROFILE={{profile}} uv run python scripts/train_loop.py

# 端到端 smoke：先产一批分片，再让 trainer 消费训练并导出模型
smoke: export-config
    cd datagen && cargo run --release -p selfplay -- ../data/config/run-config.{{profile}}.json
    cd trainer && CHESS_PROFILE={{profile}} uv run python scripts/train_loop.py --max-steps 5 --idle-poll-limit 3
