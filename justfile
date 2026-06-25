# Chinese Chess AlphaZero — Build & Run Recipes
# Usage: just <recipe> [config=config.json]

config := "config.json"

# ── Full Pipeline ──

# Start the complete training pipeline: export model → trainer + datagen
train:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "=== Step 1: Export initial model (if needed) ==="
    cd trainer && uv run python src/export_model.py --config "../{{config}}"
    cd ..
    echo "=== Step 2: Start trainer (background) ==="
    cd trainer && uv run python src/main.py --config "../{{config}}" &
    TRAINER_PID=$!
    cd ..
    echo "=== Step 3: Start datagen ==="
    cargo run --release -p datagen -- "{{config}}" &
    DATAGEN_PID=$!
    echo "trainer PID=$TRAINER_PID, datagen PID=$DATAGEN_PID"
    echo "Press Ctrl+C to stop both."
    trap "kill $TRAINER_PID $DATAGEN_PID 2>/dev/null; exit" INT TERM
    wait $TRAINER_PID $DATAGEN_PID

# ── Individual Components ──

# Run only the datagen (self-play data generation)
datagen:
    cargo run --release -p datagen -- "{{config}}"

# Run only the trainer (consumes shards, exports models)
trainer:
    cd trainer && uv run python src/main.py --config "../{{config}}"

# Export an initial random-weights ONNX model (bootstrap)
export-model:
    cd trainer && uv run python src/export_model.py --config "../{{config}}"

# Launch human-vs-AI GUI (reads latest model from models_dir)
play:
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
