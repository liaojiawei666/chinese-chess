#!/usr/bin/env python3
"""导出一个 TorchScript model.pt + 参考前向结果，供 inference crate 的 tch 前向差分测试。

产物落在 data/fixtures/torch_forward/（已 gitignore）：
  - model.pt        TorchScript 序列化的小号 PolicyValueNet
  - expected.json   若干局面（game_index/ply）经该模型前向得到的 value 与 priors

Rust 侧（cargo test -p inference --features torch）加载它们逐项对照；缺文件则跳过。

注意：tch / libtorch 版本必须与导出 model.pt 所用的 torch 版本匹配，否则
TorchScript 可能无法加载。见 README 的 libtorch 一节。

用法：
    python trainer/scripts/dump_torch_fixture.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "trainer" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from trainer.config import NetworkConfig  # noqa: E402
from trainer.network import PolicyValueNet  # noqa: E402
from trainer.reference.encoder import encode  # noqa: E402
from trainer.reference.engine import GameState, Move, Position  # noqa: E402
from trainer.reference.evaluator import priors_from_logits  # noqa: E402

OUT_DIR = REPO_ROOT / "data" / "fixtures" / "torch_forward"

# 与 dump_fixtures.py 的随机对局一致，便于 Rust 用同一 game_index/ply 复现局面。
GAMES = {
    "perpetual_check": (
        "4k4/9/3R5/9/9/9/9/9/4A4/4K4 r",
        [[3, 2, 4, 2], [4, 0, 3, 0], [4, 2, 3, 2], [3, 0, 4, 0]] * 2,
    ),
}


def random_game(seed: int, max_plies: int) -> list[list[int]]:
    import random

    rng = random.Random(seed)
    game = GameState.from_position(Position.starting())
    moves: list[list[int]] = []
    for _ in range(max_plies):
        if game.status().is_terminal:
            break
        legal = game.position.legal_moves()
        if not legal:
            break
        mv = rng.choice(legal)
        moves.append([mv.sx, mv.sy, mv.tx, mv.ty])
        game = game.make_move(mv)
    return moves


def state_at(start_fen: str, moves: list[list[int]], ply: int) -> GameState:
    game = GameState.from_position(Position.from_fen(start_fen))
    for mv in moves[:ply]:
        game = game.make_move(Move(*mv))
    return game


def main() -> None:
    torch.manual_seed(0)
    net = PolicyValueNet(NetworkConfig(hidden_channels=32, residual_blocks=2))
    net.eval()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = OUT_DIR / "model.pt"
    torch.jit.script(net).save(str(model_path))

    start_fen = Position.starting().to_fen()
    random_moves = random_game(seed=7, max_plies=40)

    specs = [
        {"name": "start", "start_fen": start_fen, "moves": [], "ply": 0},
        {"name": "random_a_ply1", "start_fen": start_fen, "moves": random_moves, "ply": 1},
        {"name": "random_a_ply10", "start_fen": start_fen, "moves": random_moves, "ply": 10},
    ]

    cases = []
    with torch.no_grad():
        for spec in specs:
            state = state_at(spec["start_fen"], spec["moves"], spec["ply"])
            x = torch.from_numpy(encode(state)).unsqueeze(0)
            policy_logits, value = net(x)
            priors = priors_from_logits(policy_logits[0].numpy().astype(np.float32), state)
            cases.append(
                {
                    "name": spec["name"],
                    "start_fen": spec["start_fen"],
                    "moves": spec["moves"],
                    "ply": spec["ply"],
                    "value": float(value.item()),
                    "priors": [[[m.sx, m.sy, m.tx, m.ty], float(p)] for m, p in priors.items()],
                }
            )

    expected_path = OUT_DIR / "expected.json"
    expected_path.write_text(
        json.dumps({"model": "model.pt", "cases": cases}, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {model_path} and {expected_path} ({len(cases)} cases, torch {torch.__version__})")


if __name__ == "__main__":
    main()
