#!/usr/bin/env python3
"""用 Python 参考引擎产出差分测试夹具，供 Rust engine crate 逐位对照。

覆盖：
  - 精选局面（来自 tests/test_rules.py 的规则边界用例）：legal_moves / in_check / status。
  - 随机走子快照（固定种子）：广覆盖 legal_moves / status。
  - 对局序列（含长将/长捉/互打 + 随机长对局）：逐手 make_move + GameState.status，
    校验 make_move 链、重复/自然限着/步数上限、长将长捉判定。

输出默认写到 datagen/crates/engine/tests/fixtures/engine.json（已纳入版本库，
Rust 测试直接读取，无需重新生成）。

用法：
    python trainer/scripts/dump_fixtures.py
    python trainer/scripts/dump_fixtures.py --out some/path.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "trainer" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np  # noqa: E402

from trainer.reference.engine import (  # noqa: E402
    ACTION_SPACE_SIZE,
    Color,
    GameState,
    Move,
    Position,
)
from trainer.reference.encoder import (  # noqa: E402
    encode,
    from_canonical_action,
    to_canonical_action,
)
from trainer.reference.evaluator import priors_from_logits  # noqa: E402
from trainer.config import MCTSConfig  # noqa: E402
from trainer.reference.mcts import MCTS  # noqa: E402

DEFAULT_OUT = REPO_ROOT / "datagen" / "crates" / "engine" / "tests" / "fixtures" / "engine.json"

# 精选局面（规则边界），取自 trainer/tests/test_rules.py。
CURATED_FENS = [
    "rheakaehr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RHEAKAEHR r",
    "4k4/9/9/9/9/9/4P4/4H4/9/4K4 r",
    "4k4/9/9/9/9/9/9/9/3P5/2E1K4 r",
    "4k4/9/9/9/9/2E6/9/9/9/4K4 r",
    "4k4/9/9/9/9/9/9/9/3A5/4K4 r",
    "k8/9/9/9/9/9/4P4/9/9/4K4 r",
    "k8/9/9/9/4P4/9/9/9/9/4K4 r",
    "4k4/9/9/9/4p4/9/4P4/9/4C4/4K4 r",
    "3k5/4R4/3R5/9/9/9/9/9/9/4K4 b",
    "4k4/9/9/9/9/9/9/9/9/4K4 r - - 0 1",
]


def status_dict(status) -> dict:
    return {
        "is_terminal": status.is_terminal,
        "reason": status.reason.value if status.reason is not None else None,
        "winner": status.winner.value if status.winner is not None else None,
    }


def moves_to_list(moves) -> list[list[int]]:
    return [[m.sx, m.sy, m.tx, m.ty] for m in moves]


def position_case(fen: str) -> dict:
    pos = Position.from_fen(fen)
    # 将帅照面属于非法局面，is_in_check 会抛错；此处置 null，Rust 侧跳过该断言。
    in_check = None if pos._kings_face() else pos.is_in_check(pos.side_to_move)
    return {
        "fen": fen,
        "to_fen": pos.to_fen(),
        "side_to_move": pos.side_to_move.value,
        "in_check": in_check,
        "legal_moves": moves_to_list(pos.legal_moves()),
        "status": status_dict(pos.status()),
    }


def random_walk_positions(seed: int, num_walks: int, max_plies: int) -> list[dict]:
    cases: list[dict] = []
    for walk in range(num_walks):
        rng = random.Random(seed + walk)
        pos = Position.starting()
        for _ in range(max_plies):
            cases.append(position_case(pos.to_fen()))
            legal = pos.legal_moves()
            if not legal:
                break
            pos = pos.make_move(rng.choice(legal))
    return cases


def game_case(name: str, start_fen: str, moves: list[list[int]]) -> dict:
    game = GameState.from_position(Position.from_fen(start_fen))
    statuses: list[dict] = []
    for mv in moves:
        game = game.make_move(Move(*mv))
        statuses.append(status_dict(game.status()))
    return {
        "name": name,
        "start_fen": start_fen,
        "moves": moves,
        "final_fen": game.position.to_fen(),
        "final_status": status_dict(game.status()),
        "statuses": statuses,
    }


def random_game_case(name: str, seed: int, max_plies: int) -> dict:
    rng = random.Random(seed)
    start_fen = Position.starting().to_fen()
    game = GameState.from_position(Position.starting())
    moves: list[list[int]] = []
    statuses: list[dict] = []
    for _ in range(max_plies):
        if game.status().is_terminal:
            break
        legal = game.position.legal_moves()
        if not legal:
            break
        mv = rng.choice(legal)
        moves.append([mv.sx, mv.sy, mv.tx, mv.ty])
        game = game.make_move(mv)
        statuses.append(status_dict(game.status()))
    return {
        "name": name,
        "start_fen": start_fen,
        "moves": moves,
        "final_fen": game.position.to_fen(),
        "final_status": status_dict(game.status()),
        "statuses": statuses,
    }


def cycle_moves(cycle: list[list[int]], repeats: int) -> list[list[int]]:
    return [list(mv) for _ in range(repeats) for mv in cycle]


def state_at_ply(start_fen: str, moves: list[list[int]], ply: int) -> GameState:
    game = GameState.from_position(Position.from_fen(start_fen))
    for mv in moves[:ply]:
        game = game.make_move(Move(*mv))
    return game


def encoding_case(game_index: int, game: dict, ply: int) -> dict:
    state = state_at_ply(game["start_fen"], game["moves"], ply)
    tensor = encode(state)
    flat = tensor.reshape(-1)
    nz = np.nonzero(flat)[0]
    nonzero = [[int(i), float(flat[i])] for i in nz]
    return {
        "game_index": game_index,
        "ply": ply,
        "shape": list(tensor.shape),
        "side_to_move": state.position.side_to_move.value,
        "nonzero": nonzero,
    }


def infer_case(game_index: int, game: dict, ply: int, rng: np.random.Generator) -> dict:
    """随机 policy logits → priors_from_logits 的对照样本。

    只 dump 合法走法对应的 canonical id 上的 logit（priors 只读这些位置即可复现），
    Rust 侧据此重建稀疏 logits 再跑 priors_from_logits 逐项对照。
    """
    state = state_at_ply(game["start_fen"], game["moves"], ply)
    logits = rng.standard_normal(ACTION_SPACE_SIZE).astype(np.float32)
    priors = priors_from_logits(logits, state)

    perspective = state.position.side_to_move
    legal = state.position.legal_moves()
    logits_sparse = [
        [to_canonical_action(m, perspective), float(logits[to_canonical_action(m, perspective)])]
        for m in legal
    ]
    priors_list = [[[m.sx, m.sy, m.tx, m.ty], float(p)] for m, p in priors.items()]
    return {
        "game_index": game_index,
        "ply": ply,
        "logits_sparse": logits_sparse,
        "priors": priors_list,
    }


def fnv1a_64(s: str) -> int:
    """64 位 FNV-1a 哈希（与 Rust 侧逐字节一致），用于构造确定性 mock 评估。"""
    h = 0xCBF29CE484222325
    for b in s.encode("utf-8"):
        h ^= b
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def _mock_value(fen: str) -> float:
    return (fnv1a_64(fen + "#v") % 20000) / 10000.0 - 1.0


def _mock_logit(fen: str, move: Move) -> float:
    key = f"{fen}#m{move.sx},{move.sy},{move.tx},{move.ty}"
    return (fnv1a_64(key) % 20000) / 10000.0 - 1.0


class MockEvaluator:
    """确定性、跨语言可复现的叶子评估：先验/价值仅由 (FEN, move) 的 FNV-1a 哈希决定。

    用它替换真实网络，使 MCTS 访问分布在固定输入下逐位可比对（Rust 侧用同一哈希）。
    """

    def evaluate(self, state: GameState):
        fen = state.position.to_fen()
        value = _mock_value(fen)
        moves = state.position.legal_moves()
        if not moves:
            return {}, value
        logits = [_mock_logit(fen, m) for m in moves]
        mx = max(logits)
        exps = [math.exp(v - mx) for v in logits]
        total = sum(exps)
        priors = {m: e / total for m, e in zip(moves, exps)}
        return priors, value


def _choose_move(visits: dict):
    """从访问次数里选最大者，平局取 legal_moves 顺序中的第一个（与 Rust 一致）。"""
    best_move = None
    best_n = -1
    for move, n in visits.items():
        if n > best_n:
            best_n = n
            best_move = move
    return best_move


def mcts_case(name: str, start_fen: str, n_sim: int, c_puct: float, num_plies: int, seed: int) -> dict:
    config = MCTSConfig(
        n_simulations=n_sim, c_puct=c_puct, dirichlet_alpha=0.3, dirichlet_epsilon=0.0
    )
    mcts = MCTS(MockEvaluator(), config, rng=np.random.default_rng(seed))
    game = GameState.from_position(Position.from_fen(start_fen))

    runs: list[list] = []
    chosen: list[list[int]] = []
    for _ in range(num_plies):
        if game.status().is_terminal:
            break
        visits = mcts.run(game)
        if not visits:
            break
        runs.append([[[m.sx, m.sy, m.tx, m.ty], n] for m, n in visits.items()])
        move = _choose_move(visits)
        chosen.append([move.sx, move.sy, move.tx, move.ty])
        mcts.advance(move)
        game = game.make_move(move)

    return {
        "name": name,
        "start_fen": start_fen,
        "n_simulations": n_sim,
        "c_puct": c_puct,
        "dirichlet_alpha": 0.3,
        "dirichlet_epsilon": 0.0,
        "runs": runs,
        "chosen": chosen,
    }


def action_cases() -> list[dict]:
    """对起手局面的全部合法走法，在红/黑两种视角下记录 canonical action 与往返还原。"""
    cases: list[dict] = []
    pos = Position.starting()
    for move in pos.legal_moves():
        for perspective in (Color.RED, Color.BLACK):
            action_id = to_canonical_action(move, perspective)
            restored = from_canonical_action(action_id, perspective)
            cases.append(
                {
                    "move": [move.sx, move.sy, move.tx, move.ty],
                    "perspective": perspective.value,
                    "canonical_action": action_id,
                    "restored": [restored.sx, restored.sy, restored.tx, restored.ty],
                }
            )
    return cases


def build_fixtures() -> dict:
    positions = [position_case(fen) for fen in CURATED_FENS]
    positions += random_walk_positions(seed=1234, num_walks=6, max_plies=40)

    games = [
        game_case(
            "perpetual_check",
            "4k4/9/3R5/9/9/9/9/9/4A4/4K4 r",
            cycle_moves([[3, 2, 4, 2], [4, 0, 3, 0], [4, 2, 3, 2], [3, 0, 4, 0]], 2),
        ),
        game_case(
            "perpetual_chase",
            "k3h3h/9/9/9/4R4/9/9/9/9/1K7 r",
            cycle_moves([[4, 4, 4, 5], [8, 0, 7, 2], [4, 5, 4, 4], [7, 2, 8, 0]], 2),
        ),
        game_case(
            "mutual_perpetual",
            "k3h4/8r/9/9/4R4/9/9/9/9/1K6H r",
            cycle_moves([[4, 4, 4, 5], [8, 1, 8, 2], [4, 5, 4, 4], [8, 2, 8, 1]], 2),
        ),
        random_game_case("random_game_a", seed=7, max_plies=120),
        random_game_case("random_game_b", seed=99, max_plies=120),
        random_game_case("random_game_c", seed=2024, max_plies=200),
    ]

    # encode 检查：覆盖红/黑视角、不足 7 帧的开局、以及含历史的中后段。
    encodings: list[dict] = []
    probe_plies = [0, 1, 2, 3, 5, 7, 12, 20]
    for game_index, game in enumerate(games):
        n_moves = len(game["moves"])
        for ply in probe_plies:
            if ply <= n_moves:
                encodings.append(encoding_case(game_index, game, ply))

    # 推理（priors_from_logits）对照：覆盖红/黑视角、不同合法走法数量。
    infer_rng = np.random.default_rng(42)
    infer_cases: list[dict] = []
    infer_specs = [(0, 0), (3, 1), (3, 4), (4, 10), (5, 20)]
    for game_index, ply in infer_specs:
        if ply <= len(games[game_index]["moves"]):
            infer_cases.append(infer_case(game_index, games[game_index], ply, infer_rng))

    # MCTS：固定 mock 评估 + epsilon=0（噪声不改先验，避免跨语言 Dirichlet 不可复现）。
    start_fen = Position.starting().to_fen()
    mcts_cases = [
        mcts_case("start_small", start_fen, n_sim=24, c_puct=1.5, num_plies=6, seed=0),
        mcts_case("start_more_sims", start_fen, n_sim=48, c_puct=2.0, num_plies=4, seed=1),
        mcts_case("endgame", "3k5/9/9/9/9/9/9/9/4R4/4K4 r", n_sim=40, c_puct=1.5, num_plies=6, seed=2),
    ]

    return {
        "positions": positions,
        "games": games,
        "encodings": encodings,
        "action_cases": action_cases(),
        "infer_cases": infer_cases,
        "mcts_cases": mcts_cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    fixtures = build_fixtures()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(fixtures, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    print(
        f"wrote {out_path} ("
        f"{len(fixtures['positions'])} positions, "
        f"{len(fixtures['games'])} games, "
        f"{len(fixtures['encodings'])} encodings, "
        f"{len(fixtures['action_cases'])} action cases, "
        f"{len(fixtures['infer_cases'])} infer cases, "
        f"{len(fixtures['mcts_cases'])} mcts cases)"
    )


if __name__ == "__main__":
    main()
