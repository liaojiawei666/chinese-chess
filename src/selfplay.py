from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import ACTION_SPACE_SIZE, Config
from encoder import encode, to_canonical_action
from engine import Color, GameState, Move, Position
from evaluator import Evaluator, priors_from_logits
from mcts import MCTS


@dataclass
class Sample:
    """一条训练样本：局面张量 + 搜索策略 + 该局面走棋方 + 终局回填的价值。

    z 在开局时占位 0，整盘结束后按 player 视角回填 +1/0/-1。
    """

    state: np.ndarray   # encode 后的 canonical 张量 (INPUT_CHANNELS, 10, 9)
    pi: np.ndarray      # MCTS 访问分布，canonical 动作空间 (ACTION_SPACE_SIZE,)
    player: Color       # 该样本走棋方（落子前的行棋方）
    z: float = 0.0      # 终局回填：该走棋方视角的 +1/0/-1


def play_game(
    evaluator: Evaluator,
    config: Config,
    rng: np.random.Generator | None = None,
) -> list[Sample]:
    """跑一盘自对弈，返回带回填 z 的样本列表。

    流程：每手 mcts.run 取访问次数 → 记 (s, π) 样本 → 按温度选招落子 →
    子树复用 advance → 直到终局；终局后按真实胜负回填每条样本的 z。
    """
    rng = rng if rng is not None else np.random.default_rng()
    mcts = MCTS(evaluator, config.mcts, rng)
    mcts.reset()

    state = GameState.from_position(Position.starting())
    samples: list[Sample] = []
    pending: list[Sample] = []  # 待回填 z 的本局样本

    ply = 0
    while True:
        counts = mcts.run(state)
        if not counts:  # 根节点终局，无可走
            break

        player = state.position.side_to_move
        pi = _visit_distribution(counts, player)
        pending.append(Sample(state=encode(state), pi=pi, player=player))

        move = _select_move(counts, ply, config.selfplay.temperature_moves, rng)
        state = state.make_move(move)
        mcts.advance(move)
        ply += 1

    _fill_outcome(pending, state)
    samples.extend(pending)
    return samples


def _visit_distribution(counts: dict, player: Color) -> np.ndarray:
    """把访问次数按 τ=1 归一化，落到 canonical 动作空间。训练目标固定用 τ=1。"""
    pi = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
    total = sum(counts.values())
    if total == 0:
        return pi
    for move, n in counts.items():
        pi[to_canonical_action(move, player)] = n / total
    return pi


def _select_move(counts: dict, ply: int, temperature_moves: int, rng: np.random.Generator):
    """选招：前 temperature_moves 手按访问次数比例采样（τ=1），之后取 argmax。"""
    moves = list(counts.keys())
    visits = np.array([counts[m] for m in moves], dtype=np.float64)
    if ply < temperature_moves:
        probs = visits / visits.sum()
        return moves[int(rng.choice(len(moves), p=probs))]
    return moves[int(visits.argmax())]


def _fill_outcome(pending: list[Sample], terminal_state: GameState) -> None:
    """按终局真实胜负，给每条样本回填该走棋方视角的 z。"""
    winner = terminal_state.status().winner
    for sample in pending:
        if winner is None:
            sample.z = 0.0
        else:
            sample.z = 1.0 if winner is sample.player else -1.0


class RemoteEvaluator:
    """跨进程推理适配层：与 Evaluator.evaluate 同签名，但把前向丢给推理服务进程。

    自对弈 worker 不持有网络，每个叶子局面把 encode 后的张量塞进 request_q，
    阻塞等自己专属 response_q 拿回 (logits, value)，再在本地做合法过滤/softmax。
    这样 N 个 worker 的单点请求会在服务端被聚合成一个 batch，喂满 GPU。
    """

    def __init__(self, worker_id: int, request_q, response_q, timeout: float = 120.0) -> None:
        self.worker_id = worker_id
        self.request_q = request_q
        self.response_q = response_q
        self.timeout = timeout

    def evaluate(self, state: GameState) -> tuple[dict[Move, float], float]:
        self.request_q.put((self.worker_id, encode(state)))
        policy_logits, value = self.response_q.get(timeout=self.timeout)
        return priors_from_logits(policy_logits, state), float(value)


def worker_loop(worker_id, config, seed, request_q, response_q, sample_q, stop_event) -> None:
    """自对弈 worker 进程入口：不停跑整局自对弈，把样本整局推给 sample_q。

    stop_event 置位后不再开新的一局（当前这局会跑完，需推理服务保持在线）。
    """
    rng = np.random.default_rng(seed)
    evaluator = RemoteEvaluator(worker_id, request_q, response_q)
    while not stop_event.is_set():
        samples = play_game(evaluator, config, rng)
        sample_q.put(samples)
