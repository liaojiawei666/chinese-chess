from __future__ import annotations

import math

import numpy as np

from config import MCTSConfig
from engine import GameState, GameStatus, Color, Move
from evaluator import Evaluator


class Node:
    """搜索树节点：持有一个 GameState，边的统计按当前合法走法存。

    用 GameState（而非 Position）是因为长将/长捉/重复判和都依赖历史，
    终局判定必须走 GameState.status()。边统计 (P, N, W) 以真实 Move 为 key，
    规模约几十，远小于 8100 的动作空间。
    """

    __slots__ = (
        "state",
        "to_play",
        "is_expanded",
        "is_terminal",
        "terminal_value",
        "noise_added",
        "children",
        "priors",
        "child_N",
        "child_W",
    )

    def __init__(self, state: GameState) -> None:
        self.state = state
        self.to_play: Color = state.position.side_to_move
        self.is_expanded = False
        self.is_terminal = False
        self.terminal_value = 0.0
        self.noise_added = False
        self.children: dict[Move, Node] = {}
        self.priors: dict[Move, float] = {}
        self.child_N: dict[Move, int] = {}
        self.child_W: dict[Move, float] = {}

    def total_visits(self) -> int:
        return sum(self.child_N.values())


class MCTS:
    """带网络先验的蒙特卡洛树搜索。

    持有当前根节点以支持子树复用：run() 跑搜索并返回根的原始访问次数，
    selfplay 落子后调 advance() 把对应子树提为新根。温度/选招不在这里——
    run() 只吐 N(s,a)，怎么选交给上层。
    """

    def __init__(
        self,
        evaluator: Evaluator,
        config: MCTSConfig | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.evaluator = evaluator
        self.config = config or MCTSConfig()
        self.rng = rng if rng is not None else np.random.default_rng()
        self.root: Node | None = None

    def run(self, root_state: GameState) -> dict[Move, int]:
        """跑 n_simulations 次模拟，返回根节点各真实走法的原始访问次数 N(s,a)。

        终局根节点返回空 dict。根节点会撒一次 Dirichlet 噪声。
        """
        self.root = self._ensure_root(root_state)
        if not self.root.is_expanded:
            self._evaluate(self.root)
        if self.root.is_terminal:
            return {}

        self._add_dirichlet_noise(self.root)
        for _ in range(self.config.n_simulations):
            self._simulate(self.root)
        return dict(self.root.child_N)

    def advance(self, move: Move) -> None:
        """落子后把对应子树提为新根（子树复用）。

        若该走法在树中没有展开过的子节点，则置空，下次 run 会从新局面重建。
        """
        if self.root is None:
            self.root = None
            return
        self.root = self.root.children.get(move)

    def reset(self) -> None:
        """开新一局时清空树。"""
        self.root = None

    # 内部实现 ----------------------------------------------------------------

    def _ensure_root(self, root_state: GameState) -> Node:
        if self.root is not None and self._same_state(self.root.state, root_state):
            return self.root
        return Node(root_state)

    @staticmethod
    def _same_state(a: GameState, b: GameState) -> bool:
        # 同一局内用 FEN + 历史步数足以判定是否同一局面（区分重复局面）。
        return (
            a.position.to_fen() == b.position.to_fen()
            and len(a.history) == len(b.history)
        )

    def _simulate(self, root: Node) -> None:
        node = root
        path: list[tuple[Node, Move]] = []

        # 沿 PUCT 下降，直到遇到未展开的叶子或终局节点。
        while node.is_expanded and not node.is_terminal:
            move = self._select(node)
            path.append((node, move))
            child = node.children.get(move)
            if child is None:
                child = Node(node.state.make_move(move))
                node.children[move] = child
            node = child

        value = node.terminal_value if node.is_expanded else self._evaluate(node)
        self._backup(path, value)

    def _select(self, node: Node) -> Move:
        # PUCT：score = Q + c_puct · P · sqrt(1 + ΣN) / (1 + N)。
        # 分子用 sqrt(1 + ΣN) 而非 sqrt(ΣN)，保证首次模拟也按先验展开，避免退化。
        total = node.total_visits()
        explore = self.config.c_puct * math.sqrt(1 + total)
        best_move: Move | None = None
        best_score = -math.inf
        for move, prior in node.priors.items():
            n = node.child_N[move]
            q = node.child_W[move] / n if n > 0 else 0.0
            score = q + explore * prior / (1 + n)
            if score > best_score:
                best_score = score
                best_move = move
        assert best_move is not None
        return best_move

    def _evaluate(self, node: Node) -> float:
        """展开/求值一个节点，返回「该节点走棋方视角」的 value。

        终局：用引擎真实胜负换算成 ±1/0；非终局：调网络取先验与估值。
        """
        status = node.state.status()
        if status.is_terminal:
            node.is_terminal = True
            node.is_expanded = True
            node.terminal_value = self._terminal_value(node, status)
            return node.terminal_value

        priors, value = self.evaluator.evaluate(node.state)
        node.priors = priors
        node.child_N = {move: 0 for move in priors}
        node.child_W = {move: 0.0 for move in priors}
        node.is_expanded = True
        return value

    @staticmethod
    def _terminal_value(node: Node, status: GameStatus) -> float:
        if status.winner is None:
            return 0.0
        return 1.0 if status.winner is node.to_play else -1.0

    @staticmethod
    def _backup(path: list[tuple[Node, Move]], value: float) -> None:
        # value 是叶子走棋方视角；交替走子，逐层取反成各自视角。
        v = value
        for node, move in reversed(path):
            v = -v
            node.child_N[move] += 1
            node.child_W[move] += v

    def _add_dirichlet_noise(self, node: Node) -> None:
        if node.noise_added or not node.priors:
            return
        moves = list(node.priors.keys())
        noise = self.rng.dirichlet([self.config.dirichlet_alpha] * len(moves))
        eps = self.config.dirichlet_epsilon
        for move, nz in zip(moves, noise):
            node.priors[move] = (1 - eps) * node.priors[move] + eps * float(nz)
        node.noise_added = True
