from __future__ import annotations

import numpy as np
import torch

from encoder import encode, to_canonical_action
from engine import GameState, Move
from network import PolicyValueNet


def resolve_device(name: str | torch.device) -> torch.device:
    """把配置里的设备字符串解析成 torch.device；"auto" 自动选 cuda/cpu。"""
    if isinstance(name, torch.device):
        return name
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def priors_from_logits(
    policy_logits: np.ndarray, state: GameState
) -> dict[Move, float]:
    """把网络 policy logits（canonical 动作空间，shape (ACTION_SPACE_SIZE,)）
    映射成合法走法上的先验概率字典。

    与 Evaluator.evaluate 内部逻辑一致，抽出来给跨进程推理（RemoteEvaluator）复用：
    推理服务只回传原始 logits（纯张量好序列化），合法过滤/softmax/还原 Move 在 worker 侧做。
    """
    moves = state.position.legal_moves()
    if not moves:
        return {}
    perspective = state.position.side_to_move
    canonical_ids = [to_canonical_action(move, perspective) for move in moves]
    legal = policy_logits[canonical_ids].astype(np.float64)
    legal -= legal.max()  # 数值稳定
    exp = np.exp(legal)
    probs = exp / exp.sum()
    return {move: float(p) for move, p in zip(moves, probs)}


class Evaluator:
    """网络与 MCTS 之间的推理适配层：局面进、(先验, value) 出。

    职责（功能 6–11）：把 GameState 编码成 canonical 张量、搬上 device、前向、
    在合法动作上做 softmax，并把 canonical 动作还原成真实 Move 再返回先验。
    这样 canonical 视角完全封装在这一层内，MCTS 只看到真实 Move，无需关心翻转/编码。

    注意：本类用于推理（自对弈/搜索/评估），构造时即把网络置 eval 模式。
    训练侧请直接使用 PolicyValueNet 计算梯度，不要经过本类。
    """

    def __init__(self, net: PolicyValueNet, device: str | torch.device = "cpu") -> None:
        self.device = torch.device(device)
        self.net = net.to(self.device)
        self.net.eval()

    @torch.no_grad()
    def evaluate(self, state: GameState) -> tuple[dict[Move, float], float]:
        """评估单个局面。

        返回 (priors, value)：
        - priors: {真实 Move: 先验概率}，仅覆盖当前合法走法，概率和为 1（无合法走法时为空）。
        - value: 当前走棋方视角的胜负倾向，范围 [-1, 1]。
        """
        x = torch.from_numpy(encode(state)).unsqueeze(0).to(self.device)
        policy_logits, value = self.net(x)
        value_scalar = float(value.item())

        moves = state.position.legal_moves()
        if not moves:
            return {}, value_scalar

        # 网络 policy 在 canonical 坐标系，按每个走法的 canonical action_id 取对应 logit。
        perspective = state.position.side_to_move
        canonical_ids = [to_canonical_action(move, perspective) for move in moves]
        legal_logits = policy_logits[0, canonical_ids]
        probs = torch.softmax(legal_logits, dim=0)
        priors = {move: float(prob) for move, prob in zip(moves, probs)}
        return priors, value_scalar

    def evaluate_batch(
        self, states: list[GameState]
    ) -> list[tuple[dict[Move, float], float]]:
        """批量评估：把多个局面拼成一个 batch 一次前向，降低调用开销。

        第一阶段先只用单局面 evaluate 跑通正确性，这里预留接口待性能优化阶段实现。
        """
        raise NotImplementedError("evaluate_batch 待性能优化阶段实现；当前用 evaluate 逐个评估")
