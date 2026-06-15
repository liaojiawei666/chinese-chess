from __future__ import annotations

import numpy as np

from config import (
    ACTION_SPACE_SIZE,
    BOARD_HEIGHT,
    BOARD_WIDTH,
    INPUT_CHANNELS,
    N_HISTORY,
    NO_CAPTURE_DRAW_PLIES,
    PLANES_PER_FRAME,
)
from engine import (
    Color,
    GameState,
    Move,
    PieceKind,
    Position,
    action_id_to_move,
    move_to_action_id,
)


# 子种在单帧内的通道顺序（对应文档：帅、车、马、炮、仕、相、兵）。
KIND_ORDER: tuple[PieceKind, ...] = (
    PieceKind.KING,
    PieceKind.ROOK,
    PieceKind.HORSE,
    PieceKind.CANNON,
    PieceKind.ADVISOR,
    PieceKind.ELEPHANT,
    PieceKind.PAWN,
)

KIND_TO_INDEX = {kind: index for index, kind in enumerate(KIND_ORDER)}


def _flip_y(y: int) -> int:
    """上下翻转一行坐标。翻转是自身的逆运算（翻两次回到原值）。"""
    return BOARD_HEIGHT - 1 - y


def canonical_square(x: int, y: int, perspective: Color) -> tuple[int, int]:
    """把真实坐标转成当前走棋方视角的坐标。

    红方视角不变；黑方视角上下翻转，使己方永远在棋盘底部。
    因为翻转是对合运算，本函数同时充当正变换和逆变换。
    """
    if perspective is Color.BLACK:
        return x, _flip_y(y)
    return x, y


def canonical_move(move: Move, perspective: Color) -> Move:
    """把一步真实走法转成 canonical 视角下的走法。"""
    sx, sy = canonical_square(move.sx, move.sy, perspective)
    tx, ty = canonical_square(move.tx, move.ty, perspective)
    return Move(sx, sy, tx, ty)


def to_canonical_action(move: Move, perspective: Color) -> int:
    """真实走法 -> canonical 视角下的 action_id（与网络 policy 同坐标系）。"""
    return move_to_action_id(canonical_move(move, perspective))


def from_canonical_action(action_id: int, perspective: Color) -> Move:
    """canonical 视角下的 action_id -> 真实走法（用于把网络选的动作还原到真实棋盘）。"""
    return canonical_move(action_id_to_move(action_id), perspective)


def legal_mask(position: Position) -> np.ndarray:
    """返回 (8100,) 的 bool mask，标记 canonical 视角下的合法动作。

    用当前走棋方视角，使其与网络输出的 policy logits 对齐，便于屏蔽非法动作。
    """
    perspective = position.side_to_move
    mask = np.zeros(ACTION_SPACE_SIZE, dtype=bool)
    for move in position.legal_moves():
        mask[to_canonical_action(move, perspective)] = True
    return mask


def _encode_frame(position: Position, perspective: Color) -> np.ndarray:
    """把单个局面编码成 (14, 10, 9) 的帧：前 7 通道己方，后 7 通道对方。

    perspective 决定哪一方算“己方”，以及是否上下翻转，与该局面自身轮到谁走无关。
    """
    frame = np.zeros((PLANES_PER_FRAME, BOARD_HEIGHT, BOARD_WIDTH), dtype=np.float32)
    flip = perspective is Color.BLACK
    for y in range(BOARD_HEIGHT):
        for x in range(BOARD_WIDTH):
            piece = position.board[y][x]
            if piece is None:
                continue
            kind_index = KIND_TO_INDEX[piece.kind]
            channel = kind_index if piece.color is perspective else PLANES_PER_FRAME // 2 + kind_index
            ry = _flip_y(y) if flip else y
            frame[channel, ry, x] = 1.0
    return frame


def _recent_positions(state: GameState) -> list[Position]:
    """取最近 N_HISTORY 个局面，最新的在前；不足的留给调用方补空帧。"""
    positions = [state.position]
    for record in reversed(state.history):
        if len(positions) == N_HISTORY:
            break
        positions.append(record.position_before)
    return positions


def encode(state: GameState) -> np.ndarray:
    """把对局状态编码成 (99, 10, 9) 的 canonical 张量。

    视角统一为当前走棋方：红方走棋按原样，黑方走棋上下翻转 + 红黑通道互换。
    7 帧历史都用同一视角编码，开局不足 7 步时较旧的帧保持全 0。
    最后 1 个通道是归一化的连续未吃子步数。
    """
    perspective = state.position.side_to_move
    tensor = np.zeros((INPUT_CHANNELS, BOARD_HEIGHT, BOARD_WIDTH), dtype=np.float32)

    for index, position in enumerate(_recent_positions(state)):
        start = index * PLANES_PER_FRAME
        tensor[start:start + PLANES_PER_FRAME] = _encode_frame(position, perspective)

    no_capture_ratio = min(state.position.no_capture_plies / NO_CAPTURE_DRAW_PLIES, 1.0)
    tensor[N_HISTORY * PLANES_PER_FRAME] = no_capture_ratio
    return tensor
