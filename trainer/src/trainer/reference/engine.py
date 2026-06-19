from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TypeAlias

# 棋盘几何与规则上限常量统一在 config 维护；此处重新导出以保持 engine 的公开接口不变。
from ..config import (
    ACTION_SPACE_SIZE,
    BOARD_HEIGHT,
    BOARD_WIDTH,
    MAX_TOTAL_PLIES,
    NO_CAPTURE_DRAW_PLIES,
    SQUARE_COUNT,
)

# 中国象棋 FEN 字符串："<10 行棋盘> <行棋方> [保留字段...] <未吃子半回合数> <完整回合数>"。
# 当前解析器主要使用棋盘、行棋方、未吃子半回合数和完整回合数。
ChineseChessFen: TypeAlias = str

# 捉子关系表：被攻击方棋子 id -> 被攻击方及其攻击方信息。
AttackedTargetsById: TypeAlias = dict[int, "AttackedTarget"]


class Color(Enum):
    RED = "red"
    BLACK = "black"

    def opposite(self) -> "Color":
        return Color.BLACK if self is Color.RED else Color.RED

    @property
    def fen_side(self) -> str:
        return "r" if self is Color.RED else "b"

    @classmethod
    def from_fen_side(cls, value: str) -> "Color":
        if value == "r":
            return cls.RED
        if value == "b":
            return cls.BLACK
        raise ValueError(f"Invalid side to move: {value}")


class PieceKind(Enum):
    KING = "king"
    ADVISOR = "advisor"
    ELEPHANT = "elephant"
    HORSE = "horse"
    ROOK = "rook"
    CANNON = "cannon"
    PAWN = "pawn"


FEN_TO_KIND = {
    "k": PieceKind.KING,
    "a": PieceKind.ADVISOR,
    "e": PieceKind.ELEPHANT,
    "h": PieceKind.HORSE,
    "r": PieceKind.ROOK,
    "c": PieceKind.CANNON, 
    "p": PieceKind.PAWN,
}

KIND_TO_FEN = {kind: fen for fen, kind in FEN_TO_KIND.items()}


@dataclass(frozen=True)
class Piece:
    """棋子实体：piece_id 标识同一个实体棋子，color/kind 描述棋子的值。

    两个红车的 color/kind 相同，但 piece_id 不同；长将、长捉等规则靠
    piece_id 跨局面追踪“同一个”棋子。
    """

    piece_id: int
    color: Color
    kind: PieceKind

    @classmethod
    def from_fen(cls, piece_id: int, value: str) -> "Piece":
        kind = FEN_TO_KIND.get(value.lower())
        if kind is None:
            raise ValueError(f"Invalid piece: {value}")
        color = Color.RED if value.isupper() else Color.BLACK
        return cls(piece_id, color, kind)

    def to_fen(self) -> str:
        value = KIND_TO_FEN[self.kind]
        return value.upper() if self.color is Color.RED else value

    def is_same_kind_as(self, other: "Piece") -> bool:
        return self.kind is other.kind


@dataclass(frozen=True)
class Move:
    """一步棋：从源格 (sx, sy) 走到目标格 (tx, ty)。"""

    sx: int
    sy: int
    tx: int
    ty: int

    @property
    def source(self) -> tuple[int, int]:
        return (self.sx, self.sy)

    @property
    def target(self) -> tuple[int, int]:
        return (self.tx, self.ty)


@dataclass(frozen=True)
class Attacker:
    """某个能合法吃掉目标子的攻击子。"""

    piece: Piece
    square: tuple[int, int]
    target_can_recapture: bool


@dataclass(frozen=True)
class AttackedTarget:
    """某个被 actor 攻击可吃的目标子，以及相关规则事实。"""

    target: Piece
    square: tuple[int, int]
    attackers: tuple[Attacker, ...]
    has_true_root: bool


class MoveValidationReason(Enum):
    """走法不合法的原因。"""

    # 起点或终点不在棋盘范围内。
    OUTSIDE_BOARD = "outside_board"

    # 起点没有棋子。
    NO_PIECE = "no_piece"

    # 起点棋子不是当前行棋方。
    WRONG_SIDE = "wrong_side"

    # 目标格是己方棋子。
    CAPTURE_OWN_PIECE = "capture_own_piece"

    # 不符合该棋子的基础走法规则。
    INVALID_PIECE_MOVE = "invalid_piece_move"

    # 走完后将帅直接照面，局面非法。
    KINGS_FACE = "kings_face"

    # 走完后本方将帅仍被攻击，或主动暴露在攻击下。
    LEAVES_KING_IN_CHECK = "leaves_king_in_check"


@dataclass(frozen=True)
class MoveValidationResult:
    """走法校验结果；普通非法走法用 reason 表达，不靠异常控制流程。"""

    is_legal: bool
    reason: MoveValidationReason | None = None


class GameStatusReason(Enum):
    """终局原因。"""

    # 将死：当前行棋方被将军，且没有任何合法走法可以解除将军。
    CHECKMATE = "checkmate"

    # 困毙：当前行棋方没有任何合法走法，但并未处于被将军状态。
    STALEMATE = "stalemate"

    # 长将：重复归原中一方持续将军，且按循环禁例判负。
    PERPETUAL_CHECK = "perpetual_check"

    # 长捉：重复归原中一方持续捉无真根棋子，且按循环禁例判负。
    PERPETUAL_CHASE = "perpetual_chase"

    # 双方同等犯例，或重复归原但双方均不构成长将/长捉，判和。
    MUTUAL_PERPETUAL = "mutual_perpetual"

    # 自然限着：连续达到规定半回合数均未吃子，判和。
    FIFTY_MOVE = "fifty_move"

    # 工程步数上限：达到最大半回合数仍未分出胜负，判和。
    MAX_MOVES = "max_moves"


@dataclass(frozen=True)
class GameStatus:
    """当前局面的终局状态。"""

    is_terminal: bool
    reason: GameStatusReason | None = None
    winner: Color | None = None


def square_index(x: int, y: int) -> int:
    return y * BOARD_WIDTH + x


def move_to_action_id(move: Move) -> int:
    return square_index(move.sx, move.sy) * SQUARE_COUNT + square_index(move.tx, move.ty)


def action_id_to_move(action_id: int) -> Move:
    if not 0 <= action_id < ACTION_SPACE_SIZE:
        raise ValueError(f"Invalid action id: {action_id}")
    source, target = divmod(action_id, SQUARE_COUNT)
    sy, sx = divmod(source, BOARD_WIDTH)
    ty, tx = divmod(target, BOARD_WIDTH)
    return Move(sx, sy, tx, ty)


def in_board(x: int, y: int) -> bool:
    return 0 <= x < BOARD_WIDTH and 0 <= y < BOARD_HEIGHT


def in_palace(color: Color, x: int, y: int) -> bool:
    if x < 3 or x > 5:
        return False
    return 7 <= y <= 9 if color is Color.RED else 0 <= y <= 2


def crossed_river(color: Color, y: int) -> bool:
    return y <= 4 if color is Color.RED else y >= 5


@dataclass(frozen=True)
class Position:
    """不可变的棋盘局面。

    坐标使用 (x, y)：x 是从左到右的列，范围 0..8；y 是从黑方到红方的行，
    范围 0..9。二维棋盘按 [y][x] 访问。
    """

    # 主棋盘：每个格子存棋子实体（含 piece_id）；空格为 None。
    board: tuple[tuple[Piece | None, ...], ...]

    # 当前轮到哪一方走棋。
    side_to_move: Color                                                                                                                                                                                                                                                                                                                                                                       

    # 反向索引，方便快速查找：棋子 id -> 当前 (x, y)。被吃掉的棋子会从这里移除。
    piece_positions: dict[int, tuple[int, int]]

    # 缓存双方将帅的位置，判断是否被将军时不用先扫描整张棋盘。
    king_square_by_color: dict[Color, tuple[int, int]]

    # 连续未吃子的半回合数，用于无吃子和棋规则。
    no_capture_plies: int = 0

    # 完整回合数，黑方走完后递增。
    fullmove_number: int = 1

    @classmethod
    def starting(cls) -> "Position":
        return cls.from_fen("rheakaehr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RHEAKAEHR r")

    @classmethod
    def from_fen(cls, fen: ChineseChessFen) -> "Position":
        parts = fen.split()
        if len(parts) < 2:
            raise ValueError(f"Invalid FEN: {fen}")
        rows = parts[0].split("/")
        if len(rows) != BOARD_HEIGHT:
            raise ValueError(f"Invalid FEN board height: {fen}")

        board_rows: list[list[Piece | None]] = []
        piece_positions: dict[int, tuple[int, int]] = {}
        king_square_by_color: dict[Color, tuple[int, int]] = {}
        next_id = 0

        for y, row in enumerate(rows):
            board_row: list[Piece | None] = []
            for char in row:
                if char.isdigit():
                    empty_count = int(char)
                    board_row.extend([None] * empty_count)
                    continue

                piece = Piece.from_fen(next_id, char)
                x = len(board_row)
                board_row.append(piece)
                piece_positions[next_id] = (x, y)
                if piece.kind is PieceKind.KING:
                    king_square_by_color[piece.color] = (x, y)
                next_id += 1

            if len(board_row) != BOARD_WIDTH:
                raise ValueError(f"Invalid FEN row width: {row}")
            board_rows.append(board_row)

        side_to_move = Color.from_fen_side(parts[1])
        no_capture_plies = int(parts[4]) if len(parts) >= 5 else 0
        fullmove_number = int(parts[5]) if len(parts) >= 6 else 1

        return cls(
            tuple(tuple(row) for row in board_rows),
            side_to_move,
            piece_positions,
            king_square_by_color,
            no_capture_plies,
            fullmove_number,
        )
                 
    def to_fen(self) -> ChineseChessFen:
        rows: list[str] = []
        for y in range(BOARD_HEIGHT):
            row = ""
            empty_count = 0
            for x in range(BOARD_WIDTH):
                piece = self.board[y][x]
                if piece is None:
                    empty_count += 1
                    continue
                if empty_count:
                    row += str(empty_count)
                    empty_count = 0
                row += piece.to_fen()
            if empty_count:
                row += str(empty_count)
            rows.append(row)
        return f"{'/'.join(rows)} {self.side_to_move.fen_side}"

    def repetition_key(self) -> ChineseChessFen:
        return self.to_fen()

    def piece_at(self, x: int, y: int) -> Piece | None:
        if not in_board(x, y):
            return None
        return self.board[y][x]

    def require_piece_at(self, x: int, y: int) -> Piece:
        piece = self.piece_at(x, y)
        if piece is None:
            raise ValueError(f"No piece at {(x, y)}")
        return piece

    def _apply_move(self, move: Move, piece: Piece) -> "Position":
        captured = self.piece_at(move.tx, move.ty)

        board = [list(row) for row in self.board]
        piece_positions = dict(self.piece_positions)
        king_square_by_color = dict(self.king_square_by_color)

        board[move.sy][move.sx] = None
        board[move.ty][move.tx] = piece
        piece_positions[piece.piece_id] = move.target
        if captured is not None:
            piece_positions.pop(captured.piece_id, None)
        if piece.kind is PieceKind.KING:
            king_square_by_color[piece.color] = move.target

        next_side = self.side_to_move.opposite()
        next_no_capture_plies = 0 if captured is not None else self.no_capture_plies + 1
        next_fullmove = self.fullmove_number + (1 if self.side_to_move is Color.BLACK else 0)
        return Position(
            tuple(tuple(row) for row in board),
            next_side,
            piece_positions,
            king_square_by_color,
            next_no_capture_plies,
            next_fullmove,
        )

    def validate_move(self, move: Move, check_side_to_move: bool = True) -> MoveValidationResult:
        if not in_board(move.sx, move.sy) or not in_board(move.tx, move.ty):
            return MoveValidationResult(False, MoveValidationReason.OUTSIDE_BOARD)
        piece = self.piece_at(move.sx, move.sy)
        if piece is None:
            return MoveValidationResult(False, MoveValidationReason.NO_PIECE)
        if check_side_to_move and piece.color is not self.side_to_move:
            return MoveValidationResult(False, MoveValidationReason.WRONG_SIDE)
        captured = self.piece_at(move.tx, move.ty)
        if captured is not None and captured.color is piece.color:
            return MoveValidationResult(False, MoveValidationReason.CAPTURE_OWN_PIECE)
        if move not in self._pseudo_moves_for_piece(move.sx, move.sy, piece):
            return MoveValidationResult(False, MoveValidationReason.INVALID_PIECE_MOVE)

        next_position = self._apply_move(move, piece)
        if next_position._kings_face():
            return MoveValidationResult(False, MoveValidationReason.KINGS_FACE)
        if next_position.is_in_check(piece.color):
            return MoveValidationResult(False, MoveValidationReason.LEAVES_KING_IN_CHECK)
        return MoveValidationResult(True)

    def is_legal_move(self, move: Move, check_side_to_move: bool = True) -> bool:
        return self.validate_move(move, check_side_to_move).is_legal

    def make_move(self, move: Move) -> "Position":
        validation = self.validate_move(move)
        if not validation.is_legal:
            reason = validation.reason.value if validation.reason is not None else "unknown"
            raise ValueError(f"Illegal move ({reason}): {move}")
        piece = self.require_piece_at(move.sx, move.sy)
        return self._apply_move(move, piece)

    def legal_move_mask(self) -> list[bool]:
        mask = [False] * ACTION_SPACE_SIZE
        for move in self.legal_moves():
            mask[move_to_action_id(move)] = True
        return mask

    def legal_moves(self) -> list[Move]:
        moves: list[Move] = []
        for y in range(BOARD_HEIGHT):
            for x in range(BOARD_WIDTH):
                piece = self.piece_at(x, y)
                if piece is None or piece.color is not self.side_to_move:
                    continue
                for move in self._pseudo_moves_for_piece(x, y, piece):
                    if self.is_legal_move(move):
                        moves.append(move)
        return moves

    def _pseudo_moves_for_piece(self, x: int, y: int, piece: Piece) -> list[Move]:
        match piece.kind:
            case PieceKind.KING:
                return self._king_moves(x, y, piece)
            case PieceKind.ADVISOR:
                return self._advisor_moves(x, y, piece)
            case PieceKind.ELEPHANT:
                return self._elephant_moves(x, y, piece)
            case PieceKind.HORSE:
                return self._horse_moves(x, y, piece)
            case PieceKind.ROOK:
                return self._rook_moves(x, y, piece)
            case PieceKind.CANNON:
                return self._cannon_moves(x, y, piece)
            case PieceKind.PAWN:
                return self._pawn_moves(x, y, piece)

    def _can_land(self, x: int, y: int, color: Color) -> bool:
        target = self.piece_at(x, y)
        return target is None or target.color is not color

    def _king_moves(self, x: int, y: int, piece: Piece) -> list[Move]:
        moves = []
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            tx, ty = x + dx, y + dy
            if in_palace(piece.color, tx, ty) and self._can_land(tx, ty, piece.color):
                moves.append(Move(x, y, tx, ty))
        return moves

    def _advisor_moves(self, x: int, y: int, piece: Piece) -> list[Move]:
        moves = []
        for dx, dy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            tx, ty = x + dx, y + dy
            if in_palace(piece.color, tx, ty) and self._can_land(tx, ty, piece.color):
                moves.append(Move(x, y, tx, ty))
        return moves

    def _elephant_moves(self, x: int, y: int, piece: Piece) -> list[Move]:
        moves = []
        for dx, dy in ((2, 2), (2, -2), (-2, 2), (-2, -2)):
            tx, ty = x + dx, y + dy
            eye_x, eye_y = x + dx // 2, y + dy // 2
            if not in_board(tx, ty):
                continue
            if piece.color is Color.RED and ty < 5:
                continue
            if piece.color is Color.BLACK and ty > 4:
                continue
            if self.piece_at(eye_x, eye_y) is None and self._can_land(tx, ty, piece.color):
                moves.append(Move(x, y, tx, ty))
        return moves

    def _horse_moves(self, x: int, y: int, piece: Piece) -> list[Move]:
        candidates = (
            (1, 2, 0, 1),
            (-1, 2, 0, 1),
            (1, -2, 0, -1),
            (-1, -2, 0, -1),
            (2, 1, 1, 0),
            (2, -1, 1, 0),
            (-2, 1, -1, 0),
            (-2, -1, -1, 0),
        )
        moves = []
        for dx, dy, leg_dx, leg_dy in candidates:
            tx, ty = x + dx, y + dy
            leg_x, leg_y = x + leg_dx, y + leg_dy
            if in_board(tx, ty) and self.piece_at(leg_x, leg_y) is None and self._can_land(tx, ty, piece.color):
                moves.append(Move(x, y, tx, ty))
        return moves

    def _rook_moves(self, x: int, y: int, piece: Piece) -> list[Move]:
        return self._sliding_moves(x, y, piece, cannon=False)

    def _cannon_moves(self, x: int, y: int, piece: Piece) -> list[Move]:
        return self._sliding_moves(x, y, piece, cannon=True)

    def _sliding_moves(self, x: int, y: int, piece: Piece, cannon: bool) -> list[Move]:
        moves = []
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            tx, ty = x + dx, y + dy
            seen_screen = False
            while in_board(tx, ty):
                target = self.piece_at(tx, ty)
                if not cannon:
                    if target is None:
                        moves.append(Move(x, y, tx, ty))
                    else:
                        if target.color is not piece.color:
                            moves.append(Move(x, y, tx, ty))
                        break
                else:
                    if not seen_screen:
                        if target is None:
                            moves.append(Move(x, y, tx, ty))
                        else:
                            seen_screen = True
                    else:
                        if target is not None:
                            if target.color is not piece.color:
                                moves.append(Move(x, y, tx, ty))
                            break
                tx += dx
                ty += dy
        return moves

    def _pawn_moves(self, x: int, y: int, piece: Piece) -> list[Move]:
        directions = [(0, -1)] if piece.color is Color.RED else [(0, 1)]
        if crossed_river(piece.color, y):
            directions += [(-1, 0), (1, 0)]
        moves = []
        for dx, dy in directions:
            tx, ty = x + dx, y + dy
            if in_board(tx, ty) and self._can_land(tx, ty, piece.color):
                moves.append(Move(x, y, tx, ty))
        return moves

    def is_in_check(self, color: Color) -> bool:
        king_square = self.king_square_by_color.get(color)
        if king_square is None:
            raise ValueError(f"Missing king for {color.value}")
        if self._kings_face():
            raise ValueError("Illegal position: kings face each other")

        kx, ky = king_square
        attacker = color.opposite()
        for y in range(BOARD_HEIGHT):
            for x in range(BOARD_WIDTH):
                piece = self.piece_at(x, y)
                if piece is None or piece.color is not attacker:
                    continue
                if self._piece_attacks_square(x, y, piece, kx, ky):
                    return True
        return False
        

    def _kings_face(self) -> bool:
        red_king = self.king_square_by_color.get(Color.RED)
        black_king = self.king_square_by_color.get(Color.BLACK)
        if red_king is None or black_king is None or red_king[0] != black_king[0]:
            return False
        x = red_king[0]
        y1, y2 = sorted((red_king[1], black_king[1]))
        return all(self.piece_at(x, y) is None for y in range(y1 + 1, y2))

    def _piece_attacks_square(self, x: int, y: int, piece: Piece, tx: int, ty: int) -> bool:
        target = self.piece_at(tx, ty)
        if target is not None and target.color is piece.color:
            return False
        if piece.kind is PieceKind.CANNON:
            # 这里判定的是能否攻击目标格。炮的移动和吃子规则不同：
            # 其他棋子的“能走到”基本等价于“能攻击到”，可复用伪合法走法。
            return self._cannon_attacks(x, y, tx, ty)
        return any(move.target == (tx, ty) for move in self._pseudo_moves_for_piece(x, y, piece))
 
    def _cannon_attacks(self, x: int, y: int, tx: int, ty: int) -> bool:
        if x != tx and y != ty:
            return False
        if (x, y) == (tx, ty):
            return False
        between = self._pieces_between(x, y, tx, ty)
        return between == 1

    def _pieces_between(self, x: int, y: int, tx: int, ty: int) -> int:
        dx = 0 if x == tx else (1 if tx > x else -1)
        dy = 0 if y == ty else (1 if ty > y else -1)
        cx, cy = x + dx, y + dy
        count = 0
        while (cx, cy) != (tx, ty):
            if self.piece_at(cx, cy) is not None:
                count += 1
            cx += dx
            cy += dy
        return count

    def status(self) -> GameStatus:
        legal_moves = self.legal_moves()
        if legal_moves:
            return GameStatus(False)
        winner = self.side_to_move.opposite()
        if self.is_in_check(self.side_to_move):
            return GameStatus(True, GameStatusReason.CHECKMATE, winner)
        return GameStatus(True, GameStatusReason.STALEMATE, winner)


@dataclass(frozen=True)
class MoveRecord:
    position_before: Position
    position_after: Position
    move: Move
    actor: Color

    def attacked_targets(self) -> AttackedTargetsById:
        attacked_targets: AttackedTargetsById = {}
        attackers_by_target: dict[int, list[Attacker]] = {}
        for attacker_id, (ax, ay) in self.position_after.piece_positions.items():
            attacker = self.position_after.require_piece_at(ax, ay)
            if attacker.color is not self.actor:
                continue
            for target_id, (tx, ty) in self.position_after.piece_positions.items():
                target = self.position_after.require_piece_at(tx, ty)
                if target.color is self.actor or target.kind is PieceKind.KING:
                    continue
                if self.position_after.is_legal_move(Move(ax, ay, tx, ty), check_side_to_move=False):
                    attackers_by_target.setdefault(target_id, []).append(
                        Attacker(
                            attacker,
                            (ax, ay),
                            self.position_after.is_legal_move(Move(tx, ty, ax, ay), check_side_to_move=False),
                        )
                    )
        for target_id, attackers in attackers_by_target.items():
            target_square = self.position_after.piece_positions[target_id]
            target = self.position_after.require_piece_at(*target_square)
            attacked_targets[target_id] = AttackedTarget(
                target,
                target_square,
                tuple(attackers),
                self._has_true_root(target_id),
            )
        return attacked_targets

    def _has_true_root(self, target_id: int) -> bool:
        target_pos = self.position_after.piece_positions[target_id]
        tx, ty = target_pos
        target = self.position_after.require_piece_at(tx, ty)

        for attacker_id, attacker_pos in self.position_after.piece_positions.items():
            ax, ay = attacker_pos
            attacker = self.position_after.require_piece_at(ax, ay)
            if attacker.color is target.color:
                continue
            capture_move = Move(ax, ay, tx, ty)
            if not self.position_after.is_legal_move(capture_move, check_side_to_move=False):
                continue
            captured_position = self.position_after._apply_move(capture_move, attacker)
            for defender_id, defender_pos in captured_position.piece_positions.items():
                dx, dy = defender_pos
                defender = captured_position.require_piece_at(dx, dy)
                if defender.color is not target.color:
                    continue
                if captured_position.is_legal_move(Move(dx, dy, tx, ty), check_side_to_move=False):
                    return True
        return False


@dataclass
class GameState:
    position: Position

    # 已走过的每一步，供重复局面、长将、长捉等需要历史的规则使用。
    history: list[MoveRecord]

    # 局面 key 出现次数，用来快速判断当前局面是否重复。
    repetition_counts: dict[str, int]

    # 局面 key -> 该局面出现过的历史位置。重复时用最近两次位置截取中间走子片段，
    # 再分析这段里是否存在长将、长捉或互打。
    key_indices: dict[str, list[int]]

    @classmethod
    def from_position(cls, position: Position) -> "GameState":
        key = position.repetition_key()
        return cls(position, [], {key: 1}, {key: [0]})

    def make_move(self, move: Move) -> "GameState":
        next_position = self.position.make_move(move)
        record = MoveRecord(self.position, next_position, move, self.position.side_to_move)
        history = [*self.history, record]
        repetition_counts = dict(self.repetition_counts)
        key_indices = {key: list(value) for key, value in self.key_indices.items()}
        key = next_position.repetition_key()
        repetition_counts[key] = repetition_counts.get(key, 0) + 1
        key_indices.setdefault(key, []).append(len(history))
        return GameState(next_position, history, repetition_counts, key_indices)

    def status(self) -> GameStatus:
        position_status = self.position.status()
        if position_status.is_terminal:
            return position_status

        key = self.position.repetition_key()
        if self.repetition_counts.get(key, 0) >= 3:
            return self._repetition_status(key)
        if self.position.no_capture_plies >= NO_CAPTURE_DRAW_PLIES:
            return GameStatus(True, GameStatusReason.FIFTY_MOVE, None)
        if len(self.history) >= MAX_TOTAL_PLIES:
            return GameStatus(True, GameStatusReason.MAX_MOVES, None)
        return GameStatus(False)

    def _repetition_status(self, key: str) -> GameStatus:
        indices = self.key_indices[key]
        if len(indices) < 2:
            return GameStatus(False)
        start, end = indices[-2], indices[-1]
        segment = self.history[start:end]
        red_violation = self._violation_for(Color.RED, segment)
        black_violation = self._violation_for(Color.BLACK, segment)

        if red_violation == black_violation:
            return GameStatus(True, GameStatusReason.MUTUAL_PERPETUAL, None)
        if red_violation == "check" and black_violation != "check":
            return GameStatus(True, GameStatusReason.PERPETUAL_CHECK, Color.BLACK)
        if black_violation == "check" and red_violation != "check":
            return GameStatus(True, GameStatusReason.PERPETUAL_CHECK, Color.RED)
        if red_violation == "chase":
            return GameStatus(True, GameStatusReason.PERPETUAL_CHASE, Color.BLACK)
        if black_violation == "chase":
            return GameStatus(True, GameStatusReason.PERPETUAL_CHASE, Color.RED)
        return GameStatus(True, GameStatusReason.MUTUAL_PERPETUAL, None)

    def _violation_for(self, color: Color, segment: list[MoveRecord]) -> str | None:
        own_records = [record for record in segment if record.actor is color]
        if not own_records:
            raise ValueError(f"Repetition segment has no moves by {color.value}")
        if all(record.position_after.is_in_check(color.opposite()) for record in own_records):
            return "check"
        if self._is_perpetual_chase(own_records):
            return "chase"
        return None

    def _is_perpetual_chase(self, records: list[MoveRecord]) -> bool:
        attacked_target_ids_by_record: list[set[int]] = []
        attacked_targets_by_id_by_record: dict[int, list[AttackedTarget]] = {}

        for record in records:
            attacked_targets = record.attacked_targets()
            attacked_target_ids = set(attacked_targets)
            if not attacked_target_ids:
                return False
            attacked_target_ids_by_record.append(attacked_target_ids)
            for target_id, attacked_target in attacked_targets.items():
                attacked_targets_by_id_by_record.setdefault(target_id, []).append(attacked_target)

        candidates = set.intersection(*attacked_target_ids_by_record)
        for target_id in candidates:
            attacked_targets_by_record = attacked_targets_by_id_by_record[target_id]
            if self._target_exempt_from_chase(attacked_targets_by_record):
                continue
            # 马/炮长捉有根车仍按长捉处理，优先于真根豁免。
            if self._target_is_rook_chased_by_horse_or_cannon(attacked_targets_by_record):
                return True
            if all(not attacked_target.has_true_root for attacked_target in attacked_targets_by_record):
                return True
        return False

    def _target_exempt_from_chase(self, attacked_targets_by_record: list[AttackedTarget]) -> bool:
        final_attacked_target = attacked_targets_by_record[-1]
        target = final_attacked_target.target
        if target.kind is PieceKind.PAWN and not crossed_river(target.color, final_attacked_target.square[1]):
            return True

        # 规则允许将/帅和兵/卒长捉，全部攻击子都属于这两类时豁免。
        if all(
            attacker.piece.kind in (PieceKind.KING, PieceKind.PAWN)
            for attacked_target in attacked_targets_by_record
            for attacker in attacked_target.attackers
        ):
            return True

        return self._is_same_kind_exchange(attacked_targets_by_record)

    def _is_same_kind_exchange(self, attacked_targets_by_record: list[AttackedTarget]) -> bool:
        for attacked_target in attacked_targets_by_record:
            if not any(
                attacker.piece.is_same_kind_as(attacked_target.target)
                and attacker.target_can_recapture
                for attacker in attacked_target.attackers
            ):
                return False
        return True

    def _target_is_rook_chased_by_horse_or_cannon(self, attacked_targets_by_record: list[AttackedTarget]) -> bool:
        for attacked_target in attacked_targets_by_record:
            if attacked_target.target.kind is not PieceKind.ROOK:
                return False
            if not any(
                attacker.piece.kind in (PieceKind.HORSE, PieceKind.CANNON)
                for attacker in attacked_target.attackers
            ):
                return False
        return True