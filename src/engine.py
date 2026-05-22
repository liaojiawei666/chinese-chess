from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


BOARD_WIDTH = 9
BOARD_HEIGHT = 10
SQUARE_COUNT = BOARD_WIDTH * BOARD_HEIGHT
ACTION_SPACE_SIZE = SQUARE_COUNT * SQUARE_COUNT
MAX_TOTAL_PLIES = 300
NO_CAPTURE_DRAW_PLIES = 100


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
    color: Color
    kind: PieceKind

    @classmethod
    def from_fen(cls, value: str) -> "Piece":
        kind = FEN_TO_KIND.get(value.lower())
        if kind is None:
            raise ValueError(f"Invalid piece: {value}")
        color = Color.RED if value.isupper() else Color.BLACK
        return cls(color, kind)

    def to_fen(self) -> str:
        value = KIND_TO_FEN[self.kind]
        return value.upper() if self.color is Color.RED else value


@dataclass(frozen=True)
class Move:
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
class GameStatus:
    is_terminal: bool
    reason: str | None = None
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
    board: tuple[tuple[Piece | None, ...], ...]
    side_to_move: Color
    piece_ids: tuple[tuple[int | None, ...], ...]
    piece_positions: dict[int, tuple[int, int]]
    king_square_by_color: dict[Color, tuple[int, int]]
    halfmove_clock: int = 0
    fullmove_number: int = 1

    @classmethod
    def starting(cls) -> "Position":
        return cls.from_fen("rheakaehr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RHEAKAEHR r")

    @classmethod
    def from_fen(cls, fen: str) -> "Position":
        parts = fen.split()
        if len(parts) < 2:
            raise ValueError(f"Invalid FEN: {fen}")
        rows = parts[0].split("/")
        if len(rows) != BOARD_HEIGHT:
            raise ValueError(f"Invalid FEN board height: {fen}")

        board_rows: list[list[Piece | None]] = []
        id_rows: list[list[int | None]] = []
        piece_positions: dict[int, tuple[int, int]] = {}
        king_square_by_color: dict[Color, tuple[int, int]] = {}
        next_id = 0

        for y, row in enumerate(rows):
            board_row: list[Piece | None] = []
            id_row: list[int | None] = []
            for char in row:
                if char.isdigit():
                    empty_count = int(char)
                    board_row.extend([None] * empty_count)
                    id_row.extend([None] * empty_count)
                    continue

                piece = Piece.from_fen(char)
                x = len(board_row)
                board_row.append(piece)
                id_row.append(next_id)
                piece_positions[next_id] = (x, y)
                if piece.kind is PieceKind.KING:
                    king_square_by_color[piece.color] = (x, y)
                next_id += 1

            if len(board_row) != BOARD_WIDTH:
                raise ValueError(f"Invalid FEN row width: {row}")
            board_rows.append(board_row)
            id_rows.append(id_row)

        side_to_move = Color.from_fen_side(parts[1])
        halfmove_clock = int(parts[4]) if len(parts) >= 5 else 0
        fullmove_number = int(parts[5]) if len(parts) >= 6 else 1

        return cls(
            tuple(tuple(row) for row in board_rows),
            side_to_move,
            tuple(tuple(row) for row in id_rows),
            piece_positions,
            king_square_by_color,
            halfmove_clock,
            fullmove_number,
        )

    def to_fen(self) -> str:
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

    def repetition_key(self) -> str:
        return self.to_fen()

    def piece_at(self, x: int, y: int) -> Piece | None:
        if not in_board(x, y):
            return None
        return self.board[y][x]

    def piece_id_at(self, x: int, y: int) -> int | None:
        if not in_board(x, y):
            return None
        return self.piece_ids[y][x]

    def _set_square(
        self,
        board: list[list[Piece | None]],
        piece_ids: list[list[int | None]],
        x: int,
        y: int,
        piece: Piece | None,
        piece_id: int | None,
    ) -> None:
        board[y][x] = piece
        piece_ids[y][x] = piece_id

    def make_move(self, move: Move) -> "Position":
        if not in_board(move.sx, move.sy) or not in_board(move.tx, move.ty):
            raise ValueError(f"Move outside board: {move}")
        piece = self.piece_at(move.sx, move.sy)
        piece_id = self.piece_id_at(move.sx, move.sy)
        if piece is None or piece_id is None:
            raise ValueError(f"No piece at {move.source}")
        captured = self.piece_at(move.tx, move.ty)
        captured_id = self.piece_id_at(move.tx, move.ty)
        if captured is not None and captured.color is piece.color:
            raise ValueError(f"Cannot capture own piece: {move}")

        board = [list(row) for row in self.board]
        piece_ids = [list(row) for row in self.piece_ids]
        piece_positions = dict(self.piece_positions)
        king_square_by_color = dict(self.king_square_by_color)

        self._set_square(board, piece_ids, move.sx, move.sy, None, None)
        self._set_square(board, piece_ids, move.tx, move.ty, piece, piece_id)
        piece_positions[piece_id] = move.target
        if captured_id is not None:
            piece_positions.pop(captured_id, None)
        if piece.kind is PieceKind.KING:
            king_square_by_color[piece.color] = move.target

        next_side = self.side_to_move.opposite()
        next_halfmove = 0 if captured is not None else self.halfmove_clock + 1
        next_fullmove = self.fullmove_number + (1 if self.side_to_move is Color.BLACK else 0)
        return Position(
            tuple(tuple(row) for row in board),
            next_side,
            tuple(tuple(row) for row in piece_ids),
            piece_positions,
            king_square_by_color,
            next_halfmove,
            next_fullmove,
        )

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
                    try:
                        next_position = self.make_move(move)
                    except ValueError:
                        continue
                    if not next_position.is_in_check(piece.color):
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
            return True
        if self._kings_face():
            red_king = self.king_square_by_color.get(Color.RED)
            black_king = self.king_square_by_color.get(Color.BLACK)
            if king_square in (red_king, black_king):
                return True

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
            return GameStatus(True, "checkmate", winner)
        return GameStatus(True, "stalemate", winner)


@dataclass(frozen=True)
class MoveRecord:
    position_before: Position
    position_after: Position
    move: Move
    actor: Color


@dataclass
class GameState:
    position: Position
    history: list[MoveRecord]
    repetition_counts: dict[str, int]
    key_indices: dict[str, list[int]]

    @classmethod
    def from_position(cls, position: Position) -> "GameState":
        key = position.repetition_key()
        return cls(position, [], {key: 1}, {key: [0]})

    def make_move(self, move: Move, validate: bool = True) -> "GameState":
        if validate and move not in self.position.legal_moves():
            raise ValueError(f"Illegal move: {move}")
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
        if self.position.halfmove_clock >= NO_CAPTURE_DRAW_PLIES:
            return GameStatus(True, "fifty_move", None)
        if len(self.history) >= MAX_TOTAL_PLIES:
            return GameStatus(True, "max_moves", None)

        key = self.position.repetition_key()
        if self.repetition_counts.get(key, 0) >= 2:
            return self._repetition_status(key)
        return GameStatus(False)

    def _repetition_status(self, key: str) -> GameStatus:
        indices = self.key_indices[key]
        if len(indices) < 2:
            return GameStatus(False)
        start, end = indices[-2], indices[-1]
        segment = self.history[start:end]
        red_violation = self._violation_for(Color.RED, segment)
        black_violation = self._violation_for(Color.BLACK, segment)

        if red_violation is None and black_violation is None:
            return GameStatus(True, "mutual_perpetual", None)
        if red_violation == black_violation:
            return GameStatus(True, "mutual_perpetual", None)
        if red_violation == "check" and black_violation != "check":
            return GameStatus(True, "perpetual_check", Color.BLACK)
        if black_violation == "check" and red_violation != "check":
            return GameStatus(True, "perpetual_check", Color.RED)
        if red_violation == "chase":
            return GameStatus(True, "perpetual_chase", Color.BLACK)
        if black_violation == "chase":
            return GameStatus(True, "perpetual_chase", Color.RED)
        return GameStatus(True, "mutual_perpetual", None)

    def _violation_for(self, color: Color, segment: list[MoveRecord]) -> str | None:
        own_records = [record for record in segment if record.actor is color]
        if not own_records:
            return None
        if all(record.position_after.is_in_check(color.opposite()) for record in own_records):
            return "check"
        if self._is_perpetual_chase(color, own_records):
            return "chase"
        return None

    def _is_perpetual_chase(self, color: Color, records: list[MoveRecord]) -> bool:
        attacked_by_record: list[set[int]] = []
        attackers_by_target: dict[int, list[set[int]]] = {}

        for record in records:
            attacks = self._attacked_non_king_targets(record.position_after, color)
            attacked = set(attacks)
            if not attacked:
                return False
            attacked_by_record.append(attacked)
            for target_id, attacker_ids in attacks.items():
                attackers_by_target.setdefault(target_id, []).append(attacker_ids)

        candidates = set.intersection(*attacked_by_record)
        for target_id in candidates:
            if self._target_exempt_from_chase(records, color, target_id, attackers_by_target[target_id]):
                continue
            if all(not self._has_true_root(record.position_after, color, target_id) for record in records):
                return True
            if self._target_is_rook_chased_by_horse_or_cannon(records, color, target_id, attackers_by_target[target_id]):
                return True
        return False

    def _attacked_non_king_targets(self, position: Position, color: Color) -> dict[int, set[int]]:
        attacks: dict[int, set[int]] = {}
        for attacker_id, (ax, ay) in position.piece_positions.items():
            attacker = position.piece_at(ax, ay)
            if attacker is None or attacker.color is not color:
                continue
            for target_id, (tx, ty) in position.piece_positions.items():
                target = position.piece_at(tx, ty)
                if target is None or target.color is color or target.kind is PieceKind.KING:
                    continue
                if position._piece_attacks_square(ax, ay, attacker, tx, ty):
                    attacks.setdefault(target_id, set()).add(attacker_id)
        return attacks

    def _target_exempt_from_chase(
        self,
        records: list[MoveRecord],
        color: Color,
        target_id: int,
        attackers_by_event: list[set[int]],
    ) -> bool:
        target_position = records[-1].position_after.piece_positions.get(target_id)
        if target_position is None:
            return True
        target = records[-1].position_after.piece_at(*target_position)
        if target is None:
            return True
        if target.kind is PieceKind.PAWN and not crossed_river(target.color, target_position[1]):
            return True

        all_attackers_are_minor = True
        for record, attacker_ids in zip(records, attackers_by_event, strict=True):
            for attacker_id in attacker_ids:
                pos = record.position_after.piece_positions.get(attacker_id)
                if pos is None:
                    continue
                attacker = record.position_after.piece_at(*pos)
                if attacker is not None and attacker.kind not in (PieceKind.KING, PieceKind.PAWN):
                    all_attackers_are_minor = False
        if all_attackers_are_minor:
            return True

        return self._is_same_kind_exchange(records, target_id, attackers_by_event)

    def _is_same_kind_exchange(
        self,
        records: list[MoveRecord],
        target_id: int,
        attackers_by_event: list[set[int]],
    ) -> bool:
        for record, attacker_ids in zip(records, attackers_by_event, strict=True):
            target_pos = record.position_after.piece_positions.get(target_id)
            if target_pos is None:
                return False
            target = record.position_after.piece_at(*target_pos)
            if target is None:
                return False
            found_exchange = False
            for attacker_id in attacker_ids:
                attacker_pos = record.position_after.piece_positions.get(attacker_id)
                if attacker_pos is None:
                    continue
                attacker = record.position_after.piece_at(*attacker_pos)
                if attacker is None or attacker.kind is not target.kind:
                    continue
                if record.position_after._piece_attacks_square(target_pos[0], target_pos[1], target, attacker_pos[0], attacker_pos[1]):
                    found_exchange = True
                    break
            if not found_exchange:
                return False
        return True

    def _target_is_rook_chased_by_horse_or_cannon(
        self,
        records: list[MoveRecord],
        color: Color,
        target_id: int,
        attackers_by_event: list[set[int]],
    ) -> bool:
        for record, attacker_ids in zip(records, attackers_by_event, strict=True):
            target_pos = record.position_after.piece_positions.get(target_id)
            if target_pos is None:
                return False
            target = record.position_after.piece_at(*target_pos)
            if target is None or target.kind is not PieceKind.ROOK:
                return False
            if not any(
                (attacker := self._piece_by_id(record.position_after, attacker_id)) is not None
                and attacker.kind in (PieceKind.HORSE, PieceKind.CANNON)
                for attacker_id in attacker_ids
            ):
                return False
        return True

    def _piece_by_id(self, position: Position, piece_id: int) -> Piece | None:
        square = position.piece_positions.get(piece_id)
        if square is None:
            return None
        return position.piece_at(*square)

    def _has_true_root(self, position: Position, attacker_color: Color, target_id: int) -> bool:
        target_pos = position.piece_positions.get(target_id)
        if target_pos is None:
            return False
        tx, ty = target_pos
        target = position.piece_at(tx, ty)
        if target is None:
            return False

        for attacker_id, attacker_pos in position.piece_positions.items():
            ax, ay = attacker_pos
            attacker = position.piece_at(ax, ay)
            if attacker is None or attacker.color is not attacker_color:
                continue
            if not position._piece_attacks_square(ax, ay, attacker, tx, ty):
                continue
            captured_position = position.make_move(Move(ax, ay, tx, ty))
            for defender_id, defender_pos in captured_position.piece_positions.items():
                if defender_id == target_id:
                    continue
                dx, dy = defender_pos
                defender = captured_position.piece_at(dx, dy)
                if defender is None or defender.color is attacker_color:
                    continue
                if captured_position._piece_attacks_square(dx, dy, defender, tx, ty):
                    return True
        return False