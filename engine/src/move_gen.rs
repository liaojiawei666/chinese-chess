use crate::{
    board::{crossed_river, in_board, in_palace, Board},
    Color, Move, MoveFilter, Piece, PieceKind, Position,
};

fn can_land(board: &Board, pos: Position, color: Color, filter: MoveFilter) -> bool {
    match board.get_piece_at(pos) {
        None => filter == MoveFilter::All,
        Some(piece) => piece.color != color,
    }
}

fn king_moves(
    board: &Board,
    source_pos: Position,
    piece: Piece,
    filter: MoveFilter,
) -> Vec<Move> {
    let mut moves = Vec::new();
    for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
        let target_pos = source_pos.add(dx, dy);
        if in_palace(piece.color, target_pos) && can_land(board, target_pos, piece.color, filter) {
            moves.push(Move::new(source_pos, target_pos));
        }
    }
    moves
}

fn advisor_moves(
    board: &Board,
    source_pos: Position,
    piece: Piece,
    filter: MoveFilter,
) -> Vec<Move> {
    let mut moves = Vec::new();
    for (dx, dy) in [(1, 1), (1, -1), (-1, 1), (-1, -1)] {
        let target_pos = source_pos.add(dx, dy);
        if in_palace(piece.color, target_pos) && can_land(board, target_pos, piece.color, filter) {
            moves.push(Move::new(source_pos, target_pos));
        }
    }
    moves
}

fn elephant_moves(
    board: &Board,
    source_pos: Position,
    piece: Piece,
    filter: MoveFilter,
) -> Vec<Move> {
    let mut moves = Vec::new();
    for (dx, dy) in [(2, 2), (-2, 2), (2, -2), (-2, -2)] {
        let target_pos = source_pos.add(dx, dy);
        let eye_pos = source_pos.add(dx / 2, dy / 2);
        if in_board(target_pos)
            && board.is_pos_empty(eye_pos)
            && !crossed_river(piece.color, target_pos.y)
            && can_land(board, target_pos, piece.color, filter)
        {
            moves.push(Move::new(source_pos, target_pos));
        }
    }
    moves
}

fn horse_moves(
    board: &Board,
    source_pos: Position,
    piece: Piece,
    filter: MoveFilter,
) -> Vec<Move> {
    let mut moves = Vec::new();
    const CANDIDATES: [(i8, i8, i8, i8); 8] = [
        (1, 2, 0, 1),
        (-1, 2, 0, 1),
        (1, -2, 0, -1),
        (-1, -2, 0, -1),
        (2, 1, 1, 0),
        (2, -1, 1, 0),
        (-2, 1, -1, 0),
        (-2, -1, -1, 0),
    ];
    for (dx, dy, leg_dx, leg_dy) in CANDIDATES {
        let target_pos = source_pos.add(dx, dy);
        let leg_pos = source_pos.add(leg_dx, leg_dy);
        if in_board(target_pos)
            && board.is_pos_empty(leg_pos)
            && can_land(board, target_pos, piece.color, filter)
        {
            moves.push(Move::new(source_pos, target_pos));
        }
    }
    moves
}

fn rook_moves(
    board: &Board,
    source_pos: Position,
    piece: Piece,
    filter: MoveFilter,
) -> Vec<Move> {
    let mut moves = Vec::new();
    for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
        let mut target_pos = source_pos.add(dx, dy);
        while in_board(target_pos) {
            match board.get_piece_at(target_pos) {
                None => {
                    if filter == MoveFilter::All {
                        moves.push(Move::new(source_pos, target_pos));
                    }
                }
                Some(p) => {
                    if p.color != piece.color {
                        moves.push(Move::new(source_pos, target_pos));
                    }
                    break;
                }
            }
            target_pos = target_pos.add(dx, dy);
        }
    }
    moves
}

fn cannon_moves(
    board: &Board,
    source_pos: Position,
    piece: Piece,
    filter: MoveFilter,
) -> Vec<Move> {
    let mut moves = Vec::new();
    for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
        let mut seen_screen = false;
        let mut target_pos = source_pos.add(dx, dy);
        while in_board(target_pos) {
            if seen_screen {
                if let Some(p) = board.get_piece_at(target_pos) {
                    if p.color != piece.color {
                        moves.push(Move::new(source_pos, target_pos));
                    }
                    break;
                }
            } else {
                match board.get_piece_at(target_pos) {
                    None => {
                        if filter == MoveFilter::All {
                            moves.push(Move::new(source_pos, target_pos));
                        }
                    }
                    Some(_) => {
                        seen_screen = true;
                    }
                }
            }
            target_pos = target_pos.add(dx, dy);
        }
    }
    moves
}

fn pawn_moves(
    board: &Board,
    source_pos: Position,
    piece: Piece,
    filter: MoveFilter,
) -> Vec<Move> {
    let mut directions: Vec<(i8, i8)> = if piece.color == Color::Red {
        vec![(0, -1)]
    } else {
        vec![(0, 1)]
    };
    if crossed_river(piece.color, source_pos.y) {
        directions.push((-1, 0));
        directions.push((1, 0));
    }
    let mut moves = Vec::new();
    for (dx, dy) in directions {
        let target_pos = source_pos.add(dx, dy);
        if in_board(target_pos) && can_land(board, target_pos, piece.color, filter) {
            moves.push(Move::new(source_pos, target_pos));
        }
    }
    moves
}

fn pseudo_moves_for_piece(
    board: &Board,
    source_pos: Position,
    piece: Piece,
    filter: MoveFilter,
) -> Vec<Move> {
    match piece.kind {
        PieceKind::King => king_moves(board, source_pos, piece, filter),
        PieceKind::Advisor => advisor_moves(board, source_pos, piece, filter),
        PieceKind::Elephant => elephant_moves(board, source_pos, piece, filter),
        PieceKind::Horse => horse_moves(board, source_pos, piece, filter),
        PieceKind::Rook => rook_moves(board, source_pos, piece, filter),
        PieceKind::Cannon => cannon_moves(board, source_pos, piece, filter),
        PieceKind::Pawn => pawn_moves(board, source_pos, piece, filter),
    }
}

/// 从 target_pos 反向查找 attacker_color 方所有能合法攻击该格的棋子。
/// 返回 (棋子, 棋子位置) 列表。合法性指吃子后不会让攻击方自己的将帅被将。
pub(crate) fn attackers_of(
    board: &mut Board,
    target_pos: Position,
    attacker_color: Color,
) -> Vec<(Piece, Position)> {
    let candidates = pseudo_attackers_of(board, target_pos, attacker_color);
    let mut result = Vec::new();
    for (piece, pos) in candidates {
        let mv = Move::new(pos, target_pos);
        let move_info = board.make_move(mv);
        let legal = !is_in_check(board, attacker_color);
        board.undo_move(&move_info);
        if legal {
            result.push((piece, pos));
        }
    }
    result
}

/// 从 target_pos 反向查找 attacker_color 方所有伪合法攻击者（不检查是否送将）。
fn pseudo_attackers_of(
    board: &Board,
    target_pos: Position,
    attacker_color: Color,
) -> Vec<(Piece, Position)> {
    let mut attackers = Vec::new();

    // 将帅：只看相邻一格（九宫内），飞将在 is_in_check 中单独处理
    for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
        let attacker_pos = target_pos.add(dx, dy);
        if !in_palace(attacker_color, attacker_pos) {
            continue;
        }
        if let Some(attacker_piece) = board.get_piece_at(attacker_pos) {
            if attacker_piece.is_king() && attacker_piece.color == attacker_color {
                attackers.push((attacker_piece, attacker_pos));
                break;
            }
        }
    }

    // 直线：车、炮（隔一子）
    for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
        let mut pos = target_pos.add(dx, dy);
        let mut screened = false;
        while in_board(pos) {
            if let Some(p) = board.get_piece_at(pos) {
                if !screened {
                    if p.color == attacker_color && p.kind == PieceKind::Rook {
                        attackers.push((p, pos));
                    }
                    screened = true;
                } else {
                    if p.color == attacker_color && p.kind == PieceKind::Cannon {
                        attackers.push((p, pos));
                    }
                    break;
                }
            }
            pos = pos.add(dx, dy);
        }
    }

    // 马：反推 8 个马位。注意蹩马腿的偏移是相对于马的位置（靠近目标格一侧）。
    const HORSE: [(i8, i8, i8, i8); 8] = [
        (1, 2, 0, 1),
        (-1, 2, 0, 1),
        (1, -2, 0, -1),
        (-1, -2, 0, -1),
        (2, 1, 1, 0),
        (2, -1, 1, 0),
        (-2, 1, -1, 0),
        (-2, -1, -1, 0),
    ];
    for (hx, hy, lx, ly) in HORSE {
        let horse_pos = target_pos.add(hx, hy);
        if !in_board(horse_pos) {
            continue;
        }
        if let Some(p) = board.get_piece_at(horse_pos) {
            if p.color == attacker_color && p.kind == PieceKind::Horse {
                let leg_pos = horse_pos.add(lx, ly);
                if board.is_pos_empty(leg_pos) {
                    attackers.push((p, horse_pos));
                }
            }
        }
    }

    // 兵/卒：红兵往 y 减小方向走，所以从 target_pos 的 y+1 和左右可被红兵攻击；黑卒反之。
    let pawn_offsets: [(i8, i8); 3] = match attacker_color {
        Color::Red => [(0, 1), (-1, 0), (1, 0)],
        Color::Black => [(0, -1), (1, 0), (-1, 0)],
    };
    for (dx, dy) in pawn_offsets {
        let pawn_pos = target_pos.add(dx, dy);
        if !in_board(pawn_pos) {
            continue;
        }
        if let Some(p) = board.get_piece_at(pawn_pos) {
            if p.color == attacker_color && p.kind == PieceKind::Pawn {
                // 左右攻击要求兵已过河
                if dx != 0 && !crossed_river(attacker_color, pawn_pos.y) {
                    continue;
                }
                attackers.push((p, pawn_pos));
            }
        }
    }

    // 士：从 target_pos 的四个斜方向找
    for (dx, dy) in [(1, 1), (1, -1), (-1, 1), (-1, -1)] {
        let advisor_pos = target_pos.add(dx, dy);
        if !in_palace(attacker_color, advisor_pos) {
            continue;
        }
        if let Some(p) = board.get_piece_at(advisor_pos) {
            if p.color == attacker_color && p.kind == PieceKind::Advisor {
                attackers.push((p, advisor_pos));
            }
        }
    }

    // 象：从 target_pos 的四个田字方向找
    for (dx, dy) in [(2, 2), (-2, 2), (2, -2), (-2, -2)] {
        let elephant_pos = target_pos.add(dx, dy);
        if !in_board(elephant_pos) || crossed_river(attacker_color, elephant_pos.y) {
            continue;
        }
        if let Some(p) = board.get_piece_at(elephant_pos) {
            if p.color == attacker_color && p.kind == PieceKind::Elephant {
                let eye_pos = target_pos.add(dx / 2, dy / 2);
                if board.is_pos_empty(eye_pos) {
                    attackers.push((p, elephant_pos));
                }
            }
        }
    }

    attackers
}

/// 判断 color 方是否被将军（含飞将检测）。
pub(crate) fn is_in_check(board: &Board, color: Color) -> bool {
    let king_pos = board.king_pos(color);
    let enemy = color.opposite();

    if !pseudo_attackers_of(board, king_pos, enemy).is_empty() {
        return true;
    }

    // 飞将：两王同列且中间无子
    let enemy_king_pos = board.king_pos(enemy);
    if board.count_pieces_between(king_pos, enemy_king_pos) == Some(0) {
        return true;
    }

    false
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoveValidationReason {
    OutsideBoard,
    NoPiece,
    WrongSide,
    CaptureOwnPiece,
    InvalidPieceMove,
    KingsFace,
    LeavesKingInCheck,
}

impl MoveValidationReason {
    pub fn as_str(self) -> &'static str {
        match self {
            MoveValidationReason::OutsideBoard => "outside_board",
            MoveValidationReason::NoPiece => "no_piece",
            MoveValidationReason::WrongSide => "wrong_side",
            MoveValidationReason::CaptureOwnPiece => "capture_own_piece",
            MoveValidationReason::InvalidPieceMove => "invalid_piece_move",
            MoveValidationReason::KingsFace => "kings_face",
            MoveValidationReason::LeavesKingInCheck => "leaves_king_in_check",
        }
    }
}

pub(crate) fn validate_move(
    board: &mut Board,
    mv: Move,
    side_to_move: Option<Color>,
) -> Option<MoveValidationReason> {
    if !in_board(mv.from) || !in_board(mv.to) {
        return Some(MoveValidationReason::OutsideBoard);
    }
    let piece = match board.get_piece_at(mv.from) {
        None => return Some(MoveValidationReason::NoPiece),
        Some(p) => p,
    };
    if let Some(color) = side_to_move {
        if piece.color != color {
            return Some(MoveValidationReason::WrongSide);
        }
    }
    if let Some(captured) = board.get_piece_at(mv.to) {
        if captured.color == piece.color {
            return Some(MoveValidationReason::CaptureOwnPiece);
        }
    }
    if !pseudo_moves_for_piece(board, mv.from, piece, MoveFilter::All)
        .into_iter()
        .any(|m| m == mv)
    {
        return Some(MoveValidationReason::InvalidPieceMove);
    }

    let move_info = board.make_move(mv);
    let mut reason = None;
    if is_in_check(board, piece.color) {
        reason = Some(MoveValidationReason::LeavesKingInCheck);
    }
    board.undo_move(&move_info);
    reason
}

pub(crate) fn is_legal_move(board: &mut Board, mv: Move, side_to_move: Option<Color>) -> bool {
    validate_move(board, mv, side_to_move).is_none()
}

pub(crate) fn legal_moves(board: &mut Board, side_to_move: Color) -> Vec<Move> {
    legal_moves_filtered(board, side_to_move, MoveFilter::All)
}

pub(crate) fn legal_capture_moves(board: &mut Board, side_to_move: Color) -> Vec<Move> {
    legal_moves_filtered(board, side_to_move, MoveFilter::CaptureOnly)
}

/// 获取指定位置棋子的所有合法走法
pub(crate) fn legal_moves_for_piece(board: &mut Board, pos: Position) -> Vec<Move> {
    let Some(piece) = board.get_piece_at(pos) else {
        return Vec::new();
    };
    let mut moves = Vec::new();
    for mv in pseudo_moves_for_piece(board, pos, piece, MoveFilter::All) {
        let move_info = board.make_move(mv);
        if !is_in_check(board, piece.color) {
            moves.push(mv);
        }
        board.undo_move(&move_info);
    }
    moves
}

fn legal_moves_filtered(board: &mut Board, side_to_move: Color, filter: MoveFilter) -> Vec<Move> {
    let mut moves = Vec::new();
    let piece_pos_map = board.piece_pos_map.clone();
    for (piece, pos) in piece_pos_map.iter() {
        if piece.color != side_to_move {
            continue;
        }
        for mv in pseudo_moves_for_piece(board, *pos, *piece, filter) {
            let move_info = board.make_move(mv);
            if !is_in_check(board, side_to_move) {
                moves.push(mv);
            }
            board.undo_move(&move_info);
        }
    }
    moves
}
