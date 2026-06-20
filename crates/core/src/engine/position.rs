//! 不可变棋盘局面（Position）。

use std::collections::BTreeMap;

use super::constants::{ACTION_SPACE_SIZE, BOARD_HEIGHT, BOARD_WIDTH};
use super::types::{
    crossed_river, in_board, in_palace, move_to_action_id, Color, GameStatus, GameStatusReason,
    Move, Piece, PieceKind,
};

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

type Board = [[Option<Piece>; BOARD_WIDTH]; BOARD_HEIGHT];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Position {
    pub board: Board,
    pub side_to_move: Color,
    pub piece_positions: BTreeMap<u32, (i32, i32)>,
    pub red_king: Option<(i32, i32)>,
    pub black_king: Option<(i32, i32)>,
    pub no_capture_plies: u32,
    pub fullmove_number: u32,
}

impl Position {
    pub fn starting() -> Position {
        Position::from_fen("rheakaehr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RHEAKAEHR r")
            .expect("starting FEN is valid")
    }

    pub fn from_fen(fen: &str) -> Result<Position, String> {
        let parts: Vec<&str> = fen.split_whitespace().collect();
        if parts.len() < 2 {
            return Err(format!("Invalid FEN: {fen}"));
        }
        let rows: Vec<&str> = parts[0].split('/').collect();
        if rows.len() != BOARD_HEIGHT {
            return Err(format!("Invalid FEN board height: {fen}"));
        }

        let mut board: Board = Default::default();
        let mut piece_positions: BTreeMap<u32, (i32, i32)> = BTreeMap::new();
        let mut red_king: Option<(i32, i32)> = None;
        let mut black_king: Option<(i32, i32)> = None;
        let mut next_id: u32 = 0;

        for (y, row) in rows.iter().enumerate() {
            let mut x: usize = 0;
            for ch in row.chars() {
                if let Some(empty) = ch.to_digit(10) {
                    x += empty as usize;
                    continue;
                }
                let piece = Piece::from_fen(next_id, ch)?;
                if x >= BOARD_WIDTH {
                    return Err(format!("Invalid FEN row width: {row}"));
                }
                board[y][x] = Some(piece);
                piece_positions.insert(next_id, (x as i32, y as i32));
                if piece.kind == PieceKind::King {
                    match piece.color {
                        Color::Red => red_king = Some((x as i32, y as i32)),
                        Color::Black => black_king = Some((x as i32, y as i32)),
                    }
                }
                next_id += 1;
                x += 1;
            }
            if x != BOARD_WIDTH {
                return Err(format!("Invalid FEN row width: {row}"));
            }
        }

        let side_to_move = Color::from_fen_side(parts[1])?;
        let no_capture_plies = parts.get(4).and_then(|s| s.parse().ok()).unwrap_or(0);
        let fullmove_number = parts.get(5).and_then(|s| s.parse().ok()).unwrap_or(1);

        Ok(Position {
            board,
            side_to_move,
            piece_positions,
            red_king,
            black_king,
            no_capture_plies,
            fullmove_number,
        })
    }

    pub fn to_fen(&self) -> String {
        let mut rows: Vec<String> = Vec::with_capacity(BOARD_HEIGHT);
        for y in 0..BOARD_HEIGHT {
            let mut row = String::new();
            let mut empty = 0;
            for x in 0..BOARD_WIDTH {
                match self.board[y][x] {
                    None => empty += 1,
                    Some(piece) => {
                        if empty > 0 {
                            row.push_str(&empty.to_string());
                            empty = 0;
                        }
                        row.push(piece.to_fen());
                    }
                }
            }
            if empty > 0 {
                row.push_str(&empty.to_string());
            }
            rows.push(row);
        }
        format!("{} {}", rows.join("/"), self.side_to_move.fen_side())
    }

    pub fn repetition_key(&self) -> String {
        self.to_fen()
    }

    pub fn king_square(&self, color: Color) -> Option<(i32, i32)> {
        match color {
            Color::Red => self.red_king,
            Color::Black => self.black_king,
        }
    }

    pub fn piece_at(&self, x: i32, y: i32) -> Option<Piece> {
        if !in_board(x, y) {
            return None;
        }
        self.board[y as usize][x as usize]
    }

    pub fn require_piece_at(&self, x: i32, y: i32) -> Piece {
        self.piece_at(x, y).expect("No piece at requested square")
    }

    pub(crate) fn apply_move(&self, mv: Move, piece: Piece) -> Position {
        let captured = self.piece_at(mv.tx, mv.ty);

        let mut board = self.board;
        let mut piece_positions = self.piece_positions.clone();
        let mut red_king = self.red_king;
        let mut black_king = self.black_king;

        board[mv.sy as usize][mv.sx as usize] = None;
        board[mv.ty as usize][mv.tx as usize] = Some(piece);
        piece_positions.insert(piece.piece_id, (mv.tx, mv.ty));
        if let Some(cap) = captured {
            piece_positions.remove(&cap.piece_id);
        }
        if piece.kind == PieceKind::King {
            match piece.color {
                Color::Red => red_king = Some((mv.tx, mv.ty)),
                Color::Black => black_king = Some((mv.tx, mv.ty)),
            }
        }

        let next_side = self.side_to_move.opposite();
        let next_no_capture_plies = if captured.is_some() {
            0
        } else {
            self.no_capture_plies + 1
        };
        let next_fullmove = self.fullmove_number + if self.side_to_move == Color::Black { 1 } else { 0 };

        Position {
            board,
            side_to_move: next_side,
            piece_positions,
            red_king,
            black_king,
            no_capture_plies: next_no_capture_plies,
            fullmove_number: next_fullmove,
        }
    }

    pub fn validation_reason(
        &self,
        mv: Move,
        side_to_move_color: Option<Color>,
    ) -> Option<MoveValidationReason> {
        if !in_board(mv.sx, mv.sy) || !in_board(mv.tx, mv.ty) {
            return Some(MoveValidationReason::OutsideBoard);
        }
        let piece = match self.piece_at(mv.sx, mv.sy) {
            None => return Some(MoveValidationReason::NoPiece),
            Some(p) => p,
        };
        if let Some(color) = side_to_move_color {
            if piece.color != color {
                return Some(MoveValidationReason::WrongSide);
            }
        }
        if let Some(captured) = self.piece_at(mv.tx, mv.ty) {
            if captured.color == piece.color {
                return Some(MoveValidationReason::CaptureOwnPiece);
            }
        }
        if !self
            .pseudo_moves_for_piece(mv.sx, mv.sy, piece)
            .into_iter()
            .any(|m| m == mv)
        {
            return Some(MoveValidationReason::InvalidPieceMove);
        }

        let next_position = self.apply_move(mv, piece);
        // 飞将（将帅照面）已并入 is_in_check 的纵线扫描，无需单独判 kings_face。
        if next_position.is_in_check(piece.color) {
            return Some(MoveValidationReason::LeavesKingInCheck);
        }
        None
    }

    pub fn is_legal_move(&self, mv: Move, side_to_move_color: Option<Color>) -> bool {
        self.validation_reason(mv, side_to_move_color).is_none()
    }

    pub fn make_move(&self, mv: Move) -> Position {
        if let Some(reason) = self.validation_reason(mv, Some(self.side_to_move)) {
            panic!("Illegal move ({}): {:?}", reason.as_str(), mv);
        }
        let piece = self.require_piece_at(mv.sx, mv.sy);
        self.apply_move(mv, piece)
    }

    pub fn legal_move_mask(&self) -> Vec<bool> {
        let mut mask = vec![false; ACTION_SPACE_SIZE];
        for mv in self.legal_moves() {
            mask[move_to_action_id(mv)] = true;
        }
        mask
    }

    pub fn legal_moves(&self) -> Vec<Move> {
        let mut moves = Vec::new();
        for y in 0..BOARD_HEIGHT as i32 {
            for x in 0..BOARD_WIDTH as i32 {
                let piece = match self.piece_at(x, y) {
                    Some(p) if p.color == self.side_to_move => p,
                    _ => continue,
                };
                for mv in self.pseudo_moves_for_piece(x, y, piece) {
                    if self.is_legal_move(mv, Some(self.side_to_move)) {
                        moves.push(mv);
                    }
                }
            }
        }
        moves
    }

    fn pseudo_moves_for_piece(&self, x: i32, y: i32, piece: Piece) -> Vec<Move> {
        match piece.kind {
            PieceKind::King => self.king_moves(x, y, piece),
            PieceKind::Advisor => self.advisor_moves(x, y, piece),
            PieceKind::Elephant => self.elephant_moves(x, y, piece),
            PieceKind::Horse => self.horse_moves(x, y, piece),
            PieceKind::Rook => self.sliding_moves(x, y, piece, false),
            PieceKind::Cannon => self.sliding_moves(x, y, piece, true),
            PieceKind::Pawn => self.pawn_moves(x, y, piece),
        }
    }

    fn can_land(&self, x: i32, y: i32, color: Color) -> bool {
        match self.piece_at(x, y) {
            None => true,
            Some(target) => target.color != color,
        }
    }

    fn king_moves(&self, x: i32, y: i32, piece: Piece) -> Vec<Move> {
        let mut moves = Vec::new();
        for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
            let (tx, ty) = (x + dx, y + dy);
            if in_palace(piece.color, tx, ty) && self.can_land(tx, ty, piece.color) {
                moves.push(Move::new(x, y, tx, ty));
            }
        }
        moves
    }

    fn advisor_moves(&self, x: i32, y: i32, piece: Piece) -> Vec<Move> {
        let mut moves = Vec::new();
        for (dx, dy) in [(1, 1), (1, -1), (-1, 1), (-1, -1)] {
            let (tx, ty) = (x + dx, y + dy);
            if in_palace(piece.color, tx, ty) && self.can_land(tx, ty, piece.color) {
                moves.push(Move::new(x, y, tx, ty));
            }
        }
        moves
    }

    fn elephant_moves(&self, x: i32, y: i32, piece: Piece) -> Vec<Move> {
        let mut moves = Vec::new();
        for (dx, dy) in [(2, 2), (2, -2), (-2, 2), (-2, -2)] {
            let (tx, ty) = (x + dx, y + dy);
            let (eye_x, eye_y) = (x + dx / 2, y + dy / 2);
            if !in_board(tx, ty) {
                continue;
            }
            if piece.color == Color::Red && ty < 5 {
                continue;
            }
            if piece.color == Color::Black && ty > 4 {
                continue;
            }
            if self.piece_at(eye_x, eye_y).is_none() && self.can_land(tx, ty, piece.color) {
                moves.push(Move::new(x, y, tx, ty));
            }
        }
        moves
    }

    fn horse_moves(&self, x: i32, y: i32, piece: Piece) -> Vec<Move> {
        const CANDIDATES: [(i32, i32, i32, i32); 8] = [
            (1, 2, 0, 1),
            (-1, 2, 0, 1),
            (1, -2, 0, -1),
            (-1, -2, 0, -1),
            (2, 1, 1, 0),
            (2, -1, 1, 0),
            (-2, 1, -1, 0),
            (-2, -1, -1, 0),
        ];
        let mut moves = Vec::new();
        for (dx, dy, leg_dx, leg_dy) in CANDIDATES {
            let (tx, ty) = (x + dx, y + dy);
            let (leg_x, leg_y) = (x + leg_dx, y + leg_dy);
            if in_board(tx, ty)
                && self.piece_at(leg_x, leg_y).is_none()
                && self.can_land(tx, ty, piece.color)
            {
                moves.push(Move::new(x, y, tx, ty));
            }
        }
        moves
    }

    fn sliding_moves(&self, x: i32, y: i32, piece: Piece, cannon: bool) -> Vec<Move> {
        let mut moves = Vec::new();
        for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
            let (mut tx, mut ty) = (x + dx, y + dy);
            let mut seen_screen = false;
            while in_board(tx, ty) {
                let target = self.piece_at(tx, ty);
                if !cannon {
                    match target {
                        None => moves.push(Move::new(x, y, tx, ty)),
                        Some(t) => {
                            if t.color != piece.color {
                                moves.push(Move::new(x, y, tx, ty));
                            }
                            break;
                        }
                    }
                } else if !seen_screen {
                    match target {
                        None => moves.push(Move::new(x, y, tx, ty)),
                        Some(_) => seen_screen = true,
                    }
                } else if let Some(t) = target {
                    if t.color != piece.color {
                        moves.push(Move::new(x, y, tx, ty));
                    }
                    break;
                }
                tx += dx;
                ty += dy;
            }
        }
        moves
    }

    fn pawn_moves(&self, x: i32, y: i32, piece: Piece) -> Vec<Move> {
        let mut directions: Vec<(i32, i32)> = if piece.color == Color::Red {
            vec![(0, -1)]
        } else {
            vec![(0, 1)]
        };
        if crossed_river(piece.color, y) {
            directions.push((-1, 0));
            directions.push((1, 0));
        }
        let mut moves = Vec::new();
        for (dx, dy) in directions {
            let (tx, ty) = (x + dx, y + dy);
            if in_board(tx, ty) && self.can_land(tx, ty, piece.color) {
                moves.push(Move::new(x, y, tx, ty));
            }
        }
        moves
    }

    /// 以将为中心反向探测是否被将：只检查能攻击到将的固定方向/点位，O(1)。
    ///
    /// 含飞将（纵线第一个子是对方将即视为被将），故已涵盖将帅照面，无需单独的 kings_face。
    /// 士、象受困于己方半盘/九宫，永远够不到对方将，无需检查。
    pub fn is_in_check(&self, color: Color) -> bool {
        let (kx, ky) = self.king_square(color).expect("Missing king for color");
        let enemy = color.opposite();

        // 车 / 炮 / 飞将：沿横竖四向扫描。第一个子是对方车或对方将（飞将）即被将；
        // 任意子都可充当炮架，越过一个炮架后，下一个子若是对方炮即被将。
        for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
            let (mut x, mut y) = (kx + dx, ky + dy);
            let mut screened = false;
            while in_board(x, y) {
                if let Some(p) = self.piece_at(x, y) {
                    if !screened {
                        if p.color == enemy
                            && (p.kind == PieceKind::Rook || p.kind == PieceKind::King)
                        {
                            return true;
                        }
                        screened = true;
                    } else {
                        if p.color == enemy && p.kind == PieceKind::Cannon {
                            return true;
                        }
                        break;
                    }
                }
                x += dx;
                y += dy;
            }
        }

        // 马：反推 8 个马位，蹩马腿（靠近将一侧的格，相对将的偏移）为空才成立。
        // 每项 = (马相对将的偏移 x, y, 马腿相对将的偏移 x, y)。
        const HORSE: [(i32, i32, i32, i32); 8] = [
            (1, 2, 1, 1),
            (-1, 2, -1, 1),
            (1, -2, 1, -1),
            (-1, -2, -1, -1),
            (2, 1, 1, 1),
            (2, -1, 1, -1),
            (-2, 1, -1, 1),
            (-2, -1, -1, -1),
        ];
        for (hox, hoy, lox, loy) in HORSE {
            if let Some(p) = self.piece_at(kx + hox, ky + hoy) {
                if p.color == enemy
                    && p.kind == PieceKind::Horse
                    && self.piece_at(kx + lox, ky + loy).is_none()
                {
                    return true;
                }
            }
        }

        // 兵/卒：红兵向 y 减小方向走，故从将下方一格 + 两侧攻；黑卒反之。
        // 将位于九宫内，两侧攻击格必在对方过河区，故无需额外过河判断。
        let pawn_squares: [(i32, i32); 3] = match enemy {
            Color::Red => [(kx, ky + 1), (kx - 1, ky), (kx + 1, ky)],
            Color::Black => [(kx, ky - 1), (kx - 1, ky), (kx + 1, ky)],
        };
        for (px, py) in pawn_squares {
            if let Some(p) = self.piece_at(px, py) {
                if p.color == enemy && p.kind == PieceKind::Pawn {
                    return true;
                }
            }
        }

        false
    }

    pub fn status(&self) -> GameStatus {
        if !self.legal_moves().is_empty() {
            return GameStatus::ongoing();
        }
        let winner = self.side_to_move.opposite();
        if self.is_in_check(self.side_to_move) {
            GameStatus::terminal(GameStatusReason::Checkmate, Some(winner))
        } else {
            GameStatus::terminal(GameStatusReason::Stalemate, Some(winner))
        }
    }
}
