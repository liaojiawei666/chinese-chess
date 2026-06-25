use std::collections::HashMap;

use super::zobrist::ZobristTable;

use super::constants::*;
use super::types::*;

pub type Squares<T> = [[T; BOARD_WIDTH as usize]; BOARD_HEIGHT as usize];

pub type PieceIndexBoard = Squares<PieceIndex>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Board {
    pub squares: Squares<Option<Piece>>,
    pub piece_pos_map: HashMap<Piece, Position>,
    pub red_king_pos: Position,
    pub black_king_pos: Position,
    pub side_to_move: Color,
    pub zobrist_hash: u64,
    pub zobrist_table: &'static ZobristTable,
    pub no_capture_plies: u32,
    pub fullmove_number: u32,
}

fn parse_fen_squares(board_str: &str) -> Result<Squares<Option<Piece>>, String> {
    let mut squares: Squares<Option<Piece>> = [[None; BOARD_WIDTH as usize]; BOARD_HEIGHT as usize];
    let mut piece_id: u8 = 0;

    let rows: Vec<&str> = board_str.split('/').collect();
    if rows.len() != BOARD_HEIGHT as usize {
        return Err(format!(
            "FEN should have {} rows, got {}",
            BOARD_HEIGHT,
            rows.len()
        ));
    }

    for (y, row) in rows.iter().enumerate() {
        let mut x: usize = 0;
        for ch in row.chars() {
            if let Some(digit) = ch.to_digit(10) {
                x += digit as usize;
            } else {
                let (color, kind) = match ch {
                    'K' => (Color::Red, PieceKind::King),
                    'A' => (Color::Red, PieceKind::Advisor),
                    'B' | 'E' => (Color::Red, PieceKind::Elephant),
                    'N' | 'H' => (Color::Red, PieceKind::Horse),
                    'R' => (Color::Red, PieceKind::Rook),
                    'C' => (Color::Red, PieceKind::Cannon),
                    'P' => (Color::Red, PieceKind::Pawn),
                    'k' => (Color::Black, PieceKind::King),
                    'a' => (Color::Black, PieceKind::Advisor),
                    'b' | 'e' => (Color::Black, PieceKind::Elephant),
                    'n' | 'h' => (Color::Black, PieceKind::Horse),
                    'r' => (Color::Black, PieceKind::Rook),
                    'c' => (Color::Black, PieceKind::Cannon),
                    'p' => (Color::Black, PieceKind::Pawn),
                    _ => return Err(format!("unknown FEN char: {}", ch)),
                };
                piece_id += 1;
                squares[y][x] = Some(Piece {
                    piece_id,
                    color,
                    kind,
                });
                x += 1;
            }
        }
        if x != BOARD_WIDTH as usize {
            return Err(format!(
                "row {} has width {}, expected {}",
                y, x, BOARD_WIDTH
            ));
        }
    }

    Ok(squares)
}

impl Board {
    pub fn from_fen(fen: &str) -> Result<Board, String> {
        let parts: Vec<&str> = fen.split_whitespace().collect();
        if parts.is_empty() {
            return Err("empty FEN string".into());
        }

        let squares = parse_fen_squares(parts[0])?;

        let side_to_move = if parts.len() > 1 && parts[1] == "b" {
            Color::Black
        } else {
            Color::Red
        };

        let no_capture_plies: u32 = parts.get(4).and_then(|s| s.parse().ok()).unwrap_or(0);

        let fullmove_number: u32 = parts.get(5).and_then(|s| s.parse().ok()).unwrap_or(1);

        let mut piece_pos_map = HashMap::new();
        let mut red_king_pos = Position::new(0, 0);
        let mut black_king_pos = Position::new(0, 0);

        for y in 0..BOARD_HEIGHT as usize {
            for x in 0..BOARD_WIDTH as usize {
                if let Some(piece) = squares[y][x] {
                    let pos = Position::new(x as i8, y as i8);
                    piece_pos_map.insert(piece, pos);
                    if piece.is_king() {
                        match piece.color {
                            Color::Red => red_king_pos = pos,
                            Color::Black => black_king_pos = pos,
                        }
                    }
                }
            }
        }

        let zobrist_table = ZobristTable::get();
        let mut zobrist_hash: u64 = 0;
        for (piece, pos) in piece_pos_map.iter() {
            zobrist_hash ^= zobrist_table.get_piece_hash(*piece, *pos);
        }
        if side_to_move == Color::Black {
            zobrist_hash ^= zobrist_table.side_to_move;
        }

        Ok(Board {
            squares,
            piece_pos_map,
            red_king_pos,
            black_king_pos,
            side_to_move,
            zobrist_hash,
            zobrist_table,
            no_capture_plies,
            fullmove_number,
        })
    }

    pub fn start_pos() -> Board {
        Board::from_fen("rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1")
            .expect("start FEN is always valid")
    }

    pub(crate) fn make_move(&mut self, mv: Move) -> MoveInfo {
        let source_piece = self.get_piece_at(mv.from).unwrap();
        let target_piece = self.get_piece_at(mv.to);

        let move_info = MoveInfo {
            mv,
            source_piece,
            target_piece,
            before_zobrist_hash: self.zobrist_hash,
            before_no_capture_plies: self.no_capture_plies,
            before_fullmove_number: self.fullmove_number,
        };

        self.remove_piece_at(mv.from, source_piece);
        if let Some(captured) = target_piece {
            self.remove_piece_at(mv.to, captured);
        }
        self.place_piece_at(mv.to, source_piece);

        if target_piece.is_some() {
            self.no_capture_plies = 0;
        } else {
            self.no_capture_plies += 1;
        }
        if self.side_to_move == Color::Black {
            self.fullmove_number += 1;
        }

        self.toggle_side_to_move();
        move_info
    }

    pub(crate) fn undo_move(&mut self, info: &MoveInfo) {
        self.toggle_side_to_move();

        self.remove_piece_at(info.mv.to, info.source_piece);
        self.place_piece_at(info.mv.from, info.source_piece);
        if let Some(captured) = info.target_piece {
            self.place_piece_at(info.mv.to, captured);
        }

        self.zobrist_hash = info.before_zobrist_hash;
        self.no_capture_plies = info.before_no_capture_plies;
        self.fullmove_number = info.before_fullmove_number;
    }

    fn toggle_side_to_move(&mut self) {
        self.zobrist_hash ^= self.zobrist_table.side_to_move;
        self.side_to_move = self.side_to_move.opposite();
    }

    pub fn king_pos(&self, color: Color) -> Position {
        match color {
            Color::Red => self.red_king_pos,
            Color::Black => self.black_king_pos,
        }
    }

    pub fn get_piece_at(&self, pos: Position) -> Option<Piece> {
        self.squares[pos.y as usize][pos.x as usize]
    }

    pub fn is_pos_empty(&self, pos: Position) -> bool {
        self.squares[pos.y as usize][pos.x as usize].is_none()
    }

    pub fn get_piece_pos(&self, piece: &Piece) -> Option<&Position> {
        self.piece_pos_map.get(piece)
    }

    fn place_piece_at(&mut self, pos: Position, piece: Piece) {
        self.squares[pos.y as usize][pos.x as usize] = Some(piece);
        self.piece_pos_map.insert(piece, pos);
        self.zobrist_hash ^= self.zobrist_table.get_piece_hash(piece, pos);
        if piece.is_king() {
            match piece.color {
                Color::Red => self.red_king_pos = pos,
                Color::Black => self.black_king_pos = pos,
            }
        }
    }

    fn remove_piece_at(&mut self, pos: Position, piece: Piece) {
        self.squares[pos.y as usize][pos.x as usize] = None;
        self.zobrist_hash ^= self.zobrist_table.get_piece_hash(piece, pos);
        self.piece_pos_map.remove(&piece);
    }

    /// 统计两个坐标之间（不含端点）的棋子数量。
    /// 两点必须在同一行或同一列，否则返回 None。
    pub(crate) fn count_pieces_between(&self, a: Position, b: Position) -> Option<u8> {
        if a.x == b.x {
            let min_y = a.y.min(b.y) + 1;
            let max_y = a.y.max(b.y);
            let mut count = 0u8;
            for y in min_y..max_y {
                if !self.is_pos_empty(Position::new(a.x, y)) {
                    count += 1;
                }
            }
            Some(count)
        } else if a.y == b.y {
            let min_x = a.x.min(b.x) + 1;
            let max_x = a.x.max(b.x);
            let mut count = 0u8;
            for x in min_x..max_x {
                if !self.is_pos_empty(Position::new(x, a.y)) {
                    count += 1;
                }
            }
            Some(count)
        } else {
            None
        }
    }

    /// 获取指定颜色方的所有非将帅棋子及其位置。
    pub fn pieces_of(&self, color: Color) -> Vec<(Piece, Position)> {
        self.piece_pos_map
            .iter()
            .filter(|(p, _)| p.color == color && p.kind != PieceKind::King)
            .map(|(p, pos)| (*p, *pos))
            .collect()
    }

    /// 获取棋子在当前位置的价值（参考亚规子力价值）
    pub fn get_piece_value(piece: Piece, pos: Position) -> u8 {
        match piece.kind {
            PieceKind::King => u8::MAX,
            PieceKind::Rook => 10,
            PieceKind::Horse | PieceKind::Cannon => 5,
            PieceKind::Pawn => {
                if crossed_river(piece.color, pos.y) { 3 } else { 1 }
            }
            PieceKind::Advisor | PieceKind::Elephant => 2,
        }
    }

    pub fn get_piece_index_board(&self) -> PieceIndexBoard {
        let mut raw_board = [[0u8; BOARD_WIDTH as usize]; BOARD_HEIGHT as usize];
        for y in 0..BOARD_HEIGHT as usize {
            for x in 0..BOARD_WIDTH as usize {
                raw_board[y][x] = self.squares[y][x].map(|p| p.to_index()).unwrap_or(0);
            }
        }
        raw_board
    }
}

pub fn in_board(pos: Position) -> bool {
    pos.x >= 0 && (pos.x as i32) < BOARD_WIDTH && pos.y >= 0 && (pos.y as i32) < BOARD_HEIGHT
}

pub fn in_palace(color: Color, pos: Position) -> bool {
    if pos.x < 3 || pos.x > 5 {
        return false;
    }
    match color {
        Color::Red => (7..=9).contains(&pos.y),
        Color::Black => (0..=2).contains(&pos.y),
    }
}

pub fn crossed_river(color: Color, y: i8) -> bool {
    match color {
        Color::Red => y <= 4,
        Color::Black => y >= 5,
    }
}
