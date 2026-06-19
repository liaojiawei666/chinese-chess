//! 基础类型：颜色、子种、棋子、走法、动作编码、几何判定。
//! 与 trainer/src/trainer/reference/engine.py 的同名定义一一对应。

use crate::constants::{ACTION_SPACE_SIZE, BOARD_HEIGHT, BOARD_WIDTH, SQUARE_COUNT};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Color {
    Red,
    Black,
}

impl Color {
    pub fn opposite(self) -> Color {
        match self {
            Color::Red => Color::Black,
            Color::Black => Color::Red,
        }
    }

    pub fn fen_side(self) -> char {
        match self {
            Color::Red => 'r',
            Color::Black => 'b',
        }
    }

    pub fn from_fen_side(value: &str) -> Result<Color, String> {
        match value {
            "r" => Ok(Color::Red),
            "b" => Ok(Color::Black),
            _ => Err(format!("Invalid side to move: {value}")),
        }
    }

    /// 与 Python Color.value 对应的字符串（"red" / "black"），用于序列化对齐。
    pub fn as_str(self) -> &'static str {
        match self {
            Color::Red => "red",
            Color::Black => "black",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PieceKind {
    King,
    Advisor,
    Elephant,
    Horse,
    Rook,
    Cannon,
    Pawn,
}

impl PieceKind {
    pub fn from_fen_char(value: char) -> Option<PieceKind> {
        match value.to_ascii_lowercase() {
            'k' => Some(PieceKind::King),
            'a' => Some(PieceKind::Advisor),
            'e' => Some(PieceKind::Elephant),
            'h' => Some(PieceKind::Horse),
            'r' => Some(PieceKind::Rook),
            'c' => Some(PieceKind::Cannon),
            'p' => Some(PieceKind::Pawn),
            _ => None,
        }
    }

    pub fn fen_char(self) -> char {
        match self {
            PieceKind::King => 'k',
            PieceKind::Advisor => 'a',
            PieceKind::Elephant => 'e',
            PieceKind::Horse => 'h',
            PieceKind::Rook => 'r',
            PieceKind::Cannon => 'c',
            PieceKind::Pawn => 'p',
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Piece {
    pub piece_id: u32,
    pub color: Color,
    pub kind: PieceKind,
}

impl Piece {
    pub fn from_fen(piece_id: u32, value: char) -> Result<Piece, String> {
        let kind = PieceKind::from_fen_char(value).ok_or_else(|| format!("Invalid piece: {value}"))?;
        let color = if value.is_ascii_uppercase() {
            Color::Red
        } else {
            Color::Black
        };
        Ok(Piece { piece_id, color, kind })
    }

    pub fn to_fen(self) -> char {
        let c = self.kind.fen_char();
        if self.color == Color::Red {
            c.to_ascii_uppercase()
        } else {
            c
        }
    }

    pub fn is_same_kind_as(self, other: &Piece) -> bool {
        self.kind == other.kind
    }
}

/// 一步棋：源格 (sx, sy) → 目标格 (tx, ty)。坐标用 i32 以便走法生成时做带符号运算。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Move {
    pub sx: i32,
    pub sy: i32,
    pub tx: i32,
    pub ty: i32,
}

impl Move {
    pub fn new(sx: i32, sy: i32, tx: i32, ty: i32) -> Move {
        Move { sx, sy, tx, ty }
    }

    pub fn source(self) -> (i32, i32) {
        (self.sx, self.sy)
    }

    pub fn target(self) -> (i32, i32) {
        (self.tx, self.ty)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GameStatusReason {
    Checkmate,
    Stalemate,
    PerpetualCheck,
    PerpetualChase,
    MutualPerpetual,
    FiftyMove,
    MaxMoves,
}

impl GameStatusReason {
    pub fn as_str(self) -> &'static str {
        match self {
            GameStatusReason::Checkmate => "checkmate",
            GameStatusReason::Stalemate => "stalemate",
            GameStatusReason::PerpetualCheck => "perpetual_check",
            GameStatusReason::PerpetualChase => "perpetual_chase",
            GameStatusReason::MutualPerpetual => "mutual_perpetual",
            GameStatusReason::FiftyMove => "fifty_move",
            GameStatusReason::MaxMoves => "max_moves",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GameStatus {
    pub is_terminal: bool,
    pub reason: Option<GameStatusReason>,
    pub winner: Option<Color>,
}

impl GameStatus {
    pub fn ongoing() -> GameStatus {
        GameStatus { is_terminal: false, reason: None, winner: None }
    }

    pub fn terminal(reason: GameStatusReason, winner: Option<Color>) -> GameStatus {
        GameStatus { is_terminal: true, reason: Some(reason), winner }
    }
}

pub fn square_index(x: i32, y: i32) -> usize {
    (y as usize) * BOARD_WIDTH + (x as usize)
}

pub fn move_to_action_id(mv: Move) -> usize {
    square_index(mv.sx, mv.sy) * SQUARE_COUNT + square_index(mv.tx, mv.ty)
}

pub fn action_id_to_move(action_id: usize) -> Move {
    assert!(action_id < ACTION_SPACE_SIZE, "Invalid action id: {action_id}");
    let source = action_id / SQUARE_COUNT;
    let target = action_id % SQUARE_COUNT;
    let sy = (source / BOARD_WIDTH) as i32;
    let sx = (source % BOARD_WIDTH) as i32;
    let ty = (target / BOARD_WIDTH) as i32;
    let tx = (target % BOARD_WIDTH) as i32;
    Move::new(sx, sy, tx, ty)
}

pub fn in_board(x: i32, y: i32) -> bool {
    x >= 0 && x < BOARD_WIDTH as i32 && y >= 0 && y < BOARD_HEIGHT as i32
}

pub fn in_palace(color: Color, x: i32, y: i32) -> bool {
    if x < 3 || x > 5 {
        return false;
    }
    match color {
        Color::Red => (7..=9).contains(&y),
        Color::Black => (0..=2).contains(&y),
    }
}

pub fn crossed_river(color: Color, y: i32) -> bool {
    match color {
        Color::Red => y <= 4,
        Color::Black => y >= 5,
    }
}
