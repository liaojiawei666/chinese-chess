use crate::constants::BOARD_WIDTH;

pub type ActionId = i32;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum PieceKind {
    King = 1,
    Advisor = 2,
    Elephant = 3,
    Horse = 4,
    Rook = 5,
    Cannon = 6,
    Pawn = 7,
}

pub type PieceIndex = u8; //红方1-7，即PieceKind，黑方8-14,即7+PieceKind

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Color {
    Red,
    Black,
}

impl Color {
    pub const fn all() -> [Color; 2] {
        [Color::Red, Color::Black]
    }

    pub fn opposite(self) -> Color {
        match self {
            Color::Red => Color::Black,
            Color::Black => Color::Red,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ByColor<T> {
    pub red: T,
    pub black: T,
}

impl<T> ByColor<T> {
    pub fn new(red: T, black: T) -> Self {
        ByColor { red, black }
    }

    pub fn get(&self, color: Color) -> &T {
        match color {
            Color::Red => &self.red,
            Color::Black => &self.black,
        }
    }

    pub fn get_mut(&mut self, color: Color) -> &mut T {
        match color {
            Color::Red => &mut self.red,
            Color::Black => &mut self.black,
        }
    }
}

impl<T: Default> Default for ByColor<T> {
    fn default() -> Self {
        ByColor {
            red: T::default(),
            black: T::default(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Piece {
    pub piece_id: u8,
    pub color: Color,
    pub kind: PieceKind,
}

impl Piece {
    pub fn is_same_kind_as(self, other: &Piece) -> bool {
        self.kind == other.kind
    }
    pub fn is_king(&self) -> bool {
        self.kind == PieceKind::King
    }
    pub fn to_index(&self) -> PieceIndex {
        match self.color {
            Color::Red => self.kind as u8,
            Color::Black => self.kind as u8 + 7,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Position {
    pub x: i8,
    pub y: i8,
}

impl Position {
    pub fn new(x: i8, y: i8) -> Position {
        Position { x, y }
    }
    pub fn add(&self, x: i8, y: i8) -> Position {
        Position {
            x: self.x + x,
            y: self.y + y,
        }
    }
    pub fn to_index(&self) -> i32 {
        (self.y as i32) * BOARD_WIDTH + (self.x as i32)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Move {
    pub from: Position,
    pub to: Position,
}

impl Move {
    pub fn new(from: Position, to: Position) -> Move {
        Move { from, to }
    }
    pub fn opposite(self) -> Move {
        Move {
            from: self.to,
            to: self.from,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GameStatusReason {
    Checkmate,       // 将杀
    Stalemate,       // 困毙（无合法走法）
    PerpetualCheck,  // 长将
    PerpetualChase,  // 长捉
    MutualPerpetual, // 双方犯规相同，或者双方都不犯规
    MaxNonCapture,   // 连续未吃子回合数达到上限
    MaxMoves,        // 总步数达到上限
}

impl GameStatusReason {
    pub fn as_str(self) -> &'static str {
        match self {
            GameStatusReason::Checkmate => "checkmate",
            GameStatusReason::Stalemate => "stalemate",
            GameStatusReason::PerpetualCheck => "perpetual_check",
            GameStatusReason::PerpetualChase => "perpetual_chase",
            GameStatusReason::MutualPerpetual => "mutual_perpetual",
            GameStatusReason::MaxNonCapture => "max_non_capture",
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
        GameStatus {
            is_terminal: false,
            reason: None,
            winner: None,
        }
    }

    pub fn terminal(reason: GameStatusReason, winner: Option<Color>) -> GameStatus {
        GameStatus {
            is_terminal: true,
            reason: Some(reason),
            winner,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoveFilter {
    All,
    CaptureOnly,
}

#[derive(Debug, Clone)]
pub struct MoveInfo {
    pub mv: Move,
    pub source_piece: Piece,
    pub target_piece: Option<Piece>,
    pub before_zobrist_hash: u64,
    pub before_no_capture_plies: u32,
    pub before_fullmove_number: u32,
}
