use crate::{board::PieceIndexBoard, constants::*, game::GameState, types::*};

const PIECE_KINDS: usize = PLANES_PER_FRAME / 2; // 7, 单方棋子种类数
const HW: usize = BOARD_HEIGHT as usize * BOARD_WIDTH as usize;
const EMPTY_BOARD: PieceIndexBoard = [[0u8; BOARD_WIDTH as usize]; BOARD_HEIGHT as usize];

/// channels 0-97 的总元素数（98 × 10 × 9 = 8820）
pub const PIECES_SIZE: usize = (INPUT_CHANNELS - 1) * HW;
/// 完整 state 的 f32 元素数（99 × 10 × 9 = 8910）
pub const STATE_F32_SIZE: usize = INPUT_CHANNELS * HW;

/// 编码后的局面，紧凑二进制形式。
///
/// - `pieces`: channels 0-97 的二值平面（0/1），按 CHW 展平
/// - `no_capture_plies`: 原始未吃子步数（0-100）
pub struct EncodedState {
    pub pieces: [u8; PIECES_SIZE],
    pub no_capture_plies: u8,
}

impl EncodedState {
    /// 还原为 inference 用的 f32 tensor `[INPUT_CHANNELS × BOARD_HEIGHT × BOARD_WIDTH]`。
    pub fn to_float_tensor(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(STATE_F32_SIZE);
        for &b in &self.pieces {
            out.push(b as f32);
        }
        let normalized = self.no_capture_plies as f32 / NO_CAPTURE_DRAW_PLIES as f32;
        for _ in 0..HW {
            out.push(normalized);
        }
        out
    }
}

fn view_coords(y: usize, x: usize, side_to_move: Color) -> (usize, usize) {
    match side_to_move {
        Color::Red => (y, x),
        Color::Black => (BOARD_HEIGHT as usize - 1 - y, x),
    }
}

fn piece_color(piece_index: u8) -> Color {
    if piece_index <= 7 {
        Color::Red
    } else {
        Color::Black
    }
}

fn piece_kind_index(piece_index: u8) -> usize {
    ((piece_index - 1) % 7) as usize
}

/// 将历史局面编码为紧凑的 `EncodedState`。
/// channels 0-97 为二值棋子平面，`no_capture_plies` 独立存储。
pub fn encode_state(game: &GameState) -> EncodedState {
    let side_to_move = game.board.side_to_move;
    let history = &game.history;

    let len = history.len();
    let mut frames: [&PieceIndexBoard; N_HISTORY] = [&EMPTY_BOARD; N_HISTORY];
    let start = len.saturating_sub(N_HISTORY);
    for (i, board) in history[start..].iter().enumerate() {
        frames[N_HISTORY - (len - start) + i] = board;
    }

    let mut pieces = [0u8; PIECES_SIZE];

    for (frame_idx, board) in frames.iter().enumerate() {
        for y in 0..BOARD_HEIGHT as usize {
            for x in 0..BOARD_WIDTH as usize {
                let (sy, sx) = view_coords(y, x, side_to_move);
                let piece_index = board[sy][sx];
                if piece_index == 0 {
                    continue;
                }

                let is_active = piece_color(piece_index) == side_to_move;
                let group_offset = if is_active {
                    0
                } else {
                    N_HISTORY * PIECE_KINDS
                };
                let kind = piece_kind_index(piece_index);
                let channel = group_offset + frame_idx * PIECE_KINDS + kind;

                pieces[channel * HW + y * BOARD_WIDTH as usize + x] = 1;
            }
        }
    }

    EncodedState {
        pieces,
        no_capture_plies: game.board.no_capture_plies.min(255) as u8,
    }
}

/// 将 Move 编码为 action_id（视角相关）。`action_id_to_move` 的逆运算。
pub fn move_to_action_id(mv: Move, side_to_move: Color) -> ActionId {
    let (from_vy, to_vy) = match side_to_move {
        Color::Red => (mv.from.y as i32, mv.to.y as i32),
        Color::Black => (
            BOARD_HEIGHT - 1 - mv.from.y as i32,
            BOARD_HEIGHT - 1 - mv.to.y as i32,
        ),
    };
    let from_view = from_vy * BOARD_WIDTH + mv.from.x as i32;
    let to_view = to_vy * BOARD_WIDTH + mv.to.x as i32;
    from_view * SQUARE_COUNT + to_view
}

/// 将 action_id 解码为 Move（视角相关）。
pub fn action_id_to_move(action_id: ActionId, side_to_move: Color) -> Move {
    let from_view = action_id / SQUARE_COUNT;
    let to_view = action_id % SQUARE_COUNT;

    let from_vx = from_view % BOARD_WIDTH;
    let from_vy = from_view / BOARD_WIDTH;
    let to_vx = to_view % BOARD_WIDTH;
    let to_vy = to_view / BOARD_WIDTH;

    let (from_x, from_y, to_x, to_y) = match side_to_move {
        Color::Red => (from_vx, from_vy, to_vx, to_vy),
        Color::Black => (
            from_vx,
            BOARD_HEIGHT - 1 - from_vy,
            to_vx,
            BOARD_HEIGHT - 1 - to_vy,
        ),
    };

    Move::new(
        Position::new(from_x as i8, from_y as i8),
        Position::new(to_x as i8, to_y as i8),
    )
}
