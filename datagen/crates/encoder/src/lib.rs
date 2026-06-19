//! 局面编码 + 动作映射（移植自 trainer/src/trainer/reference/encoder.py）。
//!
//! - `encode`：把 GameState 编成 canonical 视角的 (99,10,9) 张量，按 C 序展平成 Vec<f32>
//!   （flat = channel*90 + y*9 + x，与 numpy reshape(-1) 一致）。
//! - 视角统一为当前走棋方：红方按原样，黑方上下翻转 + 红黑通道互换。

use engine::{
    action_id_to_move, move_to_action_id, Color, GameState, Move, PieceKind, Position,
    ACTION_SPACE_SIZE, BOARD_HEIGHT, BOARD_WIDTH, INPUT_CHANNELS, N_HISTORY, NO_CAPTURE_DRAW_PLIES,
    PLANES_PER_FRAME,
};

/// 子种在单帧内的通道顺序：帅、车、马、炮、仕、相、兵。
pub fn kind_index(kind: PieceKind) -> usize {
    match kind {
        PieceKind::King => 0,
        PieceKind::Rook => 1,
        PieceKind::Horse => 2,
        PieceKind::Cannon => 3,
        PieceKind::Advisor => 4,
        PieceKind::Elephant => 5,
        PieceKind::Pawn => 6,
    }
}

fn flip_y(y: i32) -> i32 {
    BOARD_HEIGHT as i32 - 1 - y
}

/// 真实坐标 → 当前走棋方视角坐标。红方不变；黑方上下翻转（对合运算，正逆同形）。
pub fn canonical_square(x: i32, y: i32, perspective: Color) -> (i32, i32) {
    match perspective {
        Color::Black => (x, flip_y(y)),
        Color::Red => (x, y),
    }
}

pub fn canonical_move(mv: Move, perspective: Color) -> Move {
    let (sx, sy) = canonical_square(mv.sx, mv.sy, perspective);
    let (tx, ty) = canonical_square(mv.tx, mv.ty, perspective);
    Move::new(sx, sy, tx, ty)
}

/// 真实走法 → canonical 视角下的 action_id（与网络 policy 同坐标系）。
pub fn to_canonical_action(mv: Move, perspective: Color) -> usize {
    move_to_action_id(canonical_move(mv, perspective))
}

/// canonical action_id → 真实走法（把网络选的动作还原到真实棋盘）。
pub fn from_canonical_action(action_id: usize, perspective: Color) -> Move {
    canonical_move(action_id_to_move(action_id), perspective)
}

/// canonical 视角下的合法动作掩码，长度 ACTION_SPACE_SIZE。
pub fn legal_mask(position: &Position) -> Vec<bool> {
    let perspective = position.side_to_move;
    let mut mask = vec![false; ACTION_SPACE_SIZE];
    for mv in position.legal_moves() {
        mask[to_canonical_action(mv, perspective)] = true;
    }
    mask
}

const PLANE_SIZE: usize = BOARD_HEIGHT * BOARD_WIDTH;

fn encode_frame_into(tensor: &mut [f32], position: &Position, perspective: Color, start_channel: usize) {
    let flip = perspective == Color::Black;
    for y in 0..BOARD_HEIGHT {
        for x in 0..BOARD_WIDTH {
            let piece = match position.board[y][x] {
                Some(p) => p,
                None => continue,
            };
            let ki = kind_index(piece.kind);
            let channel = start_channel
                + if piece.color == perspective {
                    ki
                } else {
                    PLANES_PER_FRAME / 2 + ki
                };
            let ry = if flip { flip_y(y as i32) as usize } else { y };
            tensor[channel * PLANE_SIZE + ry * BOARD_WIDTH + x] = 1.0;
        }
    }
}

/// 最近 N_HISTORY 个局面，最新在前；不足留空（调用方补 0 帧）。
fn recent_positions(state: &GameState) -> Vec<&Position> {
    let mut positions: Vec<&Position> = vec![&state.position];
    for record in state.history.iter().rev() {
        if positions.len() == N_HISTORY {
            break;
        }
        positions.push(&record.position_before);
    }
    positions
}

/// 把对局状态编成 canonical 视角的 (99,10,9) 张量（C 序展平）。
pub fn encode(state: &GameState) -> Vec<f32> {
    let perspective = state.position.side_to_move;
    let mut tensor = vec![0f32; INPUT_CHANNELS * PLANE_SIZE];

    for (index, position) in recent_positions(state).into_iter().enumerate() {
        let start_channel = index * PLANES_PER_FRAME;
        encode_frame_into(&mut tensor, position, perspective, start_channel);
    }

    let ratio =
        (state.position.no_capture_plies as f32 / NO_CAPTURE_DRAW_PLIES as f32).min(1.0);
    let last_channel = N_HISTORY * PLANES_PER_FRAME; // 98
    let base = last_channel * PLANE_SIZE;
    for cell in tensor[base..base + PLANE_SIZE].iter_mut() {
        *cell = ratio;
    }
    tensor
}
