use std::collections::HashMap;

use crate::attack::analyze_cycle;
use crate::board::*;
use crate::constants::*;
use crate::move_gen::{is_in_check, legal_moves};
use crate::types::*;

#[derive(Clone)]
pub struct GameState {
    pub board: Board,
    pub status: GameStatus,
    pub history: Vec<PieceIndexBoard>,
    pub zobrist_hash_history: HashMap<u64, Vec<u32>>,
    pub move_info_history: Vec<MoveInfo>,
}

impl GameState {
    pub fn new(board: Board) -> GameState {
        let history = vec![board.get_piece_index_board()];
        let mut zobrist_hash_history = HashMap::new();
        zobrist_hash_history.insert(board.zobrist_hash, vec![0u32]);
        GameState {
            board,
            status: GameStatus::ongoing(),
            history,
            zobrist_hash_history,
            move_info_history: Vec::new(),
        }
    }

    pub fn from_fen(fen: &str) -> Result<GameState, String> {
        let board = Board::from_fen(fen)?;
        Ok(GameState::new(board))
    }

    pub fn start_pos() -> GameState {
        GameState::new(Board::start_pos())
    }

    /// 走一步棋并更新游戏状态。仅在 status 为 ongoing 时允许调用。
    pub fn make_move(&mut self, mv: Move) {
        let move_info = self.board.make_move(mv);
        self.move_info_history.push(move_info);
        self.history.push(self.board.get_piece_index_board());

        let state_idx = self.move_info_history.len() as u32;
        self.zobrist_hash_history
            .entry(self.board.zobrist_hash)
            .or_default()
            .push(state_idx);

        self.status = self.check_status();
    }

    /// 悔棋，恢复到上一步状态。
    pub fn undo_move(&mut self) {
        if let Some(move_info) = self.move_info_history.pop() {
            let indices = self
                .zobrist_hash_history
                .get_mut(&self.board.zobrist_hash)
                .unwrap();
            indices.pop();
            if indices.is_empty() {
                self.zobrist_hash_history.remove(&self.board.zobrist_hash);
            }

            self.history.pop();
            self.board.undo_move(&move_info);
            self.status = GameStatus::ongoing();
        }
    }
    pub fn legal_moves(&mut self) -> Vec<Move> {
        let slide_to_move = self.board.side_to_move;
        legal_moves(&mut self.board, slide_to_move)
    }

    /// 走完一步后检查游戏状态
    fn check_status(&mut self) -> GameStatus {
        let side_to_move = self.board.side_to_move;
        let moves = legal_moves(&mut self.board, side_to_move);

        // 1. 将杀 / 困毙
        if moves.is_empty() {
            let reason = if is_in_check(&self.board, side_to_move) {
                GameStatusReason::Checkmate
            } else {
                GameStatusReason::Stalemate
            };
            return GameStatus::terminal(reason, Some(side_to_move.opposite()));
        }

        // 2. 禁着循环：同一局面第三次出现时触发分析
        let repetition_count = self
            .zobrist_hash_history
            .get(&self.board.zobrist_hash)
            .unwrap()
            .len();
        if repetition_count >= 3 {
            if let Some(status) = self.check_cycle() {
                return status;
            }
        }

        // 3. 自然限着：连续 NO_CAPTURE_DRAW_PLIES 步未吃子
        if self.board.no_capture_plies >= NO_CAPTURE_DRAW_PLIES {
            return GameStatus::terminal(GameStatusReason::MaxNonCapture, None);
        }

        // 4. 总步数限制
        let total_plies = (self.board.fullmove_number - 1) * 2
            + if self.board.side_to_move == Color::Black {
                1
            } else {
                0
            };
        if total_plies >= MAX_TOTAL_PLIES {
            return GameStatus::terminal(GameStatusReason::MaxMoves, None);
        }

        GameStatus::ongoing()
    }

    /// 检测循环禁着，返回 Some(status) 如果应终止比赛
    fn check_cycle(&self) -> Option<GameStatus> {
        let indices = self.zobrist_hash_history.get(&self.board.zobrist_hash)?;
        if indices.len() < 3 {
            return None;
        }

        // 取最近两次相同局面之间的走法作为循环
        let current_idx = *indices.last().unwrap() as usize;
        let prev_idx = indices[indices.len() - 2] as usize;
        let cycle_moves = &self.move_info_history[prev_idx..current_idx];
        let violations = analyze_cycle(self.board.clone(), cycle_moves);

        let red = violations.get(Color::Red);
        let black = violations.get(Color::Black);

        match (red, black) {
            // 一方犯规，另一方不犯规 → 犯规方负
            (Some(_), None) => Some(GameStatus::terminal(red.unwrap(), Some(Color::Black))),
            (None, Some(_)) => Some(GameStatus::terminal(black.unwrap(), Some(Color::Red))),
            // 双方均犯规
            (Some(r), Some(b)) => {
                let r_severity = violation_severity(*r);
                let b_severity = violation_severity(*b);
                if r_severity > b_severity {
                    Some(GameStatus::terminal(*r, Some(Color::Black)))
                } else if b_severity > r_severity {
                    Some(GameStatus::terminal(*b, Some(Color::Red)))
                } else {
                    Some(GameStatus::terminal(
                        GameStatusReason::MutualPerpetual,
                        None,
                    ))
                }
            }
            // 双方均不犯规 → 和
            (None, None) => Some(GameStatus::terminal(
                GameStatusReason::MutualPerpetual,
                None,
            )),
        }
    }
}

/// 犯规严重程度（数值越大越严重）
fn violation_severity(reason: GameStatusReason) -> u8 {
    match reason {
        GameStatusReason::PerpetualCheck => 2,
        GameStatusReason::PerpetualChase => 1,
        _ => 0,
    }
}
