//! 中国象棋规则引擎。
//!
//! 结构常量是与 trainer 侧的硬契约（经 run-config 一致性检查校验），同时被 `encoder` /
//! `mcts` 复用。Position/GameState/legal_moves/make_move/status/fen 在此实现，规则正确性
//! 由 tests/rules.rs 的原生单测保证。

pub mod constants;
pub mod game;
pub mod position;
pub mod types;

pub use constants::*;
pub use game::{AttackedTarget, Attacker, GameState, MoveRecord};
pub use position::{MoveValidationReason, Position};
pub use types::{
    action_id_to_move, crossed_river, in_board, in_palace, move_to_action_id, square_index, Color,
    GameStatus, GameStatusReason, Move, Piece, PieceKind,
};
