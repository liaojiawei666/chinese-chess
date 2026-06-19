//! 中国象棋规则引擎（移植自 trainer/src/trainer/reference/engine.py）。
//!
//! 结构常量是与 Python 侧的硬契约，同时被 `encoder` / `mcts` 以及 run-config 一致性
//! 检查复用。Position/GameState/legal_moves/make_move/status/fen 在此实现，并由
//! trainer/scripts/dump_fixtures.py 产出的夹具做差分测试（见 tests/differential.rs）。

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
