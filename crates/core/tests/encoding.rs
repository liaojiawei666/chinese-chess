//! encoder 原生单测：编码张量基本性质 + canonical 动作往返（对合）。

use cc_core::encode::{encode, from_canonical_action, to_canonical_action};
use cc_core::engine::{move_to_action_id, Color, GameState, Position, INPUT_CHANNELS};

const PLANE_SIZE: usize = 10 * 9;
const PLANES_PER_FRAME: usize = 14;

#[test]
fn encode_has_expected_shape() {
    let state = GameState::from_position(Position::starting());
    assert_eq!(encode(&state).len(), INPUT_CHANNELS * PLANE_SIZE);
}

#[test]
fn starting_frame0_has_one_cell_per_piece() {
    let state = GameState::from_position(Position::starting());
    let tensor = encode(&state);
    // 帧0 = 前 14 个通道（自己 7 + 对方 7 种子），起手 32 子各置 1，其余 0。
    let frame0_sum: f32 = tensor[..PLANES_PER_FRAME * PLANE_SIZE].iter().sum();
    assert_eq!(frame0_sum as i32, 32);
}

#[test]
fn starting_kings_on_expected_planes() {
    let state = GameState::from_position(Position::starting());
    let tensor = encode(&state);
    // perspective=Red：红帅(4,9) 在通道 0（自己·帅）；黑将(4,0) 在通道 7（对方·帅）。
    assert_eq!(tensor[9 * 9 + 4], 1.0);
    assert_eq!(tensor[7 * PLANE_SIZE + 4], 1.0);
}

#[test]
fn canonical_action_round_trips_both_perspectives() {
    let pos = Position::starting();
    for mv in pos.legal_moves() {
        for perspective in [Color::Red, Color::Black] {
            let action = to_canonical_action(mv, perspective);
            assert_eq!(from_canonical_action(action, perspective), mv);
        }
    }
}

#[test]
fn red_perspective_action_equals_raw_action_id() {
    let pos = Position::starting();
    for mv in pos.legal_moves() {
        assert_eq!(to_canonical_action(mv, Color::Red), move_to_action_id(mv));
    }
}
