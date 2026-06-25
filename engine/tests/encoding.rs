use engine::constants::*;
use engine::encode::*;
use engine::game::GameState;
use engine::types::*;

#[test]
fn test_action_id_roundtrip_red() {
    let from = Position::new(4, 9);
    let to = Position::new(4, 8);
    let mv = Move::new(from, to);

    let aid = move_to_action_id(mv, Color::Red);
    let decoded = action_id_to_move(aid, Color::Red);

    assert_eq!(decoded.from, mv.from);
    assert_eq!(decoded.to, mv.to);
}

#[test]
fn test_action_id_roundtrip_black() {
    let from = Position::new(4, 0);
    let to = Position::new(4, 1);
    let mv = Move::new(from, to);

    let aid = move_to_action_id(mv, Color::Black);
    let decoded = action_id_to_move(aid, Color::Black);

    assert_eq!(decoded.from, mv.from);
    assert_eq!(decoded.to, mv.to);
}

#[test]
fn test_action_id_roundtrip_all_legal_moves() {
    let mut game = GameState::start_pos();
    let side = game.board.side_to_move;
    let moves = game.legal_moves();

    for mv in &moves {
        let aid = move_to_action_id(*mv, side);
        assert!(aid >= 0 && (aid as usize) < ACTION_SPACE_SIZE);
        let decoded = action_id_to_move(aid, side);
        assert_eq!(decoded.from, mv.from, "from mismatch for {:?}", mv);
        assert_eq!(decoded.to, mv.to, "to mismatch for {:?}", mv);
    }
}

#[test]
fn test_action_id_black_flip() {
    // Same physical move, different action_id for red vs black due to Y-flip
    let from = Position::new(4, 0);
    let to = Position::new(4, 1);
    let mv = Move::new(from, to);

    let aid_red = move_to_action_id(mv, Color::Red);
    let aid_black = move_to_action_id(mv, Color::Black);
    assert_ne!(aid_red, aid_black, "same physical move should differ by side");
}

#[test]
fn test_encode_start_pos() {
    let game = GameState::start_pos();
    let encoded = encode_state(&game);

    // Pieces are in channels 0..98 (8820 bytes)
    assert_eq!(encoded.pieces.len(), PIECES_SIZE);

    // At start, no_capture_plies = 0
    assert_eq!(encoded.no_capture_plies, 0);

    // Float tensor should have 99*10*9 = 8910 elements
    let tensor = encoded.to_float_tensor();
    assert_eq!(tensor.len(), STATE_F32_SIZE);

    // Channel 98 should be all 0 (no_capture_plies = 0)
    let ch98_start = 98 * (BOARD_HEIGHT as usize * BOARD_WIDTH as usize);
    for i in ch98_start..tensor.len() {
        assert_eq!(tensor[i], 0.0, "channel 98 should be 0 at start");
    }

    // There should be some non-zero values in the piece channels
    let has_pieces = encoded.pieces.iter().any(|&b| b != 0);
    assert!(has_pieces, "start position should have piece data");
}

#[test]
fn test_encode_state_has_pieces_in_latest_frame() {
    let game = GameState::start_pos();
    let encoded = encode_state(&game);

    // The latest frame for side-to-move (red) is at channels (N_HISTORY-1)*PIECE_KINDS .. N_HISTORY*PIECE_KINDS
    // = channels 42..49
    let hw = BOARD_HEIGHT as usize * BOARD_WIDTH as usize;
    let frame_start = (N_HISTORY - 1) * (PLANES_PER_FRAME / 2) * hw;
    let frame_end = N_HISTORY * (PLANES_PER_FRAME / 2) * hw;

    let has_active_pieces = encoded.pieces[frame_start..frame_end]
        .iter()
        .any(|&b| b != 0);
    assert!(has_active_pieces, "latest frame should have active pieces");
}

#[test]
fn test_history_padding() {
    // Start position has 1 history frame (initial board).
    // After 1 move, 2 frames. Still < N_HISTORY (7), so early channels should be padded with 0.
    let mut game = GameState::start_pos();
    let moves = game.legal_moves();
    game.make_move(moves[0]);

    let encoded = encode_state(&game);

    // With only 2 frames (initial + after 1 move), frames 0..4 should be empty.
    let hw = BOARD_HEIGHT as usize * BOARD_WIDTH as usize;
    let piece_kinds = PLANES_PER_FRAME / 2;
    let empty_frames = N_HISTORY - 2;
    let empty_end = empty_frames * piece_kinds * hw;
    for &b in &encoded.pieces[..empty_end] {
        assert_eq!(b, 0, "early padded frames should be all zeros");
    }
}
