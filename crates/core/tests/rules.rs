//! engine 规则原生单测（取自原 Python tests/test_rules.py 的人工核对用例）。

use std::collections::BTreeSet;

use cc_core::engine::{
    action_id_to_move, move_to_action_id, Color, GameState, GameStatusReason, Move, PieceKind,
    Position, ACTION_SPACE_SIZE,
};

#[test]
fn starting_position_round_trips_and_has_expected_moves() {
    let pos = Position::starting();
    assert_eq!(
        pos.to_fen(),
        "rheakaehr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RHEAKAEHR r"
    );
    assert_eq!(pos.legal_moves().len(), 44);
    let mask = pos.legal_move_mask();
    assert_eq!(mask.iter().filter(|&&b| b).count(), 44);
    assert_eq!(mask.len(), ACTION_SPACE_SIZE);
}

#[test]
fn action_id_round_trip() {
    let mv = Move::new(1, 9, 2, 7);
    assert_eq!(action_id_to_move(move_to_action_id(mv)), mv);
}

#[test]
fn make_move_returns_new_position() {
    let pos = Position::starting();
    let next = pos.make_move(Move::new(1, 9, 2, 7));

    assert!(pos.piece_at(2, 7).is_none());
    let moved = next.piece_at(2, 7).expect("piece moved to (2,7)");
    assert_eq!(moved.color, Color::Red);
    assert_eq!(moved.kind, PieceKind::Horse);
    assert!(next.piece_at(1, 9).is_none());
    assert_eq!(next.side_to_move, Color::Black);
}

#[test]
fn horse_leg_blocks_move() {
    let pos = Position::from_fen("4k4/9/9/9/9/9/4P4/4H4/9/4K4 r").unwrap();
    assert!(!pos.legal_moves().contains(&Move::new(4, 7, 5, 5)));
    assert!(pos.legal_moves().contains(&Move::new(4, 7, 6, 8)));
}

#[test]
fn elephant_cannot_cross_or_jump_eye() {
    let blocked_eye = Position::from_fen("4k4/9/9/9/9/9/9/9/3P5/2E1K4 r").unwrap();
    let river_edge = Position::from_fen("4k4/9/9/9/9/2E6/9/9/9/4K4 r").unwrap();
    assert!(!blocked_eye.legal_moves().contains(&Move::new(2, 9, 4, 7)));
    assert!(!river_edge.legal_moves().contains(&Move::new(2, 5, 4, 3)));
}

#[test]
fn advisor_stays_inside_palace() {
    let pos = Position::from_fen("4k4/9/9/9/9/9/9/9/3A5/4K4 r").unwrap();
    assert!(pos.legal_moves().contains(&Move::new(3, 8, 4, 7)));
    assert!(!pos.legal_moves().contains(&Move::new(3, 8, 2, 7)));
}

#[test]
fn pawn_moves_sideways_only_after_crossing_river() {
    let before = Position::from_fen("k8/9/9/9/9/9/4P4/9/9/4K4 r").unwrap();
    let after = Position::from_fen("k8/9/9/9/4P4/9/9/9/9/4K4 r").unwrap();

    let before_targets: BTreeSet<(i32, i32)> = before
        .legal_moves()
        .into_iter()
        .filter(|m| m.source() == (4, 6))
        .map(|m| m.target())
        .collect();
    let before_expected: BTreeSet<(i32, i32)> = [(4, 5)].into_iter().collect();
    assert_eq!(before_targets, before_expected);

    let after_targets: BTreeSet<(i32, i32)> = after
        .legal_moves()
        .into_iter()
        .filter(|m| m.source() == (4, 4))
        .map(|m| m.target())
        .collect();
    let after_expected: BTreeSet<(i32, i32)> = [(4, 3), (3, 4), (5, 4)].into_iter().collect();
    assert_eq!(after_targets, after_expected);
}

#[test]
fn cannon_needs_exactly_one_screen_to_capture() {
    let pos = Position::from_fen("4k4/9/9/9/4p4/9/4P4/9/4C4/4K4 r").unwrap();
    assert!(pos.legal_moves().contains(&Move::new(4, 8, 4, 4)));
    assert!(!pos.legal_moves().contains(&Move::new(4, 8, 4, 0)));
}

#[test]
fn flying_general_blocks_exposing_own_king() {
    let pos = Position::from_fen("4k4/9/4R4/9/9/9/9/9/9/4K4 r").unwrap();
    assert!(!pos.is_in_check(Color::Red));
    // 车横移离开 4 路 → 红帅黑将照面 → 走子方自曝于飞将，非法。
    assert!(!pos.legal_moves().contains(&Move::new(4, 2, 3, 2)));
    // 车沿 4 路上行逼近黑将（将军），合法。
    assert!(pos.legal_moves().contains(&Move::new(4, 2, 4, 1)));
    // 注：不断言“吃将”这步（4,2→4,0）。实战中轮到走子方能吃对方将的局面不可达，
    // 这里不去复刻 Python 参考实现吃将后残留将位、误触飞将的副作用。
}

#[test]
fn checkmate_status() {
    let pos = Position::from_fen("3k5/4R4/3R5/9/9/9/9/9/9/4K4 b").unwrap();
    let status = pos.status();
    assert!(status.is_terminal);
    assert_eq!(status.reason, Some(GameStatusReason::Checkmate));
    assert_eq!(status.winner, Some(Color::Red));
}

#[test]
fn repetition_key_ignores_counters() {
    let first = Position::from_fen("4k4/9/9/9/9/9/9/9/9/4K4 r - - 0 1").unwrap();
    let later = Position::from_fen("4k4/9/9/9/9/9/9/9/9/4K4 r - - 8 20").unwrap();
    assert_eq!(first.repetition_key(), later.repetition_key());
}

#[test]
fn piece_indexes_follow_moves() {
    let pos = Position::starting();
    let moving_id = pos.piece_at(1, 9).expect("horse at (1,9)").piece_id;

    let next = pos.make_move(Move::new(1, 9, 2, 7));
    let moved = next.piece_at(2, 7).expect("horse at (2,7)");
    assert!(next.piece_at(1, 9).is_none());
    assert_eq!(moved.piece_id, moving_id);
    assert_eq!(next.piece_positions[&moving_id], (2, 7));
    assert_eq!(next.king_square(Color::Red), Some((4, 9)));
}

/// 把一个走法环重复两遍，构造三次重复局面以触发长将/长捉/互打判定。
fn play_cycle(start_fen: &str, cycle: &[Move]) -> GameState {
    let mut game = GameState::from_position(Position::from_fen(start_fen).unwrap());
    for _ in 0..2 {
        for &mv in cycle {
            game = game.make_move(mv);
        }
    }
    game
}

#[test]
fn detects_perpetual_check() {
    let game = play_cycle(
        "4k4/9/3R5/9/9/9/9/9/4A4/4K4 r",
        &[
            Move::new(3, 2, 4, 2),
            Move::new(4, 0, 3, 0),
            Move::new(4, 2, 3, 2),
            Move::new(3, 0, 4, 0),
        ],
    );
    let status = game.status();
    assert!(status.is_terminal);
    assert_eq!(status.reason, Some(GameStatusReason::PerpetualCheck));
    assert_eq!(status.winner, Some(Color::Black));
}

#[test]
fn detects_perpetual_chase() {
    let game = play_cycle(
        "k3h3h/9/9/9/4R4/9/9/9/9/1K7 r",
        &[
            Move::new(4, 4, 4, 5),
            Move::new(8, 0, 7, 2),
            Move::new(4, 5, 4, 4),
            Move::new(7, 2, 8, 0),
        ],
    );
    let status = game.status();
    assert!(status.is_terminal);
    assert_eq!(status.reason, Some(GameStatusReason::PerpetualChase));
    assert_eq!(status.winner, Some(Color::Black));
}

#[test]
fn detects_mutual_perpetual() {
    let game = play_cycle(
        "k3h4/8r/9/9/4R4/9/9/9/9/1K6H r",
        &[
            Move::new(4, 4, 4, 5),
            Move::new(8, 1, 8, 2),
            Move::new(4, 5, 4, 4),
            Move::new(8, 2, 8, 1),
        ],
    );
    let status = game.status();
    assert!(status.is_terminal);
    assert_eq!(status.reason, Some(GameStatusReason::MutualPerpetual));
    assert_eq!(status.winner, None);
}

#[test]
fn bench_legal_moves() {
    use std::time::Instant;

    let positions = vec![
        ("开局", Position::starting()),
        ("中局", Position::from_fen("2eak4/4a4/4e3c/p3C1p1p/4p4/2P1R4/P3P1P1P/2H1E1H2/4A4/2EAK4 r").unwrap()),
        ("残局", Position::from_fen("4k4/9/9/9/9/9/4r4/9/4A4/3AK4 r").unwrap()),
    ];

    let warmup = 500;
    let iters = 5000;

    for (label, pos) in &positions {
        // warmup
        for _ in 0..warmup {
            let _ = std::hint::black_box(pos.legal_moves());
        }

        let t0 = Instant::now();
        let mut total_moves = 0u64;
        for _ in 0..iters {
            let moves = pos.legal_moves();
            total_moves += moves.len() as u64;
            std::hint::black_box(&moves);
        }
        let elapsed = t0.elapsed();
        let per_call = elapsed / iters;
        let avg_moves = total_moves / iters as u64;

        eprintln!(
            "[{label}] legal_moves: {per_call:?}/次 | 合法走法 {avg_moves} 个 | {iters} 次共 {elapsed:?}"
        );
    }

    // 模拟一局对弈中连续 legal_moves 调用(含 make_move 推进局面)
    let mut state = GameState::from_position(Position::starting());
    let mut ply = 0u32;
    let sim_moves = 60; // 模拟前 60 手
    let t0 = Instant::now();
    while ply < sim_moves {
        let moves = state.position.legal_moves();
        if moves.is_empty() {
            break;
        }
        state = state.make_move(moves[0]); // 每次走第一个合法走法
        ply += 1;
    }
    let elapsed = t0.elapsed();
    eprintln!(
        "[对弈模拟] {ply} 手 legal_moves+make_move: 共 {elapsed:?} | 均 {:?}/手",
        elapsed / ply
    );
}
