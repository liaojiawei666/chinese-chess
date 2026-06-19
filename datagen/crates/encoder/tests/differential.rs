//! 差分测试：encode 张量、canonical 动作映射逐位与 Python 参考实现对齐。
//! 复用 engine crate 下由 dump_fixtures.py 产出的同一份夹具。

use encoder::{encode, from_canonical_action, to_canonical_action};
use engine::{Color, GameState, Move, Position, INPUT_CHANNELS};
use serde_json::Value;

fn load_fixtures() -> Value {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../engine/tests/fixtures/engine.json"
    );
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("读取夹具失败 {path}: {e}（先跑 trainer/scripts/dump_fixtures.py）"));
    serde_json::from_str(&text).expect("解析夹具 JSON 失败")
}

fn move_from_json(v: &Value) -> Move {
    let a = v.as_array().unwrap();
    Move::new(
        a[0].as_i64().unwrap() as i32,
        a[1].as_i64().unwrap() as i32,
        a[2].as_i64().unwrap() as i32,
        a[3].as_i64().unwrap() as i32,
    )
}

fn color_from_str(s: &str) -> Color {
    match s {
        "red" => Color::Red,
        "black" => Color::Black,
        other => panic!("bad perspective {other}"),
    }
}

fn state_at_ply(game: &Value, ply: usize) -> GameState {
    let start_fen = game["start_fen"].as_str().unwrap();
    let moves = game["moves"].as_array().unwrap();
    let mut state = GameState::from_position(Position::from_fen(start_fen).unwrap());
    for mv in moves.iter().take(ply) {
        state = state.make_move(move_from_json(mv));
    }
    state
}

#[test]
fn encodings_match_python() {
    let fixtures = load_fixtures();
    let games = fixtures["games"].as_array().unwrap();
    let encodings = fixtures["encodings"].as_array().unwrap();
    let expected_len = INPUT_CHANNELS * 10 * 9;

    for case in encodings {
        let game_index = case["game_index"].as_u64().unwrap() as usize;
        let ply = case["ply"].as_u64().unwrap() as usize;
        let ctx = format!("encoding game={game_index} ply={ply}");

        let state = state_at_ply(&games[game_index], ply);
        let tensor = encode(&state);
        assert_eq!(tensor.len(), expected_len, "{ctx}: tensor length");

        // 从稀疏 nonzero 还原期望稠密张量，逐位对照。
        let mut expected = vec![0f32; expected_len];
        for pair in case["nonzero"].as_array().unwrap() {
            let p = pair.as_array().unwrap();
            let idx = p[0].as_u64().unwrap() as usize;
            let val = p[1].as_f64().unwrap() as f32;
            expected[idx] = val;
        }
        for (i, (a, e)) in tensor.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() < 1e-6,
                "{ctx}: tensor[{i}] = {a} != {e}"
            );
        }
    }
}

#[test]
fn canonical_actions_match_python() {
    let fixtures = load_fixtures();
    let cases = fixtures["action_cases"].as_array().unwrap();

    for case in cases {
        let mv = move_from_json(&case["move"]);
        let perspective = color_from_str(case["perspective"].as_str().unwrap());
        let expected_action = case["canonical_action"].as_u64().unwrap() as usize;
        let restored = move_from_json(&case["restored"]);

        let action = to_canonical_action(mv, perspective);
        assert_eq!(action, expected_action, "to_canonical_action {mv:?} {perspective:?}");

        let round_trip = from_canonical_action(action, perspective);
        assert_eq!(round_trip, restored, "from_canonical_action round trip");
        assert_eq!(round_trip, mv, "canonical action is an involution round trip");
    }
}
