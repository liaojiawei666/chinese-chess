//! 差分测试：priors_from_logits（合法掩码 + canonical id + f64 softmax）与 Python 对齐。
//! 不需要 libtorch：从夹具里重建稀疏 logits，比较先验分布。
//! tch 加载 model.pt 的前向对照见 tests/torch_forward.rs（需 --features torch + libtorch）。

use std::collections::BTreeMap;

use engine::{move_to_action_id, GameState, Move, Position, ACTION_SPACE_SIZE};
use inference::priors_from_logits;
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
fn priors_match_python() {
    let fixtures = load_fixtures();
    let games = fixtures["games"].as_array().unwrap();
    let cases = fixtures["infer_cases"].as_array().unwrap();
    assert!(!cases.is_empty(), "no infer cases in fixtures");

    for case in cases {
        let game_index = case["game_index"].as_u64().unwrap() as usize;
        let ply = case["ply"].as_u64().unwrap() as usize;
        let ctx = format!("infer game={game_index} ply={ply}");
        let state = state_at_ply(&games[game_index], ply);

        // 重建稀疏 logits。
        let mut logits = vec![0f32; ACTION_SPACE_SIZE];
        for pair in case["logits_sparse"].as_array().unwrap() {
            let p = pair.as_array().unwrap();
            let id = p[0].as_u64().unwrap() as usize;
            logits[id] = p[1].as_f64().unwrap() as f32;
        }

        let actual = priors_from_logits(&logits, &state);
        let actual_map: BTreeMap<usize, f32> = actual
            .iter()
            .map(|(m, p)| (move_to_action_id(*m), *p))
            .collect();

        let expected = case["priors"].as_array().unwrap();
        assert_eq!(actual_map.len(), expected.len(), "{ctx}: prior count");

        let mut prob_sum = 0.0f32;
        for entry in expected {
            let e = entry.as_array().unwrap();
            let mv = move_from_json(&e[0]);
            let prob = e[1].as_f64().unwrap() as f32;
            let key = move_to_action_id(mv);
            let got = actual_map
                .get(&key)
                .unwrap_or_else(|| panic!("{ctx}: missing prior for {mv:?}"));
            assert!((got - prob).abs() < 1e-6, "{ctx}: prior {mv:?} {got} != {prob}");
            prob_sum += prob;
        }
        assert!((prob_sum - 1.0).abs() < 1e-5, "{ctx}: priors sum {prob_sum}");
    }
}
