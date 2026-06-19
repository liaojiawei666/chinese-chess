//! 差分测试：固定 mock 评估 + epsilon=0 下，MCTS 访问分布 N(s,a) 与 Python 逐局对照。
//! 同时覆盖子树复用：每手 run → 取最多访问走法 → advance → make_move，逐手比对。

use std::collections::BTreeMap;

use engine::{move_to_action_id, GameState, Move, Position};
use mcts::{Evaluator, Mcts, MctsConfig};
use rand::rngs::StdRng;
use rand::SeedableRng;
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

// 与 dump_fixtures.py 的 MockEvaluator 逐字节一致的确定性评估。
fn fnv1a_64(s: &str) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn mock_value(fen: &str) -> f64 {
    (fnv1a_64(&format!("{fen}#v")) % 20000) as f64 / 10000.0 - 1.0
}

fn mock_logit(fen: &str, m: Move) -> f64 {
    let key = format!("{fen}#m{},{},{},{}", m.sx, m.sy, m.tx, m.ty);
    (fnv1a_64(&key) % 20000) as f64 / 10000.0 - 1.0
}

struct MockEvaluator;

impl Evaluator for MockEvaluator {
    fn evaluate(&self, state: &GameState) -> (Vec<(Move, f64)>, f64) {
        let fen = state.position.to_fen();
        let value = mock_value(&fen);
        let moves = state.position.legal_moves();
        if moves.is_empty() {
            return (Vec::new(), value);
        }
        let logits: Vec<f64> = moves.iter().map(|&m| mock_logit(&fen, m)).collect();
        let mx = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = logits.iter().map(|l| (l - mx).exp()).collect();
        let total: f64 = exps.iter().sum();
        let priors = moves
            .iter()
            .cloned()
            .zip(exps.iter().map(|e| e / total))
            .collect();
        (priors, value)
    }
}

fn choose_move(visits: &[(Move, u32)]) -> Move {
    let mut best = visits[0].0;
    let mut best_n = -1i64;
    for &(mv, n) in visits {
        if n as i64 > best_n {
            best_n = n as i64;
            best = mv;
        }
    }
    best
}

#[test]
fn mcts_visits_match_python() {
    let fixtures = load_fixtures();
    let cases = fixtures["mcts_cases"].as_array().unwrap();
    assert!(!cases.is_empty(), "no mcts cases");

    for case in cases {
        let name = case["name"].as_str().unwrap();
        let start_fen = case["start_fen"].as_str().unwrap();
        let config = MctsConfig {
            n_simulations: case["n_simulations"].as_u64().unwrap() as u32,
            c_puct: case["c_puct"].as_f64().unwrap(),
            dirichlet_alpha: case["dirichlet_alpha"].as_f64().unwrap(),
            dirichlet_epsilon: case["dirichlet_epsilon"].as_f64().unwrap(),
        };

        let mut mcts = Mcts::new(MockEvaluator, config, StdRng::seed_from_u64(0));
        let mut game = GameState::from_position(Position::from_fen(start_fen).unwrap());

        let runs = case["runs"].as_array().unwrap();
        let chosen = case["chosen"].as_array().unwrap();

        for (ply, expected_run) in runs.iter().enumerate() {
            let visits = mcts.run(game.clone());

            let actual: BTreeMap<usize, u32> =
                visits.iter().map(|(m, n)| (move_to_action_id(*m), *n)).collect();
            let expected: BTreeMap<usize, u32> = expected_run
                .as_array()
                .unwrap()
                .iter()
                .map(|pair| {
                    let p = pair.as_array().unwrap();
                    (move_to_action_id(move_from_json(&p[0])), p[1].as_u64().unwrap() as u32)
                })
                .collect();
            assert_eq!(actual, expected, "{name} ply {ply}: visit distribution");

            let mv = choose_move(&visits);
            assert_eq!(
                move_to_action_id(mv),
                move_to_action_id(move_from_json(&chosen[ply])),
                "{name} ply {ply}: chosen move"
            );
            mcts.advance(mv);
            game = game.make_move(mv);
        }
    }
}
