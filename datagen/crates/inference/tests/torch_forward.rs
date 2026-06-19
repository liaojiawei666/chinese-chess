//! tch 前向差分测试：加载 trainer/scripts/dump_torch_fixture.py 导出的 model.pt，
//! 与 Python 同一模型的 (value, priors) 逐项对照。
//!
//! 仅在 `--features torch` 下编译，且需要：
//!   1. libtorch（版本与导出 model.pt 的 torch 一致，见 README）；
//!   2. 先跑 `python trainer/scripts/dump_torch_fixture.py` 生成
//!      data/fixtures/torch_forward/{model.pt,expected.json}。
//! 缺少 model.pt / expected.json 时自动跳过（视作未配置该可选环节）。

#![cfg(feature = "torch")]

use std::collections::BTreeMap;
use std::path::PathBuf;

use engine::{move_to_action_id, GameState, Move, Position};
use inference::torch_model::TorchModel;
use inference::LeafEvaluator;
use serde_json::Value;
use tch::Device;

fn fixture_dir() -> PathBuf {
    PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR")))
        .join("../../../data/fixtures/torch_forward")
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

fn state_at(start_fen: &str, moves: &[Value], ply: usize) -> GameState {
    let mut state = GameState::from_position(Position::from_fen(start_fen).unwrap());
    for mv in moves.iter().take(ply) {
        state = state.make_move(move_from_json(mv));
    }
    state
}

#[test]
fn torch_forward_matches_python() {
    let dir = fixture_dir();
    let model_path = dir.join("model.pt");
    let expected_path = dir.join("expected.json");
    if !model_path.exists() || !expected_path.exists() {
        eprintln!(
            "skip torch_forward_matches_python: 缺少 {} / expected.json（先跑 dump_torch_fixture.py）",
            model_path.display()
        );
        return;
    }

    let model = TorchModel::load(model_path.to_str().unwrap(), Device::Cpu)
        .expect("加载 model.pt 失败（检查 libtorch 版本是否与导出 torch 匹配）");
    let fixtures: Value =
        serde_json::from_str(&std::fs::read_to_string(&expected_path).unwrap()).unwrap();

    for case in fixtures["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let start_fen = case["start_fen"].as_str().unwrap();
        let moves = case["moves"].as_array().unwrap();
        let ply = case["ply"].as_u64().unwrap() as usize;
        let state = state_at(start_fen, moves, ply);

        let (priors, value) = model.evaluate(&state);
        let expected_value = case["value"].as_f64().unwrap() as f32;
        assert!(
            (value - expected_value).abs() < 1e-4,
            "{name}: value {value} != {expected_value}"
        );

        let actual: BTreeMap<usize, f32> =
            priors.iter().map(|(m, p)| (move_to_action_id(*m), *p)).collect();
        let expected = case["priors"].as_array().unwrap();
        assert_eq!(actual.len(), expected.len(), "{name}: prior count");
        for entry in expected {
            let e = entry.as_array().unwrap();
            let mv = move_from_json(&e[0]);
            let prob = e[1].as_f64().unwrap() as f32;
            let got = actual[&move_to_action_id(mv)];
            assert!((got - prob).abs() < 1e-4, "{name}: prior {mv:?} {got} != {prob}");
        }
    }
}
