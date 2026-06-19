//! 差分测试：加载 trainer/scripts/dump_fixtures.py 产出的夹具，逐位断言 Rust engine
//! 与 Python 参考引擎行为一致（legal_moves / in_check / status / make_move 链）。

use std::collections::BTreeSet;

use engine::{Color, GameState, GameStatus, Move, Position};
use serde_json::Value;

fn load_fixtures() -> Value {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/engine.json");
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("读取夹具失败 {path}: {e}（先跑 trainer/scripts/dump_fixtures.py）"));
    serde_json::from_str(&text).expect("解析夹具 JSON 失败")
}

fn move_from_json(v: &Value) -> Move {
    let a = v.as_array().expect("move is array");
    Move::new(
        a[0].as_i64().unwrap() as i32,
        a[1].as_i64().unwrap() as i32,
        a[2].as_i64().unwrap() as i32,
        a[3].as_i64().unwrap() as i32,
    )
}

fn move_key(m: Move) -> (i32, i32, i32, i32) {
    (m.sx, m.sy, m.tx, m.ty)
}

fn color_str(c: Option<Color>) -> Option<&'static str> {
    c.map(|c| c.as_str())
}

/// 把 Rust GameStatus 转成与 Python status_dict 相同的三元组并和期望对照。
fn assert_status_eq(actual: GameStatus, expected: &Value, ctx: &str) {
    let exp_terminal = expected["is_terminal"].as_bool().unwrap();
    let exp_reason = expected["reason"].as_str();
    let exp_winner = expected["winner"].as_str();

    assert_eq!(actual.is_terminal, exp_terminal, "{ctx}: is_terminal");
    let act_reason = actual.reason.map(|r| r.as_str());
    assert_eq!(act_reason, exp_reason, "{ctx}: reason");
    assert_eq!(color_str(actual.winner), exp_winner, "{ctx}: winner");
}

#[test]
fn positions_match_python() {
    let fixtures = load_fixtures();
    let positions = fixtures["positions"].as_array().expect("positions array");

    for (i, case) in positions.iter().enumerate() {
        let fen = case["fen"].as_str().unwrap();
        let pos = Position::from_fen(fen).unwrap_or_else(|e| panic!("from_fen({fen}): {e}"));
        let ctx = format!("position[{i}] fen={fen}");

        // FEN 往返
        assert_eq!(pos.to_fen(), case["to_fen"].as_str().unwrap(), "{ctx}: to_fen");

        // 行棋方
        assert_eq!(
            pos.side_to_move.fen_side().to_string(),
            // side_to_move 在夹具里是 "red"/"black"，这里用首字母对照不可靠，改用 as_str。
            match case["side_to_move"].as_str().unwrap() {
                "red" => "r",
                "black" => "b",
                other => panic!("bad side {other}"),
            },
            "{ctx}: side_to_move"
        );

        // legal_moves（按集合对照，避免顺序脆弱）
        let expected_moves: BTreeSet<(i32, i32, i32, i32)> = case["legal_moves"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| move_key(move_from_json(m)))
            .collect();
        let actual_moves: BTreeSet<(i32, i32, i32, i32)> =
            pos.legal_moves().into_iter().map(move_key).collect();
        assert_eq!(actual_moves, expected_moves, "{ctx}: legal_moves");

        // in_check（kings 照面时夹具为 null，跳过）
        if let Some(expected_in_check) = case["in_check"].as_bool() {
            assert_eq!(
                pos.is_in_check(pos.side_to_move),
                expected_in_check,
                "{ctx}: in_check"
            );
        }

        // Position 级 status
        assert_status_eq(pos.status(), &case["status"], &format!("{ctx}: status"));
    }
}

#[test]
fn games_match_python() {
    let fixtures = load_fixtures();
    let games = fixtures["games"].as_array().expect("games array");

    for game_case in games {
        let name = game_case["name"].as_str().unwrap();
        let start_fen = game_case["start_fen"].as_str().unwrap();
        let moves = game_case["moves"].as_array().unwrap();
        let statuses = game_case["statuses"].as_array().unwrap();
        assert_eq!(moves.len(), statuses.len(), "game {name}: moves/statuses len");

        let mut game = GameState::from_position(
            Position::from_fen(start_fen).unwrap_or_else(|e| panic!("from_fen({start_fen}): {e}")),
        );
        for (i, (mv, expected_status)) in moves.iter().zip(statuses.iter()).enumerate() {
            game = game.make_move(move_from_json(mv));
            assert_status_eq(
                game.status(),
                expected_status,
                &format!("game {name} after move {i}"),
            );
        }

        assert_eq!(
            game.position.to_fen(),
            game_case["final_fen"].as_str().unwrap(),
            "game {name}: final_fen"
        );
        assert_status_eq(
            game.status(),
            &game_case["final_status"],
            &format!("game {name}: final_status"),
        );
    }
}
