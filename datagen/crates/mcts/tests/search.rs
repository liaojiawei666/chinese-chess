//! mcts 原生单测：固定（均匀）评估下访问分布的结构性质，不依赖网络/夹具。

use engine::{GameState, Move, Position};
use mcts::{Evaluator, Mcts, MctsConfig};
use rand::rngs::StdRng;
use rand::SeedableRng;

/// 均匀先验 + value 0 的确定性评估器，用来单独验证搜索骨架。
struct Uniform;

impl Evaluator for Uniform {
    fn evaluate(&self, state: &GameState) -> (Vec<(Move, f64)>, f64) {
        let moves = state.position.legal_moves();
        let n = moves.len();
        let p = if n > 0 { 1.0 / n as f64 } else { 0.0 };
        (moves.into_iter().map(|m| (m, p)).collect(), 0.0)
    }
}

fn config(n_simulations: u32) -> MctsConfig {
    MctsConfig {
        n_simulations,
        c_puct: 1.5,
        dirichlet_alpha: 0.3,
        dirichlet_epsilon: 0.0,
    }
}

#[test]
fn visit_counts_sum_to_simulations() {
    let mut mcts = Mcts::new(Uniform, config(64), StdRng::seed_from_u64(0));
    let counts = mcts.run(GameState::from_position(Position::starting()));
    let total: u32 = counts.iter().map(|(_, n)| *n).sum();
    assert_eq!(total, 64);
}

#[test]
fn returns_one_entry_per_legal_move_and_all_legal() {
    let state = GameState::from_position(Position::starting());
    let legal = state.position.legal_moves();
    let mut mcts = Mcts::new(Uniform, config(32), StdRng::seed_from_u64(1));
    let counts = mcts.run(state);

    assert_eq!(counts.len(), legal.len());
    for (mv, _) in &counts {
        assert!(legal.contains(mv));
    }
}

#[test]
fn terminal_root_returns_empty() {
    let state =
        GameState::from_position(Position::from_fen("3k5/4R4/3R5/9/9/9/9/9/9/4K4 b").unwrap());
    let mut mcts = Mcts::new(Uniform, config(16), StdRng::seed_from_u64(2));
    assert!(mcts.run(state).is_empty());
}
