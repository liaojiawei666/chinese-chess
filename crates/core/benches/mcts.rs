use cc_core::engine::{GameState, Move, Position};
use cc_core::mcts::{Evaluator, Mcts, MctsConfig, StepResult};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::rngs::StdRng;
use rand::SeedableRng;

struct Uniform;

impl Evaluator for Uniform {
    fn evaluate(&self, state: &GameState) -> (Vec<(Move, f64)>, f64) {
        let moves = state.position.legal_moves();
        let n = moves.len();
        let p = if n > 0 { 1.0 / n as f64 } else { 0.0 };
        (moves.into_iter().map(|m| (m, p)).collect(), 0.0)
    }
}

fn run_step_cycle(n_simulations: u32) {
    let config = MctsConfig {
        n_simulations,
        c_puct: 1.5,
        dirichlet_alpha: 0.3,
        dirichlet_epsilon: 0.0,
        collect_batch_size: 1,
    };
    let state = GameState::from_position(Position::starting());
    let mut mcts = Mcts::new(config, StdRng::seed_from_u64(42));
    let uniform = Uniform;

    if mcts.init_root(state.clone()) {
        let (priors, value) = uniform.evaluate(mcts.root_state());
        mcts.feed_root_eval(priors, value);
    }

    while mcts.simulations_done() < n_simulations {
        match mcts.step() {
            StepResult::NeedEval { state } => {
                let (priors, value) = uniform.evaluate(state);
                mcts.feed_eval(priors, value);
            }
            StepResult::Done => {}
        }
    }
    black_box(mcts.visit_counts());
}

fn bench_step_cycle(c: &mut Criterion) {
    c.bench_function("mcts_step_cycle_64", |b| b.iter(|| run_step_cycle(64)));
}

criterion_group!(mcts_benches, bench_step_cycle);
criterion_main!(mcts_benches);
