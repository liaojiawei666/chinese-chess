use std::time::{Duration, Instant};

use cc_core::encode::{encode, legal_mask};
use cc_core::engine::{GameState, Move, Position};
use cc_core::mcts::{Evaluator, Mcts, MctsConfig, StepResult};
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

fn main() {
    let n_simulations: u32 = 128;
    let n_rounds = 500;

    println!("=== MCTS 性能分解 (n_simulations={n_simulations}, rounds={n_rounds}) ===\n");

    // 1) legal_moves 单次耗时
    {
        let pos = Position::starting();
        let start = Instant::now();
        for _ in 0..10_000 {
            std::hint::black_box(pos.legal_moves());
        }
        let elapsed = start.elapsed();
        println!("legal_moves (starting):  {:>8.2} µs/call", elapsed.as_nanos() as f64 / 10_000.0 / 1000.0);
    }

    // 2) make_move + status
    {
        let state = GameState::from_position(Position::starting());
        let moves = state.position.legal_moves();
        let mv = moves[0];
        let start = Instant::now();
        for _ in 0..10_000 {
            let s = state.clone();
            let s2 = s.make_move(mv);
            std::hint::black_box(s2.status());
        }
        let elapsed = start.elapsed();
        println!("make_move + status:      {:>8.2} µs/call", elapsed.as_nanos() as f64 / 10_000.0 / 1000.0);
    }

    // 3) encode 耗时
    {
        let state = GameState::from_position(Position::starting());
        let start = Instant::now();
        for _ in 0..10_000 {
            std::hint::black_box(encode(&state));
        }
        let elapsed = start.elapsed();
        println!("encode (starting):       {:>8.2} µs/call", elapsed.as_nanos() as f64 / 10_000.0 / 1000.0);
    }

    // 4) legal_mask 耗时
    {
        let pos = Position::starting();
        let start = Instant::now();
        for _ in 0..10_000 {
            std::hint::black_box(legal_mask(&pos));
        }
        let elapsed = start.elapsed();
        println!("legal_mask:              {:>8.2} µs/call", elapsed.as_nanos() as f64 / 10_000.0 / 1000.0);
    }

    println!();

    // 5) MCTS step-by-step 细分
    let config = MctsConfig {
        n_simulations,
        c_puct: 1.5,
        dirichlet_alpha: 0.3,
        dirichlet_epsilon: 0.0,
        collect_batch_size: 1,
    };
    let uniform = Uniform;

    let mut t_init = Duration::ZERO;
    let mut t_step = Duration::ZERO;
    let mut t_eval = Duration::ZERO;
    let mut t_feed = Duration::ZERO;
    let mut t_visit = Duration::ZERO;
    let mut total_evals = 0u64;
    let mut total_terminals = 0u64;

    let total_start = Instant::now();

    for round in 0..n_rounds {
        let state = GameState::from_position(Position::starting());
        let mut mcts = Mcts::new(config, StdRng::seed_from_u64(42 + round as u64));

        // init_root + feed_root_eval
        let t0 = Instant::now();
        let need = mcts.init_root(state.clone());
        if need {
            let (priors, value) = uniform.evaluate(mcts.root_state());
            mcts.feed_root_eval(priors, value);
        }
        t_init += t0.elapsed();

        // step loop
        while mcts.simulations_done() < n_simulations {
            let t1 = Instant::now();
            let result = mcts.step();
            t_step += t1.elapsed();

            match result {
                StepResult::NeedEval { state } => {
                    let t2 = Instant::now();
                    let (priors, value) = uniform.evaluate(state);
                    t_eval += t2.elapsed();

                    let t3 = Instant::now();
                    mcts.feed_eval(priors, value);
                    t_feed += t3.elapsed();

                    total_evals += 1;
                }
                StepResult::Done => {
                    total_terminals += 1;
                }
            }
        }

        let t4 = Instant::now();
        std::hint::black_box(mcts.visit_counts());
        t_visit += t4.elapsed();
    }

    let total_elapsed = total_start.elapsed();
    let n = n_rounds as f64;
    let per_round_us = total_elapsed.as_micros() as f64 / n;

    println!("--- 每局 MCTS ({n_simulations} sims) 平均耗时分解 ---\n");
    println!("  init_root + eval:  {:>8.1} µs  ({:>5.1}%)", t_init.as_micros() as f64 / n, t_init.as_secs_f64() / total_elapsed.as_secs_f64() * 100.0);
    println!("  step (select):     {:>8.1} µs  ({:>5.1}%)", t_step.as_micros() as f64 / n, t_step.as_secs_f64() / total_elapsed.as_secs_f64() * 100.0);
    println!("  evaluate (CPU):    {:>8.1} µs  ({:>5.1}%)", t_eval.as_micros() as f64 / n, t_eval.as_secs_f64() / total_elapsed.as_secs_f64() * 100.0);
    println!("  feed_eval (expand):  {:>6.1} µs  ({:>5.1}%)", t_feed.as_micros() as f64 / n, t_feed.as_secs_f64() / total_elapsed.as_secs_f64() * 100.0);
    println!("  visit_counts:      {:>8.1} µs  ({:>5.1}%)", t_visit.as_micros() as f64 / n, t_visit.as_secs_f64() / total_elapsed.as_secs_f64() * 100.0);
    println!("  ────────────────────────────────────");
    println!("  TOTAL:             {:>8.1} µs/round", per_round_us);
    println!();
    println!("  avg evals/round:   {:>6.1}", total_evals as f64 / n);
    println!("  avg terminals/round: {:>4.1}", total_terminals as f64 / n);
    println!("  eval latency:      {:>8.2} µs/eval", t_eval.as_nanos() as f64 / total_evals as f64 / 1000.0);

    println!();

    // 6) evaluate 细分：legal_moves vs 其余
    {
        let state = GameState::from_position(Position::starting());
        let n_iter = 10_000;

        let mut t_lm = Duration::ZERO;
        let mut t_rest = Duration::ZERO;

        for _ in 0..n_iter {
            let t0 = Instant::now();
            let moves = state.position.legal_moves();
            t_lm += t0.elapsed();

            let t1 = Instant::now();
            let n = moves.len();
            let p = if n > 0 { 1.0 / n as f64 } else { 0.0 };
            let result: Vec<(Move, f64)> = moves.into_iter().map(|m| (m, p)).collect();
            std::hint::black_box(result);
            t_rest += t1.elapsed();
        }

        println!("--- Evaluator::evaluate 内部分解 (starting) ---");
        println!("  legal_moves:  {:>8.2} µs ({:.1}%)",
            t_lm.as_nanos() as f64 / n_iter as f64 / 1000.0,
            t_lm.as_secs_f64() / (t_lm + t_rest).as_secs_f64() * 100.0);
        println!("  policy build: {:>8.2} µs ({:.1}%)",
            t_rest.as_nanos() as f64 / n_iter as f64 / 1000.0,
            t_rest.as_secs_f64() / (t_lm + t_rest).as_secs_f64() * 100.0);
    }

    println!();

    // 7) step() 内部粗估：在 step 中 select 路径遍历 vs make_move 创建子节点
    println!("--- MCTS step() 内部估算 ---");
    println!("  step 包含：PUCT select 遍历 + make_move 创建子节点 + status 检测");
    let make_move_us = 28.0; // from criterion bench
    let legal_moves_us = 24.0; // from criterion bench
    let step_per_round = t_step.as_micros() as f64 / n;
    let sims = n_simulations as f64;
    println!("  step 每模拟: {:>6.1} µs", step_per_round / sims);
    println!("  其中 make_move+status ~{make_move_us}+{legal_moves_us}={:.0} µs (占比估 {:>4.1}%)",
        make_move_us + legal_moves_us,
        (make_move_us + legal_moves_us) / (step_per_round / sims) * 100.0);

    println!();
    println!("=== 结论 ===");
    let eval_pct = t_eval.as_secs_f64() / total_elapsed.as_secs_f64() * 100.0;
    let step_pct = t_step.as_secs_f64() / total_elapsed.as_secs_f64() * 100.0;
    println!("  evaluate (含 legal_moves) 占 {eval_pct:.1}%");
    println!("  step (select+make_move+status) 占 {step_pct:.1}%");
    println!("  → legal_moves + make_move + status 是 CPU 端绝对热点");
    println!("  → GPU 推理将替代 Uniform evaluate，CPU 端热点不变");
}
