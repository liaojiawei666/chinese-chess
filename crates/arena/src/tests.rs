//! 无 torch 的管线自测：开局生成确定性/去重 + 均匀自对杀对称且可复现。

use std::collections::HashSet;

use cc_core::mcts::MctsConfig;

use crate::eval::UniformEvaluator;
use crate::match_play::run_match;
use crate::openings::generate;

fn cfg() -> MctsConfig {
    MctsConfig {
        n_simulations: 8,
        c_puct: 1.5,
        dirichlet_alpha: 0.3,
        dirichlet_epsilon: 0.0,
        collect_batch_size: 1,
    }
}

#[test]
fn openings_are_deterministic_and_unique() {
    let a = generate(5, 4, 1.2, 8, 42);
    let b = generate(5, 4, 1.2, 8, 42);
    assert_eq!(a, b, "同种子应逐位可复现");
    assert!(a.len() == 5, "应生成 5 个开局");

    let mut seen = HashSet::new();
    for line in &a {
        assert!(seen.insert(line.clone()), "开局应互不相同");
    }
}

#[test]
fn uniform_self_match_is_symmetric_and_reproducible() {
    let openings = generate(4, 4, 1.2, 8, 7);
    let ea = UniformEvaluator;
    let eb = UniformEvaluator;

    let r1 = run_match(&openings, &ea, &eb, cfg(), 40);
    // A=B：红黑互换后逐局抵消 → 胜负相等、得分率恰好 0.5。
    assert_eq!(r1.wins_a, r1.losses_a);
    assert!((r1.score_a - 0.5).abs() < 1e-9);
    assert_eq!(r1.games, openings.len() as u32 * 2);

    let r2 = run_match(&openings, &ea, &eb, cfg(), 40);
    assert_eq!(
        (r1.wins_a, r1.draws, r1.losses_a),
        (r2.wins_a, r2.draws, r2.losses_a),
        "确定性：重跑结果应一致"
    );
}
