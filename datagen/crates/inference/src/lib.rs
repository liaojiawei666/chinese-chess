//! 叶子评估（移植自 trainer/src/trainer/reference/evaluator.py）。
//!
//! 纯逻辑部分（按合法动作的 canonical id 取 logit → 数值稳定 softmax → 还原 Move）
//! 不依赖 libtorch，可直接单测并与 Python `priors_from_logits` 对齐。tch-rs 加载
//! model.pt 前向部分在 `torch` 特性后接入（torch_model.rs）。

use engine::{GameState, Move};

/// MCTS 叶子评估接口：局面进，(合法走法先验, value) 出。
/// value 为当前走棋方视角的胜负倾向，范围 [-1, 1]。
pub trait LeafEvaluator {
    fn evaluate(&self, state: &GameState) -> (Vec<(Move, f32)>, f32);
}

/// 在合法动作子集上做数值稳定 softmax（f64 累加），返回与 `legal_ids` 同序的概率。
/// 对应 Python `priors_from_logits` 的核心：取 logits[legal_ids]，减最大值后 exp 归一化。
pub fn mask_softmax(policy_logits: &[f32], legal_ids: &[usize]) -> Vec<f32> {
    if legal_ids.is_empty() {
        return Vec::new();
    }
    let mut logits: Vec<f64> = legal_ids.iter().map(|&id| policy_logits[id] as f64).collect();
    let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut sum = 0.0;
    for v in logits.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    logits.iter().map(|&v| (v / sum) as f32).collect()
}

/// 把 canonical 动作空间的 policy logits 映射成「真实 Move → 先验概率」。
/// 与 Python `priors_from_logits` 一一对应：合法走法为空时返回空。
pub fn priors_from_logits(policy_logits: &[f32], state: &GameState) -> Vec<(Move, f32)> {
    let moves = state.position.legal_moves();
    if moves.is_empty() {
        return Vec::new();
    }
    let perspective = state.position.side_to_move;
    let legal_ids: Vec<usize> = moves
        .iter()
        .map(|&mv| encoder::to_canonical_action(mv, perspective))
        .collect();
    let probs = mask_softmax(policy_logits, &legal_ids);
    moves.into_iter().zip(probs).collect()
}

#[cfg(feature = "torch")]
pub mod torch_model;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_logits_give_uniform_probs() {
        let logits = vec![0.0f32; 8100];
        let probs = mask_softmax(&logits, &[1, 5, 9]);
        assert_eq!(probs.len(), 3);
        for p in probs {
            assert!((p - 1.0 / 3.0).abs() < 1e-6);
        }
    }

    #[test]
    fn empty_legal_returns_empty() {
        assert!(mask_softmax(&[1.0, 2.0], &[]).is_empty());
    }

    #[test]
    fn probs_sum_to_one() {
        let logits: Vec<f32> = (0..20).map(|i| i as f32 * 0.1).collect();
        let probs = mask_softmax(&logits, &[0, 3, 7, 19]);
        let s: f32 = probs.iter().sum();
        assert!((s - 1.0).abs() < 1e-6);
    }
}
