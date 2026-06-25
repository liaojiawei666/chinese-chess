use std::{future::Future, pin::Pin};

use crate::{game::GameState, ACTION_SPACE_SIZE};

pub struct EvaluatorOutput {
    pub value: f32,
    /// 策略头原始 logits（未经 softmax），由 MCTS 在合法走法上做 masked softmax。
    pub policy_logits: [f32; ACTION_SPACE_SIZE],
}

pub trait Evaluator: Send {
    fn evaluate_async(
        &self,
        game: &GameState,
    ) -> Pin<Box<dyn Future<Output = EvaluatorOutput> + Send>>;
}

/// Uniform evaluator for testing: value = 0, all logits = 0 (uniform prior after
/// masked softmax in MCTS `create_edges`).
pub struct UniformEvaluator;

impl Evaluator for UniformEvaluator {
    fn evaluate_async(
        &self,
        _game: &GameState,
    ) -> Pin<Box<dyn Future<Output = EvaluatorOutput> + Send>> {
        Box::pin(async move {
            EvaluatorOutput {
                value: 0.0,
                policy_logits: [0.0f32; ACTION_SPACE_SIZE],
            }
        })
    }
}
