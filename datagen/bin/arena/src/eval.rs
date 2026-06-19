//! MCTS 叶子评估器适配：均匀评估器（无 torch，开局生成与自检用）+ 单模型 torch 评估器。
//!
//! 这里把评估器实现到 `&T` 上（而非 `T`），这样模型在对局循环外构建一次、各局借用即可，
//! 不必每局重新 load .pt（颜色互换的第二局也复用同两份模型）。

use engine::{GameState, Move};
use mcts::Evaluator;

/// 均匀先验 + value 0：不依赖任何训练参数，确定性。
pub struct UniformEvaluator;

impl Evaluator for UniformEvaluator {
    fn evaluate(&self, state: &GameState) -> (Vec<(Move, f64)>, f64) {
        let moves = state.position.legal_moves();
        let n = moves.len();
        let p = if n > 0 { 1.0 / n as f64 } else { 0.0 };
        (moves.into_iter().map(|m| (m, p)).collect(), 0.0)
    }
}

impl Evaluator for &UniformEvaluator {
    fn evaluate(&self, state: &GameState) -> (Vec<(Move, f64)>, f64) {
        (**self).evaluate(state)
    }
}

/// 加载好的单个 TorchScript 模型评估器。
#[cfg(feature = "torch")]
pub struct TorchEvaluator {
    model: inference::torch_model::TorchModel,
}

#[cfg(feature = "torch")]
impl TorchEvaluator {
    pub fn load(path: &str, device: &str) -> anyhow::Result<Self> {
        let model = inference::torch_model::TorchModel::load_str(path, device)
            .map_err(|e| anyhow::anyhow!("加载模型失败 {path}：{e}"))?;
        Ok(TorchEvaluator { model })
    }
}

#[cfg(feature = "torch")]
impl Evaluator for &TorchEvaluator {
    fn evaluate(&self, state: &GameState) -> (Vec<(Move, f64)>, f64) {
        use inference::LeafEvaluator;
        let (priors, value) = self.model.evaluate(state);
        (
            priors.into_iter().map(|(m, p)| (m, p as f64)).collect(),
            value as f64,
        )
    }
}
