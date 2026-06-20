//! MCTS 叶子评估器适配：均匀评估器（无模型/自检）+ 单模型 torch 评估器。

use cc_core::engine::{GameState, Move};
use cc_core::mcts::Evaluator;

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

/// 加载好的单个 TorchScript 模型评估器。
pub struct TorchEvaluator {
    model: cc_core::infer::torch_model::TorchModel,
}

impl TorchEvaluator {
    pub fn load(path: &str, device: &str) -> anyhow::Result<Self> {
        let model = cc_core::infer::torch_model::TorchModel::load_str(path, device)
            .map_err(|e| anyhow::anyhow!("加载模型失败 {path}：{e}"))?;
        Ok(TorchEvaluator { model })
    }
}

impl Evaluator for TorchEvaluator {
    fn evaluate(&self, state: &GameState) -> (Vec<(Move, f64)>, f64) {
        use cc_core::infer::LeafEvaluator;
        let (priors, value) = self.model.evaluate(state);
        (
            priors.into_iter().map(|(m, p)| (m, p as f64)).collect(),
            value as f64,
        )
    }
}
