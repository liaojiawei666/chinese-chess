use std::{future::Future, pin::Pin};

use engine::{encode::encode_state, evaluate::EvaluatorOutput, game::GameState, ACTION_SPACE_SIZE};
use tokio::sync::{mpsc, oneshot};

/// 从 self-play task 发往推理服务器的评估请求。
pub struct EvalRequest {
    /// 已编码的局面张量，形状 [INPUT_CHANNELS * BOARD_HEIGHT * BOARD_WIDTH]
    pub state: Vec<f32>,
    /// 用于回传推理结果的一次性通道
    pub response_tx: oneshot::Sender<EvalResponse>,
}

/// 推理服务器返回的评估结果。
pub struct EvalResponse {
    pub value: f32,
    pub policy_logits: [f32; ACTION_SPACE_SIZE],
}

/// 基于 tokio channel 的 Evaluator 实现。
/// 每次 evaluate_async 调用会：
///   1. encode_state → 编码局面
///   2. 通过 mpsc 发送 EvalRequest 到推理服务器
///   3. await oneshot 接收结果
pub struct ChannelEvaluator {
    request_tx: mpsc::Sender<EvalRequest>,
}

impl ChannelEvaluator {
    pub fn new(request_tx: mpsc::Sender<EvalRequest>) -> Self {
        Self { request_tx }
    }
}

impl engine::evaluate::Evaluator for ChannelEvaluator {
    fn evaluate_async(
        &self,
        game: &GameState,
    ) -> Pin<Box<dyn Future<Output = EvaluatorOutput> + Send>> {
        let state = encode_state(game).to_float_tensor();
        let tx = self.request_tx.clone();

        Box::pin(async move {
            let (resp_tx, resp_rx) = oneshot::channel();
            tx.send(EvalRequest {
                state,
                response_tx: resp_tx,
            })
            .await
            .expect("inference server dropped");

            let resp = resp_rx.await.expect("inference server dropped response");
            EvaluatorOutput {
                value: resp.value,
                policy_logits: resp.policy_logits,
            }
        })
    }
}
