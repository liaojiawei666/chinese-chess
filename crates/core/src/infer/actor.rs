//! 跨 worker 批量推理（scheme A）。
//!
//! 旧实现里每个 worker 各持一份模型、逐叶同步前向（batch=1），GPU 利用率极低。
//! 这里改成「单推理 actor」模式：
//!   - 一个独立线程独占模型；
//!   - 所有 worker 的 MCTS 叶子请求经 MPSC channel 汇聚到 actor；
//!   - actor 凑够 `eval_batch_size`（或超时 `inference_timeout_ms`）后一次性批量前向，
//!     再 scatter 回各请求自带的回执 channel；
//!   - 模型版本走共享 `AtomicI64`，actor 负责按版本变化热加载，worker 只读它给分片命名。
//!
//!   - worker 一轮多局 eval 用 `InferRequest::WorkerBatch` 一次发送、一次 recv，
//!     避免逐局独立回执 channel 在 barrier 下与 actor 互相等待；
//!   - 单叶 `InferRequest::One` 供 arena / BatchedEvaluator。

use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::encode::{encode, to_canonical_action};
use crate::engine::{GameState, Move};
use crate::mcts::Evaluator;
use crate::model_io::ModelStore;

/// 一次叶子评估的返回：(按 legal_moves() 顺序的先验, 当前走子方视角 value)。
pub type EvalReply = (Vec<(Move, f64)>, f64);

/// 一个叶子的评估输入。编码与合法走法都在 worker 线程算好（天然并行），
/// actor 只负责把 `input` 堆批、前向、再用 `legal_ids`/`moves` 还原成先验。
pub struct EvalInput {
    /// 展平的网络输入（99*10*9）。
    pub input: Vec<f32>,
    /// 该局面的合法走法（与 `legal_ids` 同序）。
    pub moves: Vec<Move>,
    /// 各合法走法的 canonical 动作 id（用于在 8100 维 logits 上取值）。
    pub legal_ids: Vec<usize>,
}

/// worker → actor 的一条请求：输入 + 回执发送端。
pub struct EvalRequest {
    pub input: EvalInput,
    pub reply: Sender<EvalReply>,
}

/// worker → actor 的消息：单叶或同一 worker 一轮多局打包。
pub enum InferRequest {
    /// 单叶请求（arena / BatchedEvaluator 等）。
    One(EvalRequest),
    /// 同一 worker 一轮内多局 eval 打成一包；前向完成后一次性回传 Vec（顺序与 inputs 一致）。
    WorkerBatch {
        inputs: Vec<EvalInput>,
        reply: Sender<Vec<EvalReply>>,
    },
}

struct BatchReplyGroup {
    reply: Sender<Vec<EvalReply>>,
    start_idx: usize,
    len: usize,
}

fn extend_infer_msg(
    msg: InferRequest,
    inputs: &mut Vec<EvalInput>,
    ones: &mut Vec<(usize, Sender<EvalReply>)>,
    batches: &mut Vec<BatchReplyGroup>,
) {
    match msg {
        InferRequest::One(r) => {
            let idx = inputs.len();
            inputs.push(r.input);
            ones.push((idx, r.reply));
        }
        InferRequest::WorkerBatch {
            inputs: batch,
            reply,
        } => {
            let start_idx = inputs.len();
            let len = batch.len();
            inputs.extend(batch);
            batches.push(BatchReplyGroup {
                reply,
                start_idx,
                len,
            });
        }
    }
}

fn scatter_results(
    results: Vec<EvalReply>,
    ones: Vec<(usize, Sender<EvalReply>)>,
    batches: Vec<BatchReplyGroup>,
) {
    for (idx, tx) in ones {
        if let Some(res) = results.get(idx) {
            let _ = tx.send(res.clone());
        }
    }
    for g in batches {
        let slice: Vec<EvalReply> = results[g.start_idx..g.start_idx + g.len].to_vec();
        let _ = g.reply.send(slice);
    }
}

/// 同一 worker 一轮 eval：一次 InferRequest，一次 recv 收回全部结果。
pub fn send_worker_batch(
    tx: &Sender<InferRequest>,
    inputs: Vec<EvalInput>,
) -> mpsc::Receiver<Vec<EvalReply>> {
    let (reply_tx, reply_rx) = mpsc::channel();
    tx.send(InferRequest::WorkerBatch {
        inputs,
        reply: reply_tx,
    })
    .expect("推理 actor 已退出");
    reply_rx
}

/// 批量模型：一次评估一批叶子（顺序与输入一致）。
///
/// 模型在 actor 线程内构建并独占使用、不跨线程移动，故无需 `Send`
/// （也就不依赖 `tch::CModule: Send`）。
pub trait BatchModel {
    fn evaluate_batch(&mut self, inputs: &[EvalInput]) -> Vec<EvalReply>;
}

// ---- 均匀先验（无模型 / 加载失败时的运行时回退） ----

pub struct UniformBatch;

impl BatchModel for UniformBatch {
    fn evaluate_batch(&mut self, inputs: &[EvalInput]) -> Vec<EvalReply> {
        inputs
            .iter()
            .map(|inp| {
                let n = inp.moves.len();
                if n == 0 {
                    return (Vec::new(), 0.0);
                }
                let p = 1.0 / n as f64;
                (inp.moves.iter().map(|&m| (m, p)).collect(), 0.0)
            })
            .collect()
    }
}

// ---- torch 真实网络批量前向 ----

pub struct TorchBatch {
    model: super::torch_model::TorchModel,
}

impl BatchModel for TorchBatch {
    fn evaluate_batch(&mut self, inputs: &[EvalInput]) -> Vec<EvalReply> {
        let raw: Vec<&[f32]> = inputs.iter().map(|i| i.input.as_slice()).collect();
        let outs = self.model.forward_batch(&raw);
        inputs
            .iter()
            .zip(outs)
            .map(|(inp, (logits, value))| {
                let probs = super::mask_softmax(&logits, &inp.legal_ids);
                let priors = inp
                    .moves
                    .iter()
                    .cloned()
                    .zip(probs.into_iter().map(|p| p as f64))
                    .collect();
                (priors, value as f64)
            })
            .collect()
    }
}

pub fn build_model(
    store: &dyn ModelStore,
    device: &str,
) -> anyhow::Result<(i64, Box<dyn BatchModel>)> {
    let fallback_version = store.get_version()?.unwrap_or(0);
    let Some((version, path)) = store.get_latest_path()? else {
        log::warn!("model_dir 无 latest.json，使用均匀评估器");
        return Ok((fallback_version, Box::new(UniformBatch)));
    };
    match super::torch_model::TorchModel::load_str(path.to_str().unwrap(), device) {
        Ok(model) => Ok((version, Box::new(TorchBatch { model }))),
        Err(e) => {
            log::warn!("加载 model.pt 失败（{e:#}），使用均匀评估器");
            Ok((version, Box::new(UniformBatch)))
        }
    }
}

/// 推理 actor 主循环：独占模型，凑批前向，热加载，直到所有发送端关闭后退出。
///
/// `model_version`（当前模型权重版本）由本线程在初始化与每次热加载后写入，
/// worker 读它给样本分片命名。
pub fn run_actor<S: ModelStore>(
    rx: Receiver<InferRequest>,
    model_store: S,
    device: String,
    batch_size: usize,
    timeout: Duration,
    model_version: Arc<AtomicI64>,
) {
    let (mut cur_model_version, mut model) =
        build_model(&model_store, &device).expect("推理 actor 初始化模型失败");
    model_version.store(cur_model_version, Ordering::SeqCst);

    // 热加载读盘有成本，按时间节流（每秒至多查一次 latest.json）。
    let mut last_check = Instant::now();

    loop {
        // 阻塞等第一条请求；所有发送端 drop 后 recv 返回 Err → 收工。
        let first = match rx.recv() {
            Ok(r) => r,
            Err(_) => break,
        };
        let mut inputs: Vec<EvalInput> = Vec::new();
        let mut ones: Vec<(usize, Sender<EvalReply>)> = Vec::new();
        let mut batches: Vec<BatchReplyGroup> = Vec::new();
        extend_infer_msg(first, &mut inputs, &mut ones, &mut batches);

        // 凑批：满 batch_size 或到 timeout 截止即止。timeout=0 时退化为不凑批。
        let deadline = Instant::now() + timeout;
        while inputs.len() < batch_size {
            let wait = deadline.saturating_duration_since(Instant::now());
            if wait.is_zero() {
                break;
            }
            match rx.recv_timeout(wait) {
                Ok(msg) => extend_infer_msg(msg, &mut inputs, &mut ones, &mut batches),
                Err(RecvTimeoutError::Timeout) => break,
                Err(RecvTimeoutError::Disconnected) => break,
            }
        }

        maybe_reload(
            &model_store,
            &device,
            &mut cur_model_version,
            &mut model,
            &model_version,
            &mut last_check,
        );

        let results = model.evaluate_batch(&inputs);
        scatter_results(results, ones, batches);
    }
}

fn maybe_reload<S: ModelStore>(
    model_store: &S,
    device: &str,
    cur_model_version: &mut i64,
    model: &mut Box<dyn BatchModel>,
    model_version: &Arc<AtomicI64>,
    last_check: &mut Instant,
) {
    if last_check.elapsed() < Duration::from_secs(1) {
        return;
    }
    *last_check = Instant::now();
    let latest = match model_store.get_version() {
        Ok(Some(v)) => v,
        _ => return,
    };
    if latest == *cur_model_version {
        return;
    }
    match build_model(model_store, device) {
        Ok((nv, nm)) => {
            *cur_model_version = nv;
            *model = nm;
            model_version.store(nv, Ordering::SeqCst);
        }
        Err(e) => log::error!("热加载模型失败：{e:#}"),
    }
}

/// worker 侧的评估句柄：克隆同一 `Sender` 即可，发请求后阻塞等回执。
pub struct BatchedEvaluator {
    tx: Sender<InferRequest>,
}

impl BatchedEvaluator {
    pub fn new(tx: Sender<InferRequest>) -> Self {
        BatchedEvaluator { tx }
    }
}

// 为 &BatchedEvaluator 实现 Evaluator：每局 `Mcts::new(&evaluator, ...)` 复用同一句柄。
// &T 是 fundamental 类型、BatchedEvaluator 为本地类型，孤儿规则允许此 impl。
impl BatchedEvaluator {
    /// 把一个局面打包成请求 + 一次性回执接收端（编码/合法走法在调用线程并行算）。
    fn make_request(state: &GameState) -> (InferRequest, std::sync::mpsc::Receiver<EvalReply>) {
        let input = encode(state);
        let moves = state.position.legal_moves();
        let perspective = state.position.side_to_move;
        let legal_ids: Vec<usize> = moves
            .iter()
            .map(|&m| to_canonical_action(m, perspective))
            .collect();
        let (reply_tx, reply_rx) = mpsc::channel();
        let req = EvalRequest {
            input: EvalInput { input, moves, legal_ids },
            reply: reply_tx,
        };
        (InferRequest::One(req), reply_rx)
    }
}

impl Evaluator for BatchedEvaluator {
    fn evaluate(&self, state: &GameState) -> EvalReply {
        let (req, reply_rx) = BatchedEvaluator::make_request(state);
        self.tx.send(req).expect("推理 actor 已退出");
        reply_rx.recv().expect("推理 actor 未返回结果")
    }

    /// 叶子并行：先把一波全部请求发给 actor（不阻塞），再统一收回执。这样同波多个
    /// 叶子并发凑进 actor 的批，单次 worker 只阻塞一次往返，而非每叶一次，显著提速。
    fn evaluate_batch(&self, states: &[&GameState]) -> Vec<EvalReply> {
        let mut rxs = Vec::with_capacity(states.len());
        for state in states {
            let (req, reply_rx) = BatchedEvaluator::make_request(state);
            self.tx.send(req).expect("推理 actor 已退出");
            rxs.push(reply_rx);
        }
        rxs.into_iter()
            .map(|rx| rx.recv().expect("推理 actor 未返回结果"))
            .collect()
    }
}
