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
//! 单局自身仍是「下行到一个叶子 → 阻塞等评估 → backup」的串行语义（每 worker 同时刻
//! 至多 1 个在途请求），所以跨 worker 聚合是逐位精确的，不改变任何单局结果。

use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender, SyncSender};
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
    pub reply: SyncSender<EvalReply>,
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
    rx: Receiver<EvalRequest>,
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
        let mut reqs = vec![first];

        // 凑批：满 batch_size 或到 timeout 截止即止。timeout=0 时退化为不凑批。
        let deadline = Instant::now() + timeout;
        while reqs.len() < batch_size {
            let wait = deadline.saturating_duration_since(Instant::now());
            if wait.is_zero() {
                break;
            }
            match rx.recv_timeout(wait) {
                Ok(r) => reqs.push(r),
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

        let mut inputs = Vec::with_capacity(reqs.len());
        let mut replies = Vec::with_capacity(reqs.len());
        for r in reqs {
            inputs.push(r.input);
            replies.push(r.reply);
        }
        let results = model.evaluate_batch(&inputs);
        for (tx, res) in replies.into_iter().zip(results) {
            // worker 可能已退出（reply 接收端被 drop），忽略发送失败。
            let _ = tx.send(res);
        }
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
    tx: Sender<EvalRequest>,
}

impl BatchedEvaluator {
    pub fn new(tx: Sender<EvalRequest>) -> Self {
        BatchedEvaluator { tx }
    }
}

// 为 &BatchedEvaluator 实现 Evaluator：每局 `Mcts::new(&evaluator, ...)` 复用同一句柄。
// &T 是 fundamental 类型、BatchedEvaluator 为本地类型，孤儿规则允许此 impl。
impl BatchedEvaluator {
    /// 把一个局面打包成请求 + 一次性回执接收端（编码/合法走法在调用线程并行算）。
    fn make_request(state: &GameState) -> (EvalRequest, std::sync::mpsc::Receiver<EvalReply>) {
        let input = encode(state);
        let moves = state.position.legal_moves();
        let perspective = state.position.side_to_move;
        let legal_ids: Vec<usize> = moves
            .iter()
            .map(|&m| to_canonical_action(m, perspective))
            .collect();
        let (reply_tx, reply_rx) = mpsc::sync_channel(1);
        let req = EvalRequest {
            input: EvalInput { input, moves, legal_ids },
            reply: reply_tx,
        };
        (req, reply_rx)
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
