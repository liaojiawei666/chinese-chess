use std::path::PathBuf;
use std::time::Duration;

use anyhow::{Context, Result};
use engine::{constants::*, ACTION_SPACE_SIZE};
use ort::session::Session;
use ort::value::Value;
use tokio::{sync::mpsc, time::Instant};
use tracing::{debug, info, warn};

use crate::evaluator::{EvalRequest, EvalResponse};

const STATE_SIZE: usize = INPUT_CHANNELS * BOARD_HEIGHT as usize * BOARD_WIDTH as usize;

/// 加载 ONNX 模型，优先使用 CUDA，回退到 CPU。
pub fn load_onnx_model(model_path: &str) -> Result<Session> {
    info!("loading ONNX model from {model_path}");

    let session = Session::builder()
        .map_err(|e| anyhow::anyhow!("failed to create session builder: {e}"))?
        .with_execution_providers([
            ort::execution_providers::CUDAExecutionProvider::default().build(),
            ort::execution_providers::CPUExecutionProvider::default().build(),
        ])
        .map_err(|e| anyhow::anyhow!("failed to set execution providers: {e}"))?
        .commit_from_file(model_path)
        .map_err(|e| anyhow::anyhow!("failed to load ONNX model {model_path}: {e}"))?;

    info!(
        "ONNX model loaded, inputs: {:?}, outputs: {:?}",
        session
            .inputs()
            .iter()
            .map(|i| i.name())
            .collect::<Vec<_>>(),
        session
            .outputs()
            .iter()
            .map(|o| o.name())
            .collect::<Vec<_>>(),
    );

    Ok(session)
}

/// 推理服务器主循环。
/// 从 mpsc 通道收集评估请求，攒成 batch 后送入 ONNX 模型，结果通过 oneshot 回传。
/// 每个 batch 处理完后轮询 latest.json，generation 变化时热加载新模型。
pub async fn inference_loop(
    mut request_rx: mpsc::Receiver<EvalRequest>,
    mut session: Session,
    max_batch_size: usize,
    batch_timeout: Duration,
    models_dir: PathBuf,
    initial_generation: u64,
) {
    info!(
        "inference server started, max_batch_size={max_batch_size}, timeout={batch_timeout:?}"
    );

    let mut total_evals: u64 = 0;
    let mut current_generation = initial_generation;
    let mut throughput_start = Instant::now();
    let mut throughput_evals: u64 = 0;

    loop {
        let first = match request_rx.recv().await {
            Some(req) => req,
            None => {
                info!("all senders dropped, inference server shutting down");
                break;
            }
        };

        let mut batch = Vec::with_capacity(max_batch_size);
        batch.push(first);

        let deadline = Instant::now() + batch_timeout;
        while batch.len() < max_batch_size {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                break;
            }
            match tokio::time::timeout(remaining, request_rx.recv()).await {
                Ok(Some(req)) => batch.push(req),
                _ => break,
            }
        }

        let batch_size = batch.len();
        debug!("running batch inference, size={batch_size}");

        match run_batch(&mut session, &batch) {
            Ok(responses) => {
                for (req, resp) in batch.into_iter().zip(responses) {
                    let _ = req.response_tx.send(resp);
                }
                total_evals += batch_size as u64;
                throughput_evals += batch_size as u64;

                if total_evals % 10000 < batch_size as u64 {
                    let dt = throughput_start.elapsed().as_secs_f64().max(0.001);
                    let evals_per_sec = throughput_evals as f64 / dt;
                    info!(
                        "inference: {total_evals} total evals, \
                         {evals_per_sec:.0} evals/s (last {throughput_evals} in {dt:.1}s)"
                    );
                    throughput_start = Instant::now();
                    throughput_evals = 0;
                }
            }
            Err(e) => {
                warn!("batch inference failed: {e:#}");
                drop(batch);
            }
        }

        // 热加载：检查 latest.json 的 generation 是否变化
        if let Some((new_gen, new_path)) = crate::config::read_latest(&models_dir) {
            if new_gen != current_generation {
                info!(
                    "new model detected: gen={current_generation} -> gen={new_gen}, reloading..."
                );
                let reload_start = std::time::Instant::now();
                match load_onnx_model(new_path.to_str().unwrap_or("?")) {
                    Ok(new_session) => {
                        session = new_session;
                        let took = reload_start.elapsed().as_secs_f64();
                        info!("model reloaded: gen={new_gen}, took {took:.1}s");
                        current_generation = new_gen;
                    }
                    Err(e) => {
                        warn!("failed to reload model gen={new_gen}: {e:#}");
                    }
                }
            }
        }
    }
}

/// 执行一次 batch 推理，返回每个请求的 EvalResponse。
fn run_batch(session: &mut Session, batch: &[EvalRequest]) -> Result<Vec<EvalResponse>> {
    let batch_size = batch.len();
    let h = BOARD_HEIGHT as usize;
    let w = BOARD_WIDTH as usize;

    let mut flat = vec![0.0f32; batch_size * INPUT_CHANNELS * h * w];
    for (i, req) in batch.iter().enumerate() {
        assert_eq!(req.state.len(), STATE_SIZE);
        let offset = i * STATE_SIZE;
        flat[offset..offset + STATE_SIZE].copy_from_slice(&req.state);
    }

    let input_array = ndarray::Array4::<f32>::from_shape_vec(
        (batch_size, INPUT_CHANNELS, h, w),
        flat,
    )
    .context("failed to create input array")?;

    let input_tensor = Value::from_array(input_array)
        .map_err(|e| anyhow::anyhow!("failed to create input tensor: {e}"))?;

    let outputs = session
        .run(ort::inputs![input_tensor])
        .map_err(|e| anyhow::anyhow!("inference run failed: {e}"))?;

    let (policy_shape, policy_data) = outputs[0]
        .try_extract_tensor::<f32>()
        .map_err(|e| anyhow::anyhow!("policy extraction failed: {e}"))?;
    let (value_shape, value_data) = outputs[1]
        .try_extract_tensor::<f32>()
        .map_err(|e| anyhow::anyhow!("value extraction failed: {e}"))?;

    let bs = batch_size as i64;
    let acs = ACTION_SPACE_SIZE as i64;
    anyhow::ensure!(
        policy_shape.len() == 2 && policy_shape[0] == bs && policy_shape[1] == acs,
        "unexpected policy shape: {policy_shape:?}, expected [{batch_size}, {ACTION_SPACE_SIZE}]"
    );
    anyhow::ensure!(
        value_shape.len() == 2 && value_shape[0] == bs && value_shape[1] == 1,
        "unexpected value shape: {value_shape:?}, expected [{batch_size}, 1]"
    );

    let mut responses = Vec::with_capacity(batch_size);
    for i in 0..batch_size {
        let value = value_data[i];

        let mut policy_logits = [0.0f32; ACTION_SPACE_SIZE];
        let src = &policy_data[i * ACTION_SPACE_SIZE..(i + 1) * ACTION_SPACE_SIZE];
        policy_logits.copy_from_slice(src);

        responses.push(EvalResponse {
            value,
            policy_logits,
        });
    }

    Ok(responses)
}
