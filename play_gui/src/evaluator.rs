use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use engine::constants::*;
use engine::encode::encode_state;
use engine::evaluate::{EvaluatorOutput, Evaluator};
use engine::game::GameState;
use ort::session::Session;
use ort::value::Value;

pub struct OnnxEvaluator {
    session: Arc<Mutex<Session>>,
}

impl OnnxEvaluator {
    pub fn new(session: Session) -> Self {
        Self {
            session: Arc::new(Mutex::new(session)),
        }
    }

    fn run_inference(session: &mut Session, state: &[f32]) -> Result<EvaluatorOutput> {
        let h = BOARD_HEIGHT as usize;
        let w = BOARD_WIDTH as usize;

        let input_array = ndarray::Array4::<f32>::from_shape_vec(
            (1, INPUT_CHANNELS, h, w),
            state.to_vec(),
        )
        .context("failed to create input array")?;

        let input_tensor = Value::from_array(input_array)
            .map_err(|e| anyhow::anyhow!("failed to create input tensor: {e}"))?;

        let outputs = session
            .run(ort::inputs![input_tensor])
            .map_err(|e| anyhow::anyhow!("inference failed: {e}"))?;

        let (_, policy_data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow::anyhow!("policy extraction failed: {e}"))?;
        let (_, value_data) = outputs[1]
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow::anyhow!("value extraction failed: {e}"))?;

        let mut policy_logits = [0.0f32; ACTION_SPACE_SIZE];
        policy_logits.copy_from_slice(&policy_data[..ACTION_SPACE_SIZE]);

        Ok(EvaluatorOutput {
            value: value_data[0],
            policy_logits,
        })
    }
}

impl Evaluator for OnnxEvaluator {
    fn evaluate_async(
        &self,
        game: &GameState,
    ) -> Pin<Box<dyn Future<Output = EvaluatorOutput> + Send>> {
        let state = encode_state(game).to_float_tensor();
        let session = self.session.clone();
        Box::pin(async move {
            let mut sess = session.lock().unwrap();
            Self::run_inference(&mut sess, &state).unwrap_or_else(|e| {
                log::warn!("inference failed: {e:#}, returning uniform");
                EvaluatorOutput {
                    value: 0.0,
                    policy_logits: [0.0f32; ACTION_SPACE_SIZE],
                }
            })
        })
    }
}

pub fn load_onnx_model(model_path: &str) -> Result<Session> {
    log::info!("loading ONNX model from {model_path}");
    let session = Session::builder()
        .map_err(|e| anyhow::anyhow!("session builder: {e}"))?
        .with_execution_providers([
            ort::execution_providers::CPUExecutionProvider::default().build(),
        ])
        .map_err(|e| anyhow::anyhow!("execution providers: {e}"))?
        .commit_from_file(model_path)
        .map_err(|e| anyhow::anyhow!("load model {model_path}: {e}"))?;
    Ok(session)
}
