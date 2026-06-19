//! 读取 data/config/run-config.json（由 trainer/scripts/export_run_config.py 生成）。
//!
//! 结构对应 shared/run-config.schema.json。`verify_constants` 把 JSON 里的结构常量
//! 与 engine crate 的内置 const 做一致性断言，防止两侧漂移。

use std::path::Path;

use anyhow::{bail, Context, Result};
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct RunConfig {
    pub profile: String,
    pub device: String,
    pub total_samples: u64,
    pub board: BoardConfig,
    pub rules: RulesConfig,
    pub encoding: EncodingConfig,
    pub network: NetworkConfig,
    pub mcts: MctsConfig,
    pub train: TrainConfig,
    pub selfplay: SelfPlayConfig,
    pub datagen: DataGenConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct BoardConfig {
    pub width: usize,
    pub height: usize,
    pub square_count: usize,
    pub action_space_size: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RulesConfig {
    pub max_total_plies: u32,
    pub no_capture_draw_plies: u32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct EncodingConfig {
    pub n_history: usize,
    pub planes_per_frame: usize,
    pub input_channels: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct NetworkConfig {
    pub input_channels: usize,
    pub action_space_size: usize,
    pub hidden_channels: usize,
    pub residual_blocks: usize,
    pub policy_head_channels: usize,
    pub value_head_channels: usize,
    pub value_fc_hidden: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct MctsConfig {
    pub n_simulations: u32,
    pub c_puct: f64,
    pub dirichlet_alpha: f64,
    pub dirichlet_epsilon: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TrainConfig {
    pub batch_size: usize,
    pub learning_rate: f64,
    pub weight_decay: f64,
    pub grad_clip_norm: f64,
    pub buffer_capacity: usize,
    pub min_buffer_size: usize,
    pub steps_per_iteration: u32,
    pub weight_sync_interval: u32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SelfPlayConfig {
    pub temperature_moves: u32,
    pub eval_batch_size: usize,
    pub inference_timeout_ms: f64,
    pub num_workers: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DataGenConfig {
    pub samples_dir: String,
    pub model_dir: String,
    pub run_config_path: String,
    pub shard_games: u32,
    pub max_pending_shards: usize,
    pub model_export_interval: u32,
    pub keep_recent_models: usize,
    pub checkpoint_every: u32,
    pub keep_checkpoints: usize,
    pub num_workers: usize,
}

impl RunConfig {
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("读取 run-config 失败：{}", path.display()))?;
        let config: RunConfig =
            serde_json::from_str(&text).context("解析 run-config.json 失败")?;
        config.verify_constants()?;
        Ok(config)
    }

    /// 断言 JSON 的结构常量与 engine 内置 const 一致，防止漂移。
    pub fn verify_constants(&self) -> Result<()> {
        check("board.width", self.board.width, engine::BOARD_WIDTH)?;
        check("board.height", self.board.height, engine::BOARD_HEIGHT)?;
        check("board.square_count", self.board.square_count, engine::SQUARE_COUNT)?;
        check(
            "board.action_space_size",
            self.board.action_space_size,
            engine::ACTION_SPACE_SIZE,
        )?;
        check(
            "rules.max_total_plies",
            self.rules.max_total_plies as usize,
            engine::MAX_TOTAL_PLIES as usize,
        )?;
        check(
            "rules.no_capture_draw_plies",
            self.rules.no_capture_draw_plies as usize,
            engine::NO_CAPTURE_DRAW_PLIES as usize,
        )?;
        check("encoding.n_history", self.encoding.n_history, engine::N_HISTORY)?;
        check(
            "encoding.planes_per_frame",
            self.encoding.planes_per_frame,
            engine::PLANES_PER_FRAME,
        )?;
        check(
            "encoding.input_channels",
            self.encoding.input_channels,
            engine::INPUT_CHANNELS,
        )?;
        Ok(())
    }
}

fn check(name: &str, json_value: usize, rust_const: usize) -> Result<()> {
    if json_value != rust_const {
        bail!(
            "run-config 常量 {name}={json_value} 与 engine const={rust_const} 不一致，请重新导出 run-config 或对齐两侧"
        );
    }
    Ok(())
}
