//! 读取 config/*.json（仓库内单一真相，Python/Rust 两侧共读）。
//!
//! 只包含真正可调的超参数。棋盘几何、规则上限、编码通道数等结构常量
//! 定义在 `engine` 模块的 const 中，不重复写入 JSON。

use std::path::Path;

use anyhow::{Context, Result};
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct RunConfig {
    pub profile: String,
    pub device: String,
    pub total_samples: u64,
    pub network: NetworkConfig,
    pub mcts: MctsConfig,
    pub train: TrainConfig,
    pub datagen: DataGenConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct NetworkConfig {
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
    /// 前 N 手用温度采样（训练自对弈探索用）；对战时设为 0 表示始终 argmax。
    #[serde(default)]
    pub temperature_moves: u32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TrainConfig {
    pub batch_size: usize,
    pub learning_rate: f64,
    pub weight_decay: f64,
    pub grad_clip_norm: f64,
    pub buffer_capacity: usize,
    pub min_buffer_size: usize,
    pub target_reuse: f64,
    pub steps_per_iteration: u32,
    pub weight_sync_interval: u32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DataGenConfig {
    pub samples_dir: String,
    pub model_dir: String,
    pub num_workers: usize,
    /// 每个 worker 同时进行的游戏数（流水线宽度）。
    /// 多局并行可消除 CPU/GPU 乒乓阻塞。建议 num_workers × games_per_worker ≥ eval_batch_size。
    #[serde(default = "default_games_per_worker")]
    pub games_per_worker: usize,
    pub eval_batch_size: usize,
    pub inference_timeout_ms: f64,
    pub shard_games: u32,
    pub max_pending_shards: usize,
    pub model_export_interval: u32,
    pub keep_recent_models: usize,
    pub checkpoint_every: u32,
    pub keep_checkpoints: usize,
}

fn default_games_per_worker() -> usize {
    1
}

impl RunConfig {
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("读取 run-config 失败：{}", path.display()))?;
        let config: RunConfig =
            serde_json::from_str(&text).context("解析 config JSON 失败")?;
        Ok(config)
    }
}
