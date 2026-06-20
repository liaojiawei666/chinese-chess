//! datagen 命令行参数：读 config JSON + 可选覆盖 + 性能测试停止条件。

use std::path::PathBuf;
use std::time::{Duration, Instant};

use clap::Parser;
use cc_core::config::RunConfig;

#[derive(Parser, Debug)]
#[command(name = "datagen", about = "自对弈数据生成")]
pub struct Cli {
    /// 配置文件路径（省略时自动检测 GPU：有 CUDA → config/gpu.json，否则 config/local.json）
    #[arg(long, short)]
    pub config: Option<PathBuf>,

    #[arg(long)]
    pub workers: Option<usize>,

    #[arg(long)]
    pub games_per_worker: Option<usize>,

    #[arg(long)]
    pub n_simulations: Option<u32>,

    #[arg(long)]
    pub eval_batch_size: Option<usize>,

    #[arg(long)]
    pub inference_timeout_ms: Option<f64>,

    #[arg(long)]
    pub total_samples: Option<u64>,

    #[arg(long)]
    pub shard_games: Option<u32>,

    #[arg(long)]
    pub device: Option<String>,

    /// 只跑 N 秒后停止（性能测试 / profiling 用）
    #[arg(long)]
    pub duration_secs: Option<u64>,

    /// 只跑 N 局后停止（性能测试用）
    #[arg(long)]
    pub max_games: Option<u64>,
}

/// 加载配置并应用 CLI 覆盖。省略 `--config` 时自动检测 GPU 选择配置文件。
pub fn load_config(cli: &Cli) -> anyhow::Result<RunConfig> {
    let config_path = match &cli.config {
        Some(p) => p.clone(),
        None => detect_config_path(),
    };
    log::info!("加载配置：{}", config_path.display());
    let mut rc = RunConfig::load(&config_path)?;
    apply_overrides(&mut rc, cli);
    Ok(rc)
}

fn detect_config_path() -> PathBuf {
    let has_cuda = tch::Cuda::is_available();
    let profile = if has_cuda { "gpu" } else { "local" };
    log::info!(
        "GPU 自动检测：CUDA {}，使用 {} 配置",
        if has_cuda { "可用" } else { "不可用" },
        profile
    );
    PathBuf::from(format!("config/{profile}.json"))
}

pub fn apply_overrides(rc: &mut RunConfig, cli: &Cli) {
    if let Some(v) = cli.workers {
        rc.datagen.num_workers = v;
    }
    if let Some(v) = cli.games_per_worker {
        rc.datagen.games_per_worker = v;
    }
    if let Some(v) = cli.n_simulations {
        rc.mcts.n_simulations = v;
    }
    if let Some(v) = cli.eval_batch_size {
        rc.datagen.eval_batch_size = v;
    }
    if let Some(v) = cli.inference_timeout_ms {
        rc.datagen.inference_timeout_ms = v;
    }
    if let Some(v) = cli.total_samples {
        rc.total_samples = v;
    }
    if let Some(v) = cli.shard_games {
        rc.datagen.shard_games = v;
    }
    if let Some(v) = &cli.device {
        rc.device = v.clone();
    }
}

/// 性能测试用的停止条件。
pub struct StopConditions {
    pub deadline: Option<Instant>,
    pub max_games: Option<i64>,
}

impl StopConditions {
    pub fn from_cli(cli: &Cli) -> Self {
        StopConditions {
            deadline: cli
                .duration_secs
                .map(|s| Instant::now() + Duration::from_secs(s)),
            max_games: cli.max_games.map(|g| g as i64),
        }
    }

    pub fn should_stop(&self, games: i64) -> bool {
        if let Some(deadline) = self.deadline {
            if Instant::now() >= deadline {
                return true;
            }
        }
        if let Some(max) = self.max_games {
            if games >= max {
                return true;
            }
        }
        false
    }
}
