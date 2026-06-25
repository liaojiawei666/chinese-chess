use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use engine::mcts::MctsConfig;
use serde::Deserialize;

/// play_gui 专用 MCTS 默认参数（对弈模式，比训练时更大的模拟数、关闭噪声）。
const DEFAULT_NUM_SIMULATIONS: usize = 800;
const DEFAULT_C_PUCT: f32 = 1.5;

/// 从 config.json 中仅读取 models_dir 等基础信息。
#[derive(Debug, Deserialize)]
struct FullConfig {
    datagen: DatagenSection,
}

#[derive(Debug, Deserialize)]
struct DatagenSection {
    models_dir: String,
}

#[derive(Debug, Deserialize)]
struct LatestJson {
    pub generation: u64,
    pub model_path: String,
}

/// play_gui 运行时配置。
#[derive(Debug, Clone)]
pub struct PlayConfig {
    pub models_dir: PathBuf,
    pub mcts: MctsConfig,
}

impl PlayConfig {
    /// 从 config.json 加载 models_dir，MCTS 参数使用对弈默认值。
    pub fn from_file(config_path: &str) -> Result<Self> {
        let path = Path::new(config_path)
            .canonicalize()
            .with_context(|| format!("config path: {config_path}"))?;
        let base = path.parent().unwrap();
        let text = std::fs::read_to_string(&path)?;
        let full: FullConfig = serde_json::from_str(&text).context("parse config")?;

        let models_dir = if Path::new(&full.datagen.models_dir).is_absolute() {
            PathBuf::from(&full.datagen.models_dir)
        } else {
            base.join(&full.datagen.models_dir)
        };

        Ok(PlayConfig {
            models_dir,
            mcts: MctsConfig {
                num_simulations: DEFAULT_NUM_SIMULATIONS,
                c_puct: DEFAULT_C_PUCT,
                dirichlet_alpha: 0.0,
                noise_fraction: 0.0,
                temperature: 0.0,
            },
        })
    }

    /// 覆盖 MCTS 模拟次数。
    pub fn with_simulations(mut self, n: usize) -> Self {
        self.mcts.num_simulations = n;
        self
    }

    /// 覆盖 PUCT 探索常数。
    #[allow(dead_code)]
    pub fn with_c_puct(mut self, c: f32) -> Self {
        self.mcts.c_puct = c;
        self
    }
}

/// 读取 latest.json，返回 (generation, model_absolute_path)。失败返回 None。
pub fn read_latest(models_dir: &Path) -> Option<(u64, PathBuf)> {
    let content = std::fs::read_to_string(models_dir.join("latest.json")).ok()?;
    let latest: LatestJson = serde_json::from_str(&content).ok()?;
    let model_path = Path::new(&latest.model_path);
    let abs_path = if model_path.is_absolute() {
        model_path.to_path_buf()
    } else {
        let project_root = models_dir
            .parent()
            .unwrap_or(models_dir)
            .parent()
            .unwrap_or(models_dir);
        project_root.join(model_path)
    };
    if abs_path.exists() {
        Some((latest.generation, abs_path))
    } else {
        None
    }
}
