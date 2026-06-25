use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result};
use engine::mcts::MctsConfig;
use serde::Deserialize;
use tracing::info;

#[derive(Debug, Deserialize)]
struct FullConfig {
    pub total_samples: u64,
    pub mcts: MctsSection,
    pub datagen: DatagenSectionRaw,
}

#[derive(Debug, Deserialize, Clone)]
pub struct MctsSection {
    pub n_simulations: usize,
    pub c_puct: f32,
    pub dirichlet_alpha: f32,
    pub dirichlet_epsilon: f32,
    pub temperature_moves: usize,
}

/// JSON 反序列化用（路径尚未解析）。
#[derive(Debug, Deserialize)]
struct DatagenSectionRaw {
    pub samples_dir: String,
    pub models_dir: String,
    pub num_parallel_games: usize,
    pub eval_batch_size: usize,
    pub inference_timeout_ms: f64,
    pub shard_size: usize,
}

/// 路径已解析为绝对路径的 datagen 配置。
#[derive(Debug, Clone)]
pub struct DatagenSection {
    pub samples_dir: PathBuf,
    pub models_dir: PathBuf,
    pub num_parallel_games: usize,
    pub eval_batch_size: usize,
    pub inference_timeout_ms: f64,
    pub shard_size: usize,
}

/// datagen 运行时所需的全部配置。
#[derive(Debug, Clone)]
pub struct DatagenConfig {
    pub total_samples: u64,
    pub mcts: MctsSection,
    pub datagen: DatagenSection,
}

impl DatagenConfig {
    /// 从 JSON 文件加载配置。相对路径以 config 文件所在目录为基准解析。
    pub fn from_file(path: &str) -> Result<Self> {
        let config_path = Path::new(path)
            .canonicalize()
            .with_context(|| format!("failed to resolve config path: {path}"))?;
        let base_dir = config_path.parent().unwrap();

        let content = std::fs::read_to_string(&config_path)
            .with_context(|| format!("failed to read config: {}", config_path.display()))?;
        let full: FullConfig =
            serde_json::from_str(&content).with_context(|| "failed to parse config")?;

        let resolve = |rel: &str| -> PathBuf {
            let p = Path::new(rel);
            if p.is_absolute() {
                p.to_path_buf()
            } else {
                base_dir.join(p)
            }
        };

        Ok(DatagenConfig {
            total_samples: full.total_samples,
            mcts: full.mcts,
            datagen: DatagenSection {
                samples_dir: resolve(&full.datagen.samples_dir),
                models_dir: resolve(&full.datagen.models_dir),
                num_parallel_games: full.datagen.num_parallel_games,
                eval_batch_size: full.datagen.eval_batch_size,
                inference_timeout_ms: full.datagen.inference_timeout_ms,
                shard_size: full.datagen.shard_size,
            },
        })
    }

    pub fn mcts_config(&self) -> MctsConfig {
        MctsConfig {
            num_simulations: self.mcts.n_simulations,
            c_puct: self.mcts.c_puct,
            dirichlet_alpha: self.mcts.dirichlet_alpha,
            noise_fraction: self.mcts.dirichlet_epsilon,
            temperature: 1.0,
        }
    }
}

// ── latest.json 读取 ──

#[derive(Deserialize)]
struct LatestJson {
    generation: u64,
    model_path: String,
}

/// 读取 latest.json，返回 (generation, model_absolute_path)。失败返回 None。
pub fn read_latest(models_dir: &Path) -> Option<(u64, PathBuf)> {
    let latest_path = models_dir.join("latest.json");
    let content = std::fs::read_to_string(&latest_path).ok()?;
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

/// 启动时轮询等待 latest.json 出现，返回 (model_path, generation)。
#[allow(dead_code)]
pub fn wait_for_model(models_dir: &Path) -> Result<(PathBuf, u64)> {
    if let Some((gen, path)) = read_latest(models_dir) {
        info!("model found: gen={gen}, path={}", path.display());
        return Ok((path, gen));
    }

    info!(
        "waiting for model at {}...",
        models_dir.join("latest.json").display()
    );

    loop {
        std::thread::sleep(Duration::from_secs(3));
        if let Some((gen, path)) = read_latest(models_dir) {
            info!("model found: gen={gen}, path={}", path.display());
            return Ok((path, gen));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_parse() {
        let config = DatagenConfig::from_file("../config.json")
            .expect("config.json should parse successfully");
        assert_eq!(config.total_samples, 1_200_000);
        assert_eq!(config.mcts.n_simulations, 128);
        assert_eq!(config.datagen.num_parallel_games, 64);
        assert_eq!(config.datagen.shard_size, 4096);
    }

    #[test]
    fn test_config_local_parse() {
        let config = DatagenConfig::from_file("../config.local.json")
            .expect("config.local.json should parse successfully");
        assert_eq!(config.total_samples, 2000);
        assert_eq!(config.mcts.n_simulations, 50);
        assert_eq!(config.datagen.num_parallel_games, 2);
    }

    #[test]
    fn test_mcts_config_mapping() {
        let config = DatagenConfig::from_file("../config.json").unwrap();
        let mcts = config.mcts_config();
        assert_eq!(mcts.num_simulations, 128);
        assert_eq!(mcts.c_puct, 1.5);
        assert_eq!(mcts.noise_fraction, 0.25);
    }

    #[test]
    fn test_read_latest_missing() {
        let dir = std::env::temp_dir().join("cxsg_test_no_latest");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        assert!(read_latest(&dir).is_none());

        let _ = std::fs::remove_dir_all(&dir);
    }
}
