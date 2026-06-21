//! 启动前确保 model_dir 有 TorchScript 权重（无则调 Python 导出随机初始网络）。

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result};
use cc_core::model_io::{LocalModelStore, ModelStore};

/// `model_dir` 无 `latest.json` 时，调用 `trainer/scripts/export_initial_model.py`。
pub fn ensure_initial_model(config_path: &Path, model_dir: &str) -> Result<()> {
    let store = LocalModelStore::new(model_dir);
    if store.get_latest_path()?.is_some() {
        return Ok(());
    }

    log::info!("model_dir 无 latest.json，调用 Python 导出随机初始权重…");

    let python = find_trainer_python();
    let script = Path::new("trainer/scripts/export_initial_model.py");
    anyhow::ensure!(
        script.is_file(),
        "未找到 {}（请在仓库根目录运行 datagen）",
        script.display()
    );

    let output = Command::new(&python)
        .arg(script)
        .arg("--config")
        .arg(config_path)
        .output()
        .with_context(|| format!("启动 {python:?} 失败"))?;

    if !output.status.success() {
        anyhow::bail!(
            "export_initial_model 失败 (exit {}):\n{}\n{}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
    }

    if store.get_latest_path()?.is_none() {
        anyhow::bail!("export_initial_model 完成但 {model_dir} 仍无 latest.json");
    }

    log::info!("初始模型已就绪：{model_dir}");
    Ok(())
}

fn find_trainer_python() -> PathBuf {
    for candidate in [
        PathBuf::from("trainer/.venv/bin/python"),
        PathBuf::from("trainer/.venv/bin/python3"),
    ] {
        if candidate.is_file() {
            return candidate;
        }
    }
    PathBuf::from("python3")
}
