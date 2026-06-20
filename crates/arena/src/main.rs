//! arena：两个模型版本对杀，输出 A 相对 B 的胜率与 Elo。
//!
//! 纯指标工具：**不改 latest.json、不打断训练**。训练持续产出冻结的 model_*.pt，arena 在
//! 旁边 load 两个快照对杀即可。需要门控时，外层据本工具的得分率自行决定是否改写 latest.json。
//!
//! 开局：均匀评估器 + 固定种子采样并冻结（不依赖训练参数，可复现）。
//! 对局：每开局红黑各一局（互换抵消先手），τ=0 / ε=0 全程确定性。
//!
//! 用法（选模型三选一）：
//!   # 自动：取 --model-dir（默认 data/models）里最近两版，A=最新 vs B=上一版
//!   cargo run -p arena -- --num-openings 100 --report data/arena/report.json
//!   # 按版本号
//!   cargo run -p arena -- --version-a 100 --version-b 40
//!   # 显式路径
//!   cargo run -p arena -- \
//!     --model-a data/models/model_000100.pt --model-b data/models/model_000040.pt

mod eval;
mod match_play;
mod openings;

#[cfg(test)]
mod tests;

use std::path::PathBuf;

use anyhow::{Context, Result};
use cc_core::mcts::MctsConfig;
use serde::Deserialize;

use match_play::{run_match, MatchReport};

/// 只取 arena 需要的几个字段，run-config 其余字段被 serde 忽略。
#[derive(Debug, Deserialize)]
struct RunConfigPartial {
    #[serde(default = "default_device")]
    device: String,
    mcts: MctsPartial,
}

#[derive(Debug, Deserialize)]
struct MctsPartial {
    n_simulations: u32,
    c_puct: f64,
}

fn default_device() -> String {
    "cpu".to_string()
}

struct Args {
    run_config: Option<PathBuf>,
    model_dir: PathBuf,
    model_a: Option<String>,
    model_b: Option<String>,
    version_a: Option<i64>,
    version_b: Option<i64>,
    openings_file: PathBuf,
    num_openings: usize,
    opening_plies: u32,
    opening_temp: f64,
    opening_sims: u32,
    sims: Option<u32>,
    seed: u64,
    device: Option<String>,
    report: Option<PathBuf>,
    table: Option<PathBuf>,
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

impl Default for Args {
    fn default() -> Self {
        Args {
            run_config: None,
            model_dir: PathBuf::from("data/models"),
            model_a: None,
            model_b: None,
            version_a: None,
            version_b: None,
            openings_file: PathBuf::from("data/arena/openings.json"),
            num_openings: 32,
            opening_plies: 12,
            opening_temp: 1.2,
            opening_sims: 64,
            sims: None,
            seed: 20_260_101,
            device: None,
            report: None,
            table: None,
        }
    }
}

fn need<'a>(argv: &'a [String], i: usize) -> Result<&'a String> {
    argv.get(i + 1)
        .ok_or_else(|| anyhow::anyhow!("缺少参数 {} 的值", argv[i]))
}

fn parse_args() -> Result<Args> {
    let mut args = Args::default();
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < argv.len() {
        match argv[i].as_str() {
            "--run-config" => args.run_config = Some(PathBuf::from(need(&argv, i)?)),
            "--model-dir" => args.model_dir = PathBuf::from(need(&argv, i)?),
            "--model-a" => args.model_a = Some(need(&argv, i)?.clone()),
            "--model-b" => args.model_b = Some(need(&argv, i)?.clone()),
            "--version-a" => args.version_a = Some(need(&argv, i)?.parse().context("--version-a 需整数")?),
            "--version-b" => args.version_b = Some(need(&argv, i)?.parse().context("--version-b 需整数")?),
            "--openings-file" => args.openings_file = PathBuf::from(need(&argv, i)?),
            "--num-openings" => args.num_openings = need(&argv, i)?.parse().context("--num-openings 需整数")?,
            "--opening-plies" => args.opening_plies = need(&argv, i)?.parse().context("--opening-plies 需整数")?,
            "--opening-temp" => args.opening_temp = need(&argv, i)?.parse().context("--opening-temp 需浮点")?,
            "--opening-sims" => args.opening_sims = need(&argv, i)?.parse().context("--opening-sims 需整数")?,
            "--sims" => args.sims = Some(need(&argv, i)?.parse().context("--sims 需整数")?),
            "--seed" => args.seed = need(&argv, i)?.parse().context("--seed 需整数")?,
            "--device" => args.device = Some(need(&argv, i)?.clone()),
            "--report" => args.report = Some(PathBuf::from(need(&argv, i)?)),
            "--table" => args.table = Some(PathBuf::from(need(&argv, i)?)),
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            other => anyhow::bail!("未知参数：{other}（--help 查看用法）"),
        }
        i += 2;
    }
    Ok(args)
}

fn print_help() {
    log::info!(
        "arena —— 两模型对杀，输出 A 相对 B 的 Elo（纯指标，不改 latest.json）\n\n\
         选模型（torch 特性，三选一，优先级从上到下）：\n\
         \x20 ① --model-a <pt> --model-b <pt>    显式路径\n\
         \x20 ② --version-a <N> --version-b <N>  按版本号（在 --model-dir 下找 model_{{N:06}}.pt）\n\
         \x20 ③ 都不传                            自动取 --model-dir 里最近两版：A=最新 vs B=上一版\n\
         可选：\n\
         \x20 --model-dir <path>      默认 data/models（②③ 在此目录解析版本）\n\
         \x20 --run-config <path>     自动检测 GPU 选 config/gpu.json 或 local.json（取 mcts/device）\n\
         \x20 --openings-file <path>  默认 data/arena/openings.json（存在则加载，否则生成并冻结）\n\
         \x20 --num-openings <N>      默认 32（总局数 = N×2）\n\
         \x20 --opening-plies <k>     默认 12\n\
         \x20 --opening-temp <τ>      默认 1.2\n\
         \x20 --opening-sims <n>      默认 64（开局生成用均匀评估器的模拟数）\n\
         \x20 --sims <n>              对局每手模拟数（默认取 run-config.mcts.n_simulations）\n\
         \x20 --seed <s>             默认 20260101\n\
         \x20 --device <cpu|cuda|mps> 默认取 run-config.device\n\
         \x20 --report <path>         写 JSON 报告（单次，覆盖）\n\
         \x20 --table <path>          追加一行 CSV 战绩表（可累积，便于看趋势）"
    );
}

fn main() -> Result<()> {
    env_logger::init();

    let mut args = parse_args()?;

    let config_path = args.run_config.clone().unwrap_or_else(detect_config_path);
    log::info!("加载配置：{}", config_path.display());
    let text = std::fs::read_to_string(&config_path)
        .with_context(|| format!("读取 run-config 失败：{}", config_path.display()))?;
    let rc: RunConfigPartial = serde_json::from_str(&text).context("解析 run-config 失败")?;

    let device = args.device.clone().unwrap_or_else(|| rc.device.clone());
    let sims = args.sims.unwrap_or(rc.mcts.n_simulations);
    let mcts_config = MctsConfig {
        n_simulations: sims,
        c_puct: rc.mcts.c_puct,
        dirichlet_alpha: 0.3,
        dirichlet_epsilon: 0.0,  // 评估关噪声
        collect_batch_size: 1,   // 对杀确定性：单模型同步评估，不用叶子并行
    };

    let openings = openings::load_or_generate(
        &args.openings_file,
        args.num_openings,
        args.opening_plies,
        args.opening_temp,
        args.opening_sims,
        args.seed,
    )?;
    log::info!(
        "openings: {} 个（{}）→ {} 局；每手 {} 次模拟",
        openings.len(),
        args.openings_file.display(),
        openings.len() * 2,
        sims,
    );

    let report = run_players(&mut args, &device, &openings, mcts_config, cc_core::engine::MAX_TOTAL_PLIES)?;

    log::info!(
        "结果：A 胜 {} 和 {} 负 {}（共 {} 局）| A 得分率 {:.3} | A 相对 B：{:+.1} Elo",
        report.wins_a, report.draws, report.losses_a, report.games, report.score_a, report.elo_diff,
    );

    if let Some(path) = &args.report {
        write_report(path, &report, &args, sims)?;
        log::info!("报告已写入 {}", path.display());
    }
    if let Some(path) = &args.table {
        append_table_row(path, &report, &args)?;
        log::info!("战绩已追加到 {}", path.display());
    }
    Ok(())
}

/// 解析出 A/B 两个模型 .pt 的路径：显式路径 > 显式版本 > 自动取最近两版。
fn resolve_models(args: &Args) -> Result<(String, String)> {
    use cc_core::model_io::LocalModelStore;

    // ① 显式路径。
    match (&args.model_a, &args.model_b) {
        (Some(a), Some(b)) => return Ok((a.clone(), b.clone())),
        (None, None) => {}
        _ => anyhow::bail!("--model-a / --model-b 需成对给出（或都不传，自动取最近两版）"),
    }

    let store = LocalModelStore::new(&args.model_dir);

    // ② 显式版本号。
    match (args.version_a, args.version_b) {
        (Some(va), Some(vb)) => {
            return Ok((
                store.path_for(va).to_string_lossy().into_owned(),
                store.path_for(vb).to_string_lossy().into_owned(),
            ));
        }
        (None, None) => {}
        _ => anyhow::bail!("--version-a / --version-b 需成对给出"),
    }

    // ③ 自动：取最近两版，A=最新（挑战者），B=上一版（基准）。
    let versions = store.list_versions()?;
    if versions.len() < 2 {
        anyhow::bail!(
            "模型目录 {} 不足两个版本（找到 {} 个），无法自动对杀；\
             可用 --version-a/-b 或 --model-a/-b 指定",
            args.model_dir.display(),
            versions.len(),
        );
    }
    let n = versions.len();
    let (va, vb) = (versions[n - 1], versions[n - 2]);
    log::info!("自动选版：A = v{va}（最新） vs B = v{vb}（上一版）");
    Ok((
        store.path_for(va).to_string_lossy().into_owned(),
        store.path_for(vb).to_string_lossy().into_owned(),
    ))
}

fn run_players(
    args: &mut Args,
    device: &str,
    openings: &[Vec<cc_core::engine::Move>],
    mcts_config: MctsConfig,
    max_total_plies: u32,
) -> Result<MatchReport> {
    let (a, b) = resolve_models(args)?;
    let eval_a = eval::TorchEvaluator::load(&a, device)?;
    let eval_b = eval::TorchEvaluator::load(&b, device)?;
    log::info!("A = {a}\nB = {b}\ndevice = {device}");
    // 回填解析结果，供报告/战绩表记录实际对杀的两份模型。
    args.model_a = Some(a);
    args.model_b = Some(b);
    Ok(run_match(openings, &eval_a, &eval_b, mcts_config, max_total_plies))
}

fn write_report(path: &PathBuf, report: &MatchReport, args: &Args, sims: u32) -> Result<()> {
    let json = serde_json::json!({
        "games": report.games,
        "wins_a": report.wins_a,
        "draws": report.draws,
        "losses_a": report.losses_a,
        "score_a": report.score_a,
        "elo_diff_a_vs_b": report.elo_diff,
        "model_a": args.model_a,
        "model_b": args.model_b,
        "num_openings": args.num_openings,
        "sims": sims,
        "seed": args.seed,
    });
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    std::fs::write(path, serde_json::to_string_pretty(&json)?)
        .with_context(|| format!("写报告失败：{}", path.display()))?;
    Ok(())
}

/// 追加一行 CSV 战绩表（文件不存在时先写表头）：随训练多次跑 arena 即成趋势表。
fn append_table_row(path: &PathBuf, report: &MatchReport, args: &Args) -> Result<()> {
    use std::io::Write;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let is_new = !path.exists();
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .with_context(|| format!("打开战绩表失败：{}", path.display()))?;
    if is_new {
        writeln!(
            f,
            "ts_epoch,model_a,model_b,games,wins_a,draws,losses_a,score_a,elo_diff_a_vs_b"
        )?;
    }
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    writeln!(
        f,
        "{},{},{},{},{},{},{},{:.4},{:.1}",
        ts,
        args.model_a.as_deref().unwrap_or("-"),
        args.model_b.as_deref().unwrap_or("-"),
        report.games,
        report.wins_a,
        report.draws,
        report.losses_a,
        report.score_a,
        report.elo_diff,
    )?;
    Ok(())
}
