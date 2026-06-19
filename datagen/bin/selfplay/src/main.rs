//! datagen 自对弈数据生成器入口。
//!
//! 流程：读 run-config 并校验常量 → 起 num_workers 个 rayon 任务并行自对弈 →
//! 每 shard_games 局打成一个 safetensors 稀疏分片，背压（pending > max_pending_shards）
//! 下节流写盘 → 每局开局轮询 latest.json，版本变化则热加载模型。

mod config;
mod game;
mod infer;

use std::path::PathBuf;
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use rand::rngs::StdRng;
use rand::{RngCore, SeedableRng};

use config::RunConfig;
use game::play_game;
use infer::{BatchedEvaluator, EvalRequest};
use mcts::{Mcts, MctsConfig};
use store::{
    serialize_shard, shard_name, LocalModelStore, LocalSampleStore, ModelStore, Sample,
    SampleStore,
};

fn main() -> Result<()> {
    let path = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("data/config/run-config.local.json"));

    let run_config = RunConfig::load(&path)?;
    println!(
        "loaded run-config: profile={} device={} workers={} n_sim={} total_samples={}",
        run_config.profile,
        run_config.device,
        run_config.datagen.num_workers,
        run_config.mcts.n_simulations,
        run_config.total_samples,
    );

    let sample_store = LocalSampleStore::new(&run_config.datagen.samples_dir)?;
    let model_store = LocalModelStore::new(&run_config.datagen.model_dir);

    // 生产端按「样本数」收口：累计产出 ≥ total_samples 即停。各 worker 会把当前这局打完，
    // 自然超产一点（worst case ≈ num_workers × max_plies），即消费端预算所需的 slack。
    let produced = AtomicI64::new(0);
    let num_workers = run_config.datagen.num_workers.max(1);

    // 推理 actor：独占模型，跨 worker 聚合叶子请求后批量前向（scheme A）。
    // 版本走共享 AtomicI64：actor 初始化/热加载时写，worker 读它给分片命名。
    let initial_version = model_store.get_version()?.unwrap_or(0);
    let version = Arc::new(AtomicI64::new(initial_version));
    let (tx, rx) = mpsc::channel::<EvalRequest>();
    let actor = {
        let version = Arc::clone(&version);
        let device = run_config.device.clone();
        let batch_size = run_config.selfplay.eval_batch_size.max(1);
        let timeout =
            Duration::from_micros((run_config.selfplay.inference_timeout_ms * 1000.0) as u64);
        std::thread::spawn(move || {
            infer::run_actor(rx, model_store, device, batch_size, timeout, version);
        })
    };

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(num_workers)
        .build()?;

    pool.scope(|scope| {
        for worker_id in 0..num_workers {
            let run_config = &run_config;
            let sample_store = &sample_store;
            let produced = &produced;
            let version = &version;
            let tx = tx.clone();
            scope.spawn(move |_| {
                if let Err(e) =
                    worker_loop(worker_id, run_config, sample_store, produced, version, tx)
                {
                    eprintln!("worker {worker_id} 出错：{e:#}");
                }
            });
        }
    });

    // 关掉主端发送句柄；worker 内的克隆也随 scope 结束而 drop，
    // actor 的 recv 随后返回 Err 退出，再 join。
    drop(tx);
    if actor.join().is_err() {
        eprintln!("推理 actor 线程异常退出");
    }

    let pending = sample_store.pending_count()?;
    println!("done. pending shards in {}: {pending}", run_config.datagen.samples_dir);
    Ok(())
}

fn worker_loop(
    worker_id: usize,
    run_config: &RunConfig,
    sample_store: &dyn SampleStore,
    produced: &AtomicI64,
    version: &AtomicI64,
    tx: mpsc::Sender<EvalRequest>,
) -> Result<()> {
    let mcts_config = MctsConfig {
        n_simulations: run_config.mcts.n_simulations,
        c_puct: run_config.mcts.c_puct,
        dirichlet_alpha: run_config.mcts.dirichlet_alpha,
        dirichlet_epsilon: run_config.mcts.dirichlet_epsilon,
    };
    let temperature_moves = run_config.selfplay.temperature_moves;
    let max_total_plies = run_config.rules.max_total_plies;
    let shard_games = run_config.datagen.shard_games.max(1);
    let max_pending = run_config.datagen.max_pending_shards;
    let total_samples = run_config.total_samples as i64;

    // 每个 worker 一个主 RNG（按 worker_id 派生种子，避免各 worker 同序）。
    let mut master_rng = StdRng::seed_from_u64(0x5eed_0000 ^ worker_id as u64);

    // 评估走推理 actor：每局复用同一句柄；模型与热加载都在 actor 侧统一管理。
    let evaluator = BatchedEvaluator::new(tx);

    let mut buffer: Vec<Sample> = Vec::new();
    let mut games_in_shard: u32 = 0;
    let mut seq: u64 = 0;

    loop {
        // 按样本数收口：已累计产出 ≥ total_samples 就停（各 worker 把当前局打完即可）。
        if produced.load(Ordering::SeqCst) >= total_samples {
            break;
        }

        let seed = master_rng.next_u64();
        let mut mcts = Mcts::new(&evaluator, mcts_config, StdRng::seed_from_u64(seed));
        let samples = play_game(&mut mcts, temperature_moves, max_total_plies, &mut master_rng);

        produced.fetch_add(samples.len() as i64, Ordering::SeqCst);
        buffer.extend(samples);
        games_in_shard += 1;

        if games_in_shard >= shard_games {
            // 分片名用「当前模型版本」（actor 热加载后写入的共享值）。
            let v = version.load(Ordering::SeqCst);
            flush_shard(sample_store, &mut buffer, v, worker_id, &mut seq, max_pending)?;
            games_in_shard = 0;
        }
    }

    // 收尾：剩余不足一个分片的样本也写出。
    if !buffer.is_empty() {
        let v = version.load(Ordering::SeqCst);
        flush_shard(sample_store, &mut buffer, v, worker_id, &mut seq, max_pending)?;
    }
    Ok(())
}

fn flush_shard(
    sample_store: &dyn SampleStore,
    buffer: &mut Vec<Sample>,
    version: i64,
    worker_id: usize,
    seq: &mut u64,
    max_pending: usize,
) -> Result<()> {
    // 背压：未消费分片过多时等待 trainer 消费，避免磁盘膨胀。
    while sample_store.pending_count()? > max_pending {
        std::thread::sleep(Duration::from_millis(200));
    }
    let name = shard_name(version, worker_id, *seq);
    *seq += 1;
    let bytes = serialize_shard(buffer)?;
    sample_store.put_shard(&name, &bytes)?;
    buffer.clear();
    Ok(())
}
