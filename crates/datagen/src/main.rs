//! datagen 自对弈数据生成器入口。

mod cli;
mod pipeline;

use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use clap::Parser;
use rand::rngs::StdRng;
use rand::SeedableRng;

use cc_core::infer::actor::EvalRequest;
use cc_core::mcts::MctsConfig;
use cc_core::model_io::{LocalModelStore, LocalSampleStore, ModelStore, SampleStore};

use cli::{Cli, StopConditions};

/// worker → 写盘线程的一条待写分片。
pub(crate) struct ShardMsg {
    pub name: String,
    pub bytes: Vec<u8>,
}

fn main() -> Result<()> {
    env_logger::init();

    let cli = Cli::parse();
    let run_config = cli::load_config(&cli)?;
    let stop = StopConditions::from_cli(&cli);

    log::info!(
        "loaded config: profile={} device={} workers={} n_sim={} total_samples={}",
        run_config.profile,
        run_config.device,
        run_config.datagen.num_workers,
        run_config.mcts.n_simulations,
        run_config.total_samples,
    );
    if stop.deadline.is_some() || stop.max_games.is_some() {
        log::info!(
            "perf stop: duration_secs={:?} max_games={:?}",
            cli.duration_secs, cli.max_games
        );
    }

    let model_store = LocalModelStore::new(&run_config.datagen.model_dir);
    let produced = AtomicI64::new(0);
    let games = AtomicI64::new(0);
    let total_samples = run_config.total_samples as i64;
    let num_workers = run_config.datagen.num_workers.max(1);

    let initial_model_version = model_store.get_version()?.unwrap_or(0);
    let model_version = Arc::new(AtomicI64::new(initial_model_version));
    let (tx, rx) = mpsc::channel::<EvalRequest>();
    let actor = {
        let model_version = Arc::clone(&model_version);
        let device = run_config.device.clone();
        let batch_size = run_config.datagen.eval_batch_size.max(1);
        let timeout =
            Duration::from_micros((run_config.datagen.inference_timeout_ms * 1000.0) as u64);
        std::thread::spawn(move || {
            cc_core::infer::actor::run_actor(rx, model_store, device, batch_size, timeout, model_version);
        })
    };

    let (shard_tx, shard_rx) = mpsc::sync_channel::<ShardMsg>(num_workers);
    let writer = {
        let sample_store = LocalSampleStore::new(&run_config.datagen.samples_dir)?;
        let max_pending = run_config.datagen.max_pending_shards;
        std::thread::spawn(move || run_writer(shard_rx, sample_store, max_pending))
    };

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(num_workers)
        .build()?;

    let stop_reporter = AtomicBool::new(false);
    std::thread::scope(|s| {
        s.spawn(|| {
            run_reporter(&produced, &games, total_samples, &stop_reporter, Duration::from_secs(5));
        });

        pool.scope(|scope| {
            for worker_id in 0..num_workers {
                let run_config = &run_config;
                let produced = &produced;
                let games = &games;
                let model_version = &model_version;
                let tx = tx.clone();
                let shard_tx = shard_tx.clone();
                let stop = &stop;
                scope.spawn(move |_| {
                    if let Err(e) = worker_loop(
                        worker_id, run_config, produced, games, model_version, tx, shard_tx, stop,
                    ) {
                        log::error!("worker {worker_id} 出错：{e:#}");
                    }
                });
            }
        });

        stop_reporter.store(true, Ordering::SeqCst);
    });

    drop(tx);
    drop(shard_tx);
    if actor.join().is_err() {
        log::error!("推理 actor 线程异常退出");
    }
    if writer.join().is_err() {
        log::error!("写盘线程异常退出");
    }

    let pending = LocalSampleStore::new(&run_config.datagen.samples_dir)?.pending_count()?;
    log::info!("done. pending shards in {}: {pending}", run_config.datagen.samples_dir);
    Ok(())
}

fn run_reporter(
    produced: &AtomicI64,
    games: &AtomicI64,
    total_samples: i64,
    stop: &AtomicBool,
    interval: Duration,
) {
    use std::time::Instant;

    let start = Instant::now();
    let poll = Duration::from_millis(250);
    let mut last_t = start;
    let mut last_samples: i64 = 0;
    let mut since_report = Duration::ZERO;

    while !stop.load(Ordering::Relaxed) {
        std::thread::sleep(poll);
        since_report += poll;
        if since_report < interval {
            continue;
        }
        since_report = Duration::ZERO;

        let now = Instant::now();
        let s = produced.load(Ordering::Relaxed);
        let g = games.load(Ordering::Relaxed);
        let inst_dt = now.duration_since(last_t).as_secs_f64().max(1e-6);
        let total_dt = now.duration_since(start).as_secs_f64().max(1e-6);
        let inst_rate = (s - last_samples) as f64 / inst_dt;
        let avg_rate = s as f64 / total_dt;
        let per_game = if g > 0 { s as f64 / g as f64 } else { 0.0 };
        let pct = if total_samples > 0 {
            100.0 * s as f64 / total_samples as f64
        } else {
            0.0
        };
        log::info!(
            "[datagen] {total_dt:>5.0}s | 局 {g} | 样本 {s} | 均 {per_game:.1}/局 | \
             瞬时 {inst_rate:.0} 样本/s（均 {avg_rate:.0}/s）| 进度 {s}/{total_samples} ({pct:.1}%)"
        );
        last_t = now;
        last_samples = s;
    }
}

fn run_writer(rx: mpsc::Receiver<ShardMsg>, sample_store: LocalSampleStore, max_pending: usize) {
    for msg in rx {
        while sample_store.pending_count().unwrap_or(0) > max_pending {
            std::thread::sleep(Duration::from_millis(200));
        }
        if let Err(e) = sample_store.put_shard(&msg.name, &msg.bytes) {
            log::error!("写分片失败 {}：{e:#}", msg.name);
        }
    }
}

fn worker_loop(
    worker_id: usize,
    run_config: &cc_core::config::RunConfig,
    produced: &AtomicI64,
    games: &AtomicI64,
    model_version: &AtomicI64,
    tx: mpsc::Sender<EvalRequest>,
    shard_tx: mpsc::SyncSender<ShardMsg>,
    stop: &StopConditions,
) -> Result<()> {
    let mcts_config = MctsConfig {
        n_simulations: run_config.mcts.n_simulations,
        c_puct: run_config.mcts.c_puct,
        dirichlet_alpha: run_config.mcts.dirichlet_alpha,
        dirichlet_epsilon: run_config.mcts.dirichlet_epsilon,
        collect_batch_size: 1,
    };
    let temperature_moves = run_config.mcts.temperature_moves;
    let max_total_plies = cc_core::engine::MAX_TOTAL_PLIES;
    let shard_games = run_config.datagen.shard_games.max(1);
    let total_samples = run_config.total_samples as i64;
    let games_per_worker = run_config.datagen.games_per_worker.max(1);

    let mut master_rng = StdRng::seed_from_u64(0x5eed_0000 ^ worker_id as u64);

    pipeline::run_pipeline(
        worker_id,
        games_per_worker,
        mcts_config,
        temperature_moves,
        max_total_plies,
        shard_games,
        tx,
        shard_tx,
        produced,
        games,
        total_samples,
        model_version,
        stop,
        &mut master_rng,
    )
}
