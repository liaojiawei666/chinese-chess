//! datagen 自对弈数据生成器入口。
//!
//! 流程：读 run-config 并校验常量 → 起 num_workers 个 rayon 任务并行自对弈 →
//! 每 shard_games 局在 worker 线程序列化成一个 safetensors 稀疏分片，经有界 channel
//! 交给专门的写盘线程串行落盘（计算与 IO 解耦）→ 每局开局轮询 latest.json，版本变化则
//! 热加载模型。
//!
//! 背压两级：写盘线程在 pending > max_pending_shards 时等待 trainer 消费；其前的有界
//! channel 满时 worker 发送阻塞，从而把背压回传到生产端、避免内存/磁盘膨胀。

mod config;
mod game;
mod infer;

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::time::{Duration, Instant};

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

/// worker → 写盘线程的一条待写分片：已在 worker 线程并行序列化好的字节 + 目标文件名。
struct ShardMsg {
    name: String,
    bytes: Vec<u8>,
}

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

    let model_store = LocalModelStore::new(&run_config.datagen.model_dir);

    // 生产端按「样本数」收口：累计产出 ≥ total_samples 即停。各 worker 会把当前这局打完，
    // 自然超产一点（worst case ≈ num_workers × max_plies），即消费端预算所需的 slack。
    let produced = AtomicI64::new(0);
    let games = AtomicI64::new(0);
    let total_samples = run_config.total_samples as i64;
    let num_workers = run_config.datagen.num_workers.max(1);

    // 推理 actor：独占模型，跨 worker 聚合叶子请求后批量前向（scheme A）。
    // 当前模型权重版本（latest.json / model_NNNNNN.pt 的 version）走共享 AtomicI64：
    // actor 初始化/热加载后写入，worker 读它给样本分片命名（标记数据出自哪代模型）。
    let initial_model_version = model_store.get_version()?.unwrap_or(0);
    let model_version = Arc::new(AtomicI64::new(initial_model_version));
    let (tx, rx) = mpsc::channel::<EvalRequest>();
    let actor = {
        let model_version = Arc::clone(&model_version);
        let device = run_config.device.clone();
        let batch_size = run_config.selfplay.eval_batch_size.max(1);
        let timeout =
            Duration::from_micros((run_config.selfplay.inference_timeout_ms * 1000.0) as u64);
        std::thread::spawn(move || {
            infer::run_actor(rx, model_store, device, batch_size, timeout, model_version);
        })
    };

    // 写盘线程：独占 SampleStore，串行落盘并施加磁盘背压。worker 把序列化好的分片经
    // 有界 channel 交给它（channel 满则发送阻塞，背压回传到生产端），使计算线程不被
    // fsync / 背压等待卡住。序列化在各 worker 并行完成，写盘线程只做 IO。
    let (shard_tx, shard_rx) = mpsc::sync_channel::<ShardMsg>(num_workers);
    let writer = {
        let sample_store = LocalSampleStore::new(&run_config.datagen.samples_dir)?;
        let max_pending = run_config.datagen.max_pending_shards;
        std::thread::spawn(move || run_writer(shard_rx, sample_store, max_pending))
    };

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(num_workers)
        .build()?;

    // 进度上报线程与 worker 同期运行：每 5s 打印产量/速率，workers 收工后置 stop 退出。
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
                scope.spawn(move |_| {
                    if let Err(e) = worker_loop(
                        worker_id, run_config, produced, games, model_version, tx, shard_tx,
                    ) {
                        eprintln!("worker {worker_id} 出错：{e:#}");
                    }
                });
            }
        });

        stop_reporter.store(true, Ordering::SeqCst);
    });

    // 关掉主端发送句柄；worker 内的克隆也随 scope 结束而 drop，
    // 两个后台线程的 recv 随后返回 Err 退出，再 join。
    drop(tx);
    drop(shard_tx);
    if actor.join().is_err() {
        eprintln!("推理 actor 线程异常退出");
    }
    if writer.join().is_err() {
        eprintln!("写盘线程异常退出");
    }

    let pending = LocalSampleStore::new(&run_config.datagen.samples_dir)?.pending_count()?;
    println!("done. pending shards in {}: {pending}", run_config.datagen.samples_dir);
    Ok(())
}

/// 进度上报线程：周期打印 局数/样本数/均样本/瞬时与平均速率/进度，直到 stop 置位。
/// 用 250ms 小步轮询 stop，使收工时最多多等 250ms 即退出。
fn run_reporter(
    produced: &AtomicI64,
    games: &AtomicI64,
    total_samples: i64,
    stop: &AtomicBool,
    interval: Duration,
) {
    use std::io::Write;
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
        println!(
            "[datagen] {total_dt:>5.0}s | 局 {g} | 样本 {s} | 均 {per_game:.1}/局 | \
             瞬时 {inst_rate:.0} 样本/s（均 {avg_rate:.0}/s）| 进度 {s}/{total_samples} ({pct:.1}%)"
        );
        std::io::stdout().flush().ok();
        last_t = now;
        last_samples = s;
    }
}

/// 写盘线程主循环：串行写分片并施加磁盘背压，直到所有发送端关闭后退出。
fn run_writer(rx: mpsc::Receiver<ShardMsg>, sample_store: LocalSampleStore, max_pending: usize) {
    for msg in rx {
        // 背压：未消费分片过多时等待 trainer 消费，避免磁盘膨胀。
        while sample_store.pending_count().unwrap_or(0) > max_pending {
            std::thread::sleep(Duration::from_millis(200));
        }
        if let Err(e) = sample_store.put_shard(&msg.name, &msg.bytes) {
            eprintln!("写分片失败 {}：{e:#}", msg.name);
        }
    }
}

fn worker_loop(
    worker_id: usize,
    run_config: &RunConfig,
    produced: &AtomicI64,
    games: &AtomicI64,
    model_version: &AtomicI64,
    tx: mpsc::Sender<EvalRequest>,
    shard_tx: mpsc::SyncSender<ShardMsg>,
) -> Result<()> {
    let mcts_config = MctsConfig {
        n_simulations: run_config.mcts.n_simulations,
        c_puct: run_config.mcts.c_puct,
        dirichlet_alpha: run_config.mcts.dirichlet_alpha,
        dirichlet_epsilon: run_config.mcts.dirichlet_epsilon,
        collect_batch_size: run_config.mcts.collect_batch_size,
    };
    let temperature_moves = run_config.selfplay.temperature_moves;
    let max_total_plies = run_config.rules.max_total_plies;
    let shard_games = run_config.datagen.shard_games.max(1);
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
        games.fetch_add(1, Ordering::SeqCst);
        buffer.extend(samples);
        games_in_shard += 1;

        if games_in_shard >= shard_games {
            // 分片名用「当前模型权重版本」（actor 热加载后写入的共享值）。
            let v = model_version.load(Ordering::SeqCst);
            send_shard(&shard_tx, &mut buffer, v, worker_id, &mut seq)?;
            games_in_shard = 0;
        }
    }

    // 收尾：剩余不足一个分片的样本也写出。
    if !buffer.is_empty() {
        let v = model_version.load(Ordering::SeqCst);
        send_shard(&shard_tx, &mut buffer, v, worker_id, &mut seq)?;
    }
    Ok(())
}

/// 在 worker 线程序列化当前批次（天然并行），再经有界 channel 交给写盘线程。
/// channel 满时在此阻塞，即背压回传到生产端。
fn send_shard(
    shard_tx: &mpsc::SyncSender<ShardMsg>,
    buffer: &mut Vec<Sample>,
    version: i64,
    worker_id: usize,
    seq: &mut u64,
) -> Result<()> {
    let name = shard_name(version, worker_id, *seq);
    *seq += 1;
    let bytes = serialize_shard(buffer)?;
    buffer.clear();
    shard_tx
        .send(ShardMsg { name, bytes })
        .map_err(|_| anyhow::anyhow!("写盘线程已退出，无法提交分片"))?;
    Ok(())
}
