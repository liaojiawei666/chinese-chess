mod config;
mod data;
mod evaluator;
mod inference;
mod selfplay;

use std::time::{Duration, Instant};

use anyhow::Result;
use tokio::sync::mpsc;
use tracing::info;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();

    let config_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "config.json".to_string());
    let config = config::DatagenConfig::from_file(&config_path)?;

    info!("datagen config: {config:#?}");

    let (model_path, initial_gen) = config::wait_for_model(&config.datagen.models_dir)?;
    let session = inference::load_onnx_model(model_path.to_str().unwrap())?;

    let (eval_tx, eval_rx) = mpsc::channel(config.datagen.eval_batch_size * 4);

    let (game_tx, mut game_rx) = mpsc::channel::<Vec<data::TrainingSample>>(
        config.datagen.num_parallel_games * 2,
    );

    let batch_timeout =
        Duration::from_secs_f64(config.datagen.inference_timeout_ms / 1000.0);

    let inference_handle = tokio::spawn(inference::inference_loop(
        eval_rx,
        session,
        config.datagen.eval_batch_size,
        batch_timeout,
        config.datagen.models_dir.clone(),
        initial_gen,
    ));

    let num_workers = config.datagen.num_parallel_games;
    let mut worker_handles = Vec::with_capacity(num_workers);
    for i in 0..num_workers {
        let handle = tokio::spawn(selfplay::selfplay_worker(
            i,
            eval_tx.clone(),
            game_tx.clone(),
            config.clone(),
        ));
        worker_handles.push(handle);
    }
    drop(game_tx);
    drop(eval_tx);

    info!("launched {num_workers} selfplay workers");

    let stats = collect_samples(&mut game_rx, &config).await?;

    for h in worker_handles {
        h.abort();
    }
    inference_handle.abort();

    info!(
        "datagen complete: {} games, {} samples in {:.1}s",
        stats.games_completed,
        stats.total_samples,
        stats.elapsed.as_secs_f64(),
    );

    Ok(())
}

struct CollectStats {
    games_completed: u64,
    total_samples: u64,
    elapsed: Duration,
}

async fn collect_samples(
    game_rx: &mut mpsc::Receiver<Vec<data::TrainingSample>>,
    config: &config::DatagenConfig,
) -> Result<CollectStats> {
    let mut shard_writer = data::ShardWriter::new(
        config.datagen.samples_dir.to_string_lossy().to_string(),
        config.datagen.shard_size,
    );

    let start_time = Instant::now();
    let mut games_completed = 0u64;
    let mut total_game_samples = 0u64;
    let target_samples = config.total_samples;

    let mut last_log_time = start_time;
    let mut last_log_games = 0u64;
    let mut last_log_samples = 0u64;

    while let Some(samples) = game_rx.recv().await {
        let game_sample_count = samples.len() as u64;
        shard_writer.add_game_samples(samples)?;
        games_completed += 1;
        total_game_samples += game_sample_count;

        if games_completed % 100 == 0 {
            let now = Instant::now();
            let current_samples = shard_writer.total_samples();
            let pct = current_samples as f64 / target_samples as f64 * 100.0;

            let interval = now.duration_since(last_log_time).as_secs_f64();
            let interval_games = games_completed - last_log_games;
            let interval_samples = current_samples - last_log_samples;
            let interval_gps = interval_games as f64 / interval.max(0.001);
            let interval_spg = if interval_games > 0 {
                interval_samples as f64 / interval_games as f64
            } else {
                0.0
            };

            let elapsed = now.duration_since(start_time).as_secs_f64();
            let overall_gps = games_completed as f64 / elapsed.max(0.001);
            let overall_spg = total_game_samples as f64 / games_completed as f64;

            info!(
                "[progress] {games_completed} games | \
                 {current_samples}/{target_samples} samples ({pct:.1}%)"
            );
            info!(
                "  last {interval_games}: {interval_gps:.1} games/s, \
                 {interval_spg:.1} samples/game, {interval:.1}s"
            );
            info!(
                "  overall: {overall_gps:.1} games/s, \
                 {overall_spg:.1} samples/game"
            );

            last_log_time = now;
            last_log_games = games_completed;
            last_log_samples = current_samples;
        }

        if shard_writer.total_samples() >= target_samples {
            info!(
                "target reached ({target_samples} samples), shutting down"
            );
            break;
        }
    }

    shard_writer.flush()?;

    Ok(CollectStats {
        games_completed,
        total_samples: shard_writer.total_samples(),
        elapsed: start_time.elapsed(),
    })
}
