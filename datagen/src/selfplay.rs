use engine::{encode::encode_state, game::GameState};
use engine::mcts::{Mcts, MctsConfig};
use rand::{Rng, SeedableRng};
use tokio::sync::mpsc;
use tracing::debug;

use crate::config::DatagenConfig;
use crate::data::{finalize_samples, PendingSample, TrainingSample};
use crate::evaluator::ChannelEvaluator;

/// 温度采样：根据 visit counts 和温度参数选择走法。
/// tau = 0 时退化为 argmax；tau > 0 时按 N^(1/tau) 概率采样。
fn sample_with_temperature(
    visit_counts: &[(engine::Move, u32)],
    tau: f32,
    rng: &mut impl Rng,
) -> engine::Move {
    if tau < 1e-6 {
        return Mcts::best_move(visit_counts);
    }

    let inv_tau = 1.0 / tau;
    let weights: Vec<f32> = visit_counts
        .iter()
        .map(|(_, c)| (*c as f32).powf(inv_tau))
        .collect();
    let total: f32 = weights.iter().sum();
    if total < 1e-8 {
        return visit_counts[0].0;
    }

    let mut r = rng.gen::<f32>() * total;
    for (i, w) in weights.iter().enumerate() {
        r -= w;
        if r <= 0.0 {
            return visit_counts[i].0;
        }
    }
    visit_counts.last().unwrap().0
}

/// 运行一局完整的自我对弈，返回训练样本。
async fn play_one_game(
    request_tx: mpsc::Sender<crate::evaluator::EvalRequest>,
    mcts_config: MctsConfig,
    temperature_moves: usize,
) -> Vec<TrainingSample> {
    let game = GameState::start_pos();
    let evaluator = Box::new(ChannelEvaluator::new(request_tx));
    let mut mcts = Mcts::new(game, mcts_config, evaluator);
    let mut rng = rand::rngs::SmallRng::from_entropy();

    let mut pending_samples: Vec<PendingSample> = Vec::new();
    let mut move_count = 0u32;

    loop {
        if mcts.game().status.is_terminal {
            break;
        }

        let state = encode_state(mcts.game());
        let side_to_move = mcts.game().board.side_to_move;
        let visit_counts = mcts.run().await;

        pending_samples.push(PendingSample {
            state,
            policy: mcts.policy_target(),
            side_to_move,
        });

        // 前 temperature_moves 步用高温探索，之后用低温（接近 argmax）
        let tau = if (move_count as usize) < temperature_moves {
            1.0
        } else {
            0.1
        };
        let mv = sample_with_temperature(&visit_counts, tau, &mut rng);
        mcts.advance(mv);
        move_count += 1;
    }

    let winner = mcts.game().status.winner;
    debug!(
        "game finished in {move_count} moves, winner={winner:?}, reason={:?}",
        mcts.game().status.reason
    );

    finalize_samples(pending_samples, winner)
}

/// 自我对弈 worker：循环打局，通过 game_tx 发送完成的训练样本。
pub async fn selfplay_worker(
    worker_id: usize,
    request_tx: mpsc::Sender<crate::evaluator::EvalRequest>,
    game_tx: mpsc::Sender<Vec<TrainingSample>>,
    config: DatagenConfig,
) {
    let mcts_config = config.mcts_config();

    let mut games_played = 0u32;
    loop {
        let samples = play_one_game(
            request_tx.clone(),
            mcts_config.clone(),
            config.mcts.temperature_moves,
        )
        .await;

        games_played += 1;
        debug!(
            "worker {worker_id}: game #{games_played} done, {n} samples",
            n = samples.len()
        );

        if game_tx.send(samples).await.is_err() {
            break;
        }
    }
}
