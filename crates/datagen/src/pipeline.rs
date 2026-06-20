//! 多游戏流水线：每个 worker 同时推进 N 局游戏，交替收集叶子请求，
//! 使 GPU 推理 actor 始终有足够请求凑 batch，消除 CPU/GPU 乒乓阻塞。
//!
//! 核心循环：
//! 1. 对所有非等待中的游戏调 `mcts.step()` / `init_root()` 推进搜索；
//! 2. 碰到 `NeedEval` 叶子时编码并发送到推理 actor channel；
//! 3. 批量收集所有回执，feed 回各游戏的 MCTS；
//! 4. 搜索完成的游戏记录样本、选招推进、终局则开新局。

use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::mpsc;

use anyhow::Result;
use rand::rngs::StdRng;
use rand::{RngCore, SeedableRng};

use cc_core::encode::{encode, to_canonical_action};
use cc_core::engine::{Color, GameState, Position};
use cc_core::infer::actor::{EvalInput, EvalReply, EvalRequest};
use cc_core::mcts::{Mcts, MctsConfig, StepResult};
use cc_core::model_io::{quantize_state, Sample};
use cc_core::selfplay::select_move;

use crate::ShardMsg;

struct PendingSample {
    state: Vec<u8>,
    pi_idx: Vec<i32>,
    pi_val: Vec<f32>,
    player: Color,
}

/// 等待回填的评估类型。
enum EvalKind {
    Root,
    Leaf,
}

struct GameSlot {
    mcts: Mcts<StdRng>,
    state: GameState,
    samples: Vec<PendingSample>,
    ply: u32,
    /// 当前位置是否已 init_root。
    root_initialized: bool,
    /// 等待推理结果的回执通道与类型。
    pending: Option<(EvalKind, mpsc::Receiver<EvalReply>)>,
    /// 缓存根节点编码（GPU eval 时产生），样本记录时复用，避免重复 encode。
    cached_root_encoding: Option<Vec<f32>>,
}

impl GameSlot {
    fn new(mcts_config: MctsConfig, rng: &mut StdRng) -> Self {
        let seed = rng.next_u64();
        GameSlot {
            mcts: Mcts::new(mcts_config, StdRng::seed_from_u64(seed)),
            state: GameState::from_position(Position::starting()),
            samples: Vec::new(),
            ply: 0,
            root_initialized: false,
            pending: None,
            cached_root_encoding: None,
        }
    }

    fn reset_game(&mut self, rng: &mut StdRng) {
        let seed = rng.next_u64();
        self.mcts = Mcts::new(*self.mcts.config(), StdRng::seed_from_u64(seed));
        self.state = GameState::from_position(Position::starting());
        self.samples.clear();
        self.ply = 0;
        self.root_initialized = false;
        self.cached_root_encoding = None;
    }
}

fn make_eval_input(state: &GameState) -> EvalInput {
    let input = encode(state);
    let moves = state.position.legal_moves();
    let perspective = state.position.side_to_move;
    let legal_ids: Vec<usize> = moves
        .iter()
        .map(|&m| to_canonical_action(m, perspective))
        .collect();
    EvalInput {
        input,
        moves,
        legal_ids,
    }
}

fn finalize_samples(samples: Vec<PendingSample>, winner: Option<Color>) -> Vec<Sample> {
    samples
        .into_iter()
        .map(|p| {
            let z = match winner {
                None => 0.0,
                Some(w) => {
                    if w == p.player {
                        1.0
                    } else {
                        -1.0
                    }
                }
            };
            Sample {
                state: p.state,
                pi_idx: p.pi_idx,
                pi_val: p.pi_val,
                z,
            }
        })
        .collect()
}

/// 多游戏流水线 worker 主循环。替代原有的单局串行 `worker_loop`。
#[allow(clippy::too_many_arguments)]
pub fn run_pipeline(
    worker_id: usize,
    games_per_worker: usize,
    mcts_config: MctsConfig,
    temperature_moves: u32,
    max_total_plies: u32,
    shard_games: u32,
    tx: mpsc::Sender<EvalRequest>,
    shard_tx: mpsc::SyncSender<ShardMsg>,
    produced: &AtomicI64,
    game_count: &AtomicI64,
    total_samples: i64,
    model_version: &AtomicI64,
    stop: &crate::cli::StopConditions,
    master_rng: &mut StdRng,
) -> Result<()> {
    let mut slots: Vec<GameSlot> = (0..games_per_worker)
        .map(|_| GameSlot::new(mcts_config, master_rng))
        .collect();

    let mut sample_buffer: Vec<Sample> = Vec::new();
    let mut games_in_shard: u32 = 0;
    let mut shard_seq: u64 = 0;

    loop {
        if produced.load(Ordering::SeqCst) >= total_samples {
            break;
        }
        if stop.should_stop(game_count.load(Ordering::SeqCst)) {
            break;
        }

        // Phase 1: 驱动所有非等待中的 slot 前进，直到各自需要一次 eval 或搜索完成。
        for slot in slots.iter_mut() {
            if slot.pending.is_some() {
                continue;
            }
            drive_slot(
                slot,
                &mcts_config,
                temperature_moves,
                max_total_plies,
                &tx,
                produced,
                game_count,
                &mut sample_buffer,
                &mut games_in_shard,
                master_rng,
            );
        }

        // Phase 2: 收集所有待回填的回执。
        let mut feed_list: Vec<(usize, EvalKind, EvalReply)> = Vec::new();
        for (i, slot) in slots.iter_mut().enumerate() {
            if let Some((kind, rx)) = slot.pending.take() {
                let reply = rx.recv().expect("推理 actor 已退出");
                feed_list.push((i, kind, reply));
            }
        }

        if feed_list.is_empty() {
            // 所有 slot 都没有 pending eval —— 说明全部搜索完成或已终止
            // 检查是否应该退出（produced >= total_samples 在上方已检查）
            break;
        }

        // Phase 3: 回填评估结果。
        for (i, kind, (priors, value)) in feed_list {
            match kind {
                EvalKind::Root => slots[i].mcts.feed_root_eval(priors, value),
                EvalKind::Leaf => slots[i].mcts.feed_eval(priors, value),
            }
        }

        // Phase 4: 写分片（攒够 shard_games 局）。
        if games_in_shard >= shard_games {
            let v = model_version.load(Ordering::SeqCst);
            let name = cc_core::model_io::shard_name(v, worker_id, shard_seq);
            shard_seq += 1;
            let bytes = cc_core::model_io::serialize_shard(&sample_buffer)?;
            sample_buffer.clear();
            games_in_shard = 0;
            shard_tx
                .send(ShardMsg { name, bytes })
                .map_err(|_| anyhow::anyhow!("写盘线程已退出"))?;
        }
    }

    // 收尾：剩余不足一个分片的样本也写出。
    if !sample_buffer.is_empty() {
        let v = model_version.load(Ordering::SeqCst);
        let name = cc_core::model_io::shard_name(v, worker_id, shard_seq);
        let bytes = cc_core::model_io::serialize_shard(&sample_buffer)?;
        shard_tx
            .send(ShardMsg { name, bytes })
            .map_err(|_| anyhow::anyhow!("写盘线程已退出"))?;
    }

    Ok(())
}

/// 驱动单个 slot 前进：循环调 init_root / step，直到碰到一次 NeedEval（发送请求后返回）
/// 或所有模拟已完成（处理选招/推进/开新局后继续循环）。
#[allow(clippy::too_many_arguments)]
fn drive_slot(
    slot: &mut GameSlot,
    mcts_config: &MctsConfig,
    temperature_moves: u32,
    max_total_plies: u32,
    tx: &mpsc::Sender<EvalRequest>,
    produced: &AtomicI64,
    game_count: &AtomicI64,
    sample_buffer: &mut Vec<Sample>,
    games_in_shard: &mut u32,
    rng: &mut StdRng,
) {
    loop {
        // 1. 初始化根节点（新位置首次进入时）
        if !slot.root_initialized {
            let need_eval = slot.mcts.init_root(slot.state.clone());
            slot.root_initialized = true;

            if slot.mcts.is_terminal() {
                complete_game(slot, produced, game_count, sample_buffer, games_in_shard, rng);
                continue;
            }

            if need_eval {
                let eval_input = {
                    let state = slot.mcts.root_state();
                    make_eval_input(state)
                };
                slot.cached_root_encoding = Some(eval_input.input.clone());
                send_eval(tx, eval_input, EvalKind::Root, &mut slot.pending);
                return;
            }
        }

        // 2. 检查搜索是否完成
        if slot.mcts.simulations_done() >= mcts_config.n_simulations {
            let counts = slot.mcts.visit_counts();
            if counts.is_empty() {
                complete_game(slot, produced, game_count, sample_buffer, games_in_shard, rng);
                continue;
            }

            // 记录样本
            let player = slot.state.position.side_to_move;
            let total: f32 = counts.iter().map(|(_, n)| *n as f32).sum();
            let mut pi_idx = Vec::with_capacity(counts.len());
            let mut pi_val = Vec::with_capacity(counts.len());
            for (m, n) in &counts {
                pi_idx.push(to_canonical_action(*m, player) as i32);
                pi_val.push(*n as f32 / total);
            }
            let encoded = slot
                .cached_root_encoding
                .take()
                .unwrap_or_else(|| encode(&slot.state));
            slot.samples.push(PendingSample {
                state: quantize_state(&encoded),
                pi_idx,
                pi_val,
                player,
            });

            // 选招、推进
            let mv = select_move(&counts, slot.ply, temperature_moves, rng);
            slot.state = slot.state.make_move(mv);
            slot.mcts.advance(mv);
            slot.ply += 1;
            slot.root_initialized = false;

            // 检查游戏是否结束
            if slot.ply >= max_total_plies || slot.state.status().is_terminal {
                complete_game(slot, produced, game_count, sample_buffer, games_in_shard, rng);
            }
            continue;
        }

        // 3. 一步模拟
        let eval_input = match slot.mcts.step() {
            StepResult::NeedEval { state } => Some(make_eval_input(state)),
            StepResult::Done => None,
        };

        if let Some(input) = eval_input {
            send_eval(tx, input, EvalKind::Leaf, &mut slot.pending);
            return;
        }
        // Done（终局叶）：sim 计数已自增，继续循环
    }
}

fn send_eval(
    tx: &mpsc::Sender<EvalRequest>,
    input: EvalInput,
    kind: EvalKind,
    pending: &mut Option<(EvalKind, mpsc::Receiver<EvalReply>)>,
) {
    let (reply_tx, reply_rx) = mpsc::sync_channel(1);
    tx.send(EvalRequest {
        input,
        reply: reply_tx,
    })
    .expect("推理 actor 已退出");
    *pending = Some((kind, reply_rx));
}

fn complete_game(
    slot: &mut GameSlot,
    produced: &AtomicI64,
    game_count: &AtomicI64,
    sample_buffer: &mut Vec<Sample>,
    games_in_shard: &mut u32,
    rng: &mut StdRng,
) {
    let winner = slot.state.status().winner;
    let game_samples = finalize_samples(std::mem::take(&mut slot.samples), winner);
    produced.fetch_add(game_samples.len() as i64, Ordering::SeqCst);
    game_count.fetch_add(1, Ordering::SeqCst);
    sample_buffer.extend(game_samples);
    *games_in_shard += 1;

    slot.reset_game(rng);
}
