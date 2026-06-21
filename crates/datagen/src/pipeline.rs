//! 多游戏流水线：每个 worker 同时推进 N 局游戏，一轮 eval 打成一包发给 actor，
//! worker 只阻塞一次 recv，避免逐局发送/收包导致 actor 与 worker 互相等待。

use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::mpsc;

use anyhow::Result;
use rand::rngs::StdRng;
use rand::{RngCore, SeedableRng};

use cc_core::encode::{encode, to_canonical_action};
use cc_core::engine::{Color, GameState, Position};
use cc_core::infer::actor::{send_worker_batch, EvalInput, EvalReply, InferRequest};
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
    root_initialized: bool,
    cached_root_encoding: Option<Vec<f32>>,
}

/// 同一 worker 一轮待回填的 eval（一次 WorkerBatch 对应一次 recv）。
struct PendingWave {
    items: Vec<(usize, EvalKind)>,
    rx: mpsc::Receiver<Vec<EvalReply>>,
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

/// 多游戏流水线 worker 主循环。
#[allow(clippy::too_many_arguments)]
pub fn run_pipeline(
    worker_id: usize,
    games_per_worker: usize,
    mcts_config: MctsConfig,
    temperature_moves: u32,
    max_total_plies: u32,
    shard_games: u32,
    tx: mpsc::Sender<InferRequest>,
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
    let mut pending_wave: Option<PendingWave> = None;

    loop {
        if produced.load(Ordering::SeqCst) >= total_samples {
            break;
        }
        if stop.should_stop(game_count.load(Ordering::SeqCst)) {
            break;
        }

        // Phase 1: 若无在途 batch，驱动各 slot 并一次性打包发送 WorkerBatch。
        if pending_wave.is_none() {
            let mut wave_items: Vec<(usize, EvalKind)> = Vec::new();
            let mut wave_inputs: Vec<EvalInput> = Vec::new();

            for (i, slot) in slots.iter_mut().enumerate() {
                if let Some((kind, input)) = drive_slot_until_eval(
                    slot,
                    &mcts_config,
                    temperature_moves,
                    max_total_plies,
                    produced,
                    game_count,
                    &mut sample_buffer,
                    &mut games_in_shard,
                    master_rng,
                ) {
                    wave_items.push((i, kind));
                    wave_inputs.push(input);
                }
            }

            if !wave_inputs.is_empty() {
                let rx = send_worker_batch(&tx, wave_inputs);
                pending_wave = Some(PendingWave {
                    items: wave_items,
                    rx,
                });
            }
        }

        // Phase 2: 一次 recv 收回本轮全部 eval。
        let Some(wave) = pending_wave.take() else {
            break;
        };
        let replies = wave.rx.recv().expect("推理 actor 已退出");
        assert_eq!(
            wave.items.len(),
            replies.len(),
            "WorkerBatch 回执数量与请求不一致"
        );

        // Phase 3: 回填评估结果。
        for ((i, kind), reply) in wave.items.into_iter().zip(replies) {
            match kind {
                EvalKind::Root => slots[i].mcts.feed_root_eval(reply.0, reply.1),
                EvalKind::Leaf => slots[i].mcts.feed_eval(reply.0, reply.1),
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

/// 驱动单个 slot 直到需要一次 eval，或本 slot 在本轮无可推进的工作。
fn drive_slot_until_eval(
    slot: &mut GameSlot,
    mcts_config: &MctsConfig,
    temperature_moves: u32,
    max_total_plies: u32,
    produced: &AtomicI64,
    game_count: &AtomicI64,
    sample_buffer: &mut Vec<Sample>,
    games_in_shard: &mut u32,
    rng: &mut StdRng,
) -> Option<(EvalKind, EvalInput)> {
    loop {
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
                return Some((EvalKind::Root, eval_input));
            }
        }

        if slot.mcts.simulations_done() >= mcts_config.n_simulations {
            let counts = slot.mcts.visit_counts();
            if counts.is_empty() {
                complete_game(slot, produced, game_count, sample_buffer, games_in_shard, rng);
                continue;
            }

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

            let mv = select_move(&counts, slot.ply, temperature_moves, rng);
            slot.state = slot.state.make_move(mv);
            slot.mcts.advance(mv);
            slot.ply += 1;
            slot.root_initialized = false;

            if slot.ply >= max_total_plies || slot.state.status().is_terminal {
                complete_game(slot, produced, game_count, sample_buffer, games_in_shard, rng);
            }
            continue;
        }

        let eval_input = match slot.mcts.step() {
            StepResult::NeedEval { state } => Some(make_eval_input(state)),
            StepResult::Done => None,
        };

        if let Some(input) = eval_input {
            return Some((EvalKind::Leaf, input));
        }
    }
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
