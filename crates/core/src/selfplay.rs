//! 单局自对弈（移植自 trainer/src/trainer/reference/selfplay.py 的 play_game）。
//!
//! 每手 mcts.run 取访问次数 → 记 (state, π) 样本 → 按温度选招 → 子树复用 advance →
//! 直到终局；终局后按真实胜负回填每条样本的 z。样本直接产出 store::Sample（state 量化
//! 成 uint8，π 以稀疏 (idx,val) 表示）。

use crate::encode::{encode, to_canonical_action};
use crate::engine::{Color, GameState, Move, Position};
use crate::mcts::{Evaluator, Mcts};
use crate::model_io::{quantize_state, Sample};
use rand::Rng;

struct Pending {
    state: Vec<u8>,
    pi_idx: Vec<i32>,
    pi_val: Vec<f32>,
    player: Color,
}

/// 跑一整局，返回带回填 z 的样本。`select_rng` 用于温度阶段按访问比例采样选招。
pub fn play_game<E, R, S>(
    mcts: &mut Mcts<R>,
    evaluator: &E,
    temperature_moves: u32,
    max_total_plies: u32,
    select_rng: &mut S,
) -> Vec<Sample>
where
    E: Evaluator,
    R: Rng,
    S: Rng,
{
    mcts.reset();
    let mut state = GameState::from_position(Position::starting());
    let mut pending: Vec<Pending> = Vec::new();

    let mut ply: u32 = 0;
    while ply < max_total_plies {
        let counts = mcts.run(state.clone(), evaluator);
        if counts.is_empty() {
            break; // 根节点终局，无可走
        }

        let player = state.position.side_to_move;
        let total: f32 = counts.iter().map(|(_, n)| *n as f32).sum();
        let mut pi_idx = Vec::with_capacity(counts.len());
        let mut pi_val = Vec::with_capacity(counts.len());
        for (m, n) in &counts {
            pi_idx.push(to_canonical_action(*m, player) as i32);
            pi_val.push(*n as f32 / total);
        }
        pending.push(Pending {
            state: quantize_state(&encode(&state)),
            pi_idx,
            pi_val,
            player,
        });

        let mv = select_move(&counts, ply, temperature_moves, select_rng);
        state = state.make_move(mv);
        mcts.advance(mv);
        ply += 1;
    }

    let winner = state.status().winner;
    pending
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

/// 前 temperature_moves 手按访问次数比例采样（τ=1），之后取 argmax（首个最大者）。
pub fn select_move<R: Rng>(
    counts: &[(Move, u32)],
    ply: u32,
    temperature_moves: u32,
    rng: &mut R,
) -> Move {
    if ply < temperature_moves {
        let total: u32 = counts.iter().map(|(_, n)| *n).sum();
        if total > 0 {
            let mut r = rng.gen_range(0..total);
            for (m, n) in counts {
                if r < *n {
                    return *m;
                }
                r -= *n;
            }
        }
        return counts.last().unwrap().0;
    }
    let mut best = counts[0].0;
    let mut best_n = counts[0].1;
    for (m, n) in counts {
        if *n > best_n {
            best_n = *n;
            best = *m;
        }
    }
    best
}
