//! 确定性开局生成 + 冻结。
//!
//! 关噪声（ε=0）后，固定起始局面下 MCTS 是确定性的，换种子也没用；多样性只能来自
//! **起始局面**。这里用均匀评估器（不依赖训练参数）跑浅层 MCTS，按温度对访问分布采样
//! 走 k 步得到一个开局，唯一随机源是「按分布掷骰子」，由固定种子锁死 → 可复现。
//!
//! 生成结果冻结到 JSON（move 序列，保留历史以便对局正确判重复/长将）；文件已存在则直接
//! 加载，保证每代挑战者都在同一批固定开局上受审、跨时间可比。

use std::collections::HashSet;
use std::path::Path;

use anyhow::{Context, Result};
use engine::{GameState, Move, Position};
use mcts::{Mcts, MctsConfig};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

use crate::eval::UniformEvaluator;

#[derive(Serialize, Deserialize)]
struct OpeningsFile {
    seed: u64,
    plies: u32,
    temperature: f64,
    sims: u32,
    /// 每个开局是一串 [sx, sy, tx, ty] 走法。
    openings: Vec<Vec<[i32; 4]>>,
}

/// 文件存在则加载，否则生成并冻结。返回每个开局的走法序列（从起始局面重放）。
pub fn load_or_generate(
    path: &Path,
    num: usize,
    plies: u32,
    temperature: f64,
    sims: u32,
    seed: u64,
) -> Result<Vec<Vec<Move>>> {
    if path.exists() {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("读取开局文件失败：{}", path.display()))?;
        let file: OpeningsFile = serde_json::from_str(&text).context("解析开局文件失败")?;
        return Ok(file
            .openings
            .into_iter()
            .map(|line| line.into_iter().map(arr_to_move).collect())
            .collect());
    }

    let openings = generate(num, plies, temperature, sims, seed);
    let file = OpeningsFile {
        seed,
        plies,
        temperature,
        sims,
        openings: openings
            .iter()
            .map(|line| line.iter().map(|&m| move_to_arr(m)).collect())
            .collect(),
    };
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    std::fs::write(path, serde_json::to_string_pretty(&file)?)
        .with_context(|| format!("写开局文件失败：{}", path.display()))?;
    Ok(openings)
}

/// 生成 num 个互不相同的开局（每个 plies 步）。固定种子可复现。
pub fn generate(num: usize, plies: u32, temperature: f64, sims: u32, seed: u64) -> Vec<Vec<Move>> {
    let config = MctsConfig {
        n_simulations: sims,
        c_puct: 1.5,
        dirichlet_alpha: 0.3,
        dirichlet_epsilon: 0.0, // 关噪声：走法分布确定，随机只来自温度采样
        collect_batch_size: 1,
    };
    let eval = UniformEvaluator;

    let mut openings: Vec<Vec<Move>> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    let max_attempts = (num as u64).saturating_mul(200).max(1000);

    for attempt in 0..max_attempts {
        if openings.len() >= num {
            break;
        }
        // 每个开局一个独立子种子（唯一随机源）；MCTS 自身 ε=0 不消费 rng。
        let mut sample_rng =
            StdRng::seed_from_u64(seed ^ attempt.wrapping_mul(0x9e37_79b9_7f4a_7c15));
        let mut mcts = Mcts::new(&eval, config, StdRng::seed_from_u64(seed));
        let mut state = GameState::from_position(Position::starting());
        let mut line: Vec<Move> = Vec::with_capacity(plies as usize);

        let mut ok = true;
        for _ in 0..plies {
            let counts = mcts.run(state.clone());
            if counts.is_empty() {
                ok = false; // 开局阶段就终局，丢弃
                break;
            }
            let mv = sample_by_counts(&counts, temperature, &mut sample_rng);
            line.push(mv);
            state = state.make_move(mv);
            mcts.advance(mv);
        }
        if !ok {
            continue;
        }
        let key = format!("{}|{}", state.position.to_fen(), state.history.len());
        if seen.insert(key) {
            openings.push(line);
        }
    }
    openings
}

/// 按 p(a) ∝ N(a)^(1/τ) 采样；τ→0 退化为 argmax（首个最大）。
fn sample_by_counts<R: Rng>(counts: &[(Move, u32)], temperature: f64, rng: &mut R) -> Move {
    if temperature <= 1e-6 {
        let mut best = counts[0];
        for &(m, n) in counts {
            if n > best.1 {
                best = (m, n);
            }
        }
        return best.0;
    }
    let inv = 1.0 / temperature;
    let weights: Vec<f64> = counts.iter().map(|(_, n)| (*n as f64).powf(inv)).collect();
    let total: f64 = weights.iter().sum();
    let mut r = rng.gen::<f64>() * total;
    for (i, (m, _)) in counts.iter().enumerate() {
        r -= weights[i];
        if r <= 0.0 {
            return *m;
        }
    }
    counts.last().unwrap().0
}

fn move_to_arr(m: Move) -> [i32; 4] {
    [m.sx, m.sy, m.tx, m.ty]
}

fn arr_to_move(a: [i32; 4]) -> Move {
    Move::new(a[0], a[1], a[2], a[3])
}
