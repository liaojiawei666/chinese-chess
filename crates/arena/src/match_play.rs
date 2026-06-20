//! 对杀引擎：每个开局红黑各打一局（颜色互换抵消先手优势），全程 τ=0 / ε=0 确定性，
//! 汇总成 A 相对 B 的胜率与 Elo。

use cc_core::engine::{Color, GameState, Move, Position};
use cc_core::mcts::{Evaluator, Mcts, MctsConfig};
use rand::rngs::StdRng;
use rand::SeedableRng;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Outcome {
    RedWin,
    BlackWin,
    Draw,
}

#[derive(Clone, Copy, Debug)]
pub struct MatchReport {
    pub games: u32,
    pub wins_a: u32,
    pub draws: u32,
    pub losses_a: u32,
    pub score_a: f64,
    pub elo_diff: f64,
}

/// 跑完整一场比赛：每个开局两局（A 执红 / B 执红），返回 A 视角统计。
pub fn run_match<E: Evaluator + Copy>(
    openings: &[Vec<Move>],
    eval_a: E,
    eval_b: E,
    mcts_config: MctsConfig,
    max_total_plies: u32,
) -> MatchReport {
    let mut wins_a = 0u32;
    let mut draws = 0u32;
    let mut losses_a = 0u32;

    for opening in openings {
        // 第 1 局：A 执红、B 执黑。
        match play_one(opening, eval_a, eval_b, mcts_config, max_total_plies) {
            Outcome::RedWin => wins_a += 1,
            Outcome::BlackWin => losses_a += 1,
            Outcome::Draw => draws += 1,
        }
        // 第 2 局：颜色互换（B 执红、A 执黑），结果按 A 视角解读。
        match play_one(opening, eval_b, eval_a, mcts_config, max_total_plies) {
            Outcome::RedWin => losses_a += 1, // 红方是 B → A 输
            Outcome::BlackWin => wins_a += 1,  // 黑方是 A → A 赢
            Outcome::Draw => draws += 1,
        }
    }

    let games = wins_a + draws + losses_a;
    let score_a = if games > 0 {
        (wins_a as f64 + 0.5 * draws as f64) / games as f64
    } else {
        0.5
    };
    MatchReport {
        games,
        wins_a,
        draws,
        losses_a,
        score_a,
        elo_diff: elo_diff(score_a),
    }
}

/// 从开局重放出带历史的局面，再让红/黑两个评估器各持一棵 MCTS 树对弈到终局。
fn play_one<E: Evaluator + Copy>(
    opening: &[Move],
    red: E,
    black: E,
    mcts_config: MctsConfig,
    max_total_plies: u32,
) -> Outcome {
    let mut state = GameState::from_position(Position::starting());
    for &mv in opening {
        state = state.make_move(mv);
    }

    // 双树复用：各方一棵，每步只跑当前走子方的树，落子后两树都 advance。
    let mut mcts_red = Mcts::new(mcts_config, StdRng::seed_from_u64(1));
    let mut mcts_black = Mcts::new(mcts_config, StdRng::seed_from_u64(2));

    let mut ply = state.history.len() as u32;
    while ply < max_total_plies {
        let counts = match state.position.side_to_move {
            Color::Red => mcts_red.run(state.clone(), &red),
            Color::Black => mcts_black.run(state.clone(), &black),
        };
        if counts.is_empty() {
            break; // 根节点终局
        }
        let mv = argmax_move(&counts);
        state = state.make_move(mv);
        mcts_red.advance(mv);
        mcts_black.advance(mv);
        ply += 1;
    }

    match state.status().winner {
        Some(Color::Red) => Outcome::RedWin,
        Some(Color::Black) => Outcome::BlackWin,
        None => Outcome::Draw,
    }
}

/// τ=0 选招：取访问次数最大者（并列取首个，确定性）。
fn argmax_move(counts: &[(Move, u32)]) -> Move {
    let mut best = counts[0];
    for &(m, n) in counts {
        if n > best.1 {
            best = (m, n);
        }
    }
    best.0
}

/// 由得分率换算 Elo 差（正值表示该方更强）。得分率 0/1 时夹紧避免无穷。
pub fn elo_diff(score: f64) -> f64 {
    let s = score.clamp(1e-4, 1.0 - 1e-4);
    -400.0 * (1.0 / s - 1.0).log10()
}
