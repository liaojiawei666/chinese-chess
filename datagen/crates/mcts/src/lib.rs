//! 带网络先验的蒙特卡洛树搜索（移植自 trainer/src/trainer/reference/mcts.py）。
//!
//! - PUCT 选择：score = Q + c_puct · P · sqrt(1 + ΣN) / (1 + N)。
//! - 根节点 Dirichlet 噪声（仅 epsilon>0 时生效；自对弈开局探索用）。
//! - 子树复用：advance(move) 把对应子节点提为新根。
//! - run() 只返回根各真实走法的访问次数 N(s,a)，温度/选招交给上层。
//!
//! 数值上全程用 f64（与 Python 一致），保证固定输入下访问分布逐位可复现。

use engine::{Color, GameState, GameStatus, Move};
use rand::Rng;
use rand_distr::Distribution;

/// MCTS 叶子评估接口：返回 (合法走法先验, value)，value 为当前走棋方视角，范围 [-1,1]。
/// 先验需按 `legal_moves()` 的顺序给出（与 PUCT 选择的遍历/打破平局顺序一致）。
pub trait Evaluator {
    fn evaluate(&self, state: &GameState) -> (Vec<(Move, f64)>, f64);
}

#[derive(Debug, Clone, Copy)]
pub struct MctsConfig {
    pub n_simulations: u32,
    pub c_puct: f64,
    pub dirichlet_alpha: f64,
    pub dirichlet_epsilon: f64,
}

impl Default for MctsConfig {
    fn default() -> Self {
        MctsConfig {
            n_simulations: 200,
            c_puct: 1.5,
            dirichlet_alpha: 0.3,
            dirichlet_epsilon: 0.25,
        }
    }
}

struct Edge {
    mv: Move,
    prior: f64,
    n: u32,
    w: f64,
    child: Option<Box<Node>>,
}

struct Node {
    state: GameState,
    to_play: Color,
    is_expanded: bool,
    is_terminal: bool,
    terminal_value: f64,
    noise_added: bool,
    edges: Vec<Edge>,
}

impl Node {
    fn new(state: GameState) -> Node {
        let to_play = state.position.side_to_move;
        Node {
            state,
            to_play,
            is_expanded: false,
            is_terminal: false,
            terminal_value: 0.0,
            noise_added: false,
            edges: Vec::new(),
        }
    }

    fn total_visits(&self) -> u32 {
        self.edges.iter().map(|e| e.n).sum()
    }
}

pub struct Mcts<E: Evaluator, R: Rng> {
    evaluator: E,
    config: MctsConfig,
    rng: R,
    root: Option<Box<Node>>,
}

impl<E: Evaluator, R: Rng> Mcts<E, R> {
    pub fn new(evaluator: E, config: MctsConfig, rng: R) -> Self {
        Mcts {
            evaluator,
            config,
            rng,
            root: None,
        }
    }

    /// 跑 n_simulations 次模拟，返回根各真实走法的原始访问次数 N(s,a)（顺序同 legal_moves）。
    /// 终局根返回空。
    pub fn run(&mut self, root_state: GameState) -> Vec<(Move, u32)> {
        let mut root = match self.root.take() {
            Some(r) if same_state(&r.state, &root_state) => r,
            _ => Box::new(Node::new(root_state)),
        };

        if !root.is_expanded {
            self.evaluate(&mut root);
        }
        if root.is_terminal {
            self.root = Some(root);
            return Vec::new();
        }

        self.add_dirichlet_noise(&mut root);
        for _ in 0..self.config.n_simulations {
            self.simulate(&mut root);
        }

        let counts = root.edges.iter().map(|e| (e.mv, e.n)).collect();
        self.root = Some(root);
        counts
    }

    /// 落子后把对应子树提为新根（子树复用）；该走法没有展开过的子节点则置空。
    pub fn advance(&mut self, mv: Move) {
        self.root = self.root.take().and_then(|mut r| {
            r.edges
                .iter_mut()
                .find(|e| e.mv == mv)
                .and_then(|e| e.child.take())
        });
    }

    /// 开新一局时清空树。
    pub fn reset(&mut self) {
        self.root = None;
    }

    fn simulate(&mut self, node: &mut Node) -> f64 {
        // 前置：node 已展开且非终局（根在 run 中保证；递归只进入展开非终局子节点）。
        let i = self.select(node);

        if node.edges[i].child.is_none() {
            let child_state = node.state.make_move(node.edges[i].mv);
            node.edges[i].child = Some(Box::new(Node::new(child_state)));
        }
        let child = node.edges[i].child.as_mut().unwrap();

        let child_value = if !child.is_expanded {
            self.evaluate(child)
        } else if child.is_terminal {
            child.terminal_value
        } else {
            self.simulate(child)
        };

        let edge = &mut node.edges[i];
        edge.n += 1;
        edge.w += -child_value;
        -child_value
    }

    fn select(&self, node: &Node) -> usize {
        let total = node.total_visits();
        let explore = self.config.c_puct * ((1 + total) as f64).sqrt();
        let mut best_index = 0usize;
        let mut best_score = f64::NEG_INFINITY;
        for (i, edge) in node.edges.iter().enumerate() {
            let q = if edge.n > 0 { edge.w / edge.n as f64 } else { 0.0 };
            let score = q + explore * edge.prior / (1.0 + edge.n as f64);
            if score > best_score {
                best_score = score;
                best_index = i;
            }
        }
        best_index
    }

    fn evaluate(&self, node: &mut Node) -> f64 {
        let status = node.state.status();
        if status.is_terminal {
            node.is_terminal = true;
            node.is_expanded = true;
            node.terminal_value = terminal_value(node.to_play, status);
            return node.terminal_value;
        }

        let (priors, value) = self.evaluator.evaluate(&node.state);
        node.edges = priors
            .into_iter()
            .map(|(mv, prior)| Edge {
                mv,
                prior,
                n: 0,
                w: 0.0,
                child: None,
            })
            .collect();
        node.is_expanded = true;
        value
    }

    fn add_dirichlet_noise(&mut self, node: &mut Node) {
        // 单一走法时 Dirichlet 退化（噪声必为 [1.0]，混合后先验仍是 1.0），直接跳过；
        // rand_distr 的 Dirichlet 也要求至少 2 个类别。
        if node.noise_added
            || node.edges.len() < 2
            || self.config.dirichlet_epsilon <= 0.0
        {
            return;
        }
        let alphas = vec![self.config.dirichlet_alpha; node.edges.len()];
        let dirichlet = rand_distr::Dirichlet::new(&alphas).expect("valid dirichlet alpha");
        let noise = dirichlet.sample(&mut self.rng);
        let eps = self.config.dirichlet_epsilon;
        for (edge, nz) in node.edges.iter_mut().zip(noise.into_iter()) {
            edge.prior = (1.0 - eps) * edge.prior + eps * nz;
        }
        node.noise_added = true;
    }
}

fn terminal_value(to_play: Color, status: GameStatus) -> f64 {
    match status.winner {
        None => 0.0,
        Some(winner) => {
            if winner == to_play {
                1.0
            } else {
                -1.0
            }
        }
    }
}

/// 同一局内用 FEN + 历史步数判定是否同一局面（区分重复局面），对应 Python `_same_state`。
fn same_state(a: &GameState, b: &GameState) -> bool {
    a.position.to_fen() == b.position.to_fen() && a.history.len() == b.history.len()
}
