//! 带网络先验的蒙特卡洛树搜索。
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

    /// 批量评估：默认逐个调用 `evaluate`。跨进程批量推理的实现（selfplay 的
    /// BatchedEvaluator）可重写为「一次发出全部请求、再统一收回执」，让一局内
    /// 一波收集的多个叶子并发凑成大 GPU 批，摊薄每次推理的往返延迟。
    fn evaluate_batch(&self, states: &[&GameState]) -> Vec<(Vec<(Move, f64)>, f64)> {
        states.iter().map(|s| self.evaluate(s)).collect()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MctsConfig {
    pub n_simulations: u32,
    pub c_puct: f64,
    pub dirichlet_alpha: f64,
    pub dirichlet_epsilon: f64,
    /// 叶子并行宽度：一波收集多少个叶子后再批量评估（virtual loss 防同波重复选同叶）。
    /// =1 时退化为逐叶串行，与原始（逐位可复现）行为完全一致；>1 时用 GPU 批量提速。
    pub collect_batch_size: u32,
}

impl Default for MctsConfig {
    fn default() -> Self {
        MctsConfig {
            n_simulations: 200,
            c_puct: 1.5,
            dirichlet_alpha: 0.3,
            dirichlet_epsilon: 0.25,
            collect_batch_size: 1,
        }
    }
}

/// 虚拟损失：收集一个叶子时沿途各边临时记一笔"输"（n+1, w-1），降低其 Q，
/// 促使同一波后续收集避开同一分支；评估后在 backup 时连同真实值一并修正回来。
const VIRTUAL_LOSS: f64 = 1.0;

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

        let width = self.config.collect_batch_size.max(1);
        let mut done = 0u32;
        while done < self.config.n_simulations {
            let want = (self.config.n_simulations - done).min(width) as usize;

            // 收集本波叶子：沿途记虚拟损失，使同波各次收集尽量落到不同叶子。
            let mut leaf_paths: Vec<Vec<usize>> = Vec::new();
            let mut terminals: Vec<(Vec<usize>, f64)> = Vec::new();
            for _ in 0..want {
                match self.collect_leaf(&mut root) {
                    Collected::Leaf { path } => leaf_paths.push(path),
                    Collected::Terminal { path, value } => terminals.push((path, value)),
                }
            }

            // 非终局叶子批量评估（一次前向），再各自展开 + 回传。
            if !leaf_paths.is_empty() {
                let states: Vec<&GameState> =
                    leaf_paths.iter().map(|p| &leaf_node(&root, p).state).collect();
                let results = self.evaluator.evaluate_batch(&states);
                drop(states);
                for (path, (priors, value)) in leaf_paths.iter().zip(results.into_iter()) {
                    expand_leaf(&mut root, path, priors);
                    backup_path(&mut root, path, value);
                }
            }
            // 终局叶子无需评估，直接回传其终局值。
            for (path, value) in &terminals {
                backup_path(&mut root, path, *value);
            }

            done += want as u32;
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

    /// 从根下行选到一个叶子（未展开的真实叶 / 终局），沿途对所选边记虚拟损失，
    /// 返回到达该叶的边索引路径。终局叶顺带标记并带回其终局值。
    fn collect_leaf(&self, root: &mut Node) -> Collected {
        let mut path: Vec<usize> = Vec::new();
        let mut node: &mut Node = root;
        loop {
            // 前置：node 已展开且非终局。
            let i = self.select(node);
            path.push(i);
            // 记虚拟损失（backup 时连同真实值修正回来）。
            node.edges[i].n += 1;
            node.edges[i].w -= VIRTUAL_LOSS;

            if node.edges[i].child.is_none() {
                let child_state = node.state.make_move(node.edges[i].mv);
                node.edges[i].child = Some(Box::new(Node::new(child_state)));
            }
            let child = node.edges[i].child.as_mut().unwrap();

            if !child.is_expanded {
                // 尚未展开：先判终局，否则即为待评估的真实叶子。
                let status = child.state.status();
                if status.is_terminal {
                    child.is_terminal = true;
                    child.is_expanded = true;
                    child.terminal_value = terminal_value(child.to_play, status);
                    return Collected::Terminal { path, value: child.terminal_value };
                }
                return Collected::Leaf { path };
            }
            if child.is_terminal {
                return Collected::Terminal { path, value: child.terminal_value };
            }
            node = child;
        }
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

/// 一波收集到的叶子：待评估的真实叶（只记路径）或终局叶（带回终局值）。
enum Collected {
    Leaf { path: Vec<usize> },
    Terminal { path: Vec<usize>, value: f64 },
}

/// 按边索引路径从根走到叶节点（只读），用于取叶子局面做批量评估。
fn leaf_node<'a>(root: &'a Node, path: &[usize]) -> &'a Node {
    let mut node = root;
    for &i in path {
        node = node.edges[i].child.as_ref().expect("路径上的子节点应已创建");
    }
    node
}

/// 用网络先验展开叶子（已被同波其他结果展开过则跳过，避免重复）。
fn expand_leaf(root: &mut Node, path: &[usize], priors: Vec<(Move, f64)>) {
    let mut node: &mut Node = root;
    for &i in path {
        node = node.edges[i].child.as_mut().expect("路径上的子节点应已创建");
    }
    if !node.is_expanded {
        node.edges = priors
            .into_iter()
            .map(|(mv, prior)| Edge { mv, prior, n: 0, w: 0.0, child: None })
            .collect();
        node.is_expanded = true;
    }
}

/// 沿路径回传：撤销虚拟损失并叠加真实值。叶子值为叶节点走子方视角，逐层取反，
/// 故第 j 层边（路径长 k）的真实贡献为 value·(-1)^(k-j)。访问数在收集时已 +1。
fn backup_path(root: &mut Node, path: &[usize], leaf_value: f64) {
    let k = path.len();
    let mut node: &mut Node = root;
    for (j, &i) in path.iter().enumerate() {
        let sign = if (k - j) % 2 == 1 { -1.0 } else { 1.0 };
        // +VIRTUAL_LOSS 撤销收集时记的虚拟损失，再叠加真实贡献。
        node.edges[i].w += VIRTUAL_LOSS + sign * leaf_value;
        node = node.edges[i].child.as_mut().expect("路径上的子节点应已创建");
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
