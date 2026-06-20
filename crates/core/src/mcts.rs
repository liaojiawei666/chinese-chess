//! 带网络先验的蒙特卡洛树搜索（step-by-step API）。
//!
//! - PUCT 选择：score = Q + c_puct · P · sqrt(1 + ΣN) / (1 + N)。
//! - 根节点 Dirichlet 噪声（仅 epsilon>0 时生效；自对弈开局探索用）。
//! - 子树复用：advance(move) 把对应子节点提为新根。
//!
//! 数值上全程用 f64（与 Python 一致），保证固定输入下访问分布逐位可复现。
//!
//! ## Step-by-step API
//!
//! MCTS 不再持有 Evaluator，不直接调网络。调用方负责：
//! 1. 调 `step()` → 返回 `StepResult::NeedEval { state, .. }` 表示需要叶子评估；
//! 2. 拿 `state` 去外部推理（编码 → GPU forward → softmax）；
//! 3. 调 `feed_eval(priors, value)` 回填结果，完成一次模拟。
//!
//! 重复以上直到 `simulations_done() >= n_simulations`。对于已展开/终局叶子，
//! `step()` 返回 `StepResult::Done`，无需 feed_eval。

use crate::engine::{Color, GameState, GameStatus, Move};
use rand::Rng;
use rand_distr::Distribution;

/// Evaluator 接口保留用于简单场景（arena、play_gui）中的同步评估。
pub trait Evaluator {
    fn evaluate(&self, state: &GameState) -> (Vec<(Move, f64)>, f64);

    fn evaluate_batch(&self, states: &[&GameState]) -> Vec<(Vec<(Move, f64)>, f64)> {
        states.iter().map(|s| self.evaluate(s)).collect()
    }
}

impl<T: Evaluator> Evaluator for &T {
    fn evaluate(&self, state: &GameState) -> (Vec<(Move, f64)>, f64) {
        (**self).evaluate(state)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MctsConfig {
    pub n_simulations: u32,
    pub c_puct: f64,
    pub dirichlet_alpha: f64,
    pub dirichlet_epsilon: f64,
    /// 叶子并行宽度（保留向后兼容，仅在 `run()` 同步 API 中使用）。
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

/// `step()` 的返回值：要么需要叶子评估，要么本次模拟已自行完成（终局叶）。
pub enum StepResult<'a> {
    /// 叶子需要外部评估。调用方拿 `state` 去推理后调 `feed_eval`。
    NeedEval { state: &'a GameState },
    /// 本次模拟已完成（终局或已展开叶子），无需 feed_eval。
    Done,
}

pub struct Mcts<R: Rng> {
    config: MctsConfig,
    rng: R,
    root: Option<Box<Node>>,
    simulations_done: u32,
    pending_path: Option<Vec<usize>>,
}

impl<R: Rng> Mcts<R> {
    pub fn new(config: MctsConfig, rng: R) -> Self {
        Mcts {
            config,
            rng,
            root: None,
            simulations_done: 0,
            pending_path: None,
        }
    }

    pub fn config(&self) -> &MctsConfig {
        &self.config
    }

    pub fn simulations_done(&self) -> u32 {
        self.simulations_done
    }

    /// 初始化根节点。必须在 step() 循环前调用一次。
    /// 如果根已展开且 state 相同则复用（子树复用跨 `advance` 已处理），否则创建新根。
    /// 返回 true 表示根需要评估（调用方应走 step/feed_eval 或直接 feed_root_eval）。
    pub fn init_root(&mut self, root_state: GameState) -> bool {
        self.simulations_done = 0;
        self.pending_path = None;

        let root = match self.root.take() {
            Some(r) if same_state(&r.state, &root_state) => r,
            _ => Box::new(Node::new(root_state)),
        };
        let need_eval = !root.is_expanded;
        self.root = Some(root);
        need_eval
    }

    /// 对根节点做首次评估（需要在 init_root 返回 true 时调用）。
    pub fn feed_root_eval(&mut self, priors: Vec<(Move, f64)>, value: f64) {
        let root = self.root.as_mut().expect("root not initialized");
        if root.is_expanded {
            return;
        }
        let status = root.state.status();
        if status.is_terminal {
            root.is_terminal = true;
            root.is_expanded = true;
            root.terminal_value = terminal_value(root.to_play, status);
            return;
        }
        root.edges = priors
            .into_iter()
            .map(|(mv, prior)| Edge { mv, prior, n: 0, w: 0.0, child: None })
            .collect();
        root.is_expanded = true;
        let _ = value; // 根的 value 不回传
    }

    /// 获取根状态的引用（用于编码等）。
    pub fn root_state(&self) -> &GameState {
        &self.root.as_ref().expect("root not initialized").state
    }

    /// 是否终局根（无可走子）。
    pub fn is_terminal(&self) -> bool {
        self.root.as_ref().map_or(true, |r| r.is_terminal)
    }

    /// 添加 Dirichlet 噪声到根（在第一次 step 前自动调用，也可手动调）。
    pub fn add_noise(&mut self) {
        let root = self.root.as_mut().expect("root not initialized");
        add_dirichlet_noise(root, &self.config, &mut self.rng);
    }

    /// 单步模拟：从根下行选到叶子。
    /// - 返回 `NeedEval` 时调用方须外部评估后调 `feed_eval`。
    /// - 返回 `Done` 时该模拟已自行完成（终局叶），计数自增。
    pub fn step(&mut self) -> StepResult<'_> {
        assert!(self.pending_path.is_none(), "上一次 step 的 feed_eval 尚未调用");
        let root = self.root.as_mut().expect("root not initialized");

        if !root.noise_added {
            add_dirichlet_noise(root, &self.config, &mut self.rng);
        }

        let (path, is_terminal) = collect_leaf(root, &self.config);

        if is_terminal {
            let leaf_value = leaf_node(root, &path).terminal_value;
            backup_path(root, &path, leaf_value);
            self.simulations_done += 1;
            return StepResult::Done;
        }

        self.pending_path = Some(path);

        let path_ref = self.pending_path.as_ref().unwrap();
        let leaf = leaf_node(self.root.as_ref().unwrap(), path_ref);
        StepResult::NeedEval { state: &leaf.state }
    }

    /// 回填叶子评估结果，完成一次模拟。
    pub fn feed_eval(&mut self, priors: Vec<(Move, f64)>, value: f64) {
        let path = self.pending_path.take().expect("没有待回填的叶子");
        let root = self.root.as_mut().unwrap();
        expand_leaf(root, &path, priors);
        backup_path(root, &path, value);
        self.simulations_done += 1;
    }

    /// 同步 API（保留向后兼容）：用 Evaluator 跑完所有模拟，返回访问计数。
    pub fn run<E: Evaluator>(&mut self, root_state: GameState, evaluator: &E) -> Vec<(Move, u32)> {
        let need_eval = self.init_root(root_state);
        if need_eval {
            let root = self.root.as_ref().unwrap();
            let (priors, value) = evaluator.evaluate(&root.state);
            self.feed_root_eval(priors, value);
        }
        if self.is_terminal() {
            return Vec::new();
        }

        let width = self.config.collect_batch_size.max(1);
        while self.simulations_done < self.config.n_simulations {
            let want = (self.config.n_simulations - self.simulations_done).min(width) as usize;
            let root = self.root.as_mut().unwrap();

            if !root.noise_added {
                add_dirichlet_noise(root, &self.config, &mut self.rng);
            }

            let mut leaf_paths: Vec<Vec<usize>> = Vec::new();
            let mut terminals: Vec<(Vec<usize>, f64)> = Vec::new();
            for _ in 0..want {
                let (path, is_terminal) = collect_leaf_vl(root, &self.config);
                if is_terminal {
                    let val = leaf_node(root, &path).terminal_value;
                    terminals.push((path, val));
                } else {
                    leaf_paths.push(path);
                }
            }

            if !leaf_paths.is_empty() {
                let states: Vec<&GameState> =
                    leaf_paths.iter().map(|p| &leaf_node(root, p).state).collect();
                let results = evaluator.evaluate_batch(&states);
                drop(states);
                for (path, (priors, value)) in leaf_paths.iter().zip(results.into_iter()) {
                    expand_leaf(root, path, priors);
                    backup_path_vl(root, path, value);
                }
            }
            for (path, value) in &terminals {
                backup_path_vl(root, path, *value);
            }

            self.simulations_done += want as u32;
        }

        let root = self.root.as_ref().unwrap();
        root.edges.iter().map(|e| (e.mv, e.n)).collect()
    }

    /// 获取当前访问计数（step-by-step 循环结束后调用）。
    pub fn visit_counts(&self) -> Vec<(Move, u32)> {
        match &self.root {
            Some(root) => root.edges.iter().map(|e| (e.mv, e.n)).collect(),
            None => Vec::new(),
        }
    }

    /// 落子后把对应子树提为新根（子树复用）。
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
}

fn select(node: &Node, config: &MctsConfig) -> usize {
    let total = node.total_visits();
    let explore = config.c_puct * ((1 + total) as f64).sqrt();
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

/// 从根下行选到一个叶子（不加虚拟损失，用于 step API）。
/// 返回 (边索引路径, 是否终局)。
fn collect_leaf(root: &mut Node, config: &MctsConfig) -> (Vec<usize>, bool) {
    let mut path: Vec<usize> = Vec::new();
    let mut node: &mut Node = root;
    loop {
        let i = select(node, config);
        path.push(i);
        node.edges[i].n += 1;

        if node.edges[i].child.is_none() {
            let child_state = node.state.make_move(node.edges[i].mv);
            node.edges[i].child = Some(Box::new(Node::new(child_state)));
        }
        let child = node.edges[i].child.as_mut().unwrap();

        if !child.is_expanded {
            let status = child.state.status();
            if status.is_terminal {
                child.is_terminal = true;
                child.is_expanded = true;
                child.terminal_value = terminal_value(child.to_play, status);
                return (path, true);
            }
            return (path, false);
        }
        if child.is_terminal {
            return (path, true);
        }
        node = child;
    }
}

/// 从根下行选到一个叶子（加虚拟损失，用于同步 batch API）。
fn collect_leaf_vl(root: &mut Node, config: &MctsConfig) -> (Vec<usize>, bool) {
    let mut path: Vec<usize> = Vec::new();
    let mut node: &mut Node = root;
    loop {
        let i = select(node, config);
        path.push(i);
        node.edges[i].n += 1;
        node.edges[i].w -= VIRTUAL_LOSS;

        if node.edges[i].child.is_none() {
            let child_state = node.state.make_move(node.edges[i].mv);
            node.edges[i].child = Some(Box::new(Node::new(child_state)));
        }
        let child = node.edges[i].child.as_mut().unwrap();

        if !child.is_expanded {
            let status = child.state.status();
            if status.is_terminal {
                child.is_terminal = true;
                child.is_expanded = true;
                child.terminal_value = terminal_value(child.to_play, status);
                return (path, true);
            }
            return (path, false);
        }
        if child.is_terminal {
            return (path, true);
        }
        node = child;
    }
}

fn leaf_node<'a>(root: &'a Node, path: &[usize]) -> &'a Node {
    let mut node = root;
    for &i in path {
        node = node.edges[i].child.as_ref().expect("路径上的子节点应已创建");
    }
    node
}

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

/// 回传（无虚拟损失版本，用于 step API）。访问数在 collect_leaf 已 +1。
fn backup_path(root: &mut Node, path: &[usize], leaf_value: f64) {
    let k = path.len();
    let mut node: &mut Node = root;
    for (j, &i) in path.iter().enumerate() {
        let sign = if (k - j) % 2 == 1 { -1.0 } else { 1.0 };
        node.edges[i].w += sign * leaf_value;
        node = node.edges[i].child.as_mut().expect("路径上的子节点应已创建");
    }
}

/// 回传（撤销虚拟损失，用于同步 batch API）。
fn backup_path_vl(root: &mut Node, path: &[usize], leaf_value: f64) {
    let k = path.len();
    let mut node: &mut Node = root;
    for (j, &i) in path.iter().enumerate() {
        let sign = if (k - j) % 2 == 1 { -1.0 } else { 1.0 };
        node.edges[i].w += VIRTUAL_LOSS + sign * leaf_value;
        node = node.edges[i].child.as_mut().expect("路径上的子节点应已创建");
    }
}

fn terminal_value(to_play: Color, status: GameStatus) -> f64 {
    match status.winner {
        None => 0.0,
        Some(winner) => {
            if winner == to_play { 1.0 } else { -1.0 }
        }
    }
}

fn same_state(a: &GameState, b: &GameState) -> bool {
    a.position.to_fen() == b.position.to_fen() && a.history.len() == b.history.len()
}

fn add_dirichlet_noise(node: &mut Node, config: &MctsConfig, rng: &mut impl Rng) {
    if node.noise_added
        || node.edges.len() < 2
        || config.dirichlet_epsilon <= 0.0
    {
        return;
    }
    let alphas = vec![config.dirichlet_alpha; node.edges.len()];
    let dirichlet = rand_distr::Dirichlet::new(&alphas).expect("valid dirichlet alpha");
    let noise = dirichlet.sample(rng);
    let eps = config.dirichlet_epsilon;
    for (edge, nz) in node.edges.iter_mut().zip(noise.into_iter()) {
        edge.prior = (1.0 - eps) * edge.prior + eps * nz;
    }
    node.noise_added = true;
}
