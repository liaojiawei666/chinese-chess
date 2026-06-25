use std::collections::{HashMap, HashSet, VecDeque};

use crate::{
    encode::move_to_action_id, evaluate::Evaluator, game::GameState, Color, GameStatus, Move,
    ACTION_SPACE_SIZE,
};
use rand::SeedableRng;
use rand_distr::Distribution;

type NodeId = u32;

#[derive(Debug, Clone)]
pub struct MctsConfig {
    /// 每一步棋的搜索次数 (Simulations Per Move)。
    /// 数值越大棋力越强，但思考时间越长。训练时推荐写小，实战时写大。
    pub num_simulations: usize,

    /// PUCT 公式的探索常数 (C_puct)。
    /// 控制探索 vs 利用的平衡。推荐值 1.0 ~ 2.5。
    pub c_puct: f32,

    /// 狄利克雷噪声的 alpha 参数。
    /// 控制噪声集中度，alpha 越小噪声越偏向少数走法。推荐值 0.03 ~ 0.3。
    pub dirichlet_alpha: f32,

    /// 噪声混合比例。0.0 表示不加噪声（对弈模式），0.25 为自对弈推荐值。
    pub noise_fraction: f32,

    /// 探索温度 (Temperature / Tau)。
    /// MCTS 本身不使用此值，由调用方（selfplay / play）在选招时使用。
    pub temperature: f32,
}

struct Edge {
    mv: Move,
    child_id: Option<NodeId>,
    prior: f32,
    visit_count: u32,
    total_value: f32,
}

impl Edge {
    fn q(&self) -> f32 {
        if self.visit_count > 0 {
            self.total_value / self.visit_count as f32
        } else {
            0.0
        }
    }
}

struct Node {
    edges: Vec<Edge>,
    is_terminal: bool,
    terminal_value: f32,
}

pub struct Mcts {
    nodes: HashMap<NodeId, Node>,
    next_id: NodeId,
    root_id: NodeId,
    game: GameState,
    config: MctsConfig,
    evaluator: Box<dyn Evaluator>,
    rng: rand::rngs::SmallRng,
}

/// 将终局状态转换为当前行棋方视角的价值：胜 +1、和 0、负 −1。
fn compute_terminal_value(status: &GameStatus, side_to_move: Color) -> f32 {
    match status.winner {
        Some(winner) if winner == side_to_move => 1.0,
        Some(_) => -1.0,
        None => 0.0,
    }
}

/// 从合法走法和网络 logits 构建 Edge 列表，在合法走法上做 masked softmax 得到先验。
fn create_edges(
    legal_moves: &[Move],
    policy_logits: &[f32; ACTION_SPACE_SIZE],
    side: Color,
) -> Vec<Edge> {
    if legal_moves.is_empty() {
        return Vec::new();
    }

    let mut raw: Vec<(Move, f32)> = Vec::with_capacity(legal_moves.len());
    let mut max_logit = f32::NEG_INFINITY;

    for &mv in legal_moves {
        let logit = policy_logits[move_to_action_id(mv, side) as usize];
        max_logit = max_logit.max(logit);
        raw.push((mv, logit));
    }

    // masked softmax: exp(logit - max) / sum(exp)，减 max 防止溢出
    let mut sum_exp = 0.0f32;
    let exps: Vec<f32> = raw
        .iter()
        .map(|&(_, l)| {
            let e = (l - max_logit).exp();
            sum_exp += e;
            e
        })
        .collect();

    raw.iter()
        .zip(exps)
        .map(|(&(mv, _), e)| Edge {
            mv,
            child_id: None,
            prior: e / sum_exp,
            visit_count: 0,
            total_value: 0.0,
        })
        .collect()
}

impl Mcts {
    pub fn new(game: GameState, config: MctsConfig, evaluator: Box<dyn Evaluator>) -> Self {
        Mcts {
            nodes: HashMap::new(),
            next_id: 0,
            root_id: 0,
            game,
            config,
            evaluator,
            rng: rand::rngs::SmallRng::from_entropy(),
        }
    }

    pub fn game(&self) -> &GameState {
        &self.game
    }

    /// 跑 num_simulations 次模拟，返回根节点各子边的 (Move, 访问次数)。
    pub async fn run(&mut self) -> Vec<(Move, u32)> {
        if self.nodes.is_empty() {
            self.create_root().await;
        }

        if self.config.noise_fraction > 0.0 {
            self.add_dirichlet_noise();
        }

        for _ in 0..self.config.num_simulations {
            self.simulate().await;
        }

        self.nodes[&self.root_id]
            .edges
            .iter()
            .map(|e| (e.mv, e.visit_count))
            .collect()
    }

    /// 返回根节点的策略目标（稀疏格式）：Vec<(action_id, probability)>。
    pub fn policy_target(&self) -> Vec<(u16, f32)> {
        let root = &self.nodes[&self.root_id];
        let total: u32 = root.edges.iter().map(|e| e.visit_count).sum();
        if total == 0 {
            return Vec::new();
        }
        let side = self.game.board.side_to_move;
        root.edges
            .iter()
            .filter(|e| e.visit_count > 0)
            .map(|e| {
                let aid = move_to_action_id(e.mv, side) as u16;
                let prob = e.visit_count as f32 / total as f32;
                (aid, prob)
            })
            .collect()
    }

    /// 从 visit counts 中选出访问次数最多的走法。
    pub fn best_move(visit_counts: &[(Move, u32)]) -> Move {
        visit_counts
            .iter()
            .max_by_key(|(_, c)| *c)
            .expect("visit_counts must not be empty")
            .0
    }

    /// 落子后保留对应子树作为新根，BFS 剪枝清除不可达节点。
    /// 找不到对应子树时清空整棵树，下次 run() 重建。
    pub fn advance(&mut self, mv: Move) {
        self.game.make_move(mv);

        let new_root_id = if !self.nodes.is_empty() {
            self.nodes[&self.root_id]
                .edges
                .iter()
                .find(|e| e.mv == mv)
                .and_then(|e| e.child_id)
        } else {
            None
        };

        match new_root_id {
            Some(id) => {
                let alive = self.reachable_from(id);
                self.nodes.retain(|id, _| alive.contains(id));
                self.root_id = id;
            }
            None => {
                self.nodes.clear();
            }
        }
    }

    // ---- private ----

    fn alloc_node(&mut self, node: Node) -> NodeId {
        let id = self.next_id;
        self.next_id += 1;
        self.nodes.insert(id, node);
        id
    }

    /// BFS 收集从 start 可达的所有节点 ID。
    fn reachable_from(&self, start: NodeId) -> HashSet<NodeId> {
        let mut alive = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back(start);
        alive.insert(start);
        while let Some(id) = queue.pop_front() {
            if let Some(node) = self.nodes.get(&id) {
                for edge in &node.edges {
                    if let Some(cid) = edge.child_id {
                        if alive.insert(cid) {
                            queue.push_back(cid);
                        }
                    }
                }
            }
        }
        alive
    }

    async fn create_root(&mut self) {
        let status = self.game.status;
        if status.is_terminal {
            let tv = compute_terminal_value(&status, self.game.board.side_to_move);
            self.root_id = self.alloc_node(Node {
                edges: vec![],
                is_terminal: true,
                terminal_value: tv,
            });
            return;
        }

        let output = self.evaluator.evaluate_async(&self.game).await;
        let side = self.game.board.side_to_move;
        let legal = self.game.legal_moves();
        let edges = create_edges(&legal, &output.policy_logits, side);
        self.root_id = self.alloc_node(Node {
            edges,
            is_terminal: false,
            terminal_value: 0.0,
        });
    }

    /// 单次模拟：select → expand/evaluate → backup。
    async fn simulate(&mut self) {
        let mut node_id = self.root_id;
        let mut path: Vec<(NodeId, usize)> = Vec::new();

        loop {
            if self.nodes[&node_id].is_terminal {
                let value = self.nodes[&node_id].terminal_value;
                self.backup(&path, value);
                return;
            }

            let edge_idx = self.select_edge(node_id);
            path.push((node_id, edge_idx));

            let mv = self.nodes[&node_id].edges[edge_idx].mv;
            self.game.make_move(mv);

            match self.nodes[&node_id].edges[edge_idx].child_id {
                Some(cid) => node_id = cid,
                None => {
                    let value = self.expand_edge(node_id, edge_idx).await;
                    self.backup(&path, value);
                    return;
                }
            }
        }
    }

    /// PUCT 选择：在给定节点中选出得分最高的边。
    fn select_edge(&self, node_id: NodeId) -> usize {
        let node = &self.nodes[&node_id];
        let total_visits: u32 = node.edges.iter().map(|e| e.visit_count).sum();
        let sqrt_total = ((1 + total_visits) as f32).sqrt();

        let mut best_idx = 0;
        let mut best_score = f32::NEG_INFINITY;

        for (i, edge) in node.edges.iter().enumerate() {
            let q = edge.q();
            let u = self.config.c_puct * edge.prior * sqrt_total
                / (1.0 + edge.visit_count as f32);
            let score = q + u;
            if score > best_score {
                best_score = score;
                best_idx = i;
            }
        }

        best_idx
    }

    /// 扩展一条未展开的边：调用 evaluator 创建子节点，返回叶子价值（当前行棋方视角）。
    async fn expand_edge(&mut self, parent_id: NodeId, edge_idx: usize) -> f32 {
        let status = self.game.status;

        if status.is_terminal {
            let side = self.game.board.side_to_move;
            let tv = compute_terminal_value(&status, side);
            let new_id = self.alloc_node(Node {
                edges: vec![],
                is_terminal: true,
                terminal_value: tv,
            });
            self.nodes.get_mut(&parent_id).unwrap().edges[edge_idx].child_id = Some(new_id);
            return tv;
        }

        let output = self.evaluator.evaluate_async(&self.game).await;
        let side = self.game.board.side_to_move;
        let legal = self.game.legal_moves();
        let edges = create_edges(&legal, &output.policy_logits, side);

        let new_id = self.alloc_node(Node {
            edges,
            is_terminal: false,
            terminal_value: 0.0,
        });
        self.nodes.get_mut(&parent_id).unwrap().edges[edge_idx].child_id = Some(new_id);

        output.value
    }

    /// 沿路径回传价值（逐层取反），并 undo 所有模拟中走过的着法。
    fn backup(&mut self, path: &[(NodeId, usize)], leaf_value: f32) {
        let mut value = -leaf_value;
        for &(node_id, edge_idx) in path.iter().rev() {
            let edge = &mut self.nodes.get_mut(&node_id).unwrap().edges[edge_idx];
            edge.visit_count += 1;
            edge.total_value += value;
            value = -value;
        }

        for _ in 0..path.len() {
            self.game.undo_move();
        }
    }

    /// 在根节点先验上混入 Dirichlet 噪声。
    fn add_dirichlet_noise(&mut self) {
        let root_id = self.root_id;
        if self.nodes[&root_id].edges.is_empty() {
            return;
        }

        let alpha = self.config.dirichlet_alpha as f64;
        let eps = self.config.noise_fraction;
        let n = self.nodes[&root_id].edges.len();

        let gamma = rand_distr::Gamma::new(alpha, 1.0).unwrap();
        let mut noise = Vec::with_capacity(n);
        for _ in 0..n {
            noise.push(gamma.sample(&mut self.rng) as f32);
        }
        let sum: f32 = noise.iter().sum();
        if sum > 1e-8 {
            for v in &mut noise {
                *v /= sum;
            }
        }

        let root = self.nodes.get_mut(&root_id).unwrap();
        for (edge, &nz) in root.edges.iter_mut().zip(noise.iter()) {
            edge.prior = (1.0 - eps) * edge.prior + eps * nz;
        }
    }
}
