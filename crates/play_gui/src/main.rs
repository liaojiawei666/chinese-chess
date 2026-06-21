use std::path::PathBuf;
use std::sync::mpsc::{self, Receiver};
use std::thread;

use anyhow::{Context, Result};
use eframe::egui;
use cc_core::engine::{
    Color, GameState, GameStatus, Move, Piece, PieceKind, Position, BOARD_HEIGHT, BOARD_WIDTH,
};
use cc_core::mcts::{Evaluator, Mcts, MctsConfig};
use rand::SeedableRng;
use serde::Deserialize;

#[derive(Debug, Clone)]
struct Args {
    run_config: Option<PathBuf>,
    model_dir: PathBuf,
    device: Option<String>,
    sims: Option<u32>,
}

fn detect_config_path() -> PathBuf {
    let has_cuda = tch::Cuda::is_available();
    let profile = if has_cuda { "gpu" } else { "local" };
    log::info!(
        "GPU 自动检测：CUDA {}，使用 {} 配置",
        if has_cuda { "可用" } else { "不可用" },
        profile
    );
    PathBuf::from(format!("config/{profile}.json"))
}

impl Default for Args {
    fn default() -> Self {
        Args {
            run_config: None,
            model_dir: PathBuf::from("data/models"),
            device: None,
            sims: None,
        }
    }
}

#[derive(Debug, Deserialize)]
struct RunConfigPartial {
    #[serde(default = "default_device")]
    device: String,
    mcts: MctsPartial,
}

#[derive(Debug, Deserialize)]
struct MctsPartial {
    n_simulations: u32,
    c_puct: f64,
}

fn default_device() -> String {
    "cpu".to_string()
}

#[derive(Debug, Clone)]
struct PlayConfig {
    model_dir: PathBuf,
    device: String,
    mcts: MctsConfig,
}

#[derive(Debug)]
struct AiResult {
    mv: Option<Move>,
    model_version: Option<i64>,
    visits: u32,
    error: Option<String>,
}

struct PlayGui {
    state: GameState,
    human: Color,
    selected: Option<(i32, i32)>,
    legal_targets: Vec<(i32, i32)>,
    ai_rx: Option<Receiver<AiResult>>,
    ai_thinking: bool,
    config: PlayConfig,
    status: String,
    last_model_version: Option<i64>,
    last_ai_move: Option<Move>,
}

fn main() -> Result<()> {
    let args = parse_args()?;
    let config = load_play_config(args)?;
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([760.0, 900.0]),
        ..Default::default()
    };

    eframe::run_native(
        "Chinese Chess - latest model",
        native_options,
        Box::new(move |cc| {
            install_chinese_font(&cc.egui_ctx);
            Ok(Box::new(PlayGui::new(config.clone())))
        }),
    )
    .map_err(|e| anyhow::anyhow!("启动 GUI 失败：{e}"))
}

fn install_chinese_font(ctx: &egui::Context) {
    let candidates = [
        r"/System/Library/Fonts/PingFang.ttc",
        r"/System/Library/Fonts/STHeiti Light.ttc",
        r"/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        r"/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    ];
    let Some(bytes) = candidates.iter().find_map(|path| std::fs::read(path).ok()) else {
        return;
    };

    let mut fonts = egui::FontDefinitions::default();
    fonts.font_data.insert(
        "chinese".to_string(),
        egui::FontData::from_owned(bytes).into(),
    );
    for family in [egui::FontFamily::Proportional, egui::FontFamily::Monospace] {
        fonts
            .families
            .entry(family)
            .or_default()
            .insert(0, "chinese".to_string());
    }
    ctx.set_fonts(fonts);
}

fn parse_args() -> Result<Args> {
    let mut args = Args::default();
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < argv.len() {
        match argv[i].as_str() {
            "--run-config" => args.run_config = Some(PathBuf::from(need(&argv, i)?)),
            "--model-dir" => args.model_dir = PathBuf::from(need(&argv, i)?),
            "--device" => args.device = Some(need(&argv, i)?.clone()),
            "--sims" => args.sims = Some(need(&argv, i)?.parse().context("--sims 需整数")?),
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            other => anyhow::bail!("未知参数：{other}（--help 查看用法）"),
        }
        i += 2;
    }
    Ok(args)
}

fn need<'a>(argv: &'a [String], i: usize) -> Result<&'a String> {
    argv.get(i + 1)
        .ok_or_else(|| anyhow::anyhow!("缺少参数 {} 的值", argv[i]))
}

fn print_help() {
    println!(
        "play_gui —— 人机对战 GUI，默认读取 data/models/latest.json\n\n\
         可选：\n\
         \x20 --run-config <path>  自动检测 GPU 选 config/gpu.json 或 local.json\n\
         \x20 --model-dir <path>   默认 data/models\n\
         \x20 --device <cpu|cuda|mps> 默认取 run-config.device\n\
         \x20 --sims <n>           每手 MCTS 模拟数，默认取 run-config.mcts.n_simulations"
    );
}

fn load_play_config(args: Args) -> Result<PlayConfig> {
    let config_path = args.run_config.clone().unwrap_or_else(detect_config_path);
    log::info!("加载配置：{}", config_path.display());
    let text = std::fs::read_to_string(&config_path)
        .with_context(|| format!("读取 run-config 失败：{}", config_path.display()))?;
    let rc: RunConfigPartial = serde_json::from_str(&text).context("解析 run-config 失败")?;
    let device = args.device.unwrap_or(rc.device);
    let sims = args.sims.unwrap_or(rc.mcts.n_simulations);
    Ok(PlayConfig {
        model_dir: args.model_dir,
        device,
        mcts: MctsConfig {
            n_simulations: sims,
            c_puct: rc.mcts.c_puct,
            dirichlet_alpha: 0.3,
            dirichlet_epsilon: 0.0,
            collect_batch_size: 1,
        },
    })
}

impl PlayGui {
    fn new(config: PlayConfig) -> Self {
        let mut app = PlayGui {
            state: GameState::from_position(Position::starting()),
            human: Color::Red,
            selected: None,
            legal_targets: Vec::new(),
            ai_rx: None,
            ai_thinking: false,
            config,
            status: "你执红先行。".to_string(),
            last_model_version: None,
            last_ai_move: None,
        };
        app.maybe_start_ai();
        app
    }

    fn reset(&mut self, human: Color) {
        self.state = GameState::from_position(Position::starting());
        self.human = human;
        self.selected = None;
        self.legal_targets.clear();
        self.ai_rx = None;
        self.ai_thinking = false;
        self.last_ai_move = None;
        self.status = match human {
            Color::Red => "新局：你执红先行。".to_string(),
            Color::Black => "新局：你执黑，AI 先行。".to_string(),
        };
        self.maybe_start_ai();
    }

    fn maybe_start_ai(&mut self) {
        if self.ai_thinking || self.state.status().is_terminal {
            return;
        }
        if self.state.position.side_to_move == self.human {
            return;
        }

        let state = self.state.clone();
        let config = self.config.clone();
        let (tx, rx) = mpsc::channel();
        self.ai_rx = Some(rx);
        self.ai_thinking = true;
        self.status = format!(
            "AI 思考中：{} sims，device={} ...",
            self.config.mcts.n_simulations, self.config.device
        );
        thread::spawn(move || {
            let result = search_ai_move(state, config);
            let _ = tx.send(result);
        });
    }

    fn poll_ai(&mut self) {
        let Some(rx) = &self.ai_rx else {
            return;
        };
        let Ok(result) = rx.try_recv() else {
            return;
        };
        self.ai_rx = None;
        self.ai_thinking = false;

        if let Some(error) = result.error {
            self.status = format!("AI 出错：{error}");
            return;
        }
        let Some(mv) = result.mv else {
            self.status = "AI 无合法走法。".to_string();
            return;
        };
        if self
            .state
            .position
            .is_legal_move(mv, Some(self.state.position.side_to_move))
        {
            self.state = self.state.make_move(mv);
            self.last_ai_move = Some(mv);
            self.last_model_version = result.model_version;
            self.status = format!(
                "AI: {}，访问 {}，模型 v{}。",
                move_text(mv),
                result.visits,
                result
                    .model_version
                    .map_or("-".to_string(), |v| v.to_string())
            );
            self.update_terminal_status();
        } else {
            self.status = format!("AI 返回非法走法：{}", move_text(mv));
        }
    }

    fn handle_square_click(&mut self, x: i32, y: i32) {
        if self.ai_thinking || self.state.status().is_terminal {
            return;
        }
        if self.state.position.side_to_move != self.human {
            self.maybe_start_ai();
            return;
        }

        let clicked_piece = self.state.position.piece_at(x, y);
        if let Some((sx, sy)) = self.selected {
            let mv = Move::new(sx, sy, x, y);
            if self.state.position.is_legal_move(mv, Some(self.human)) {
                self.state = self.state.make_move(mv);
                self.selected = None;
                self.legal_targets.clear();
                self.last_ai_move = None;
                self.status = format!("你: {}", move_text(mv));
                self.update_terminal_status();
                self.maybe_start_ai();
                return;
            }
        }

        match clicked_piece {
            Some(piece) if piece.color == self.human => {
                self.selected = Some((x, y));
                self.legal_targets = self
                    .state
                    .position
                    .legal_moves()
                    .into_iter()
                    .filter(|m| m.sx == x && m.sy == y)
                    .map(|m| (m.tx, m.ty))
                    .collect();
                self.status = format!("已选择 {} {}", color_name(piece.color), piece_name(piece));
            }
            _ => {
                self.selected = None;
                self.legal_targets.clear();
                self.status = "请选择自己的棋子。".to_string();
            }
        }
    }

    fn update_terminal_status(&mut self) {
        let status = self.state.status();
        if status.is_terminal {
            self.status = terminal_text(status, self.human);
        }
    }
}

impl eframe::App for PlayGui {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.poll_ai();
        if self.ai_thinking {
            ctx.request_repaint_after(std::time::Duration::from_millis(100));
        }

        egui::SidePanel::right("side_panel")
            .min_width(220.0)
            .show(ctx, |ui| {
                ui.heading("人机对战");
                ui.label(format!("你执：{}", color_name(self.human)));
                ui.label(format!(
                    "轮到：{}",
                    color_name(self.state.position.side_to_move)
                ));
                ui.label(format!("设备：{}", self.config.device));
                ui.label(format!("模拟：{}", self.config.mcts.n_simulations));
                ui.label(format!(
                    "模型：{}",
                    self.last_model_version
                        .map_or("未加载".to_string(), |v| format!("v{v}"))
                ));
                ui.separator();
                ui.label(&self.status);
                if let Some(mv) = self.last_ai_move {
                    ui.label(format!("上步 AI：{}", move_text(mv)));
                }
                ui.separator();
                if ui.button("新局：我执红").clicked() {
                    self.reset(Color::Red);
                }
                if ui.button("新局：我执黑").clicked() {
                    self.reset(Color::Black);
                }
                if ui.button("AI 现在走一步").clicked() {
                    self.maybe_start_ai();
                }
                ui.separator();
                ui.small("用法：点击自己的棋子，再点击目标格。AI 每步读取 latest.json 对应模型。");
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            draw_board(ui, self);
        });
    }
}

fn draw_board(ui: &mut egui::Ui, app: &mut PlayGui) {
    let board_size = ui.available_width().min(ui.available_height()).min(680.0);
    let margin = 36.0;
    let cell = (board_size - 2.0 * margin) / 9.0;
    let desired = egui::vec2(board_size, board_size + cell);
    let (rect, response) = ui.allocate_exact_size(desired, egui::Sense::click());
    let painter = ui.painter_at(rect);
    let origin = egui::pos2(rect.left() + margin, rect.top() + margin);
    let view_red = app.human == Color::Red;

    let board_rect = egui::Rect::from_min_max(
        origin,
        egui::pos2(origin.x + cell * 8.0, origin.y + cell * 9.0),
    );
    painter.rect_filled(rect, 0.0, egui::Color32::from_rgb(246, 213, 151));
    painter.rect_stroke(
        board_rect.expand(2.0),
        0.0,
        egui::Stroke::new(2.0, egui::Color32::from_rgb(90, 48, 20)),
        egui::StrokeKind::Outside,
    );

    for x in 0..BOARD_WIDTH {
        let sx = origin.x + x as f32 * cell;
        painter.line_segment(
            [
                egui::pos2(sx, origin.y),
                egui::pos2(sx, origin.y + 4.0 * cell),
            ],
            egui::Stroke::new(1.5, egui::Color32::from_rgb(90, 48, 20)),
        );
        painter.line_segment(
            [
                egui::pos2(sx, origin.y + 5.0 * cell),
                egui::pos2(sx, origin.y + 9.0 * cell),
            ],
            egui::Stroke::new(1.5, egui::Color32::from_rgb(90, 48, 20)),
        );
    }
    for y in 0..BOARD_HEIGHT {
        let sy = origin.y + y as f32 * cell;
        painter.line_segment(
            [
                egui::pos2(origin.x, sy),
                egui::pos2(origin.x + 8.0 * cell, sy),
            ],
            egui::Stroke::new(1.5, egui::Color32::from_rgb(90, 48, 20)),
        );
    }

    draw_palace(&painter, origin, cell, true);
    draw_palace(&painter, origin, cell, false);
    painter.text(
        egui::pos2(origin.x + 4.0 * cell, origin.y + 4.5 * cell),
        egui::Align2::CENTER_CENTER,
        "楚河        汉界",
        egui::FontId::proportional(24.0),
        egui::Color32::from_rgb(120, 66, 25),
    );

    for &(x, y) in &app.legal_targets {
        let p = board_to_screen(x, y, origin, cell, view_red);
        painter.circle_filled(
            p,
            cell * 0.16,
            egui::Color32::from_rgba_unmultiplied(60, 140, 40, 150),
        );
    }
    if let Some((x, y)) = app.selected {
        let p = board_to_screen(x, y, origin, cell, view_red);
        painter.circle_stroke(
            p,
            cell * 0.38,
            egui::Stroke::new(4.0, egui::Color32::YELLOW),
        );
    }

    for y in 0..BOARD_HEIGHT as i32 {
        for x in 0..BOARD_WIDTH as i32 {
            if let Some(piece) = app.state.position.piece_at(x, y) {
                draw_piece(
                    &painter,
                    board_to_screen(x, y, origin, cell, view_red),
                    cell,
                    piece,
                );
            }
        }
    }

    if response.clicked() {
        if let Some(pos) = response.interact_pointer_pos() {
            if let Some((x, y)) = screen_to_board(pos, origin, cell, view_red) {
                app.handle_square_click(x, y);
            }
        }
    }
}

fn draw_palace(painter: &egui::Painter, origin: egui::Pos2, cell: f32, top: bool) {
    let y0 = if top { 0.0 } else { 7.0 };
    let a = egui::pos2(origin.x + 3.0 * cell, origin.y + y0 * cell);
    let b = egui::pos2(origin.x + 5.0 * cell, origin.y + (y0 + 2.0) * cell);
    let c = egui::pos2(origin.x + 5.0 * cell, origin.y + y0 * cell);
    let d = egui::pos2(origin.x + 3.0 * cell, origin.y + (y0 + 2.0) * cell);
    let stroke = egui::Stroke::new(1.5, egui::Color32::from_rgb(90, 48, 20));
    painter.line_segment([a, b], stroke);
    painter.line_segment([c, d], stroke);
}

fn draw_piece(painter: &egui::Painter, center: egui::Pos2, cell: f32, piece: Piece) {
    let fill = egui::Color32::from_rgb(255, 238, 190);
    let stroke_color = match piece.color {
        Color::Red => egui::Color32::from_rgb(190, 20, 20),
        Color::Black => egui::Color32::from_rgb(20, 20, 20),
    };
    painter.circle_filled(center, cell * 0.36, fill);
    painter.circle_stroke(center, cell * 0.36, egui::Stroke::new(2.5, stroke_color));
    painter.text(
        center,
        egui::Align2::CENTER_CENTER,
        piece_name(piece),
        egui::FontId::proportional(cell * 0.36),
        stroke_color,
    );
}

fn board_to_screen(x: i32, y: i32, origin: egui::Pos2, cell: f32, view_red: bool) -> egui::Pos2 {
    let sx = if view_red {
        x
    } else {
        (BOARD_WIDTH as i32 - 1) - x
    };
    let sy = if view_red {
        y
    } else {
        (BOARD_HEIGHT as i32 - 1) - y
    };
    egui::pos2(origin.x + sx as f32 * cell, origin.y + sy as f32 * cell)
}

fn screen_to_board(
    pos: egui::Pos2,
    origin: egui::Pos2,
    cell: f32,
    view_red: bool,
) -> Option<(i32, i32)> {
    let sx = ((pos.x - origin.x) / cell).round() as i32;
    let sy = ((pos.y - origin.y) / cell).round() as i32;
    if sx < 0 || sx >= BOARD_WIDTH as i32 || sy < 0 || sy >= BOARD_HEIGHT as i32 {
        return None;
    }
    let x = if view_red {
        sx
    } else {
        (BOARD_WIDTH as i32 - 1) - sx
    };
    let y = if view_red {
        sy
    } else {
        (BOARD_HEIGHT as i32 - 1) - sy
    };
    Some((x, y))
}

fn piece_name(piece: Piece) -> &'static str {
    match (piece.color, piece.kind) {
        (Color::Red, PieceKind::King) => "帅",
        (Color::Black, PieceKind::King) => "将",
        (_, PieceKind::Advisor) => "士",
        (_, PieceKind::Elephant) => "象",
        (_, PieceKind::Horse) => "马",
        (_, PieceKind::Rook) => "车",
        (_, PieceKind::Cannon) => "炮",
        (Color::Red, PieceKind::Pawn) => "兵",
        (Color::Black, PieceKind::Pawn) => "卒",
    }
}

fn color_name(color: Color) -> &'static str {
    match color {
        Color::Red => "红",
        Color::Black => "黑",
    }
}

fn move_text(mv: Move) -> String {
    format!("({},{}) -> ({},{})", mv.sx, mv.sy, mv.tx, mv.ty)
}

fn terminal_text(status: GameStatus, human: Color) -> String {
    match status.winner {
        Some(winner) if winner == human => "终局：你赢了。".to_string(),
        Some(winner) => format!("终局：{}胜。", color_name(winner)),
        None => "终局：和棋。".to_string(),
    }
}

fn search_ai_move(state: GameState, config: PlayConfig) -> AiResult {
    let state2 = state.clone();
    let config2 = config.clone();
    match search_ai_move_inner(state, config) {
        Ok(result) => result,
        Err(e) => {
            log::warn!("模型推理失败，退化为均匀评估器：{e:#}");
            search_ai_move_uniform(state2, config2)
        }
    }
}

fn search_ai_move_inner(state: GameState, config: PlayConfig) -> Result<AiResult> {
    use cc_core::model_io::{LocalModelStore, ModelStore};

    let store = LocalModelStore::new(&config.model_dir);
    let (version, path) = store
        .get_latest_path()?
        .ok_or_else(|| anyhow::anyhow!("{} 下没有 latest.json", config.model_dir.display()))?;
    let path_text = path.to_string_lossy().into_owned();
    let evaluator = TorchEvaluator::load(&path_text, &config.device)?;
    let mut mcts = Mcts::new(
        config.mcts,
        rand::rngs::StdRng::seed_from_u64(20260620),
    );
    let counts = mcts.run(state, &evaluator);
    let (mv, visits) = counts
        .into_iter()
        .max_by_key(|(_, n)| *n)
        .ok_or_else(|| anyhow::anyhow!("当前局面无合法走法"))?;
    Ok(AiResult {
        mv: Some(mv),
        model_version: Some(version),
        visits,
        error: None,
    })
}

fn search_ai_move_uniform(state: GameState, config: PlayConfig) -> AiResult {
    let evaluator = UniformEvaluator;
    let mut mcts = Mcts::new(
        config.mcts,
        rand::rngs::StdRng::seed_from_u64(20260620),
    );
    let counts = mcts.run(state, &evaluator);
    match counts.into_iter().max_by_key(|(_, n)| *n) {
        Some((mv, visits)) => AiResult {
            mv: Some(mv),
            model_version: None,
            visits,
            error: Some("使用均匀评估器（无可用模型）。".to_string()),
        },
        None => AiResult {
            mv: None,
            model_version: None,
            visits: 0,
            error: Some("当前局面无合法走法".to_string()),
        },
    }
}

struct TorchEvaluator {
    model: cc_core::infer::torch_model::TorchModel,
}

impl TorchEvaluator {
    fn load(path: &str, device: &str) -> Result<Self> {
        let model = cc_core::infer::torch_model::TorchModel::load_str(path, device)
            .map_err(|e| anyhow::anyhow!("加载模型失败 {path}：{e}"))?;
        Ok(TorchEvaluator { model })
    }
}

impl Evaluator for TorchEvaluator {
    fn evaluate(&self, state: &GameState) -> (Vec<(Move, f64)>, f64) {
        use cc_core::infer::LeafEvaluator;
        let (priors, value) = self.model.evaluate(state);
        (
            priors.into_iter().map(|(m, p)| (m, p as f64)).collect(),
            value as f64,
        )
    }
}

struct UniformEvaluator;

impl Evaluator for UniformEvaluator {
    fn evaluate(&self, state: &GameState) -> (Vec<(Move, f64)>, f64) {
        let moves = state.position.legal_moves();
        let n = moves.len();
        let p = if n > 0 { 1.0 / n as f64 } else { 0.0 };
        (moves.into_iter().map(|m| (m, p)).collect(), 0.0)
    }
}
