mod config;
mod evaluator;

use std::sync::mpsc::{self, Receiver};
use std::thread;

use anyhow::Result;
use eframe::egui;
use engine::game::GameState;
use engine::mcts::Mcts;
use engine::types::*;

use config::{read_latest, PlayConfig};
use evaluator::{load_onnx_model, OnnxEvaluator};

// ── AI background thread result ──

struct AiResult {
    mv: Option<Move>,
    model_version: Option<u64>,
    visits: u32,
}

// ── Main app state ──

struct PlayGui {
    state: GameState,
    human: Color,
    selected: Option<Position>,
    legal_targets: Vec<Position>,
    ai_rx: Option<Receiver<AiResult>>,
    ai_thinking: bool,
    config: PlayConfig,
    status: String,
    model_status: ModelStatus,
    last_ai_move: Option<Move>,
}

#[derive(Clone)]
enum ModelStatus {
    NotLoaded,
    Loaded { generation: u64 },
}

fn main() -> Result<()> {
    env_logger::init();

    let args: Vec<String> = std::env::args().collect();
    let config_path = args.get(1).map(|s| s.as_str()).unwrap_or("config.json");

    let mut config = PlayConfig::from_file(config_path)?;

    // Parse optional --sims N flag
    if let Some(idx) = args.iter().position(|a| a == "--sims") {
        if let Some(val) = args.get(idx + 1) {
            let n: usize = val.parse().expect("--sims requires a number");
            config = config.with_simulations(n);
        }
    }

    log::info!(
        "play_gui: models_dir={}, sims={}",
        config.models_dir.display(),
        config.mcts.num_simulations
    );

    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([760.0, 900.0])
            // WSLg 无 xdg-desktop-portal，客户端装饰(sctk-adwaita)查询配色超时后
            // 会偶发把 Wayland 连接搞断导致崩溃；关掉装饰绕开这条路径。
            .with_decorations(false),
        ..Default::default()
    };

    eframe::run_native(
        "Chinese Chess — Human vs AI",
        native_options,
        Box::new(move |cc| {
            install_chinese_font(&cc.egui_ctx);
            Ok(Box::new(PlayGui::new(config)))
        }),
    )
    .map_err(|e| anyhow::anyhow!("GUI failed: {e}"))
}

fn install_chinese_font(ctx: &egui::Context) {
    let candidates = [
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        // WSL：直接复用 Windows 自带中文字体（无需在 Linux 侧安装）
        "/mnt/c/Windows/Fonts/simhei.ttf",
        "/mnt/c/Windows/Fonts/msyh.ttc",
        "/mnt/c/Windows/Fonts/simsun.ttc",
        "/mnt/c/Windows/Fonts/NotoSansSC-VF.ttf",
    ];
    let Some(bytes) = candidates.iter().find_map(|p| std::fs::read(p).ok()) else {
        return;
    };
    let mut fonts = egui::FontDefinitions::default();
    fonts
        .font_data
        .insert("chinese".into(), egui::FontData::from_owned(bytes).into());
    for family in [egui::FontFamily::Proportional, egui::FontFamily::Monospace] {
        fonts
            .families
            .entry(family)
            .or_default()
            .insert(0, "chinese".into());
    }
    ctx.set_fonts(fonts);
}

// ── PlayGui impl ──

impl PlayGui {
    fn new(config: PlayConfig) -> Self {
        let mut app = PlayGui {
            state: GameState::start_pos(),
            human: Color::Red,
            selected: None,
            legal_targets: Vec::new(),
            ai_rx: None,
            ai_thinking: false,
            config,
            status: "你执红先行。".into(),
            model_status: ModelStatus::NotLoaded,
            last_ai_move: None,
        };
        app.maybe_start_ai();
        app
    }

    fn reset(&mut self, human: Color) {
        self.state = GameState::start_pos();
        self.human = human;
        self.selected = None;
        self.legal_targets.clear();
        self.ai_rx = None;
        self.ai_thinking = false;
        self.last_ai_move = None;
        self.status = match human {
            Color::Red => "新局：你执红先行。".into(),
            Color::Black => "新局：你执黑，AI 先行。".into(),
        };
        self.maybe_start_ai();
    }

    fn maybe_start_ai(&mut self) {
        if self.ai_thinking || self.state.status.is_terminal {
            return;
        }
        if self.state.board.side_to_move == self.human {
            return;
        }

        let state = self.state.clone();
        let config = self.config.clone();
        let (tx, rx) = mpsc::channel();
        self.ai_rx = Some(rx);
        self.ai_thinking = true;
        self.status = format!(
            "AI 思考中：{} sims ...",
            self.config.mcts.num_simulations
        );

        thread::spawn(move || {
            let result = search_ai_move(state, &config);
            let _ = tx.send(result);
        });
    }

    fn poll_ai(&mut self) {
        let Some(rx) = &self.ai_rx else { return };
        let Ok(result) = rx.try_recv() else { return };
        self.ai_rx = None;
        self.ai_thinking = false;

        let Some(mv) = result.mv else {
            self.status = "AI 无合法走法。".into();
            return;
        };

        if let Some(gen) = result.model_version {
            self.model_status = ModelStatus::Loaded { generation: gen };
        }

        self.state.make_move(mv);
        self.last_ai_move = Some(mv);

        let model_label = match result.model_version {
            Some(v) => format!("v{v}"),
            None => "随机".into(),
        };
        self.status = format!(
            "AI: {}，访问 {}，模型 {}。",
            move_text(mv),
            result.visits,
            model_label,
        );
        self.update_terminal_status();
    }

    fn handle_square_click(&mut self, x: i8, y: i8) {
        if self.ai_thinking || self.state.status.is_terminal {
            return;
        }
        if self.state.board.side_to_move != self.human {
            self.maybe_start_ai();
            return;
        }

        let pos = Position::new(x, y);
        let clicked_piece = self.state.board.get_piece_at(pos);

        if let Some(sel) = self.selected {
            let mv = Move::new(sel, pos);
            let legal = self.state.legal_moves();
            if legal.contains(&mv) {
                self.state.make_move(mv);
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
                self.selected = Some(pos);
                let legal = self.state.legal_moves();
                self.legal_targets = legal
                    .into_iter()
                    .filter(|m| m.from == pos)
                    .map(|m| m.to)
                    .collect();
                self.status = format!(
                    "已选择 {} {}",
                    color_name(piece.color),
                    piece_name(piece.kind)
                );
            }
            _ => {
                self.selected = None;
                self.legal_targets.clear();
                self.status = "请选择自己的棋子。".into();
            }
        }
    }

    fn update_terminal_status(&mut self) {
        if self.state.status.is_terminal {
            self.status = terminal_text(&self.state.status, self.human);
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
                ui.separator();

                ui.label(format!("你执：{}", color_name(self.human)));
                ui.label(format!(
                    "轮到：{}",
                    color_name(self.state.board.side_to_move)
                ));
                ui.label(format!("模拟数：{}", self.config.mcts.num_simulations));

                ui.separator();

                match &self.model_status {
                    ModelStatus::NotLoaded => {
                        ui.colored_label(
                            egui::Color32::from_rgb(200, 120, 0),
                            "⚠ 模型：未加载（随机下棋）",
                        );
                    }
                    ModelStatus::Loaded { generation } => {
                        ui.colored_label(
                            egui::Color32::from_rgb(0, 160, 0),
                            format!("✓ 模型：v{generation}"),
                        );
                    }
                }

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
                ui.separator();
                ui.small("点击棋子后点击目标格落子。");
                ui.small("AI 每步自动尝试加载最新模型。");
                ui.small("无模型时使用随机策略。");
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            draw_board(ui, self);
        });
    }
}

// ── AI search ──

fn search_ai_move(state: GameState, config: &PlayConfig) -> AiResult {
    // Try loading the latest trained model; fall back to uniform if unavailable.
    let (evaluator, version): (Box<dyn engine::evaluate::Evaluator>, Option<u64>) =
        match try_load_model(config) {
            Some((eval, gen)) => (Box::new(eval), Some(gen)),
            None => {
                log::info!("no model found, using uniform evaluator");
                (Box::new(engine::evaluate::UniformEvaluator), None)
            }
        };

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    let mcts_config = config.mcts.clone();
    let (mv, visits) = rt.block_on(async {
        let mut mcts = Mcts::new(state, mcts_config, evaluator);
        let counts = mcts.run().await;
        if counts.is_empty() {
            return (None, 0);
        }
        let best = Mcts::best_move(&counts);
        let v = counts.iter().find(|(m, _)| *m == best).unwrap().1;
        (Some(best), v)
    });

    AiResult { mv, model_version: version, visits }
}

fn try_load_model(config: &PlayConfig) -> Option<(OnnxEvaluator, u64)> {
    let (gen, path) = read_latest(&config.models_dir)?;
    match load_onnx_model(path.to_str().unwrap_or("?")) {
        Ok(session) => {
            log::info!("loaded model gen={gen}");
            Some((OnnxEvaluator::new(session), gen))
        }
        Err(e) => {
            log::warn!("failed to load model gen={gen}: {e:#}");
            None
        }
    }
}

// ── Board drawing ──

const CELL_SIZE: f32 = 55.0;
const MARGIN: f32 = 40.0;
const PIECE_RADIUS: f32 = 22.0;

fn board_to_screen(x: i8, y: i8, flip: bool) -> egui::Pos2 {
    let (bx, by) = if flip {
        (8 - x as i32, 9 - y as i32)
    } else {
        (x as i32, y as i32)
    };
    egui::pos2(
        MARGIN + bx as f32 * CELL_SIZE,
        MARGIN + by as f32 * CELL_SIZE,
    )
}

fn screen_to_board(pos: egui::Pos2, flip: bool) -> Option<(i8, i8)> {
    let bx = ((pos.x - MARGIN + CELL_SIZE / 2.0) / CELL_SIZE).floor() as i32;
    let by = ((pos.y - MARGIN + CELL_SIZE / 2.0) / CELL_SIZE).floor() as i32;
    if bx < 0 || bx > 8 || by < 0 || by > 9 {
        return None;
    }
    if flip {
        Some(((8 - bx) as i8, (9 - by) as i8))
    } else {
        Some((bx as i8, by as i8))
    }
}

fn draw_board(ui: &mut egui::Ui, app: &mut PlayGui) {
    let flip = app.human == Color::Black;
    let board_w = MARGIN * 2.0 + 8.0 * CELL_SIZE;
    let board_h = MARGIN * 2.0 + 9.0 * CELL_SIZE;

    let (response, painter) =
        ui.allocate_painter(egui::vec2(board_w, board_h), egui::Sense::click());

    let bg = egui::Color32::from_rgb(240, 217, 181);
    painter.rect_filled(response.rect, 0.0, bg);

    let line_color = egui::Color32::from_rgb(60, 40, 20);
    let line_stroke = egui::Stroke::new(1.5, line_color);

    for x in 0..9 {
        let top = board_to_screen(x, 0, flip);
        let bot = board_to_screen(x, 9, flip);
        if x == 0 || x == 8 {
            painter.line_segment([top, bot], line_stroke);
        } else {
            let mid_top = board_to_screen(x, 4, flip);
            let mid_bot = board_to_screen(x, 5, flip);
            painter.line_segment([top, mid_top], line_stroke);
            painter.line_segment([mid_bot, bot], line_stroke);
        }
    }
    for y in 0..10 {
        let left = board_to_screen(0, y, flip);
        let right = board_to_screen(8, y, flip);
        painter.line_segment([left, right], line_stroke);
    }

    for &(x1, y1, x2, y2) in &[(3, 0, 5, 2), (3, 7, 5, 9)] {
        let a = board_to_screen(x1, y1, flip);
        let b = board_to_screen(x2, y2, flip);
        painter.line_segment([a, b], line_stroke);
        let c = board_to_screen(x2, y1, flip);
        let d = board_to_screen(x1, y2, flip);
        painter.line_segment([c, d], line_stroke);
    }

    let river_top = board_to_screen(4, 4, flip);
    let river_bot = board_to_screen(4, 5, flip);
    let river_center = egui::pos2(
        (river_top.x + river_bot.x) / 2.0,
        (river_top.y + river_bot.y) / 2.0,
    );
    painter.text(
        river_center,
        egui::Align2::CENTER_CENTER,
        "楚 河          漢 界",
        egui::FontId::proportional(16.0),
        egui::Color32::from_rgb(120, 80, 40),
    );

    if let Some(sel) = app.selected {
        let center = board_to_screen(sel.x, sel.y, flip);
        painter.circle_filled(
            center,
            PIECE_RADIUS + 3.0,
            egui::Color32::from_rgba_unmultiplied(0, 150, 255, 80),
        );
    }

    for &pos in &app.legal_targets {
        let center = board_to_screen(pos.x, pos.y, flip);
        painter.circle_filled(
            center,
            8.0,
            egui::Color32::from_rgba_unmultiplied(0, 200, 0, 120),
        );
    }

    if let Some(mv) = app.last_ai_move {
        for pos in [mv.from, mv.to] {
            let center = board_to_screen(pos.x, pos.y, flip);
            painter.circle_stroke(
                center,
                PIECE_RADIUS + 2.0,
                egui::Stroke::new(2.0, egui::Color32::from_rgb(255, 100, 0)),
            );
        }
    }

    for y in 0..10i8 {
        for x in 0..9i8 {
            if let Some(piece) = app.state.board.get_piece_at(Position::new(x, y)) {
                let center = board_to_screen(x, y, flip);
                let (bg_color, text_color) = match piece.color {
                    Color::Red => (
                        egui::Color32::from_rgb(255, 230, 200),
                        egui::Color32::from_rgb(180, 30, 30),
                    ),
                    Color::Black => (
                        egui::Color32::from_rgb(220, 220, 220),
                        egui::Color32::from_rgb(30, 30, 30),
                    ),
                };
                painter.circle_filled(center, PIECE_RADIUS, bg_color);
                painter.circle_stroke(center, PIECE_RADIUS, egui::Stroke::new(1.5, text_color));
                painter.text(
                    center,
                    egui::Align2::CENTER_CENTER,
                    piece_char(piece.kind, piece.color),
                    egui::FontId::proportional(20.0),
                    text_color,
                );
            }
        }
    }

    if response.clicked() {
        if let Some(pos) = response.interact_pointer_pos() {
            if let Some((x, y)) = screen_to_board(pos, flip) {
                app.handle_square_click(x, y);
            }
        }
    }
}

// ── Helpers ──

fn piece_char(kind: PieceKind, color: Color) -> &'static str {
    match (kind, color) {
        (PieceKind::King, Color::Red) => "帅",
        (PieceKind::King, Color::Black) => "将",
        (PieceKind::Advisor, Color::Red) => "仕",
        (PieceKind::Advisor, Color::Black) => "士",
        (PieceKind::Elephant, Color::Red) => "相",
        (PieceKind::Elephant, Color::Black) => "象",
        (PieceKind::Horse, Color::Red) => "马",
        (PieceKind::Horse, Color::Black) => "馬",
        (PieceKind::Rook, Color::Red) => "车",
        (PieceKind::Rook, Color::Black) => "車",
        (PieceKind::Cannon, Color::Red) => "炮",
        (PieceKind::Cannon, Color::Black) => "砲",
        (PieceKind::Pawn, Color::Red) => "兵",
        (PieceKind::Pawn, Color::Black) => "卒",
    }
}

fn piece_name(kind: PieceKind) -> &'static str {
    match kind {
        PieceKind::King => "将/帅",
        PieceKind::Advisor => "士/仕",
        PieceKind::Elephant => "象/相",
        PieceKind::Horse => "马",
        PieceKind::Rook => "车",
        PieceKind::Cannon => "炮",
        PieceKind::Pawn => "兵/卒",
    }
}

fn color_name(c: Color) -> &'static str {
    match c {
        Color::Red => "红",
        Color::Black => "黑",
    }
}

fn move_text(mv: Move) -> String {
    format!(
        "({},{})→({},{})",
        mv.from.x, mv.from.y, mv.to.x, mv.to.y
    )
}

fn terminal_text(status: &GameStatus, human: Color) -> String {
    let reason = status.reason.map(|r| r.as_str()).unwrap_or("unknown");
    match status.winner {
        Some(w) if w == human => format!("你赢了！({})", reason),
        Some(_) => format!("你输了。({})", reason),
        None => format!("和棋。({})", reason),
    }
}
