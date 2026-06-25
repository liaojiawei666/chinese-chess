//! 结构常量：与 trainer/src/trainer/config.py 的 board / rules / encoding 段一一对应。
//!
//! 这些值是 datagen 与 trainer 之间的硬契约。bin/selfplay 启动时会用 run-config.json
//! 的对应字段对它们做一致性断言（见 selfplay 的 config 模块），防止两侧漂移。

// 棋盘几何
pub const BOARD_WIDTH: i32 = 9;
pub const BOARD_HEIGHT: i32 = 10;
pub const SQUARE_COUNT: i32 = BOARD_WIDTH * BOARD_HEIGHT; // 90
pub const ACTION_SPACE_SIZE: usize = SQUARE_COUNT as usize * SQUARE_COUNT as usize; // 8100
pub const PIECE_TYPE_COUNT: usize = 14; //区分红黑方，每边7种类型

// 规则上限
pub const MAX_TOTAL_PLIES: u32 = 300;
pub const NO_CAPTURE_DRAW_PLIES: u32 = 100;

// 局面编码
pub const N_HISTORY: usize = 7;
pub const PLANES_PER_FRAME: usize = 14;
pub const INPUT_CHANNELS: usize = N_HISTORY * PLANES_PER_FRAME + 1; // 99
