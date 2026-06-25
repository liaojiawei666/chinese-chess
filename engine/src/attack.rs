use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};

use crate::{
    board::Board,
    move_gen::{attackers_of, is_in_check},
    ByColor, Color, GameStatusReason, Move, MoveInfo, Piece, PieceKind, Position,
};

/// 攻击者信息
#[derive(Debug, Clone)]
struct Attacker {
    pub piece: Piece,
    pub pos: Position,
    /// attacker 是否可被对方任意棋子合法吃掉（不限于 victim 本身）
    pub can_be_recaptured: bool,
    /// 保护根：attacker 吃掉 victim 后，victim 方有其他子可合法反吃 attacker
    pub victim_has_true_root: bool,
}

#[derive(Debug, Clone)]
struct Victim {
    pub piece: Piece,
    pub pos: Position,
    pub attackers: Vec<Attacker>,
}

impl PartialEq for Victim {
    fn eq(&self, other: &Self) -> bool {
        self.piece == other.piece
    }
}

impl Eq for Victim {}

impl Hash for Victim {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.piece.hash(state);
    }
}

/// 分析 attacker_color 方对对方棋子（将/帅除外）的攻击情况。
/// board 应处于 attacker_color 方刚走完、轮到对方走的状态。
fn attack_analysis(board: &mut Board, attacker_color: Color) -> HashSet<Victim> {
    let victim_color = attacker_color.opposite();
    let victim_pieces = board.pieces_of(victim_color);

    let mut result = HashSet::new();
    let mut recapture_cache: HashMap<Position, bool> = HashMap::new();

    for (victim_piece, victim_pos) in victim_pieces {
        let attacker_list = attackers_of(board, victim_pos, attacker_color);
        if attacker_list.is_empty() {
            continue;
        }

        let attackers: Vec<Attacker> = attacker_list
            .into_iter()
            .map(|(attacker_piece, attacker_pos)| {
                let cached = *recapture_cache
                    .entry(attacker_pos)
                    .or_insert_with(|| can_be_recaptured(board, attacker_pos, victim_color));
                Attacker {
                    piece: attacker_piece,
                    pos: attacker_pos,
                    can_be_recaptured: cached,
                    victim_has_true_root: victim_has_true_root(
                        board, attacker_pos, victim_pos, victim_color,
                    ),
                }
            })
            .collect();

        result.insert(Victim {
            piece: victim_piece,
            pos: victim_pos,
            attackers,
        });
    }

    result
}

/// attacker 是否可被对方任意棋子合法吃掉
fn can_be_recaptured(
    board: &mut Board,
    attacker_pos: Position,
    victim_color: Color,
) -> bool {
    !attackers_of(board, attacker_pos, victim_color).is_empty()
}

/// 保护根：模拟 attacker 吃掉 victim 后，victim 方是否有子能合法反吃 attacker
fn victim_has_true_root(
    board: &mut Board,
    attacker_pos: Position,
    victim_pos: Position,
    victim_color: Color,
) -> bool {
    let mv = Move::new(attacker_pos, victim_pos);
    let move_info = board.make_move(mv);
    let has_protector = !attackers_of(board, victim_pos, victim_color).is_empty();
    board.undo_move(&move_info);
    has_protector
}

/// 单步级别的例外（与循环无关，可以在单步判定）
fn is_chase_exception(attacker: &Attacker, victim: &Victim) -> bool {
    // 规则1：捉未过河的兵/卒，不算长捉
    if victim.piece.kind == PieceKind::Pawn
        && !crate::board::crossed_river(victim.piece.color, victim.pos.y)
    {
        return true;
    }
    // 规则3：attacker 可被反吃，且被捉子价值 ≤ 攻击子价值 → 不算捉
    if attacker.can_be_recaptured {
        let av = Board::get_piece_value(attacker.piece, attacker.pos);
        let vv = Board::get_piece_value(victim.piece, victim.pos);
        if vv <= av {
            return true;
        }
    }
    false
}

/// 判断某个攻击者对 victim 是否构成有效"捉"
fn is_valid_chase(attacker: &Attacker, victim: &Victim) -> bool {
    if is_chase_exception(attacker, victim) {
        return false;
    }
    // 规则4：低价值捉高价值，即使有真根也算捉
    let av = Board::get_piece_value(attacker.piece, attacker.pos);
    let vv = Board::get_piece_value(victim.piece, victim.pos);
    if av < vv {
        return true;
    }
    // 正常逻辑：有保护根则不算捉
    !attacker.victim_has_true_root
}

/// 判断 victim 是否被有效"捉"住：存在至少一个构成有效捉的攻击者
fn is_chased(victim: &Victim) -> bool {
    victim.attackers.iter().any(|a| is_valid_chase(a, victim))
}

/// 提取单步中被有效"捉"住的棋子集合
fn chased_pieces(victims: &HashSet<Victim>) -> HashSet<Piece> {
    victims
        .iter()
        .filter(|v| is_chased(v))
        .map(|v| v.piece)
        .collect()
}

/// 检查整个循环中，对某个子的有效攻击者是否全为将/兵（循环级别例外）
fn is_only_king_pawn_chase(piece: Piece, move_sets: &[HashSet<Victim>]) -> bool {
    move_sets.iter().all(|victims| {
        let Some(victim) = victims.iter().find(|v| v.piece == piece) else {
            return true;
        };
        victim
            .attackers
            .iter()
            .filter(|a| is_valid_chase(a, victim))
            .all(|a| matches!(a.piece.kind, PieceKind::King | PieceKind::Pawn))
    })
}

/// 判定长捉：循环中每步都在捉同一个子。
/// "仅由将/兵构成的捉"是循环级别的例外：整个循环中对该子的所有有效攻击者
/// 都是将/兵才排除，任何一步有其他攻击者参与就算长捉。
fn is_perpetual_chase(victim_per_move: &ByColor<Vec<HashSet<Victim>>>) -> ByColor<bool> {
    let mut perpetual_chase = ByColor::new(false, false);
    for color in Color::all() {
        let victims_by_step = victim_per_move.get(color);

        let chased_candidate_set = victims_by_step
            .iter()
            .map(|victims| chased_pieces(victims))
            .reduce(|acc, set| &acc & &set);
        if chased_candidate_set.is_none() {
            continue;
        }

        let has_real_chase = chased_candidate_set
            .unwrap()
            .iter()
            .any(|&piece| !is_only_king_pawn_chase(piece, victims_by_step));
        *perpetual_chase.get_mut(color) = has_real_chase;
    }
    perpetual_chase
}

/// 分析循环中双方的犯规情况，返回每方的犯规类型。
/// board 为最新局面（取所有权），cycle_move_infos 按时间正序排列，
/// 函数内部从最新局面逐步 undo 回退分析。
/// 分析的局面数 = cycle_move_infos.len() + 1（含起始局面和终止局面）。
pub(crate) fn analyze_cycle(
    mut board: Board,
    cycle_move_infos: &[MoveInfo],
) -> ByColor<Option<GameStatusReason>> {
    let mut all_check: ByColor<bool> = ByColor::new(true, true);
    let mut victim_per_move: ByColor<Vec<HashSet<Victim>>> = ByColor::default();

    // 分析 len+1 个局面：从最新局面开始，分析完后 undo，最后一次只分析不 undo
    let len = cycle_move_infos.len();
    for i in 0..=len {
        let attacker_color = board.side_to_move.opposite();
        let victim_color = board.side_to_move;

        let check = is_in_check(&board, victim_color);
        *all_check.get_mut(attacker_color) &= check;

        let victims = attack_analysis(&mut board, attacker_color);
        victim_per_move.get_mut(attacker_color).push(victims);

        if i < len {
            board.undo_move(&cycle_move_infos[len - 1 - i]);
        }
    }

    // 判定每方犯规类型
    let perpetual_chase = is_perpetual_chase(&victim_per_move);
    let mut result: ByColor<Option<GameStatusReason>> = ByColor::new(None, None);

    for color in Color::all() {
        if *all_check.get(color) {
            *result.get_mut(color) = Some(GameStatusReason::PerpetualCheck);
        } else if *perpetual_chase.get(color) {
            *result.get_mut(color) = Some(GameStatusReason::PerpetualChase);
        }
    }

    result
}
