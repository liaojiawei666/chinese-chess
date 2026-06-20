//! 对局状态与历史相关规则（MoveRecord / GameState）。
//!
//! 长将 / 长捉 / 重复判和都依赖历史，故 MCTS 节点持有 GameState 而非裸 Position。

use std::collections::{BTreeMap, BTreeSet, HashMap};

use super::constants::{MAX_TOTAL_PLIES, NO_CAPTURE_DRAW_PLIES};
use super::position::Position;
use super::types::{
    crossed_river, Color, GameStatus, GameStatusReason, Move, Piece, PieceKind,
};

#[derive(Debug, Clone)]
pub struct Attacker {
    pub piece: Piece,
    pub square: (i32, i32),
    pub target_can_recapture: bool,
}

#[derive(Debug, Clone)]
pub struct AttackedTarget {
    pub target: Piece,
    pub square: (i32, i32),
    pub attackers: Vec<Attacker>,
    pub has_true_root: bool,
}

#[derive(Debug, Clone)]
pub struct MoveRecord {
    pub position_before: Position,
    pub position_after: Position,
    pub mv: Move,
    pub actor: Color,
}

impl MoveRecord {
    /// actor 在 position_after 下能合法吃到的对方非帅目标子（按 target_id 归并）。
    pub fn attacked_targets(&self) -> BTreeMap<u32, AttackedTarget> {
        let after = &self.position_after;
        let mut attackers_by_target: BTreeMap<u32, Vec<Attacker>> = BTreeMap::new();

        for (&_attacker_id, &(ax, ay)) in after.piece_positions.iter() {
            let attacker = after.require_piece_at(ax, ay);
            if attacker.color != self.actor {
                continue;
            }
            for (&target_id, &(tx, ty)) in after.piece_positions.iter() {
                let target = after.require_piece_at(tx, ty);
                if target.color == self.actor || target.kind == PieceKind::King {
                    continue;
                }
                if after.is_legal_move(Move::new(ax, ay, tx, ty), None) {
                    let target_can_recapture =
                        after.is_legal_move(Move::new(tx, ty, ax, ay), None);
                    attackers_by_target.entry(target_id).or_default().push(Attacker {
                        piece: attacker,
                        square: (ax, ay),
                        target_can_recapture,
                    });
                }
            }
        }

        let mut attacked_targets: BTreeMap<u32, AttackedTarget> = BTreeMap::new();
        for (target_id, attackers) in attackers_by_target.into_iter() {
            let target_square = after.piece_positions[&target_id];
            let target = after.require_piece_at(target_square.0, target_square.1);
            let has_true_root = self.has_true_root(target_id);
            attacked_targets.insert(
                target_id,
                AttackedTarget {
                    target,
                    square: target_square,
                    attackers,
                    has_true_root,
                },
            );
        }
        attacked_targets
    }

    fn has_true_root(&self, target_id: u32) -> bool {
        let after = &self.position_after;
        let (tx, ty) = after.piece_positions[&target_id];
        let target = after.require_piece_at(tx, ty);

        for (&_attacker_id, &(ax, ay)) in after.piece_positions.iter() {
            let attacker = after.require_piece_at(ax, ay);
            if attacker.color == target.color {
                continue;
            }
            let capture_move = Move::new(ax, ay, tx, ty);
            if !after.is_legal_move(capture_move, None) {
                continue;
            }
            let captured_position = after.apply_move(capture_move, attacker);
            for (&_defender_id, &(dx, dy)) in captured_position.piece_positions.iter() {
                let defender = captured_position.require_piece_at(dx, dy);
                if defender.color != target.color {
                    continue;
                }
                if captured_position.is_legal_move(Move::new(dx, dy, tx, ty), None) {
                    return true;
                }
            }
        }
        false
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Violation {
    Check,
    Chase,
}

#[derive(Debug, Clone)]
pub struct GameState {
    pub position: Position,
    pub history: Vec<MoveRecord>,
    pub repetition_counts: HashMap<String, u32>,
    pub key_indices: HashMap<String, Vec<usize>>,
}

impl GameState {
    pub fn from_position(position: Position) -> GameState {
        let key = position.repetition_key();
        let mut repetition_counts = HashMap::new();
        repetition_counts.insert(key.clone(), 1);
        let mut key_indices = HashMap::new();
        key_indices.insert(key, vec![0]);
        GameState {
            position,
            history: Vec::new(),
            repetition_counts,
            key_indices,
        }
    }

    pub fn make_move(&self, mv: Move) -> GameState {
        let next_position = self.position.make_move(mv);
        let record = MoveRecord {
            position_before: self.position.clone(),
            position_after: next_position.clone(),
            mv,
            actor: self.position.side_to_move,
        };
        let mut history = self.history.clone();
        history.push(record);
        let mut repetition_counts = self.repetition_counts.clone();
        let mut key_indices = self.key_indices.clone();
        let key = next_position.repetition_key();
        *repetition_counts.entry(key.clone()).or_insert(0) += 1;
        key_indices.entry(key).or_default().push(history.len());
        GameState {
            position: next_position,
            history,
            repetition_counts,
            key_indices,
        }
    }

    pub fn status(&self) -> GameStatus {
        let position_status = self.position.status();
        if position_status.is_terminal {
            return position_status;
        }

        let key = self.position.repetition_key();
        if self.repetition_counts.get(&key).copied().unwrap_or(0) >= 3 {
            return self.repetition_status(&key);
        }
        if self.position.no_capture_plies >= NO_CAPTURE_DRAW_PLIES {
            return GameStatus::terminal(GameStatusReason::FiftyMove, None);
        }
        if self.history.len() as u32 >= MAX_TOTAL_PLIES {
            return GameStatus::terminal(GameStatusReason::MaxMoves, None);
        }
        GameStatus::ongoing()
    }

    fn repetition_status(&self, key: &str) -> GameStatus {
        let indices = &self.key_indices[key];
        if indices.len() < 2 {
            return GameStatus::ongoing();
        }
        let start = indices[indices.len() - 2];
        let end = indices[indices.len() - 1];
        let segment = &self.history[start..end];
        let red_violation = self.violation_for(Color::Red, segment);
        let black_violation = self.violation_for(Color::Black, segment);

        if red_violation == black_violation {
            return GameStatus::terminal(GameStatusReason::MutualPerpetual, None);
        }
        if red_violation == Some(Violation::Check) && black_violation != Some(Violation::Check) {
            return GameStatus::terminal(GameStatusReason::PerpetualCheck, Some(Color::Black));
        }
        if black_violation == Some(Violation::Check) && red_violation != Some(Violation::Check) {
            return GameStatus::terminal(GameStatusReason::PerpetualCheck, Some(Color::Red));
        }
        if red_violation == Some(Violation::Chase) {
            return GameStatus::terminal(GameStatusReason::PerpetualChase, Some(Color::Black));
        }
        if black_violation == Some(Violation::Chase) {
            return GameStatus::terminal(GameStatusReason::PerpetualChase, Some(Color::Red));
        }
        GameStatus::terminal(GameStatusReason::MutualPerpetual, None)
    }

    fn violation_for(&self, color: Color, segment: &[MoveRecord]) -> Option<Violation> {
        let own_records: Vec<&MoveRecord> =
            segment.iter().filter(|r| r.actor == color).collect();
        assert!(
            !own_records.is_empty(),
            "Repetition segment has no moves by {}",
            color.as_str()
        );
        if own_records
            .iter()
            .all(|r| r.position_after.is_in_check(color.opposite()))
        {
            return Some(Violation::Check);
        }
        if self.is_perpetual_chase(&own_records) {
            return Some(Violation::Chase);
        }
        None
    }

    fn is_perpetual_chase(&self, records: &[&MoveRecord]) -> bool {
        let mut attacked_target_ids_by_record: Vec<BTreeSet<u32>> = Vec::new();
        let mut attacked_targets_by_id: BTreeMap<u32, Vec<AttackedTarget>> = BTreeMap::new();

        for record in records {
            let attacked_targets = record.attacked_targets();
            if attacked_targets.is_empty() {
                return false;
            }
            let ids: BTreeSet<u32> = attacked_targets.keys().copied().collect();
            attacked_target_ids_by_record.push(ids);
            for (target_id, attacked_target) in attacked_targets.into_iter() {
                attacked_targets_by_id
                    .entry(target_id)
                    .or_default()
                    .push(attacked_target);
            }
        }

        // 所有 record 都在捉的目标子（id 交集）。
        let mut candidates: BTreeSet<u32> = match attacked_target_ids_by_record.first() {
            Some(first) => first.clone(),
            None => return false,
        };
        for ids in attacked_target_ids_by_record.iter().skip(1) {
            candidates = candidates.intersection(ids).copied().collect();
        }

        for target_id in candidates {
            let by_record = &attacked_targets_by_id[&target_id];
            if Self::target_exempt_from_chase(by_record) {
                continue;
            }
            // 马/炮长捉有根车仍按长捉处理，优先于真根豁免。
            if Self::target_is_rook_chased_by_horse_or_cannon(by_record) {
                return true;
            }
            if by_record.iter().all(|t| !t.has_true_root) {
                return true;
            }
        }
        false
    }

    fn target_exempt_from_chase(by_record: &[AttackedTarget]) -> bool {
        let final_target = by_record.last().expect("non-empty by_record");
        let target = final_target.target;
        if target.kind == PieceKind::Pawn && !crossed_river(target.color, final_target.square.1) {
            return true;
        }

        // 规则允许将/帅和兵/卒长捉，全部攻击子都属于这两类时豁免。
        let all_king_or_pawn = by_record.iter().all(|t| {
            t.attackers
                .iter()
                .all(|a| matches!(a.piece.kind, PieceKind::King | PieceKind::Pawn))
        });
        if all_king_or_pawn {
            return true;
        }

        Self::is_same_kind_exchange(by_record)
    }

    fn is_same_kind_exchange(by_record: &[AttackedTarget]) -> bool {
        for attacked_target in by_record {
            let any_same_kind_recapture = attacked_target.attackers.iter().any(|a| {
                a.piece.is_same_kind_as(&attacked_target.target) && a.target_can_recapture
            });
            if !any_same_kind_recapture {
                return false;
            }
        }
        true
    }

    fn target_is_rook_chased_by_horse_or_cannon(by_record: &[AttackedTarget]) -> bool {
        for attacked_target in by_record {
            if attacked_target.target.kind != PieceKind::Rook {
                return false;
            }
            let any_horse_or_cannon = attacked_target
                .attackers
                .iter()
                .any(|a| matches!(a.piece.kind, PieceKind::Horse | PieceKind::Cannon));
            if !any_horse_or_cannon {
                return false;
            }
        }
        true
    }
}
