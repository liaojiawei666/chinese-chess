use rand::rngs::SmallRng;
use rand::{RngCore, SeedableRng};
use std::sync::OnceLock;

use crate::{Piece, Position, PIECE_TYPE_COUNT, SQUARE_COUNT};

#[derive(Debug, PartialEq, Eq)]
pub struct ZobristTable {
    pieces: [[u64; PIECE_TYPE_COUNT]; SQUARE_COUNT as usize],
    pub side_to_move: u64,
}

static ZOBRIST_TABLE: OnceLock<ZobristTable> = OnceLock::new();

impl ZobristTable {
    fn new() -> ZobristTable {
        let mut rng = SmallRng::seed_from_u64(123456);
        let mut pieces = [[0u64; PIECE_TYPE_COUNT]; SQUARE_COUNT as usize];
        for i in 0..SQUARE_COUNT as usize {
            for j in 0..PIECE_TYPE_COUNT {
                pieces[i][j] = rng.next_u64();
            }
        }
        ZobristTable {
            pieces,
            side_to_move: rng.next_u64(),
        }
    }
    pub fn get() -> &'static ZobristTable {
        ZOBRIST_TABLE.get_or_init(|| ZobristTable::new())
    }

    pub fn get_piece_hash(&self, piece: Piece, pos: Position) -> u64 {
        self.pieces[pos.to_index() as usize][piece.to_index() as usize - 1]
    }
    pub fn get_side_to_move_hash(&self) -> u64 {
        self.side_to_move
    }
}
