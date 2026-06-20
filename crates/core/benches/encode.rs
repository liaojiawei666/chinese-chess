use cc_core::encode::{encode, legal_mask};
use cc_core::engine::{GameState, Position};
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_encode_starting(c: &mut Criterion) {
    let state = GameState::from_position(Position::starting());
    c.bench_function("encode_starting", |b| {
        b.iter(|| black_box(encode(&state)))
    });
}

fn bench_encode_with_history(c: &mut Criterion) {
    let mut state = GameState::from_position(Position::starting());
    for _ in 0..7 {
        let mv = state.position.legal_moves()[0];
        state = state.make_move(mv);
    }
    c.bench_function("encode_with_history", |b| {
        b.iter(|| black_box(encode(&state)))
    });
}

fn bench_legal_mask(c: &mut Criterion) {
    let pos = Position::starting();
    c.bench_function("legal_mask", |b| {
        b.iter(|| black_box(legal_mask(&pos)))
    });
}

criterion_group!(
    encode_benches,
    bench_encode_starting,
    bench_encode_with_history,
    bench_legal_mask
);
criterion_main!(encode_benches);
