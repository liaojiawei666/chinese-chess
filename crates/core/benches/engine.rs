use cc_core::engine::{GameState, Position};
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_legal_moves_starting(c: &mut Criterion) {
    let pos = Position::starting();
    c.bench_function("legal_moves_starting", |b| {
        b.iter(|| black_box(pos.legal_moves()))
    });
}

fn bench_legal_moves_midgame(c: &mut Criterion) {
    let pos = Position::from_fen(
        "r1ea1a3/4kh3/2h1e4/p1p1p1p1p/4c4/6P2/P1P1P3P/1C4H2/9/R1EAKAEHR r",
    )
    .expect("midgame fen");
    c.bench_function("legal_moves_midgame", |b| {
        b.iter(|| black_box(pos.legal_moves()))
    });
}

fn bench_make_move(c: &mut Criterion) {
    let state = GameState::from_position(Position::starting());
    let moves = state.position.legal_moves();
    let mv = moves[0];
    c.bench_function("make_move", |b| {
        b.iter(|| {
            let s = state.clone();
            black_box(s.make_move(mv).status())
        })
    });
}

fn bench_status(c: &mut Criterion) {
    let state = GameState::from_position(Position::starting());
    c.bench_function("status", |b| {
        b.iter(|| black_box(state.status()))
    });
}

criterion_group!(
    engine_benches,
    bench_legal_moves_starting,
    bench_legal_moves_midgame,
    bench_make_move,
    bench_status
);
criterion_main!(engine_benches);
