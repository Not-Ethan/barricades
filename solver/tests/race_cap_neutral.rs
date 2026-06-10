//! Neutrality gate for the BOUNDED, config-granular LRU race memo.
//!
//! The race memo (`endgame.rs::RaceTt`) is now capped by `QS_RACE_MB` and
//! evicted at WHOLE-CONFIG granularity in LRU order, so high-wall solves no
//! longer grow it without bound. The memo is a PURE cache of EXACT race values:
//! a cap-induced eviction only forces the (cheap, deterministic) retrograde to
//! re-run for that one frozen-wall config and recompute the identical value.
//! Capping can therefore NEVER change a returned value — only the work to
//! obtain it.
//!
//! This pins that directly: the full game value of a board solved with a TINY
//! race cap (1 MiB — heavy, near-continuous config eviction) equals the value
//! with a LARGE cap (no eviction). If any value differed, the eviction would be
//! dropping live information rather than a pure cache; fix the cache, never
//! weaken this gate.
//!
//! The cap is set via the `set_race_cap_mb` test hook (not the `QS_RACE_MB`
//! env var) so the harness can vary it within one process without racy env
//! mutation. Solves use the default (multi-thread) worker count; the race memo
//! is shared and thread-safe, so the tiny cap also stress-tests eviction under
//! concurrent fills.

use quoridor_solver::board::Board;
use quoridor_solver::solver::{Solver, Value};

/// Solve the initial position of `w x h` with `walls` walls and a race memo
/// capped at `race_mb` MiB (fresh solver).
fn solve_with_race_cap(w: u8, h: u8, walls: u8, race_mb: usize) -> Value {
    let b = Board::new(w, h, walls);
    let mut sol = Solver::new(&b);
    sol.set_race_cap_mb(race_mb);
    sol.solve(&b.initial())
}

#[test]
fn race_cap_neutral_5x5_w2() {
    // 5x5 W2: fast. Tiny cap (1 MiB, heavy eviction) == large cap (4 GiB).
    let tiny = solve_with_race_cap(5, 5, 2, 1);
    let large = solve_with_race_cap(5, 5, 2, 4096);
    assert_eq!(
        tiny, large,
        "5x5 W2: tiny-race-cap value {tiny:?} != large-cap value {large:?} \
         (race-memo eviction is NOT value-neutral — bug in the LRU cache)"
    );
}

/// 6x5 W2: the larger of the two required race-cap gates. Heavier than 5x5 W2,
/// so it runs with the default multi-thread worker count to stay tractable.
/// Tiny cap (1 MiB, near-continuous whole-config eviction) must equal the large
/// cap exactly.
#[test]
fn race_cap_neutral_6x5_w2() {
    let tiny = solve_with_race_cap(6, 5, 2, 1);
    let large = solve_with_race_cap(6, 5, 2, 4096);
    assert_eq!(
        tiny, large,
        "6x5 W2: tiny-race-cap value {tiny:?} != large-cap value {large:?} \
         (race-memo eviction is NOT value-neutral — bug in the LRU cache)"
    );
}

/// Tiny-SHARD eviction stress + reused-vs-fresh equality.
///
/// The race memo is sharded (16 shards, per-shard budget slice, per-shard
/// whole-config LRU via atomic last-used ticks). A 1 MiB TOTAL cap leaves each
/// shard only ~64 KiB — a handful of configs — so every shard evicts heavily
/// and concurrently under the default multi-thread worker count. This gate
/// pins, against a fresh LARGE-cap solver (no eviction):
///
///   1. the tiny-shard-cap value (fresh solver), and
///   2. the value from RE-solving with the SAME tiny-cap solver (reused,
///      warm-but-heavily-evicted shard state: post-eviction maps, advanced
///      LRU clocks, surviving resident configs).
///
/// Any difference would mean per-shard eviction or the atomic-tick LRU is
/// dropping/corrupting live information rather than acting as a pure cache;
/// fix the cache, never weaken this gate.
fn assert_tiny_shard_stress(w: u8, h: u8, walls: u8) {
    let large = solve_with_race_cap(w, h, walls, 4096);

    let b = Board::new(w, h, walls);
    let mut sol = Solver::new(&b);
    sol.set_race_cap_mb(1); // ~64 KiB per shard: heavy per-shard eviction.
    let fresh = sol.solve(&b.initial());
    assert_eq!(
        fresh, large,
        "{w}x{h} W{walls}: tiny-shard-cap value {fresh:?} != large-cap value \
         {large:?} (per-shard race eviction is NOT value-neutral)"
    );

    let reused = sol.solve(&b.initial());
    assert_eq!(
        reused, large,
        "{w}x{h} W{walls}: REUSED tiny-shard-cap solver value {reused:?} != \
         large-cap value {large:?} (post-eviction shard state is NOT value-neutral)"
    );
}

#[test]
fn race_cap_tiny_shard_stress_5x5_w2() {
    assert_tiny_shard_stress(5, 5, 2);
}

#[test]
fn race_cap_tiny_shard_stress_6x5_w2() {
    assert_tiny_shard_stress(6, 5, 2);
}
