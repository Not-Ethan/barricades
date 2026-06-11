//! Exactness gate for the LAZY-SMP parallel search.
//!
//! The novel-result requirement is absolute: the parallel solve MUST return the
//! IDENTICAL game-theoretic value the single-threaded solve returns, on every
//! input. Parallel alpha-beta over a SHARED transposition table is exact —
//! every worker thread searches the root with the full `(Loss, Win)` window, so
//! each returns the EXACT minimax value regardless of which (always-sound)
//! bounds it happens to observe in the shared TT. The TT only prunes nodes; it
//! never alters the value alpha-beta computes. Threads share discovered bounds
//! (the speedup) but never a value.
//!
//! This file pins that directly:
//!   * `parallel_value_equals_serial_*`: for thread counts {1, 2, 8}, the value
//!     of every one of >=80 seeded random reachable positions per thread-count
//!     equals the QS_THREADS=1 value, across several boards (incl. an even-width
//!     6x4). VALUE equality only — parallel NODE counts vary run-to-run and are
//!     deliberately NOT asserted.
//!   * `full_solves_agree_8_vs_1`: the headline full solves 4x4-w2=Win,
//!     5x5-w1=Loss, 3x3-w1=Loss are identical under 8 threads and 1 thread (and
//!     match their known values).
//!
//! Thread count is set via the `set_threads` test hook (not the `QS_THREADS`
//! env var) so the harness can vary it safely within one process without racy
//! env mutation.

use quoridor_solver::board::Board;
use quoridor_solver::movegen::{apply, legal_moves};
use quoridor_solver::solver::{Solver, Value};
use quoridor_solver::state::State;

/// Tiny deterministic LCG for reproducible random play.
struct Lcg(u64);
impl Lcg {
    fn next(&mut self, n: usize) -> usize {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((self.0 >> 33) as usize) % n
    }
}

/// Collect seeded random reachable, non-terminal states by playing random legal
/// games from the initial position. Walls-biased so some positions reach the
/// race regime (which exercises the shared bounded race memo under threads).
fn random_states(b: &Board, seed: u64, games: usize, plies: usize) -> Vec<State> {
    let mut rng = Lcg(seed);
    let mut out = Vec::new();
    for _ in 0..games {
        let mut s = b.initial();
        for _ in 0..plies {
            if b.is_terminal(&s) {
                break;
            }
            out.push(s);
            let ms = legal_moves(b, &s);
            if ms.is_empty() {
                break;
            }
            s = apply(b, &s, ms[rng.next(ms.len())]);
        }
    }
    out
}

/// Solve `s` with exactly `threads` worker threads on a FRESH solver (so the
/// only variable under test is the thread count, not cross-call TT reuse).
fn solve_with_threads(b: &Board, s: &State, threads: usize) -> Value {
    let mut sol = Solver::new(b);
    sol.set_threads(threads);
    sol.solve(s)
}

/// The boards under test. Includes an even-width board (6x4) where wall legality
/// is subtlest, and a few small wall-rich boards.
const BOARDS: &[(u8, u8, u8, u64)] = &[
    (3, 3, 1, 0x3311_A5A5),
    (4, 4, 2, 0x4422_C0FF),
    (5, 5, 1, 0x5511_BEEF),
    (6, 4, 1, 0x6411_1234), // even width
    (3, 5, 2, 0x3522_FACE),
];

/// Core gate: for each thread count in {1, 2, 8}, every checked position's
/// parallel value equals the single-thread (1-thread) value. Asserts >=80
/// positions per thread-count.
fn check_parallel_equals_serial(threads: usize) {
    let mut checked = 0usize;
    for &(w, h, walls, seed) in BOARDS {
        let b = Board::new(w, h, walls);
        // Enough games/plies that each board contributes a healthy batch; the
        // total across boards comfortably clears the >=80 per-thread-count bar.
        for s in random_states(&b, seed, 24, 18) {
            let want = solve_with_threads(&b, &s, 1);
            let got = solve_with_threads(&b, &s, threads);
            assert_eq!(
                got, want,
                "parallel value (threads={threads}) != serial on {w}x{h}-w{walls}: \
                 pawns={:?} h={:#x} v={:#x} wl={:?} turn={} (got={got:?} want={want:?})",
                s.pawn, s.h_walls, s.v_walls, s.walls_left, s.turn
            );
            checked += 1;
        }
    }
    assert!(
        checked >= 80,
        "only {checked} positions checked for threads={threads}, need >= 80"
    );
}

#[test]
fn parallel_value_equals_serial_1_thread() {
    // threads=1 vs the 1-thread reference: a trivial identity, but it pins that
    // the single-worker path is the canonical reference and is self-consistent.
    check_parallel_equals_serial(1);
}

#[test]
fn parallel_value_equals_serial_2_threads() {
    check_parallel_equals_serial(2);
}

#[test]
fn parallel_value_equals_serial_8_threads() {
    check_parallel_equals_serial(8);
}

/// Headline full solves: identical under 8 threads and 1 thread, and equal to
/// their established game values.
#[test]
fn full_solves_agree_8_vs_1() {
    let cases: &[(u8, u8, u8, Value)] = &[
        (4, 4, 2, Value::Win),
        (5, 5, 1, Value::Loss),
        (3, 3, 1, Value::Loss),
    ];
    for &(w, h, walls, expected) in cases {
        let b = Board::new(w, h, walls);
        let s = b.initial();
        let v1 = solve_with_threads(&b, &s, 1);
        let v8 = solve_with_threads(&b, &s, 8);
        assert_eq!(v1, expected, "{w}x{h}-w{walls}: 1-thread value wrong");
        assert_eq!(
            v8, v1,
            "{w}x{h}-w{walls}: 8-thread value {v8:?} != 1-thread value {v1:?}"
        );
    }
}
