//! Gates for THEOREM 4 — one-sided frozen-race bounds as depth-infinity TT
//! synthesis (`docs/superpowers/solver-pruning-theorems.md` §B.4).
//!
//! The theorem: at a non-terminal node where EXACTLY ONE side has exhausted its
//! walls, the frozen-race value `r = race_value(s with walls_left := [0,0])`
//! bounds the node's value UNIFORMLY IN DEPTH — a Lower bound for the side to
//! move when the OPPONENT is exhausted, an Upper bound when the MOVER is.
//! Decisive bounds (Win-Lower / Loss-Upper) replace the whole subtree.
//! Falsification-validated by exact whole-graph relabeling at 2,121,148 states
//! across five boards, zero mismatches.
//!
//! Gates pinned here:
//!   (a) A/B EXACT-VALUE REGRESSION: `solve` with the feature ON vs OFF (the
//!       `set_use_t4` hook; env knob `QS_T4=1` enables, default OFF) returns the
//!       IDENTICAL value on >=150 seeded random reachable positions across
//!       3x3-w1, 4x3-w2, 4x4-w2, 3x5-w2, 6x4-w1 — plus every known-value gate
//!       (writeup defaults, 6x5 W0/W1/W2 = Loss, keystone 6x4 = Win, blockade
//!       7x5 = Draw) and the doc's P-suite positions (P1-P4, P5a/b/c).
//!   (b) COUNTERS: `t4_fires`/`t4_cutoffs` are >0 on wall-heavy boards with the
//!       feature ON, and identically 0 with it OFF.
//!
//! The A/B toggle uses the `set_use_t4` test hook (not racy env mutation); the
//! `QS_T4` env knob funnels to the same flag in `Solver::with_tt_mb`.

use quoridor_solver::board::Board;
use quoridor_solver::movegen::{apply, legal_moves};
use quoridor_solver::solver::{Solver, Value};
use quoridor_solver::state::{Move, State};
use std::collections::HashSet;

/// Per-test main-TT cap in MiB: plenty for every board here, small enough that
/// the suite's many short-lived solvers do not each zero gigabytes.
const TT_MB: usize = 64;

/// Tiny deterministic LCG for reproducible random play.
struct Lcg(u64);
impl Lcg {
    fn next(&mut self, n: usize) -> usize {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((self.0 >> 33) as usize) % n
    }
}

/// Solve `s` with Theorem 4 ON or OFF (fresh solver either way).
fn solve_t4(b: &Board, s: &State, on: bool) -> Value {
    let mut sol = Solver::with_tt_mb(b, TT_MB);
    sol.set_use_t4(on);
    sol.solve(s)
}

/// Assert the A/B exact-value contract on one position and return the value.
fn assert_on_off(b: &Board, s: &State, label: &str) -> Value {
    let v_on = solve_t4(b, s, true);
    let v_off = solve_t4(b, s, false);
    assert_eq!(
        v_on, v_off,
        "Theorem-4 ON/OFF value mismatch at {label}: pawn={:?} h={:#x} v={:#x} wl={:?} turn={}",
        s.pawn, s.h_walls, s.v_walls, s.walls_left, s.turn
    );
    v_on
}

/// Collect DISTINCT seeded random reachable, non-terminal states by playing
/// random legal games from the initial position (same scheme as the
/// parallel-exactness gate; deduped so each position is solved once).
fn random_states(b: &Board, seed: u64, games: usize, plies: usize) -> Vec<State> {
    let mut rng = Lcg(seed);
    let mut seen: HashSet<State> = HashSet::new();
    let mut out = Vec::new();
    for _ in 0..games {
        let mut s = b.initial();
        for _ in 0..plies {
            if b.is_terminal(&s) {
                break;
            }
            if seen.insert(s) {
                out.push(s);
            }
            let ms = legal_moves(b, &s);
            if ms.is_empty() {
                break;
            }
            s = apply(b, &s, ms[rng.next(ms.len())]);
        }
    }
    out
}

// ---------------------------------------------------------------------------
// (a) A/B exact-value regression: random reachable positions, >=150 total.
// ---------------------------------------------------------------------------

#[test]
fn t4_ab_regression_random_positions() {
    // (board, seed, games, plies) — five boards spanning odd/even widths, wall
    // budgets 1-2, and the 6x4 even-width keystone geometry.
    let cases: &[(u8, u8, u8, u64)] = &[
        (3, 3, 1, 0xA11CE),
        (4, 3, 2, 0xB0B),
        (4, 4, 2, 0xC0FFEE),
        (3, 5, 2, 0xD00D),
        (6, 4, 1, 0xE66),
    ];
    let mut total = 0usize;
    for &(w, h, walls, seed) in cases {
        let b = Board::new(w, h, walls);
        for s in random_states(&b, seed, 10, 8) {
            assert_on_off(&b, &s, &format!("{w}x{h}-w{walls} random"));
            total += 1;
        }
    }
    assert!(total >= 150, "A/B regression must cover >=150 positions, got {total}");
    eprintln!("Theorem-4 A/B regression: {total} positions, 0 mismatches");
}

// ---------------------------------------------------------------------------
// (a) A/B on the known-value gates.
// ---------------------------------------------------------------------------

#[test]
fn t4_known_value_gates() {
    // Writeup defaults + 6x5 ladder (W0..W2 here; W3 is minutes-scale and runs
    // in the ignored extended gate below).
    let gates: &[(u8, u8, u8, Value)] = &[
        (3, 3, 1, Value::Loss),
        (4, 4, 1, Value::Win),
        (4, 4, 2, Value::Win),
        (5, 5, 0, Value::Loss),
        (5, 5, 1, Value::Loss),
        (6, 5, 0, Value::Loss),
        (6, 5, 1, Value::Loss),
        (6, 5, 2, Value::Loss),
    ];
    for &(w, h, walls, expect) in gates {
        let b = Board::new(w, h, walls);
        let v = assert_on_off(&b, &b.initial(), &format!("{w}x{h}-w{walls} initial"));
        assert_eq!(v, expect, "{w}x{h}-w{walls} initial must be {expect:?}");
    }
}

/// Extended gate (minutes-scale): 6x5 W3 = Loss, ON vs OFF. Run explicitly:
/// `cargo test --release --test theorem4 -- --ignored`.
#[ignore]
#[test]
fn t4_known_value_gate_6x5_w3() {
    let b = Board::new(6, 5, 3);
    let v = assert_on_off(&b, &b.initial(), "6x5-w3 initial");
    assert_eq!(v, Value::Loss, "6x5-w3 initial must be Loss");
}

#[test]
fn t4_keystone_6x4_win() {
    // Keystone 6x4: the even-width board-spanning wall-legality repro.
    let b = Board::new(6, 4, 1);
    let s = State {
        pawn: [8, 6],
        h_walls: 0x220,
        v_walls: 0,
        walls_left: [1, 1],
        turn: 0,
    };
    assert_eq!(assert_on_off(&b, &s, "keystone 6x4"), Value::Win);
}

#[test]
fn t4_blockade_7x5_draw() {
    // The documented genuine frozen-wall blockade draw ([0,0] race).
    let b = Board::new(7, 5, 4);
    let s = State {
        pawn: [18, 24],
        h_walls: 0x280240,
        v_walls: 0x500400,
        walls_left: [0, 0],
        turn: 1,
    };
    assert_eq!(assert_on_off(&b, &s, "blockade 7x5"), Value::Draw);
}

// ---------------------------------------------------------------------------
// (a) A/B on the doc's P-suite (Track-B falsification positions).
// ---------------------------------------------------------------------------

/// P1 (2x2 dead-wall pocket, 6x5-w4): parent value Win for p0.
#[test]
fn t4_p1_pocket() {
    let b = Board::new(6, 5, 4);
    let s = State {
        pawn: [14, 20],
        h_walls: 0x4000,  // H(4,2)
        v_walls: 0x40000, // V(3,3)
        walls_left: [2, 2],
        turn: 0,
    };
    assert_eq!(assert_on_off(&b, &s, "P1"), Value::Win);
    // P3 variants of P1: turn flipped, budgets [1,1], parity-tight pawns.
    for (label, s3) in [
        ("P3: P1 turn:1", State { turn: 1, ..s }),
        ("P3: P1 wl [1,1]", State { walls_left: [1, 1], ..s }),
        ("P3: P1 pawn [6,18]", State { pawn: [6, 18], ..s }),
    ] {
        assert_on_off(&b, &s3, label);
    }
}

/// P2 (4-wide pocket, six dead slots, 6x5-w4): parent value Loss for p0.
#[test]
fn t4_p2_pocket() {
    let b = Board::new(6, 5, 4);
    let s = State {
        pawn: [8, 3],
        h_walls: 0x2800,  // H(1,2), H(3,2)
        v_walls: 0x88000, // V(0,3), V(4,3)
        walls_left: [2, 2],
        turn: 0,
    };
    assert_eq!(assert_on_off(&b, &s, "P2"), Value::Loss);
    // P3 variants of P2.
    for (label, s3) in [
        ("P3: P2 turn:1", State { turn: 1, ..s }),
        ("P3: P2 wl [1,1]", State { walls_left: [1, 1], ..s }),
    ] {
        assert_on_off(&b, &s3, label);
    }
}

/// P4 (race-tempo dominance counterexample, 5x4-w2): V = Win, yet EVERY pawn
/// step loses — only a wall attains the value. Must keep standing with T4 on.
#[test]
fn t4_p4_tempo_trap() {
    let b = Board::new(5, 4, 2);
    let s = State {
        pawn: [7, 18],
        h_walls: 0x400, // H(2,2)
        v_walls: 0x80,  // V(3,1)
        walls_left: [1, 1],
        turn: 0,
    };
    assert_eq!(assert_on_off(&b, &s, "P4"), Value::Win);
    let mut wall_wins = 0usize;
    for m in legal_moves(&b, &s) {
        let child = apply(&b, &s, m);
        let v = assert_on_off(&b, &child, "P4 child").negate();
        match m {
            Move::Step(_) => {
                assert_eq!(v, Value::Loss, "P4: every pawn step must lose ({m:?})");
            }
            Move::Wall { .. } => {
                if v == Value::Win {
                    wall_wins += 1;
                }
            }
        }
    }
    assert!(wall_wins > 0, "P4: a wall placement must attain the Win");
}

/// P5a/P5b (6x5-w3 initial pawns, one-sided budgets): the bound directions.
/// `[2,0]` (opp exhausted)   => V >= race_value;
/// `[0,2]` (mover exhausted) => V <= race_value.
#[test]
fn t4_p5ab_bound_directions() {
    let b = Board::new(6, 5, 3);
    let base = State {
        pawn: [3, 27],
        h_walls: 0,
        v_walls: 0,
        walls_left: [0, 0],
        turn: 0,
    };
    // Frozen-race value (exact, via the [0,0] retrograde path).
    let r = assert_on_off(&b, &base, "P5 race");
    let v_a = assert_on_off(&b, &State { walls_left: [2, 0], ..base }, "P5a [2,0]");
    let v_b = assert_on_off(&b, &State { walls_left: [0, 2], ..base }, "P5b [0,2]");
    assert!(v_a >= r, "P5a: V(wl=[2,0]) = {v_a:?} must be >= race {r:?}");
    assert!(v_b <= r, "P5b: V(wl=[0,2]) = {v_b:?} must be <= race {r:?}");
}

/// P5c (sharp Draw-boundary case on the 7x5 blockade, attacks trap 2): with
/// mover p1 exhausted (`wl=[2,0]`, turn 1) the value is Loss (<= Draw — p0's
/// two unused walls cannot hand p1 a win even via diagonal-jump creation);
/// with `[0,2]` it is Win (>= Draw).
#[test]
fn t4_p5c_draw_boundary() {
    let b = Board::new(7, 5, 4);
    let base = State {
        pawn: [18, 24],
        h_walls: 0x280240,
        v_walls: 0x500400,
        walls_left: [0, 0],
        turn: 1,
    };
    let v_mover_exhausted = assert_on_off(&b, &State { walls_left: [2, 0], ..base }, "P5c [2,0]");
    assert_eq!(v_mover_exhausted, Value::Loss, "P5c [2,0] must be Loss (<= Draw bound)");
    let v_opp_exhausted = assert_on_off(&b, &State { walls_left: [0, 2], ..base }, "P5c [0,2]");
    assert_eq!(v_opp_exhausted, Value::Win, "P5c [0,2] must be Win (>= Draw bound)");
}

// ---------------------------------------------------------------------------
// (b) Instrumentation: the counters fire on wall-heavy boards.
// ---------------------------------------------------------------------------

#[test]
fn t4_counters_fire_wall_heavy() {
    // Single-thread for a deterministic traversal: the search of these
    // wall-bearing boards must reach one-sided-exhaustion nodes.
    for (w, h, walls, expect) in [(4u8, 4u8, 2u8, Value::Win), (6, 5, 1, Value::Loss)] {
        let b = Board::new(w, h, walls);
        let mut sol = Solver::with_tt_mb(&b, TT_MB);
        sol.set_threads(1);
        sol.set_use_t4(true);
        assert_eq!(sol.solve(&b.initial()), expect);
        assert!(
            sol.t4_fires > 0,
            "{w}x{h}-w{walls}: Theorem 4 must fire at least once (fires=0)"
        );
        assert!(
            sol.t4_cutoffs > 0,
            "{w}x{h}-w{walls}: Theorem 4 must produce at least one cutoff (fires={}, cutoffs=0)",
            sol.t4_fires
        );
        eprintln!(
            "{w}x{h}-w{walls}: t4_fires={} t4_cutoffs={} nodes={}",
            sol.t4_fires, sol.t4_cutoffs, sol.nodes
        );
    }
    // And under the default lazy-SMP thread count (counters are summed).
    let b = Board::new(4, 4, 2);
    let mut sol = Solver::with_tt_mb(&b, TT_MB);
    sol.set_use_t4(true);
    assert_eq!(sol.solve(&b.initial()), Value::Win);
    assert!(sol.t4_fires > 0, "4x4-w2 (parallel): Theorem 4 must fire");
}

#[test]
fn t4_counters_zero_when_off() {
    let b = Board::new(4, 4, 2);
    let mut sol = Solver::with_tt_mb(&b, TT_MB);
    sol.set_use_t4(false);
    assert_eq!(sol.solve(&b.initial()), Value::Win);
    assert_eq!(sol.t4_fires, 0, "OFF must never evaluate the bound");
    assert_eq!(sol.t4_cutoffs, 0, "OFF must never cut");
}
