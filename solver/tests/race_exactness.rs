//! Bug-2 regression: the race endgame used a depth floor `2*(w+h)` that could
//! fire on a long-path frozen maze and return a bogus `Draw` — but a wall-less
//! race is never a true draw, so that Draw was a wrong game value that escaped
//! `ab()` and flipped ancestors (and was order-dependent: reused vs fresh Solver
//! disagreed). The fix replaces the floor with DFS-path cycle detection plus a
//! panic-on-unresolved guard. These tests pin the two audit repros and the
//! reused-vs-fresh differential that exposed the bug.

use quoridor_solver::board::Board;
use quoridor_solver::movegen::{apply, legal_moves};
use quoridor_solver::solver::{Solver, Value};
use quoridor_solver::state::{Move, State};

/// Minimal LCG for reproducible playouts.
struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Lcg(seed ^ 0x9E37_79B9_7F4A_7C15)
    }
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 16
    }
}

/// Bug-2 audit repro A: 5x5 walls=3, frozen maze (walls_left=[0,0]).
/// Was `Draw` (floor artifact); true value is `Loss` for BOTH turns.
#[test]
fn race_repro_a_5x5() {
    let b = Board::new(5, 5, 3);
    for turn in 0u8..2 {
        let s = State {
            pawn: [2, 22],
            h_walls: 0x5a5,
            v_walls: 0,
            walls_left: [0, 0],
            turn,
        };
        let mut sol = Solver::new(&b);
        assert_eq!(
            sol.solve(&s),
            Value::Loss,
            "5x5 W3 race repro A must be Loss for turn={turn} (was Draw under the bug)"
        );
    }
}

/// Bug-2 audit repro B: 3x5 walls=2, frozen maze. Was `Draw`; true value `Loss`.
#[test]
fn race_repro_b_3x5() {
    let b = Board::new(3, 5, 2);
    let s = State {
        pawn: [1, 13],
        h_walls: 0x81,
        v_walls: 0x18,
        walls_left: [0, 0],
        turn: 0,
    };
    let mut sol = Solver::new(&b);
    assert_eq!(
        sol.solve(&s),
        Value::Loss,
        "3x5 W2 race repro B must be Loss (was Draw under the bug)"
    );
}

/// Bug-2 differential: a single REUSED Solver must return the same value as a
/// FRESH Solver per call, over seeded random games. This is the harness that
/// exposed Bug 2's order-dependence — the bug let a tainted (depth-floor) `Draw`
/// escape into the search, and a reused Solver (whose persistent race memo had
/// accumulated state) disagreed with a fresh one. >= 50 positions.
///
/// We compare only on positions in (or close to) the race regime — total
/// remaining walls `<= max_walls` — for two reasons: (1) that is exactly where
/// Bug 2 lived (the race endgame and its persistent memo), and (2) it keeps each
/// `solve` tractable (a full-walls 5x5-W3 solve from the opening is minutes-long;
/// race-regime positions resolve in well under a second). The random walk biases
/// toward wall placement so games actually reach that regime. This mirrors the
/// `race_memo_persistence` harness that the audit used.
#[allow(clippy::too_many_arguments)]
fn run_reused_vs_fresh(
    w: u8,
    h: u8,
    walls: u8,
    max_walls: u32,
    games: usize,
    plies: usize,
    max_checks: usize,
    seed: u64,
) -> usize {
    let b = Board::new(w, h, walls);
    // ONE reused solver across all positions and games (persistence under test).
    let mut reused = Solver::new(&b);
    let mut rng = Lcg::new(seed);
    let mut checked = 0usize;
    for _ in 0..games {
        if checked >= max_checks {
            break;
        }
        let mut s: State = b.initial();
        for _ in 0..plies {
            if b.is_terminal(&s) || checked >= max_checks {
                break;
            }
            let walls_remaining = s.walls_left[0] as u32 + s.walls_left[1] as u32;
            if walls_remaining <= max_walls {
                let got_reused = reused.solve(&s);
                let got_fresh = Solver::new(&b).solve(&s);
                assert_eq!(
                    got_reused, got_fresh,
                    "{w}x{h} W{walls}: reused-vs-fresh disagree at pawns={:?} \
                     h={:#x} v={:#x} walls_left={:?} turn={} (reused={:?} fresh={:?})",
                    s.pawn, s.h_walls, s.v_walls, s.walls_left, s.turn, got_reused, got_fresh
                );
                checked += 1;
            }

            // Bias toward wall placement (to reach the frozen-maze race regime).
            let moves = legal_moves(&b, &s);
            if moves.is_empty() {
                break;
            }
            let walls_m: Vec<Move> = moves
                .iter()
                .copied()
                .filter(|m| matches!(m, Move::Wall { .. }))
                .collect();
            let pool = if !walls_m.is_empty() && rng.next() % 10 < 7 {
                &walls_m
            } else {
                &moves
            };
            let pick = (rng.next() % pool.len() as u64) as usize;
            s = apply(&b, &s, pool[pick]);
        }
    }
    checked
}

#[test]
fn reused_vs_fresh_solver_agree() {
    // 3x5 W2: race regime is positions with <= 2 walls remaining (solve is cheap
    // there); the bulk of coverage comes from here. 5x5 W3: only compare once
    // walls are fully exhausted (the pure race, where Bug 2's depth-floor Draw
    // arose). A FRESH 5x5 race solve has no memo and is relatively heavy, so we
    // cap the number of 5x5 checks; the heavy direct 5x5-W3 race coverage is
    // provided by `race_repro_a_5x5`.
    let c1 = run_reused_vs_fresh(3, 5, 2, 2, 200, 60, 200, 0xBEEF);
    let c2 = run_reused_vs_fresh(5, 5, 3, 0, 300, 80, 8, 0xF00D);
    assert!(c1 >= 42, "3x5 W2: only {c1} positions checked");
    assert!(c2 >= 4, "5x5 W3: only {c2} race positions checked");
    assert!(c1 + c2 >= 50, "only {} positions total, need >= 50", c1 + c2);
}
