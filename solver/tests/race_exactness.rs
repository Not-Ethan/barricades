//! Exactness gate for the race endgame solver (`endgame.rs::race_value`).
//!
//! The race is `walls_left == [0, 0]` (no MORE walls), but FROZEN walls remain
//! on the board. A frozen-wall maze CAN be a genuine draw (one pawn perpetually
//! body-blocks the only corridor the other must cross), with legal moves
//! available — not zugzwang. The solver computes Win/Loss/**Draw** by EXACT
//! retrograde (backward-induction) labeling over the finite `(pawn0, pawn1,
//! turn)` graph: no depth bound, no panic.
//!
//! This was preceded by two earlier-and-wrong race fixes: a fixed depth floor
//! `2*(w+h)` (Bug 2, returned a bogus Draw on long-path mazes), then iterative
//! deepening + a panic-on-unresolved guard (rested on the FALSE invariant "a
//! wall-less race is never a true draw" — it panicked / hung for minutes on a
//! real blockade draw, reachable on 7x5 W>=4).
//!
//! Tests pinned here:
//!   * `race_repro_a_5x5`, `race_repro_b_3x5` — the two decisive audit repros
//!     (still `Loss`).
//!   * `race_blockade_draw_7x5` — the NEW critical case: a genuine frozen-wall
//!     blockade DRAW that the old code panicked on; must be `Draw` AND fast.
//!   * `retrograde_vs_reference_negamax` — >=200 seeded frozen-wall configs +
//!     a constructed blockade, retrograde value == convention-matching exact
//!     negamax (no-move=Loss, exact depth `2*(w*h)^2`).
//!   * `reused_vs_fresh_solver_agree`, `reused_vs_fresh_reachable` — reused
//!     `Solver` value == fresh `Solver` value (persistent-memo soundness).

use quoridor_solver::board::Board;
use quoridor_solver::movegen::{apply, legal_moves, legal_steps, legal_walls};
use quoridor_solver::solver::{Solver, Value};
use quoridor_solver::state::{Move, State};
use std::collections::HashMap;
use std::time::Instant;

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

/// NEW critical case: a GENUINE frozen-wall blockade DRAW.
///
/// `walls_left == [0, 0]` (no more walls), but the frozen maze lets pawn 0
/// perpetually body-block the single corridor pawn 1 must cross, so neither side
/// can force a goal — a true Draw with legal moves available (not zugzwang). The
/// previous iterative-deepening + panic code deepened to its ceiling and PANICKED
/// (hanging for minutes) on this position; the exact retrograde solver labels it
/// `Draw` in microseconds. We assert both the value AND that it returns fast
/// (retrograde is O(states); a regression to the deepening search would blow this
/// generous budget).
#[test]
fn race_blockade_draw_7x5() {
    let b = Board::new(7, 5, 4);
    let s = State {
        pawn: [18, 24],
        h_walls: 0x280240,
        v_walls: 0x500400,
        walls_left: [0, 0],
        turn: 1,
    };
    let t0 = Instant::now();
    let mut sol = Solver::new(&b);
    let v = sol.solve(&s);
    let dt = t0.elapsed();
    assert_eq!(
        v,
        Value::Draw,
        "7x5 frozen-wall blockade must be a genuine Draw (old code panicked here)"
    );
    // Retrograde is O(race states) ~ a few thousand nodes here; well under a
    // second. A multi-minute deepening regression would fail this.
    assert!(
        dt.as_secs() < 5,
        "blockade race must resolve fast (retrograde is O(states)); took {dt:?}"
    );
}

/// Convention-matching EXACT reference: plain negamax over steps only, no-move =
/// Loss for the mover, depth-bounded with a `Draw` floor. Memoized on
/// `(state, depth)` so it is tractable at the exact depth. At depth `2*(w*h)^2`
/// (one more than the longest possible simple line over the `(w*h)^2 * 2` race
/// states) every Win/Loss is fully proven, so the floor only ever yields the
/// TRUE game-theoretic `Draw` — i.e. this reference is exact, independent of the
/// retrograde implementation under test.
fn ref_race_nega(
    b: &Board,
    s: &State,
    depth: u32,
    memo: &mut HashMap<(State, u32), Value>,
) -> Value {
    if let Some(p) = b.winner(s) {
        return if p == s.turn { Value::Win } else { Value::Loss };
    }
    if depth == 0 {
        return Value::Draw;
    }
    if let Some(&v) = memo.get(&(*s, depth)) {
        return v;
    }
    let mut best = Value::Loss; // no legal step => Loss for the mover.
    for d in legal_steps(b, s) {
        let v = ref_race_nega(b, &apply(b, s, Move::Step(d)), depth - 1, memo).negate();
        if v > best {
            best = v;
        }
        if best == Value::Win {
            break;
        }
    }
    memo.insert((*s, depth), best);
    best
}

/// Differential: retrograde race value == convention-matching exact negamax over
/// many seeded frozen-wall race configs (random legal walls, `walls_left=[0,0]`,
/// random distinct pawns) on small boards, PLUS a constructed blockade -> Draw.
/// >= 200 configs.
#[test]
fn retrograde_vs_reference_negamax() {
    let boards = [(3u8, 3u8), (4, 3), (3, 4), (4, 4), (5, 4), (3, 5)];
    let mut rng = Lcg::new(0xA11CE);
    let mut total = 0usize;
    let mut draws = 0usize;

    for &(w, h) in &boards {
        let b = Board::new(w, h, 4);
        let cells = (w as u32) * (h as u32);
        let depth = 2 * cells * cells; // exact reference depth (no truncation).
        for _ in 0..60 {
            // Build a frozen wall config: a few random NON-overlapping LEGAL walls.
            let mut base = b.initial();
            let nwalls = (rng.next() % 4) as usize;
            for _ in 0..nwalls {
                let ws = legal_walls(&b, &base);
                if ws.is_empty() {
                    break;
                }
                let m = ws[(rng.next() as usize) % ws.len()];
                base = apply(&b, &base, m);
            }
            base.walls_left = [0, 0];

            // A handful of random distinct on-board pawn placements + turn.
            let cn = cells as u8;
            for _ in 0..6 {
                let p0 = (rng.next() as u8) % cn;
                let p1 = (rng.next() as u8) % cn;
                if p0 == p1 {
                    continue;
                }
                let turn = (rng.next() % 2) as u8;
                let mut q = base;
                q.pawn = [p0, p1];
                q.turn = turn;

                let mut sol = Solver::new(&b);
                let got = sol.solve(&q);
                let mut memo = HashMap::new();
                let want = ref_race_nega(&b, &q, depth, &mut memo);
                assert_eq!(
                    got, want,
                    "retrograde != reference on {w}x{h} pawns={:?} h={:#x} v={:#x} turn={turn}",
                    q.pawn, q.h_walls, q.v_walls
                );
                if got == Value::Draw {
                    draws += 1;
                }
                total += 1;
            }
        }
    }
    assert!(total >= 200, "only {total} configs checked, need >= 200");

    // Constructed blockade -> Draw (the 7x5 repro), also matched against the
    // reference negamax to prove the reference itself certifies the Draw.
    {
        let b = Board::new(7, 5, 4);
        let q = State {
            pawn: [18, 24],
            h_walls: 0x280240,
            v_walls: 0x500400,
            walls_left: [0, 0],
            turn: 1,
        };
        let mut sol = Solver::new(&b);
        let got = sol.solve(&q);
        assert_eq!(got, Value::Draw, "constructed blockade must be Draw");
        let cells = 7u32 * 5;
        let depth = 2 * cells * cells;
        let mut memo = HashMap::new();
        let want = ref_race_nega(&b, &q, depth, &mut memo);
        assert_eq!(want, Value::Draw, "reference negamax must also certify Draw");
        draws += 1;
    }
    assert!(
        draws >= 1,
        "differential must include at least one Draw (got {draws})"
    );
}

/// Reused-vs-fresh `Solver` agreement over REACHABLE random-game positions on
/// 5x5 W2 and 6x5 W2 (the persistent race memo must give identical values
/// whether warm or cold). >= 50 checks. Complements `reused_vs_fresh_solver_agree`
/// (which biases hard toward the frozen-wall regime); here we walk reachable
/// positions and compare in the low-wall regime where each solve is tractable.
#[test]
fn reused_vs_fresh_reachable() {
    let c1 = run_reused_vs_fresh(5, 5, 2, 2, 200, 60, 60, 0x1357);
    let c2 = run_reused_vs_fresh(6, 5, 2, 2, 200, 60, 60, 0x2468);
    assert!(c1 >= 20, "5x5 W2: only {c1} positions checked");
    assert!(c2 >= 20, "6x5 W2: only {c2} positions checked");
    assert!(c1 + c2 >= 50, "only {} positions total, need >= 50", c1 + c2);
}
