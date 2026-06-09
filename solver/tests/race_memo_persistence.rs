//! Soundness gate for the PERSISTENT exact race memo.
//!
//! The race endgame value is memoized across every walls-exhausted leaf of a
//! `solve()` (and across `solve()` calls on the same `Solver`). A persistent
//! memo is only sound if every stored value is the position's EXACT
//! game-theoretic value. This test pins that directly: it walks seeded random
//! games and asserts, at checked nodes, that a **single REUSED `Solver`
//! instance** (so the persistent race memo is exercised and accumulates across
//! positions and games) returns exactly the unpruned brute-force oracle value
//! `brute_value`.
//!
//! Because the same `Solver` is reused for ALL positions across ALL games,
//! later positions hit race-memo entries populated by earlier ones — if any
//! stored entry were a bound rather than an exact value, the equality would
//! fail. We bias the random walk toward wall placement and forward steps so
//! games actually reach the walls-exhausted race regime that feeds the memo.
//!
//! Oracle affordability: `brute_value` is (almost) unpruned, so it is only
//! tractable on positions whose game tree is small — i.e. cheap everywhere on
//! 3x3 W1, but on larger/wall-rich boards only once walls are nearly/fully
//! exhausted (small branching, near-terminal). We therefore gate the expensive
//! comparison on a small remaining-wall budget; those low-wall and race
//! positions are exactly the ones that populate and read the race memo, so the
//! persistence path is fully exercised.

use quoridor_solver::board::Board;
use quoridor_solver::movegen::{apply, legal_moves};
use quoridor_solver::solver::{brute_value, Solver};
use quoridor_solver::state::{Move, State};

/// Minimal LCG for reproducible playouts (Numerical Recipes constants).
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

/// Walk seeded random games on `board`, asserting `sol.solve(s) ==
/// brute_value(s, depth)` at every checked node. A node is checked iff the total
/// remaining walls `<= max_walls_for_oracle` (keeps the unpruned oracle
/// tractable; these are the race-regime positions under test). The SAME `sol` is
/// reused across every position and game (persistence under test). To reach the
/// race regime, the playout prefers wall placements, then forward steps.
/// Returns the number of nodes actually checked.
fn run_board(
    w: u8,
    h: u8,
    walls: u8,
    depth: u32,
    max_walls_for_oracle: u32,
    games: usize,
    plies: usize,
    seed: u64,
) -> usize {
    let b = Board::new(w, h, walls);
    let mut sol = Solver::new(&b); // REUSED across all positions and all games.
    let mut rng = Lcg::new(seed);
    let mut checked = 0usize;
    for _ in 0..games {
        let mut s: State = b.initial();
        for _ in 0..plies {
            if b.is_terminal(&s) {
                break;
            }
            let walls_remaining = s.walls_left[0] as u32 + s.walls_left[1] as u32;
            if walls_remaining <= max_walls_for_oracle {
                let got = sol.solve(&s);
                let want = brute_value(&b, &s, depth);
                assert_eq!(
                    got, want,
                    "value mismatch on {w}x{h} W{walls} at pawns={:?} h={:#x} v={:#x} \
                     walls_left={:?} turn={} (got={:?} want={:?})",
                    s.pawn, s.h_walls, s.v_walls, s.walls_left, s.turn, got, want
                );
                checked += 1;
            }

            // Bias toward reaching the race regime: place a wall if possible,
            // else step. Among same-class moves pick uniformly at random.
            let moves = legal_moves(&b, &s);
            if moves.is_empty() {
                break;
            }
            let walls: Vec<Move> = moves
                .iter()
                .copied()
                .filter(|m| matches!(m, Move::Wall { .. }))
                .collect();
            // 70% of the time prefer a wall (to exhaust walls) when available.
            let pool = if !walls.is_empty() && rng.next() % 10 < 7 {
                &walls
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
fn race_memo_exact_under_persistence_3x3_w1() {
    // 3x3 W1: oracle is cheap at every position, so check them all (max budget
    // = total walls = 2). Depth 16 fully resolves this board.
    let checked = run_board(3, 3, 1, 16, 2, 60, 12, 0xC0FFEE);
    assert!(checked > 50, "only {checked} nodes checked, need > 50");
}

#[test]
fn race_memo_exact_under_persistence_4x4_w2() {
    // 4x4 W2: the unpruned oracle is intractable from wall-rich positions, so
    // we only compare once walls are exhausted (the pure race regime, where the
    // oracle is cheap and resolves to an exact Win/Loss). These are exactly the
    // positions the persistent race memo serves. Reuses one Solver across games.
    let checked = run_board(4, 4, 2, 24, 0, 120, 20, 0x5EED_1234);
    assert!(checked > 50, "only {checked} race nodes checked, need > 50");
}

/// Explicit check that the persistent race memo actually fills (proves the path
/// under test is exercised) AND that a Solver reused across two different start
/// positions agrees with the oracle on both — i.e. cross-call persistence is
/// sound, not just within a single solve.
#[test]
fn race_memo_persists_across_solve_calls() {
    let b = Board::new(4, 4, 2);
    let mut sol = Solver::new(&b);

    // Two race (walls-exhausted) start positions; the oracle is cheap on both.
    let mut a = b.initial();
    a.walls_left = [0, 0];
    let va = sol.solve(&a);
    assert_eq!(va, brute_value(&b, &a, 24), "first race position mismatch");
    let after_first = sol.race_tt_len();
    assert!(after_first > 0, "race memo should have filled during first solve");

    // A different race position reuses the same (already-populated) race memo.
    // If any persisted entry were a non-exact bound, this would fail.
    let mut c = b.initial();
    c.walls_left = [0, 0];
    c.pawn[0] = b.idx(b.w / 2, 1); // nudge p0 forward one cell
    c.turn = 1;
    assert_eq!(sol.solve(&c), brute_value(&b, &c, 24), "second race position mismatch");
    assert!(
        sol.race_tt_len() >= after_first,
        "race memo must persist (and may grow) across solve calls"
    );
}
