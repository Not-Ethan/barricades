//! Horizontal-mirror equivariance + canonicalization gates.
//!
//! The board is left-right symmetric: reflecting columns `c -> w-1-c` (and wall
//! anchors `wc -> w-2-wc`) is a graph automorphism of Quoridor that preserves
//! the side to move, both goals, and therefore the game-theoretic value. This
//! file pins that property:
//!   1. `mirror(mirror(s)) == s` (involution).
//!   2. `Solver::solve(s) == Solver::solve(mirror(s))` over >=100 seeded random
//!      reachable states across several boards (fresh solver each, so the only
//!      thing being tested is the value-preserving map, not TT reuse).
//!   3. Reused-vs-fresh: with TT canonicalization ON, a reused `Solver`'s value
//!      must equal a fresh `Solver`'s value over random games — catches a bad
//!      canonical key or a mirror collision that conflates unequal states.
//!   4. Brute differential (reused solver): `Solver::solve == brute_value` with
//!      0 inversions on 4x3-w1 (full BFS) + 6x4-w1 (sampled). The oracle runs
//!      at a bounded depth and is trusted only on DECISIVE (Win/Loss) verdicts,
//!      which are exact at any depth (only `Draw` is depth-limited) — this keeps
//!      the un-memoized brute oracle tractable on the larger board.
//!
//! If ANY value differs, the symmetry map is wrong; fix the map, never weaken
//! this gate.

use quoridor_solver::board::Board;
use quoridor_solver::movegen::{apply, legal_moves};
use quoridor_solver::solver::{brute_value, mirror, Solver, Value};
use quoridor_solver::state::{Move, State};

/// Render a `State` for assertion messages (`State` derives no `Debug`).
fn fmt_state(s: &State) -> String {
    format!(
        "pawns={:?} h={:#x} v={:#x} wl={:?} turn={}",
        s.pawn, s.h_walls, s.v_walls, s.walls_left, s.turn
    )
}

/// Tiny deterministic LCG for reproducible random play.
struct Lcg(u64);
impl Lcg {
    fn next(&mut self, n: usize) -> usize {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((self.0 >> 33) as usize) % n
    }
}

/// Collect seeded random reachable, non-terminal states by playing random legal
/// games from the initial position.
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

#[test]
fn mirror_is_an_involution() {
    let mut total = 0usize;
    for (w, h, walls) in [(3, 3, 1), (4, 3, 1), (5, 5, 1), (6, 4, 1)] {
        let b = Board::new(w, h, walls);
        for s in random_states(&b, 0xABCD1234 ^ (w as u64), 20, 16) {
            let m = mirror(&b, &s);
            let mm = mirror(&b, &m);
            assert!(
                mm == s,
                "mirror not involutive on {w}x{h}-w{walls}: {}",
                fmt_state(&s)
            );
            total += 1;
        }
    }
    assert!(total >= 100, "only {total} involution checks");
}

#[test]
fn solve_is_mirror_equivariant() {
    let mut total = 0usize;
    for (w, h, walls) in [(3, 3, 1), (4, 3, 1), (5, 5, 1), (6, 4, 1)] {
        let b = Board::new(w, h, walls);
        for s in random_states(&b, 0x5EED00 ^ ((w as u64) << 8 | h as u64), 16, 14) {
            let m = mirror(&b, &s);
            // Fresh solver for each: isolates the value-preserving map from any
            // TT-reuse effect.
            let v_s = Solver::new(&b).solve(&s);
            let v_m = Solver::new(&b).solve(&m);
            assert_eq!(
                v_s,
                v_m,
                "value not mirror-invariant on {w}x{h}-w{walls}: {v_s:?} vs {v_m:?} for {}",
                fmt_state(&s)
            );
            total += 1;
        }
    }
    assert!(total >= 100, "only {total} equivariance checks, need >= 100");
}

#[test]
fn reused_solver_matches_fresh_with_canonicalization() {
    // With canonicalization ON, reusing a Solver (shared TT keyed on canonical
    // representatives) must give the SAME value as a fresh Solver. A faulty
    // canonical key or a mirror collision would surface here as a divergence.
    let mut total = 0usize;
    let mut inversions = 0usize;
    for (w, h, walls) in [(3, 3, 1), (4, 3, 1), (5, 5, 1), (6, 4, 1)] {
        let b = Board::new(w, h, walls);
        let mut reused = Solver::new(&b);
        for s in random_states(&b, 0xC0FFEE ^ ((w as u64) << 4 | h as u64), 14, 14) {
            let v_reused = reused.solve(&s);
            let v_fresh = Solver::new(&b).solve(&s);
            if v_reused != v_fresh {
                inversions += 1;
                eprintln!(
                    "reuse mismatch on {w}x{h}-w{walls}: reused={v_reused:?} fresh={v_fresh:?} {}",
                    fmt_state(&s)
                );
            }
            total += 1;
        }
    }
    assert_eq!(inversions, 0, "{inversions}/{total} reused-vs-fresh inversions");
    assert!(total >= 100, "only {total} reuse checks");
}

#[test]
fn brute_differential_4x3_w1_full_reused() {
    // FULL BFS over every reachable non-terminal state on 4x3-w1, REUSED solver
    // (shared canonicalized TT). The oracle runs at a modest depth and is
    // trusted only on DECISIVE (Win/Loss) verdicts — sound at any depth, since a
    // forced result proven by bounded negamax is the true value (only `Draw` is
    // depth-limited). This keeps full STATE coverage while bounding the
    // un-memoized oracle's cost (a deep brute over every with-wall state is
    // otherwise minutes-long). 4x3-w1 is decisive almost everywhere, so nearly
    // every state is verified. 0 inversions required.
    let b = Board::new(4, 3, 1);
    let probe_depth = 16; // ample for this 12-cell board's tactics.
    let mut sol = Solver::new(&b);

    let mut seen = std::collections::HashSet::new();
    let mut queue = std::collections::VecDeque::new();
    let start = b.initial();
    seen.insert(start);
    queue.push_back(start);

    let mut visited = 0usize;
    let mut decisive = 0usize;
    let mut inversions = 0usize;
    while let Some(s) = queue.pop_front() {
        if !b.is_terminal(&s) {
            let got = sol.solve(&s); // exact; warms canonical TT.
            let probe = brute_value(&b, &s, probe_depth);
            if probe != Value::Draw {
                decisive += 1;
                if got != probe {
                    inversions += 1;
                    eprintln!("4x3-w1 inversion: solver={got:?} oracle={probe:?} {}", fmt_state(&s));
                }
            }
            visited += 1;
        }
        for m in legal_moves(&b, &s) {
            let s2 = apply(&b, &s, m);
            if seen.insert(s2) {
                queue.push_back(s2);
            }
        }
    }
    assert_eq!(inversions, 0, "{inversions} brute inversions on 4x3-w1 (of {decisive} decisive)");
    assert!(visited >= 100, "only {visited} states visited on 4x3-w1");
    assert!(decisive >= 100, "only {decisive} decisive states on 4x3-w1");
    eprintln!("4x3-w1 full BFS brute differential: {visited} visited, {decisive} decisive, 0 inversions");
}

#[test]
fn brute_differential_6x4_w1_reused() {
    // 6x4-w1 with walls present is intractable for the (un-pruned) brute oracle
    // at the full ceiling, so we run the oracle at a MODEST depth and only trust
    // it when it returns a DECISIVE value. Soundness: a Win/Loss proven by
    // negamax within ANY depth bound is the true game-theoretic value (a forced
    // result is never mis-proven by truncation — only `Draw` is depth-limited).
    // So when `brute_value(s, d)` is Win or Loss we may compare it to the
    // Solver's exact value; when it is `Draw` we SKIP (undetermined at depth d,
    // not necessarily a true draw). This makes the differential sound at any
    // depth while staying cheap. The REUSED solver solves every visited position
    // (warming the canonicalized TT over wall configs and the race region the
    // games traverse). 0 inversions required.
    let b = Board::new(6, 4, 1);
    let probe_depth = 12; // modest: enough that many tactical positions resolve.
    let mut sol = Solver::new(&b);
    let mut rng = Lcg(0xBEEF_F00D);

    let mut checked = 0usize;
    let mut decisive = 0usize;
    let mut inversions = 0usize;
    for _ in 0..200 {
        let mut s = b.initial();
        for ply in 0..28 {
            if b.is_terminal(&s) {
                break;
            }
            let got = sol.solve(&s); // exact; also warms the canonical TT.
            let probe = brute_value(&b, &s, probe_depth);
            if probe != Value::Draw {
                // Decisive at bounded depth => exact => must match the Solver.
                decisive += 1;
                if got != probe {
                    inversions += 1;
                    eprintln!("6x4-w1 inversion: solver={got:?} oracle={probe:?} {}", fmt_state(&s));
                }
            }
            checked += 1;

            let ms = legal_moves(&b, &s);
            if ms.is_empty() {
                break;
            }
            // Early plies: bias toward placing a wall (exercise canonical TT on
            // real wall configs). Later: random, to advance pawns toward goals
            // and produce decisive near-terminal probe points.
            let walls: Vec<usize> = ms
                .iter()
                .enumerate()
                .filter(|(_, m)| matches!(m, Move::Wall { .. }))
                .map(|(i, _)| i)
                .collect();
            let pick = if ply < 4 && !walls.is_empty() && rng.next(2) == 0 {
                walls[rng.next(walls.len())]
            } else {
                rng.next(ms.len())
            };
            s = apply(&b, &s, ms[pick]);
        }
    }
    assert_eq!(inversions, 0, "{inversions} brute inversions on 6x4-w1 (of {decisive} decisive)");
    assert!(checked >= 100, "only {checked} states visited on 6x4-w1");
    assert!(decisive >= 50, "only {decisive} decisive probe points on 6x4-w1");
    eprintln!("6x4-w1 brute differential: {checked} visited, {decisive} decisive, 0 inversions");
}
