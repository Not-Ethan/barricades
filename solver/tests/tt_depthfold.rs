//! Exactness gate for the dense, packed-key, DEPTH-FOLDED, fixed-capacity main
//! transposition table.
//!
//! The depth-fold (one entry per CANONICAL position, with the remaining search
//! depth stored IN the entry and a stored result reused only when
//! `stored.depth >= query_depth`) is the correctness-sensitive change. If it is
//! wrong, a reused `Solver` (which accumulates entries across positions and
//! depths) would diverge from a FRESH `Solver` per position, or from the
//! un-pruned brute oracle. These tests pin that it does NOT:
//!
//!   (a) reused-vs-fresh equality over seeded random games on several boards;
//!   (b) brute_value differential (definite verdicts only) on 4x3-w1 FULL and a
//!       6x4-w1 sample, 0 inversions;
//!   (c) TINY-cap stress (`QS_TT_MB`-equivalent of 1 MiB, heavy eviction) equals
//!       a large-cap solve — proving eviction is correctness-neutral;
//!   (d) the two critical repros (keystone 6x4 => Win, blockade 7x5 => Draw).
//!
//! If ANY value differs the depth-fold or the u128 pack is wrong; fix it, never
//! weaken this gate.

use quoridor_solver::board::Board;
use quoridor_solver::movegen::{apply, legal_moves};
use quoridor_solver::solver::{brute_value, Solver, Value};
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

fn fmt(s: &State) -> String {
    format!(
        "pawns={:?} h={:#x} v={:#x} wl={:?} turn={}",
        s.pawn, s.h_walls, s.v_walls, s.walls_left, s.turn
    )
}

/// (a) A single REUSED solver (its dense depth-fold TT persists and accumulates
/// entries across every position and depth) must return the SAME value as a
/// FRESH solver per position over seeded random games on several boards
/// (including even-width 6x4). Catches an unsound depth-fold reuse, a bad u128
/// pack collision, or a faulty eviction. >= 200 positions total.
#[test]
fn reused_vs_fresh_depthfold() {
    let boards = [(3, 3, 1), (4, 4, 2), (5, 5, 1), (6, 4, 1), (3, 5, 2)];
    let mut total = 0usize;
    let mut inversions = 0usize;
    for (w, h, walls) in boards {
        let b = Board::new(w, h, walls);
        // One reused solver across ALL positions of this board: maximizes
        // cross-position / cross-depth TT reuse, the thing under test.
        let mut reused = Solver::new(&b);
        for s in random_states(&b, 0xD00D ^ ((w as u64) << 8 | h as u64), 16, 16) {
            let v_reused = reused.solve(&s);
            let v_fresh = Solver::new(&b).solve(&s);
            if v_reused != v_fresh {
                inversions += 1;
                eprintln!(
                    "reuse mismatch {w}x{h}-w{walls}: reused={v_reused:?} fresh={v_fresh:?} {}",
                    fmt(&s)
                );
            }
            total += 1;
        }
    }
    assert_eq!(inversions, 0, "{inversions}/{total} reused-vs-fresh inversions");
    assert!(total >= 200, "only {total} positions checked, need >= 200");
    eprintln!("reused-vs-fresh depth-fold: {total} positions, 0 inversions");
}

/// (b1) FULL BFS over every reachable non-terminal 4x3-w1 state, REUSED solver.
/// The brute oracle runs at a modest depth and is trusted only on DECISIVE
/// (Win/Loss) verdicts — exact at any depth (only `Draw` is depth-limited). 0
/// inversions.
#[test]
fn brute_differential_4x3_w1_full() {
    let b = Board::new(4, 3, 1);
    let probe_depth = 16;
    let mut sol = Solver::new(&b);

    let mut seen = std::collections::HashSet::new();
    let mut queue = std::collections::VecDeque::new();
    let start = b.initial();
    seen.insert(start);
    queue.push_back(start);

    let (mut visited, mut decisive, mut inversions) = (0usize, 0usize, 0usize);
    while let Some(s) = queue.pop_front() {
        if !b.is_terminal(&s) {
            let got = sol.solve(&s);
            let probe = brute_value(&b, &s, probe_depth);
            if probe != Value::Draw {
                decisive += 1;
                if got != probe {
                    inversions += 1;
                    eprintln!("4x3-w1 inversion: solver={got:?} oracle={probe:?} {}", fmt(&s));
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
    assert!(visited >= 100 && decisive >= 100, "4x3-w1: {visited} visited, {decisive} decisive");
    eprintln!("4x3-w1 full brute differential: {visited} visited, {decisive} decisive, 0 inversions");
}

/// (b2) 6x4-w1 SAMPLE (seeded random games), REUSED solver, DECISIVE-only brute
/// oracle at a modest depth. 0 inversions.
#[test]
fn brute_differential_6x4_w1_sample() {
    let b = Board::new(6, 4, 1);
    let probe_depth = 12;
    let mut sol = Solver::new(&b);
    let (mut decisive, mut inversions, mut visited) = (0usize, 0usize, 0usize);
    for s in random_states(&b, 0xBEEF_06A4, 24, 18) {
        let got = sol.solve(&s);
        let probe = brute_value(&b, &s, probe_depth);
        if probe != Value::Draw {
            decisive += 1;
            if got != probe {
                inversions += 1;
                eprintln!("6x4-w1 inversion: solver={got:?} oracle={probe:?} {}", fmt(&s));
            }
        }
        visited += 1;
    }
    assert_eq!(inversions, 0, "{inversions} brute inversions on 6x4-w1 (of {decisive} decisive)");
    assert!(visited >= 100, "only {visited} 6x4-w1 positions visited");
    eprintln!("6x4-w1 sample brute differential: {visited} visited, {decisive} decisive, 0 inversions");
}

/// (c) TINY-cap stress: a deliberately tiny capacity (1 MiB) forces heavy
/// eviction in the dense fixed-capacity table. Every value must match a
/// generous-cap solve (proving eviction / capacity is correctness-neutral). Run
/// on the same boards as the reused-vs-fresh gate plus 5x5-w2.
#[test]
fn tiny_cap_eviction_stress() {
    let boards = [(3, 3, 1), (4, 4, 2), (5, 5, 1), (5, 5, 2)];
    let mut total = 0usize;
    let mut inversions = 0usize;
    for (w, h, walls) in boards {
        let b = Board::new(w, h, walls);
        // Tiny-cap reused solver (heavy eviction) vs large-cap reused solver.
        let mut tiny = Solver::with_tt_mb(&b, 1);
        let mut big = Solver::with_tt_mb(&b, 4096);
        for s in random_states(&b, 0xE71C7 ^ ((w as u64) << 8 | h as u64), 12, 16) {
            let v_tiny = tiny.solve(&s);
            let v_big = big.solve(&s);
            if v_tiny != v_big {
                inversions += 1;
                eprintln!("tiny/big mismatch {w}x{h}-w{walls}: tiny={v_tiny:?} big={v_big:?} {}", fmt(&s));
            }
            total += 1;
        }
    }
    assert_eq!(inversions, 0, "{inversions}/{total} tiny-cap eviction inversions");
    assert!(total >= 100, "only {total} tiny-cap positions checked");
    eprintln!("tiny-cap eviction stress: {total} positions, 0 inversions");
}

/// (d) The critical repros. These exact positions exercised the two prior
/// critical bugs; with the new TT they MUST resolve to the same verdicts.
///   * keystone 6x4: pawn=[8,6], h_walls=0x220, walls_left=[1,1], turn 0 => Win.
///   * blockade 7x5: pawn=[18,24], h_walls=0x280240, v_walls=0x500400,
///     walls_left=[0,0], turn 1 => Draw.
#[test]
fn critical_repro_keystone_6x4_win() {
    let b = Board::new(6, 4, 1);
    let s = State {
        pawn: [8, 6],
        h_walls: 0x220,
        v_walls: 0,
        walls_left: [1, 1],
        turn: 0,
    };
    assert_eq!(Solver::new(&b).solve(&s), Value::Win, "keystone 6x4 must be Win");
    // Also from a reused solver (cross-position TT) to exercise the depth-fold.
    let mut reused = Solver::new(&b);
    for w in random_states(&b, 0x123, 4, 8) {
        reused.solve(&w);
    }
    assert_eq!(reused.solve(&s), Value::Win, "keystone 6x4 must be Win (reused)");
}

#[test]
fn critical_repro_blockade_7x5_draw() {
    let b = Board::new(7, 5, 0);
    let s = State {
        pawn: [18, 24],
        h_walls: 0x280240,
        v_walls: 0x500400,
        walls_left: [0, 0],
        turn: 1,
    };
    assert_eq!(Solver::new(&b).solve(&s), Value::Draw, "blockade 7x5 must be Draw");
}
