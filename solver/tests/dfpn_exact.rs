//! Stage-1 exactness gates for the df-pn engine (`src/dfpn.rs`).
//!
//! The df-pn engine is a SECOND driver beside the verified alpha-beta solver;
//! these gates pin it to the AB oracle and to the independently known values:
//!
//!   * `dfpn_equals_ab_full_graph_*`: df-pn value == AB value on EVERY
//!     reachable position of the complete 3x3-w1 and 4x3-w2 game graphs
//!     (exhaustive BFS from the initial position — not a sample).
//!   * `dfpn_writeup_defaults` / `dfpn_keystone_6x4_win` /
//!     `dfpn_blockade_7x5_draw` / `dfpn_6x5_w0_w1_w2_loss`: the known-value
//!     gates (writeup defaults, the 6x4 keystone Win, the 7x5 blockade-race
//!     Draw, and 6x5 W0..W2 = Loss).
//!   * `ghi_machinery_*`: repetition-heavy constructions. The full
//!     Kishimoto–Müller machinery must (a) agree with the AB oracle and (b)
//!     demonstrably ENGAGE (repetition adjudications and path-dependent twin
//!     entries observed). A naive GHI-OFF df-pn (repetition disproofs stored
//!     path-independently — exactly the unsoundness K-M fixed) is also run
//!     across a reachable-position sweep; any value it gets wrong must be a
//!     position the full machinery gets right.
//!   * `dfpn_vs_ab_nodes_5x5w2_6x5w2`: full-board agreement on 5x5-w2 and
//!     6x5-w2 with node counts for both engines printed (run with
//!     `--nocapture` to see the comparison).
//!   * `fdfpn_*` (Stage 2): FDFPN dynamic widening gates. The whole suite
//!     already runs with widening ON at the defaults (base=4, fraction=0.25);
//!     these add (a) the default-config pin, (b) full-graph df-pn == AB with
//!     widening OFF (the measurement baseline of the same binary), and (c)
//!     full-graph df-pn == AB under the maximally aggressive single-child
//!     window (base=1, fraction=0) — the hardest stress on the "every child
//!     is eventually considered" exactness argument.
//!
//! All df-pn solvers here use explicit small TT budgets (`with_tt_mb`) so the
//! suite stays memory-sane under parallel test threads; TT size is
//! value-neutral by design (least-work eviction only forces re-search).

use quoridor_solver::board::Board;
use quoridor_solver::dfpn::DfpnSolver;
use quoridor_solver::movegen::{apply, legal_moves};
use quoridor_solver::solver::{Solver, Value};
use quoridor_solver::state::State;
use std::collections::{HashSet, VecDeque};

/// Exhaustive BFS of the full reachable game graph from the initial position
/// (terminals included but not expanded). `cap` guards against an unexpected
/// blow-up making the test silently enormous.
fn reachable_graph(b: &Board, cap: usize) -> Vec<State> {
    let mut seen: HashSet<State> = HashSet::new();
    let mut queue: VecDeque<State> = VecDeque::new();
    let s0 = b.initial();
    seen.insert(s0);
    queue.push_back(s0);
    let mut out = Vec::new();
    while let Some(s) = queue.pop_front() {
        out.push(s);
        assert!(out.len() <= cap, "reachable graph exceeded cap {cap}");
        if b.is_terminal(&s) {
            continue;
        }
        for m in legal_moves(b, &s) {
            let c = apply(b, &s, m);
            if seen.insert(c) {
                queue.push_back(c);
            }
        }
    }
    out
}

fn fmt_state(s: &State) -> String {
    format!(
        "pawn={:?} h=0x{:x} v=0x{:x} wl={:?} turn={}",
        s.pawn, s.h_walls, s.v_walls, s.walls_left, s.turn
    )
}

/// df-pn == AB on every non-terminal reachable position of the FULL graph.
/// One shared solver pair (cross-call TT reuse is part of the contract being
/// gated: base entries are path-independent, twins are signature-matched).
/// `cfg` configures the df-pn engine before the sweep (widening variants).
fn check_full_graph_with(
    w: u8,
    h: u8,
    walls: u8,
    cap: usize,
    label: &str,
    cfg: impl Fn(&mut DfpnSolver),
) {
    let b = Board::new(w, h, walls);
    let states = reachable_graph(&b, cap);
    let mut ab = Solver::new(&b);
    ab.set_threads(1);
    let mut dfpn = DfpnSolver::with_tt_mb(&b, 64);
    cfg(&mut dfpn);
    let mut checked = 0usize;
    for s in &states {
        if b.is_terminal(s) {
            continue;
        }
        let want = ab.solve(s);
        let got = dfpn.solve(s);
        assert_eq!(
            got,
            want,
            "df-pn [{label}] disagrees with AB on {w}x{h} w{walls}: {}",
            fmt_state(s)
        );
        checked += 1;
    }
    println!(
        "full-graph {w}x{h} w{walls} [{label}]: {} reachable, {} non-terminal checked, \
         dfpn nodes={} rep_hits={} twins={} sims={} fallbacks={}",
        states.len(),
        checked,
        dfpn.stats.total_nodes(),
        dfpn.stats.rep_hits,
        dfpn.stats.twin_stores,
        dfpn.stats.sim_calls,
        dfpn.stats.fallbacks(),
    );
    assert!(checked > 100, "graph suspiciously small: {checked}");
}

/// Default engine configuration (env defaults: widening ON, base=4, frac=0.25).
fn check_full_graph(w: u8, h: u8, walls: u8, cap: usize) {
    check_full_graph_with(w, h, walls, cap, "default", |_| {});
}

#[test]
fn dfpn_equals_ab_full_graph_3x3_w1() {
    check_full_graph(3, 3, 1, 2_000_000);
}

#[test]
fn dfpn_equals_ab_full_graph_4x3_w2() {
    check_full_graph(4, 3, 2, 20_000_000);
}

/// Stage 2: the FDFPN defaults are wired as documented (the rest of the suite
/// then gates the widened engine everywhere it runs the default config).
#[test]
fn fdfpn_widening_default_config() {
    let b = Board::new(4, 4, 2);
    let dfpn = DfpnSolver::with_tt_mb(&b, 8);
    assert_eq!(
        dfpn.widening(),
        Some((4, 0.25)),
        "FDFPN defaults changed (or QS_DFPN_WIDEN* is set in the test env)"
    );
}

/// Stage 2: widening OFF (the ON-vs-OFF measurement baseline of the SAME
/// binary) — df-pn == AB on every position of the full 4x3-w2 graph.
#[test]
fn fdfpn_widening_off_full_graph_4x3_w2() {
    check_full_graph_with(4, 3, 2, 20_000_000, "widen-off", |d| {
        d.set_widening(None);
    });
}

/// Stage 2: the maximally aggressive single-unsolved-child window (base=1,
/// fraction=0) — the hardest stress on "every child is eventually considered"
/// (any unsoundly hidden child would flip some value in a FULL-graph sweep).
#[test]
fn fdfpn_widening_aggressive_window_full_graph_3x3_w1() {
    check_full_graph_with(3, 3, 1, 2_000_000, "widen(1,0.0)", |d| {
        d.set_widening(Some((1, 0.0)));
    });
}

#[test]
fn fdfpn_widening_aggressive_window_full_graph_4x3_w2() {
    check_full_graph_with(4, 3, 2, 20_000_000, "widen(1,0.0)", |d| {
        d.set_widening(Some((1, 0.0)));
    });
}

/// Writeup-default known values.
#[test]
fn dfpn_writeup_defaults() {
    for &(w, h, walls, want) in &[
        (3u8, 3u8, 1u8, Value::Loss),
        (4, 4, 1, Value::Win),
        (4, 4, 2, Value::Win),
        (5, 5, 0, Value::Loss),
        (5, 5, 1, Value::Loss),
    ] {
        let b = Board::new(w, h, walls);
        let mut dfpn = DfpnSolver::with_tt_mb(&b, 128);
        assert_eq!(dfpn.solve(&b.initial()), want, "{w}x{h} w{walls}");
        // NB: cycle-guard AB fallbacks may fire (they are exactness-preserving
        // by construction and logged/counted); the VALUE is the gate here.
    }
}

/// Keystone: 6x4, pawns [8,6], one horizontal-wall pair (anchors 5 and 9),
/// one wall in hand each, player 0 to move => Win for the side to move.
#[test]
fn dfpn_keystone_6x4_win() {
    let b = Board::new(6, 4, 1);
    let s = State {
        pawn: [8, 6],
        h_walls: 0x220,
        v_walls: 0,
        walls_left: [1, 1],
        turn: 0,
    };
    let mut dfpn = DfpnSolver::with_tt_mb(&b, 128);
    assert_eq!(dfpn.solve(&s), Value::Win);
}

/// Blockade race: 7x5, walls exhausted, mutual blockade => Draw. Exercises
/// the direct retrograde-race fold (the df-pn root short-circuit).
#[test]
fn dfpn_blockade_7x5_draw() {
    let b = Board::new(7, 5, 4);
    let s = State {
        pawn: [18, 24],
        h_walls: 0x280240,
        v_walls: 0x500400,
        walls_left: [0, 0],
        turn: 1,
    };
    let mut dfpn = DfpnSolver::with_tt_mb(&b, 64);
    assert_eq!(dfpn.solve(&s), Value::Draw);
}

/// 6x5 W0..W2 are all Loss for the first player (previously computed and
/// pinned by the AB solver's own gates).
#[test]
fn dfpn_6x5_w0_w1_w2_loss() {
    for walls in 0..=2u8 {
        let b = Board::new(6, 5, walls);
        let mut dfpn = DfpnSolver::with_tt_mb(&b, 512);
        assert_eq!(dfpn.solve(&b.initial()), Value::Loss, "6x5 w{walls}");
    }
}

/// Repetition-heavy constructions: near-blockade corridors with walls still in
/// hand, where optimal play shuffles (the loopy regime GHI is about). The full
/// machinery must agree with the AB oracle on every one AND visibly engage.
#[test]
fn ghi_machinery_engages_and_agrees_on_repetition_heavy_positions() {
    // 5x5 with a wall corridor: two horizontal wall pairs walling rows, one
    // wall in hand for player 0 — play degenerates into shuffling battles.
    let b5 = Board::new(5, 5, 3);
    let mut cases: Vec<(Board, State)> = Vec::new();
    cases.push((
        b5,
        State {
            pawn: [2, 22],
            h_walls: 0x5a0,
            v_walls: 0,
            walls_left: [1, 1],
            turn: 0,
        },
    ));
    // 7x5 blockade with one wall re-armed: the blockade race draw plus a
    // single wall in hand each (repetition-rich, race-adjacent).
    let b7 = Board::new(7, 5, 4);
    cases.push((
        b7,
        State {
            pawn: [18, 24],
            h_walls: 0x280240,
            v_walls: 0x500400,
            walls_left: [1, 1],
            turn: 1,
        },
    ));
    // 4x4 facing pawns with walls forming a pocket.
    let b4 = Board::new(4, 4, 2);
    cases.push((
        b4,
        State {
            pawn: [5, 9],
            h_walls: 0x9,
            v_walls: 0,
            walls_left: [1, 1],
            turn: 0,
        },
    ));

    let mut rep_hits = 0u64;
    let mut twins = 0u64;
    for (b, s) in &cases {
        let mut ab = Solver::new(b);
        ab.set_threads(1);
        let want = ab.solve(s);
        let mut dfpn = DfpnSolver::with_tt_mb(b, 64);
        let got = dfpn.solve(s);
        assert_eq!(got, want, "repetition-heavy case {}", fmt_state(s));
        rep_hits += dfpn.stats.rep_hits;
        twins += dfpn.stats.twin_stores;
    }
    // The machinery must actually have been exercised, or this gate is vacuous.
    assert!(rep_hits > 0, "no repetition adjudications observed");
    assert!(twins > 0, "no path-dependent twin entries stored");
}

/// The unsoundness the K-M machinery exists to fix, demonstrated: sweep
/// reachable positions; wherever the NAIVE (GHI-off) df-pn disagrees with the
/// AB oracle, the FULL df-pn must agree. (Vanilla df-pn with repetition
/// (dis)proofs stored path-independently produced 18/200 wrong disproofs in
/// K-M's checkers experiments; the same failure mode is reproduced here on
/// Quoridor positions if it occurs in this sweep — and the full machinery is
/// shown immune on exactly those positions.)
#[test]
fn ghi_naive_dfpn_errors_are_fixed_by_full_machinery() {
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self, n: usize) -> usize {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((self.0 >> 33) as usize) % n
        }
    }
    let boards: &[(u8, u8, u8, u64)] = &[
        (4, 4, 2, 0x4422_C0FF),
        (5, 4, 2, 0x5422_2222),
        (5, 5, 2, 0x5522_0042),
        (4, 3, 2, 0x4322_1111),
    ];
    let mut checked = 0usize;
    let mut naive_wrong = 0usize;
    for &(w, h, walls, seed) in boards {
        let b = Board::new(w, h, walls);
        let mut ab = Solver::new(&b);
        ab.set_threads(1);
        let mut rng = Lcg(seed);
        let mut seen: HashSet<State> = HashSet::new();
        for _ in 0..64 {
            let mut s = b.initial();
            for _ in 0..40 {
                if b.is_terminal(&s) {
                    break;
                }
                // The GHI failure mode lives where repetition shapes values:
                // few-or-no walls in hand for the mover but NOT a pure race
                // (some search above the race layer). Restrict the sweep to
                // that regime; the full-graph gates above already cover the
                // wall-rich early game exhaustively.
                let in_hand = s.walls_left[0] as u32 + s.walls_left[1] as u32;
                if (1..=2).contains(&in_hand) && seen.insert(s) {
                    let truth = ab.solve(&s);
                    // Fresh small solvers per position: no cross-position TT
                    // effects masking (or faking) a divergence.
                    let mut full = DfpnSolver::with_tt_mb(&b, 8);
                    assert_eq!(
                        full.solve(&s),
                        truth,
                        "FULL df-pn wrong on {w}x{h} w{walls}: {}",
                        fmt_state(&s)
                    );
                    let mut naive = DfpnSolver::with_tt_mb(&b, 8);
                    naive.set_ghi_for_unsound_demo(false);
                    let nv = naive.solve(&s);
                    if nv != truth {
                        naive_wrong += 1;
                        println!(
                            "naive WRONG on {w}x{h} w{walls}: truth={truth:?} naive={nv:?} {}",
                            fmt_state(&s)
                        );
                    }
                    checked += 1;
                }
                let ms = legal_moves(&b, &s);
                if ms.is_empty() {
                    break;
                }
                s = apply(&b, &s, ms[rng.next(ms.len())]);
            }
        }
    }
    println!("naive-vs-full sweep: checked={checked} naive_wrong={naive_wrong}");
    assert!(checked >= 400, "sweep too small: {checked}");
}

/// A pinned position (found by the sweep above) where the NAIVE GHI-ignoring
/// df-pn provably returns the WRONG value (Draw) while the full Kishimoto–
/// Müller machinery returns the AB-verified truth (Win). This is the live
/// Quoridor reproduction of the unsoundness K-M documented on checkers (18/200
/// wrong disproofs): the naive engine stores a repetition-based disproof
/// path-independently and later trusts it on a path it is not valid for.
/// (Both engines here are deterministic single-threaded with a fixed TT size,
/// so the divergence is stable; if a future change makes the naive engine
/// accidentally right on this position, re-hunt with the sweep test.)
#[test]
fn ghi_pinned_naive_misevaluation_fixed_by_full_machinery() {
    let b = Board::new(5, 4, 2);
    let s = State {
        pawn: [7, 17],
        h_walls: 0x100,
        v_walls: 0x80,
        walls_left: [1, 1],
        turn: 1,
    };
    let mut ab = Solver::new(&b);
    ab.set_threads(1);
    assert_eq!(ab.solve(&s), Value::Win, "oracle value changed?");
    let mut full = DfpnSolver::with_tt_mb(&b, 8);
    assert_eq!(full.solve(&s), Value::Win, "full K-M df-pn must match the oracle");
    let mut naive = DfpnSolver::with_tt_mb(&b, 8);
    naive.set_ghi_for_unsound_demo(false);
    assert_eq!(
        naive.solve(&s),
        Value::Draw,
        "the naive engine is EXPECTED to be wrong here (that is the demo); \
         if it is now right, the pinned position no longer demonstrates GHI — re-hunt"
    );
}

/// Full-board 5x5-w2 and 6x5-w2: df-pn == AB, node counts printed for both
/// engines (the first measurement of the df-pn-vs-AB node-count hope).
#[test]
fn dfpn_vs_ab_nodes_5x5w2_6x5w2() {
    for &(w, h, walls) in &[(5u8, 5u8, 2u8), (6, 5, 2)] {
        let b = Board::new(w, h, walls);
        let s = b.initial();
        let mut ab = Solver::new(&b);
        ab.set_threads(1);
        let t0 = std::time::Instant::now();
        let want = ab.solve(&s);
        let t_ab = t0.elapsed();
        let mut dfpn = DfpnSolver::with_tt_mb(&b, 512);
        let t1 = std::time::Instant::now();
        let got = dfpn.solve(&s);
        let t_dfpn = t1.elapsed();
        assert_eq!(got, want, "{w}x{h} w{walls}");
        println!(
            "{w}x{h} w{walls}: value={want:?}  AB(1-thread) nodes={} time={:.2}s  \
             df-pn nodes={} (mid={} race={} sim={}) time={:.2}s  twins={} sims={} fallbacks={}",
            ab.nodes,
            t_ab.as_secs_f64(),
            dfpn.stats.total_nodes(),
            dfpn.stats.nodes,
            dfpn.stats.race_nodes,
            dfpn.stats.sim_nodes,
            t_dfpn.as_secs_f64(),
            dfpn.stats.twin_stores,
            dfpn.stats.sim_calls,
            dfpn.stats.fallbacks(),
        );
    }
}
