//! Gates for THEOREM 1 (Wall-Insertion Invariance), Win direction — the
//! wall-relevance footprint ("mustplay") pruning
//! (`docs/superpowers/solver-pruning-theorems.md` §A, final post-falsification
//! form including the §A.5 rank amendment).
//!
//! The theorem: a Definition-1 certificate `P` proving `V(flip(s)) = Win` for
//! `Y` (the opponent of the side to move at `s`) compiles to a Definition-2
//! footprint `R(P)` — two wall-anchor masks. Any Z wall anchored OUTSIDE the
//! masks leaves `P` valid verbatim, so its child value is EXACTLY `Win` for Y
//! (`Loss` for Z) and the refutation loop may skip it with zero search.
//!
//! Gates pinned here:
//!   (a) A/B EXACT-VALUE REGRESSION: `solve` with the feature ON vs OFF (the
//!       `set_use_footprint` hook; env knob `QS_FOOTPRINT=1` enables, default OFF)
//!       returns the IDENTICAL value on >=150 seeded random reachable
//!       positions across the five falsifier boards (3x3-w1, 4x3-w2, 4x4-w2,
//!       3x5-w2, 6x4-w1), plus every known-value gate (writeup defaults,
//!       6x5 W0/W1/W2 = Loss, keystone 6x4 = Win, blockade 7x5 = Draw).
//!   (b) THE DOC'S T-SUITE: T1/T2 extraction yields EXACTLY the verified
//!       masks; every T3 far-wall value-flipper lands INSIDE the extracted
//!       footprint; T5's footprint contains `Conflict(H(3,1))`; T4 (failed
//!       precondition) extracts NOTHING. Every out-of-footprint wall is
//!       re-solved and must preserve the Win (the falsification harness
//!       recipe, §6 of the raw note).
//!   (c) CONSTRUCTED FIRE: a position where the in-search pruning actually
//!       fires (`fp_prunes > 0`) and the ON value equals the OFF value; and
//!       counters are identically zero with the feature OFF.

use quoridor_solver::board::Board;
use quoridor_solver::movegen::{apply, legal_moves, legal_walls};
use quoridor_solver::solver::{Solver, Value};
use quoridor_solver::state::{Move, State};
use std::collections::HashSet;

/// Per-test main-TT cap in MiB (plenty for these boards; keeps the suite's
/// many short-lived solvers from each zeroing gigabytes).
const TT_MB: usize = 64;

/// Tiny deterministic LCG for reproducible random play.
struct Lcg(u64);
impl Lcg {
    fn next(&mut self, n: usize) -> usize {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((self.0 >> 33) as usize) % n
    }
}

/// Solve `s` with footprint pruning ON or OFF (fresh solver either way).
fn solve_fp(b: &Board, s: &State, on: bool) -> Value {
    let mut sol = Solver::with_tt_mb(b, TT_MB);
    sol.set_use_footprint(on);
    sol.solve(s)
}

/// Assert the A/B exact-value contract on one position and return the value.
fn assert_on_off(b: &Board, s: &State, label: &str) -> Value {
    let v_on = solve_fp(b, s, true);
    let v_off = solve_fp(b, s, false);
    assert_eq!(
        v_on, v_off,
        "footprint ON/OFF value mismatch at {label}: pawn={:?} h={:#x} v={:#x} wl={:?} turn={}",
        s.pawn, s.h_walls, s.v_walls, s.walls_left, s.turn
    );
    v_on
}

/// Anchor bit of a wall move.
fn wall_bit(b: &Board, wc: u8, wr: u8) -> u64 {
    1u64 << ((wr as u32) * (b.w as u32 - 1) + wc as u32)
}

/// The falsification-harness soundness sweep (raw note §6): for every legal
/// Z wall at `s` whose anchor is OUTSIDE the masks, the child position (Y to
/// move) must still be a Win for Y — Theorem 1's claim, checked against the
/// real solver. Returns (outside, total) wall counts.
fn sweep_outside_walls(b: &Board, s: &State, hm: u64, vm: u64) -> (usize, usize) {
    let mut oracle = Solver::with_tt_mb(b, TT_MB);
    oracle.set_use_footprint(false); // plain solver as the oracle
    let mut outside = 0usize;
    let mut total = 0usize;
    for m in legal_walls(b, s) {
        let Move::Wall { wc, wr, horiz } = m else {
            continue;
        };
        total += 1;
        let bit = wall_bit(b, wc, wr);
        let inside = if horiz { hm & bit != 0 } else { vm & bit != 0 };
        if inside {
            continue;
        }
        outside += 1;
        let child = apply(b, s, m);
        assert_eq!(
            oracle.solve(&child),
            Value::Win,
            "UNSOUND FOOTPRINT: out-of-footprint wall ({wc},{wr},horiz={horiz}) at \
             pawn={:?} h={:#x} v={:#x} wl={:?} turn={} does NOT preserve the Win",
            s.pawn,
            s.h_walls,
            s.v_walls,
            s.walls_left,
            s.turn
        );
    }
    (outside, total)
}

/// Collect DISTINCT seeded random reachable, non-terminal states by playing
/// random legal games from the initial position (same scheme as the Theorem-4
/// gate; deduped so each position is solved once).
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
// (b) T1/T2: extraction reproduces the falsifier's EXACT verified masks.
// ---------------------------------------------------------------------------

/// T1 (corridor win): footprint EXACTLY {H(1,3), H(2,3)} — the certificate is
/// the single step (2,3)->(2,4), so component (A)'s one edge compiles to its
/// two blocking anchors and nothing else. Both in-footprint walls genuinely
/// flip the value (the footprint isolates exactly the value-critical walls),
/// and all 30 out-of-footprint walls preserve the Win.
#[test]
fn fp_t1_exact_masks_and_sweep() {
    let b = Board::new(5, 5, 3);
    let s = State {
        pawn: [17, 7],
        h_walls: 0,
        v_walls: 0,
        walls_left: [3, 3],
        turn: 1,
    };
    let flip = State { turn: 0, ..s };
    let mut sol = Solver::with_tt_mb(&b, TT_MB);
    let (hm, vm) = sol.extract_footprint(&flip).expect("T1 must extract a certificate");
    assert_eq!(hm, wall_bit(&b, 1, 3) | wall_bit(&b, 2, 3), "T1 H-mask must be {{H(1,3),H(2,3)}}");
    assert_eq!(vm, 0, "T1 V-mask must be empty");
    let (outside, total) = sweep_outside_walls(&b, &s, hm, vm);
    assert_eq!((outside, total), (30, 32), "T1 prunes 30 of 32 walls");
    // The two in-footprint walls are genuinely value-critical (flip to Loss).
    let mut oracle = Solver::with_tt_mb(&b, TT_MB);
    for wc in [1u8, 2u8] {
        let child = apply(&b, &s, Move::Wall { wc, wr: 3, horiz: true });
        assert_eq!(oracle.solve(&child), Value::Loss, "H({wc},3) must flip the win");
    }
}

/// T2 (jump win): footprint EXACTLY the 4 anchors of the straight-jump edges
/// (2,2)-(2,3) and (2,3)-(2,4); 28 of 32 walls pruned, all preserving Win.
#[test]
fn fp_t2_exact_masks_and_sweep() {
    let b = Board::new(5, 5, 3);
    let s = State {
        pawn: [12, 17],
        h_walls: 0,
        v_walls: 0,
        walls_left: [3, 3],
        turn: 1,
    };
    let flip = State { turn: 0, ..s };
    let mut sol = Solver::with_tt_mb(&b, TT_MB);
    let (hm, vm) = sol.extract_footprint(&flip).expect("T2 must extract a certificate");
    let expect = wall_bit(&b, 1, 2) | wall_bit(&b, 2, 2) | wall_bit(&b, 1, 3) | wall_bit(&b, 2, 3);
    assert_eq!(hm, expect, "T2 H-mask must be {{H(1,2),H(2,2),H(1,3),H(2,3)}}");
    assert_eq!(vm, 0, "T2 V-mask must be empty");
    let (outside, total) = sweep_outside_walls(&b, &s, hm, vm);
    assert_eq!((outside, total), (28, 32), "T2 prunes 28 of 32 walls");
}

// ---------------------------------------------------------------------------
// (b) T3: the six adversarial far-wall value-flippers (walls Chebyshev >= 2
// from both pawns and from every shortest-path cell that flip Win -> Loss)
// must land INSIDE the extracted footprint; everything outside stays Win.
// Any geometric/shortest-path footprint fails here — Definition 2 must not.
// ---------------------------------------------------------------------------

#[test]
fn fp_t3_far_wall_flippers_inside_footprint() {
    let b = Board::new(5, 5, 3);
    // (pawn, h_walls, v_walls, flipper walls [(wc, wr, horiz)])
    type Flip = (u8, u8, bool);
    let cases: &[([u8; 2], u64, u64, &[Flip])] = &[
        ([4, 14], 0x0, 0xc00, &[(0, 1, false)]),
        ([4, 21], 0x4, 0x2000, &[(1, 1, false)]),
        ([4, 5], 0x0, 0x28, &[(1, 3, false)]),
        ([5, 6], 0x0, 0x1080, &[(3, 0, true), (3, 2, true), (3, 3, false)]),
    ];
    for &(pawn, h_walls, v_walls, flippers) in cases {
        let s = State {
            pawn,
            h_walls,
            v_walls,
            walls_left: [2, 2],
            turn: 1,
        };
        let flip = State { turn: 0, ..s };
        let mut sol = Solver::with_tt_mb(&b, TT_MB);
        let (hm, vm) = sol
            .extract_footprint(&flip)
            .unwrap_or_else(|| panic!("T3 {pawn:?} must extract a certificate"));
        for &(wc, wr, horiz) in flippers {
            let bit = wall_bit(&b, wc, wr);
            let inside = if horiz { hm & bit != 0 } else { vm & bit != 0 };
            assert!(
                inside,
                "THEOREM 1 FALSIFIED: T3 flipper ({wc},{wr},horiz={horiz}) at pawn={pawn:?} \
                 is OUTSIDE the extracted footprint (hm={hm:#x} vm={vm:#x})"
            );
        }
        sweep_outside_walls(&b, &s, hm, vm);
    }
}

// ---------------------------------------------------------------------------
// (b) T5: wall-forced wins — components (C)/(D) are live. The footprint must
// contain Conflict(H(3,1)) for the unique-winning-wall instance, and every
// out-of-footprint wall must preserve the Win on all four instances.
// ---------------------------------------------------------------------------

#[test]
fn fp_t5_wall_forced_wins() {
    let b = Board::new(5, 5, 3);
    let cases: &[([u8; 2], u64, u64)] = &[
        ([6, 13], 0x0, 0x1400),  // unique winning move H(3,1)
        ([4, 11], 0x0, 0x404),   // wins: H(0,0), H(0,1), H(1,0)
        ([7, 9], 0x1, 0x0),      // H(3,0)
        ([5, 10], 0x28, 0x0),    // H(0,0)
    ];
    for (i, &(pawn, h_walls, v_walls)) in cases.iter().enumerate() {
        let s = State {
            pawn,
            h_walls,
            v_walls,
            walls_left: [2, 2],
            turn: 1,
        };
        let flip = State { turn: 0, ..s };
        let mut sol = Solver::with_tt_mb(&b, TT_MB);
        let (hm, vm) = sol
            .extract_footprint(&flip)
            .unwrap_or_else(|| panic!("T5.{i} {pawn:?} must extract a certificate"));
        if i == 0 {
            // Conflict(H(3,1)) = {H(2,1), H(3,1), V(3,1)} must be protected
            // (component (C) of the certificate's opening wall move).
            let need_h = wall_bit(&b, 2, 1) | wall_bit(&b, 3, 1);
            let need_v = wall_bit(&b, 3, 1);
            assert_eq!(hm & need_h, need_h, "T5.0 footprint must contain H(2,1),H(3,1)");
            assert_eq!(vm & need_v, need_v, "T5.0 footprint must contain V(3,1)");
        }
        sweep_outside_walls(&b, &s, hm, vm);
    }
}

// ---------------------------------------------------------------------------
// (b) T4: negative control — the Theorem-1 precondition FAILS (V(flip(s)) is
// not a Win for Y), so extraction must yield NOTHING (any implementation that
// prunes here is buggy: nearly every Z wall wins for Z).
// ---------------------------------------------------------------------------

#[test]
fn fp_t4_negative_control_no_certificate() {
    let b = Board::new(7, 5, 4);
    let s = State {
        pawn: [18, 24],
        h_walls: 0x280240,
        v_walls: 0x500400,
        walls_left: [0, 1],
        turn: 1,
    };
    let flip = State { turn: 0, ..s };
    let mut sol = Solver::with_tt_mb(&b, TT_MB);
    assert!(
        sol.extract_footprint(&flip).is_none(),
        "T4: V(flip(s)) is a LOSS for Y — extraction must refuse a certificate"
    );
    // Z (p1, holding the wall) genuinely wins here; ON/OFF must agree.
    assert_eq!(assert_on_off(&b, &s, "T4 control"), Value::Win);
}

// ---------------------------------------------------------------------------
// (b) Race-leaf extraction: a certificate whose Z replies burn the LAST wall,
// driving the closure into [0,0] race nodes (exact retrograde witnesses).
// ---------------------------------------------------------------------------

#[test]
fn fp_race_leaf_extraction() {
    let b = Board::new(5, 5, 3);
    // Y = p0 at (2,2) (dist 2, to move in flip), Z = p1 at (0,3) (dist 3)
    // holding ONE wall: every Z wall reply reaches walls_left == [0,0].
    let s = State {
        pawn: [12, 15],
        h_walls: 0,
        v_walls: 0,
        walls_left: [0, 1],
        turn: 1,
    };
    let flip = State { turn: 0, ..s };
    let mut sol = Solver::with_tt_mb(&b, TT_MB);
    if let Some((hm, vm)) = sol.extract_footprint(&flip) {
        let (outside, total) = sweep_outside_walls(&b, &s, hm, vm);
        eprintln!("race-leaf extraction: {outside}/{total} walls outside (hm={hm:#x} vm={vm:#x})");
    } else {
        // The position must then NOT be a flip-win at all — otherwise the
        // extractor missed a certificate it was designed to build.
        let mut oracle = Solver::with_tt_mb(&b, TT_MB);
        assert_ne!(
            oracle.solve(&flip),
            Value::Win,
            "flip(s) is a Win but the race-leaf extraction failed"
        );
        panic!("expected a Win certificate for the race-leaf fixture");
    }
}

// ---------------------------------------------------------------------------
// (a) A/B on the known-value gates.
// ---------------------------------------------------------------------------

#[test]
fn fp_known_value_gates() {
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
/// `cargo test --release --test footprint -- --ignored`.
#[ignore]
#[test]
fn fp_known_value_gate_6x5_w3() {
    let b = Board::new(6, 5, 3);
    let v = assert_on_off(&b, &b.initial(), "6x5-w3 initial");
    assert_eq!(v, Value::Loss, "6x5-w3 initial must be Loss");
}

#[test]
fn fp_keystone_6x4_win() {
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
fn fp_blockade_7x5_draw() {
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
// (a) A/B exact-value regression: >=150 seeded random reachable positions on
// the five falsifier boards.
// ---------------------------------------------------------------------------

#[test]
fn fp_ab_regression_random_positions() {
    let cases: &[(u8, u8, u8, u64)] = &[
        (3, 3, 1, 0xF00F),
        (4, 3, 2, 0xBEEF),
        (4, 4, 2, 0xFACE),
        (3, 5, 2, 0xD1CE),
        (6, 4, 1, 0xFEED),
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
    eprintln!("footprint A/B regression: {total} positions, 0 mismatches");
}

// ---------------------------------------------------------------------------
// (c) Constructed fire: in-search pruning fires and the value matches OFF.
// ---------------------------------------------------------------------------

#[test]
fn fp_prunes_fire_and_value_matches() {
    // 5x5-w3, Z = p1 to move at (2,2), Y = p0 at (2,3) one step from goal.
    // flip(s) is an immediate Win for Y (step to the goal row), so the root's
    // wall iteration runs under the T1-style footprint {H(1,3), H(2,3)}: all
    // other wall replies are exact Losses and must be skipped.
    let b = Board::new(5, 5, 3);
    let s = State {
        pawn: [17, 12],
        h_walls: 0,
        v_walls: 0,
        walls_left: [3, 3],
        turn: 1,
    };
    let v_off = solve_fp(&b, &s, false);

    let mut sol = Solver::with_tt_mb(&b, TT_MB);
    sol.set_threads(1); // deterministic traversal for the counter assertions
    sol.set_use_footprint(true);
    // Undeferred gates so the root's wall fan is pruned from the first wall
    // (the deferred default is covered by the A/B and known-value suites).
    sol.set_footprint_gates(4, 1, u32::MAX, 0);
    let v_on = sol.solve(&s);
    assert_eq!(v_on, v_off, "constructed-fire position: ON value must equal OFF");
    assert!(sol.fp_extractions > 0, "at least one certificate must be extracted");
    assert!(
        sol.fp_prunes >= 25,
        "the root's wall fan must be mustplay-pruned (fp_prunes={}, expected >=25)",
        sol.fp_prunes
    );
    assert!(sol.fp_mask_bits > 0, "successful extractions must record mask sizes");
    eprintln!(
        "constructed fire: value={v_on:?} fp_attempts={} fp_extractions={} fp_prunes={} avg_bits={:.1}",
        sol.fp_attempts,
        sol.fp_extractions,
        sol.fp_prunes,
        sol.fp_mask_bits as f64 / sol.fp_extractions as f64
    );
}

#[test]
fn fp_counters_zero_when_off() {
    let b = Board::new(4, 4, 2);
    let mut sol = Solver::with_tt_mb(&b, TT_MB);
    sol.set_use_footprint(false);
    assert_eq!(sol.solve(&b.initial()), Value::Win);
    assert_eq!(sol.fp_attempts, 0, "OFF must never attempt an extraction");
    assert_eq!(sol.fp_extractions, 0, "OFF must never extract");
    assert_eq!(sol.fp_prunes, 0, "OFF must never prune");
}
