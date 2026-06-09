//! THROWAWAY adversarial sign/perspective audit. Not committed.
use quoridor_solver::board::Board;
use quoridor_solver::movegen::{apply, legal_moves};
use quoridor_solver::solver::{brute_value, Solver, Value};
use quoridor_solver::state::State;
use std::collections::{HashMap, HashSet, VecDeque};

/// Enumerate every state reachable from the initial position by legal play,
/// stopping expansion at terminal states. Returns all reachable states.
fn reachable(b: &Board) -> Vec<State> {
    let mut seen: HashSet<State> = HashSet::new();
    let mut q: VecDeque<State> = VecDeque::new();
    let s0 = b.initial();
    seen.insert(s0);
    q.push_back(s0);
    while let Some(s) = q.pop_front() {
        if b.is_terminal(&s) {
            continue;
        }
        for m in legal_moves(b, &s) {
            let s2 = apply(b, &s, m);
            if seen.insert(s2) {
                q.push_back(s2);
            }
        }
    }
    seen.into_iter().collect()
}

/// Differential: a *reused* Solver must match the unpruned oracle on every
/// reachable non-terminal node. Reuse is the contamination risk for the TT and
/// the persistent race memo. The brute depth must be >= the longest forced line;
/// we use a depth large enough that brute_value resolves all non-draws.
fn diff_board(w: u8, h: u8, walls: u8, brute_depth: u32) {
    let b = Board::new(w, h, walls);
    let states = reachable(&b);
    // One solver reused across ALL positions (worst case for reuse bugs).
    let mut sol = Solver::new(&b);
    let mut checked = 0usize;
    for s in &states {
        if b.is_terminal(s) {
            continue;
        }
        let got = sol.solve(s);
        let want = brute_value(&b, s, brute_depth);
        assert_eq!(
            got, want,
            "SIGN/VALUE MISMATCH on {}x{} w{} at pawns={:?} turn={} walls_left={:?} hw={:#x} vw={:#x}: solver={:?} brute={:?}",
            w, h, walls, s.pawn, s.turn, s.walls_left, s.h_walls, s.v_walls, got, want
        );
        checked += 1;
    }
    eprintln!("diff {}x{} w{}: {} reachable, {} non-terminal checked", w, h, walls, states.len(), checked);
    assert!(checked > 0);
}

#[test]
fn diff_3x3_w1() { diff_board(3, 3, 1, 18); }
#[test]
fn diff_4x4_w1() { diff_board(4, 4, 1, 26); }
#[test]
fn diff_4x3_w1() { diff_board(4, 3, 1, 22); }
#[test]
fn diff_3x4_w1() { diff_board(3, 4, 1, 22); }
#[test]
fn diff_2x3_w1() { diff_board(2, 3, 1, 18); }
#[test]
fn diff_3x2_w1() { diff_board(3, 2, 1, 18); }
#[test]
fn diff_4x4_w2() { diff_board(4, 4, 2, 30); }
#[test]
fn diff_5x3_w1() { diff_board(5, 3, 1, 26); }
#[test]
fn diff_2x4_w1() { diff_board(2, 4, 1, 22); }

/// Reachability probe of the terminal assumption. The terminal rule assumes that
/// at any reachable search node WITH a winner, that winner is `1 - turn` (the
/// player who just moved), so `winner(s) == turn` should never occur at a
/// reachable node. Also: never both-on-goal. Enumerate and assert.
fn probe_terminal_assumption(w: u8, h: u8, walls: u8) {
    let b = Board::new(w, h, walls);
    let states = reachable(&b);
    for s in &states {
        let (_, r0) = b.cr(s.pawn[0]);
        let (_, r1) = b.cr(s.pawn[1]);
        let p0_on = r0 == b.goal_row(0);
        let p1_on = r1 == b.goal_row(1);
        // Never both on goal at a reachable node.
        assert!(
            !(p0_on && p1_on),
            "BOTH on goal reachable on {}x{}: pawns={:?} turn={}",
            w, h, s.pawn, s.turn
        );
        // If side-to-move is on its OWN goal, the `if p==turn { Win }` branch
        // fires. Report whether this is reachable.
        let stm_on_own = if s.turn == 0 { p0_on } else { p1_on };
        if stm_on_own {
            eprintln!(
                "REACHABLE: side-to-move {} already on own goal! {}x{} pawns={:?} walls_left={:?}",
                s.turn, w, h, s.pawn, s.walls_left
            );
            panic!("side-to-move-on-own-goal reachable");
        }
    }
    eprintln!("probe {}x{} w{}: {} states, terminal assumption holds", w, h, walls, states.len());
}

#[test]
fn probe_3x3() { probe_terminal_assumption(3, 3, 1); }
#[test]
fn probe_4x4() { probe_terminal_assumption(4, 4, 1); }
#[test]
fn probe_4x3() { probe_terminal_assumption(4, 3, 1); }
#[test]
fn probe_2x3() { probe_terminal_assumption(2, 3, 1); }

/// Sign-known boards: an even-HEIGHT board (both pawns equidistant from goal,
/// player 0 moves first) is a first-player Win by a strategy-stealing / tempo
/// argument in the wall-less case, and remains a P1 win for these small boards
/// per the writeup. Value is side-to-move-relative and p0 is to move at the
/// start, so the start value MUST be `Win` for an even-height first-player win.
/// A sign flip would report `Loss`. Cross-check with brute oracle too.
#[test]
fn sign_known_even_height() {
    // 4x4 w0: pure race, even height. First player reaches row 3 in 3 moves;
    // by symmetry/tempo p0 wins the race. Independently known sign = Win.
    for &(w, h) in &[(4u8, 4u8), (3, 4), (5, 4), (4, 2), (3, 2), (5, 2)] {
        let b = Board::new(w, h, 0);
        let v = Solver::new(&b).solve(&b.initial());
        assert_eq!(v, Value::Win, "even-height {}x{} w0 must be P1 Win, got {:?}", w, h, v);
        // brute cross-check
        let bv = brute_value(&b, &b.initial(), 4 * (w as u32 + h as u32));
        assert_eq!(bv, Value::Win, "brute even-height {}x{} w0 must be Win", w, h);
    }
}

/// Direct test of the terminal Win branch: a HAND-BUILT (possibly unreachable)
/// state where the side-to-move is standing on its own goal row. The code
/// returns Win for it. Confirm brute_value AGREES (both use the same terminal
/// rule), and confirm that's the only place the branch matters. This documents
/// that IF such a node were ever generated as a child, both oracle and solver
/// treat it identically (so the differential is faithful).
#[test]
fn terminal_branch_consistency() {
    let b = Board::new(3, 3, 1);
    let mut s = b.initial();
    // Force p0 (side to move) onto its own goal row h-1=2.
    s.pawn[0] = b.idx(1, 2);
    s.turn = 0;
    assert_eq!(b.winner(&s), Some(0));
    // Both solver and brute use the same terminal rule -> Win.
    assert_eq!(brute_value(&b, &s, 4), Value::Win);
    let mut sol = Solver::new(&b);
    assert_eq!(sol.solve(&s), Value::Win);
}

/// Negate / Ord composition sanity: negate must be an order-reversing involution
/// on the 3-value lattice, and alpha-beta window negation must round-trip.
#[test]
fn negate_is_order_reversing_involution() {
    use Value::*;
    let all = [Loss, Draw, Win];
    for &x in &all {
        assert_eq!(x.negate().negate(), x, "involution");
    }
    // order-reversing
    for &x in &all {
        for &y in &all {
            assert_eq!(x < y, y.negate() < x.negate(), "order reversal {:?} {:?}", x, y);
        }
    }
    // window negation in ab(): child called with (beta.negate(), alpha.negate())
    // then result .negate(). The window [alpha,beta] must map to a valid window
    // [beta.negate(), alpha.negate()] i.e. lower <= upper preserved.
    for &alpha in &all {
        for &beta in &all {
            if alpha <= beta {
                assert!(beta.negate() <= alpha.negate(), "window negation preserves order");
            }
        }
    }
}
