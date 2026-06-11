//! THROWAWAY adversarial audit test for the TT bound-flag logic.
//! Compares Solver::solve (reused across positions, exercising cross-position
//! TT reuse) against brute_value (the unpruned oracle) on many reachable
//! positions of small boards. Delete after auditing.

use quoridor_solver::board::Board;
use quoridor_solver::movegen::{apply, legal_moves};
use quoridor_solver::solver::{brute_value, Solver, Value};
use quoridor_solver::state::State;

/// The exact ceiling solve() uses internally, so brute_value's depth-limited
/// Draw semantics match solve()'s exactly.
fn ceiling(b: &Board) -> u32 {
    let w = b.w as u32;
    let h = b.h as u32;
    let walls = b.walls as u32;
    4 * (w + h) + 2 * walls + 8
}

/// BFS over the reachable game graph from the initial state, up to `max_states`
/// distinct states. For each, assert solve() (on the REUSED solver) == oracle.
fn audit_board(w: u8, h: u8, walls: u8, max_states: usize) -> usize {
    let b = Board::new(w, h, walls);
    let cap = ceiling(&b);

    // Single reused solver -> exercises cross-position TT reuse, the risk.
    let mut sol = Solver::new(&b);

    let mut seen = std::collections::HashSet::new();
    let mut queue = std::collections::VecDeque::new();
    let start = b.initial();
    seen.insert(start);
    queue.push_back(start);

    let mut checked = 0usize;
    while let Some(s) = queue.pop_front() {
        if checked >= max_states {
            break;
        }
        if b.is_terminal(&s) {
            continue;
        }
        let got = sol.solve(&s);
        let want = brute_value(&b, &s, cap);
        assert_eq!(
            got, want,
            "MISMATCH on {w}x{h} w{walls}: pawns={:?} h_walls={:#x} v_walls={:#x} wl={:?} turn={} -> solver={:?} oracle={:?}",
            s.pawn, s.h_walls, s.v_walls, s.walls_left, s.turn, got, want
        );
        checked += 1;

        for m in legal_moves(&b, &s) {
            let s2 = apply(&b, &s, m);
            if seen.insert(s2) {
                queue.push_back(s2);
            }
        }
    }
    checked
}

/// Same as audit_board but visits states in a RANDOMIZED (non-BFS) order and
/// interleaves solve() calls so the TT is populated with assorted depths/windows
/// before/after each check — maximizes the chance a stale or mis-flagged TT
/// entry is reused for a different window.
fn audit_board_shuffled(w: u8, h: u8, walls: u8, max_states: usize, seed: u64) -> usize {
    let b = Board::new(w, h, walls);
    let cap = ceiling(&b);
    let mut sol = Solver::new(&b);

    // Collect reachable states first.
    let mut seen = std::collections::HashSet::new();
    let mut order: Vec<State> = Vec::new();
    let mut queue = std::collections::VecDeque::new();
    let start = b.initial();
    seen.insert(start);
    queue.push_back(start);
    while let Some(s) = queue.pop_front() {
        if order.len() >= max_states {
            break;
        }
        if !b.is_terminal(&s) {
            order.push(s);
        }
        for m in legal_moves(&b, &s) {
            let s2 = apply(&b, &s, m);
            if seen.insert(s2) {
                queue.push_back(s2);
            }
        }
    }

    // LCG shuffle.
    let mut st = seed ^ 0x9E37_79B9_7F4A_7C15;
    let mut rnd = || {
        st = st.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        st >> 16
    };
    for i in (1..order.len()).rev() {
        let j = (rnd() % (i as u64 + 1)) as usize;
        order.swap(i, j);
    }

    let mut checked = 0;
    for s in &order {
        let got = sol.solve(s);
        let want = brute_value(&b, s, cap);
        assert_eq!(
            got, want,
            "SHUFFLED MISMATCH {w}x{h} w{walls}: pawns={:?} h={:#x} v={:#x} wl={:?} turn={} -> solver={:?} oracle={:?}",
            s.pawn, s.h_walls, s.v_walls, s.walls_left, s.turn, got, want
        );
        checked += 1;
    }
    checked
}

#[test]
fn audit_4x4_w2_reused_solver() {
    let n = audit_board(4, 4, 2, 6000);
    assert!(n > 100, "only {n} states checked");
    eprintln!("4x4 w2 BFS checked {n} states");
}

#[test]
fn audit_3x5_w2_reused_solver() {
    let n = audit_board(3, 5, 2, 6000);
    assert!(n > 100, "only {n} states checked");
    eprintln!("3x5 w2 BFS checked {n} states");
}

#[test]
fn audit_4x4_w1_reused_solver() {
    let n = audit_board(4, 4, 1, 8000);
    assert!(n > 100, "only {n} states checked");
    eprintln!("4x4 w1 BFS checked {n} states");
}

#[test]
fn audit_3x3_w1_reused_solver() {
    let n = audit_board(3, 3, 1, 20000);
    eprintln!("3x3 w1 BFS checked {n} states (full graph)");
}

#[test]
fn audit_shuffled_4x4_w2() {
    let n = audit_board_shuffled(4, 4, 2, 4000, 0xDEAD_BEEF);
    assert!(n > 100);
    eprintln!("4x4 w2 shuffled checked {n} states");
}

#[test]
fn audit_shuffled_3x5_w2() {
    let n = audit_board_shuffled(3, 5, 2, 4000, 0x1357_9BDF);
    assert!(n > 100);
    eprintln!("3x5 w2 shuffled checked {n} states");
}
