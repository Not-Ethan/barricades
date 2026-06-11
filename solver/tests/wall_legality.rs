//! Bug-1 regression: the floating-wall fast-path used to admit an illegal
//! board-spanning "keystone" wall on even-width boards, inverting solver values.
//! The fast-path is now deleted — `legal_walls` always runs the two-player
//! connectivity BFS. These tests pin both the exact keystone exclusion (and the
//! value it was corrupting) and a general legality invariant over random games.

use quoridor_solver::board::Board;
use quoridor_solver::movegen::{apply, legal_moves, legal_walls, legal_walls_bruteforce};
use quoridor_solver::solver::{Solver, Value};
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

/// Place a wall anchor bit WITHOUT flipping turn / decrementing counts, so we
/// can probe connectivity exactly the way `legal_walls` does internally.
fn with_wall(b: &Board, s: &State, m: Move) -> State {
    let mut t = *s;
    match m {
        Move::Wall { wc, wr, horiz: true } => t.h_walls |= 1u64 << b.hbit(wc, wr),
        Move::Wall { wc, wr, horiz: false } => t.v_walls |= 1u64 << b.vbit(wc, wr),
        Move::Step(_) => unreachable!(),
    }
    t
}

fn wall_set(moves: &[Move]) -> std::collections::HashSet<(bool, u8, u8)> {
    moves
        .iter()
        .filter_map(|m| match *m {
            Move::Wall { wc, wr, horiz } => Some((horiz, wc, wr)),
            Move::Step(_) => None,
        })
        .collect()
}

/// Bug-1 test 1: exact keystone repro through `Solver::solve`.
///
/// Board 6x4, walls=5. Two horizontal flanks at H(0,1) and H(4,1) (anchors
/// Chebyshev-2 apart along the axis, sharing lattice endpoints with the gap).
/// The keystone H(2,1) completes a goal-spanning barrier that strands a pawn —
/// it must be EXCLUDED from legal_walls. With the old fast-path it was admitted
/// (it is interior and Chebyshev-2 from both flanks), inverting the value.
#[test]
fn keystone_excluded_and_value_uninverted_6x4() {
    let b = Board::new(6, 4, 5);
    let s = State {
        pawn: [b.idx(2, 1), b.idx(0, 1)],
        h_walls: (1u64 << b.hbit(0, 1)) | (1u64 << b.hbit(4, 1)),
        v_walls: 0,
        walls_left: [1, 1],
        turn: 0,
    };

    let walls = wall_set(&legal_walls(&b, &s));
    assert!(
        !walls.contains(&(true, 2, 1)),
        "keystone wall H(2,1) must be excluded (it strands a pawn); legal walls = {:?}",
        walls
    );

    // The keystone was the move that flipped the value: with the bug, solve
    // returned Loss; the true value is Win.
    let mut sol = Solver::new(&b);
    assert_eq!(
        sol.solve(&s),
        Value::Win,
        "6x4 W5 keystone position must be a first-player Win (was Loss under the bug)"
    );
}

/// Bug-1 test 1b: target-board witness. Board 6x5 with horizontal flanks at
/// H(0,2) and H(4,2); candidate H(2,2) is the keystone and must be excluded.
#[test]
fn keystone_excluded_6x5_witness() {
    let b = Board::new(6, 5, 5);
    let s = State {
        pawn: [b.idx(2, 1), b.idx(2, 3)],
        h_walls: (1u64 << b.hbit(0, 2)) | (1u64 << b.hbit(4, 2)),
        v_walls: 0,
        walls_left: [1, 1],
        turn: 0,
    };
    let walls = wall_set(&legal_walls(&b, &s));
    assert!(
        !walls.contains(&(true, 2, 2)),
        "keystone wall H(2,2) must be excluded on 6x5; legal walls = {:?}",
        walls
    );
    // And it really would strand a pawn (independent confirmation).
    let probed = with_wall(&b, &s, Move::Wall { wc: 2, wr: 2, horiz: true });
    assert!(
        !(b.has_path(&probed, 0) && b.has_path(&probed, 1)),
        "H(2,2) should disconnect a pawn here"
    );
}

/// Bug-1 test 2: legality invariant over seeded random games on even boards.
///
/// At every node: (a) every wall returned by `legal_walls` keeps BOTH pawns'
/// paths open after placement, and (b) `legal_walls == legal_walls_bruteforce`
/// as a set. >= 60 nodes per board.
#[test]
fn legal_walls_always_keep_paths_open_even_boards() {
    let boards: [(u8, u8); 5] = [(6, 4), (6, 5), (8, 3), (5, 6), (6, 6)];
    for &(w, h) in &boards {
        for &walls in &[4u8, 5] {
            let b = Board::new(w, h, walls);
            let mut rng = Lcg::new(0xA11CE ^ ((w as u64) << 16 ^ (h as u64) << 8 ^ walls as u64));
            let mut nodes = 0usize;
            // Several games until we have checked >= 60 nodes.
            'games: for _ in 0..200 {
                let mut s = b.initial();
                for _ in 0..40 {
                    if b.is_terminal(&s) {
                        break;
                    }
                    let lw = legal_walls(&b, &s);

                    // (a) Every returned wall keeps both pawns connected.
                    for &m in &lw {
                        let probed = with_wall(&b, &s, m);
                        assert!(
                            b.has_path(&probed, 0) && b.has_path(&probed, 1),
                            "{w}x{h} W{walls}: legal_walls returned {:?} which strands a pawn \
                             at pawns={:?} h={:#x} v={:#x} turn={}",
                            m, s.pawn, s.h_walls, s.v_walls, s.turn
                        );
                    }

                    // (b) legal_walls == legal_walls_bruteforce as a set.
                    let bf = legal_walls_bruteforce(&b, &s);
                    assert_eq!(
                        wall_set(&lw),
                        wall_set(&bf),
                        "{w}x{h} W{walls}: legal_walls != bruteforce at pawns={:?} h={:#x} v={:#x} turn={}",
                        s.pawn, s.h_walls, s.v_walls, s.turn
                    );

                    nodes += 1;

                    let moves = legal_moves(&b, &s);
                    if moves.is_empty() {
                        break;
                    }
                    let pick = (rng.next() % moves.len() as u64) as usize;
                    s = apply(&b, &s, moves[pick]);
                }
                if nodes >= 80 {
                    break 'games;
                }
            }
            assert!(
                nodes >= 60,
                "{w}x{h} W{walls}: only {nodes} nodes checked, need >= 60"
            );
        }
    }
}
