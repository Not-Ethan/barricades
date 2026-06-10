//! Gates for the DSU-on-posts wall-legality fast path.
//!
//! `legal_walls` (DSU fast path, default ON) must be SET-EQUAL to
//! `legal_walls_bruteforce` (always-BFS authority) at EVERY position of
//! complete seeded wall-dense random playouts — the same harness style that
//! caught the keystone bug, including even-width boards — plus targeted
//! curve cases: floating loops (enclosing a pawn or not), border peninsulas
//! (one vs two border contacts), the collinear keystone geometry, and the
//! center-post contact case that proves the THREE-post rule is required
//! (an extremes-only rule would wrongly skip it).

use quoridor_solver::board::Board;
use quoridor_solver::movegen::{
    apply, dsu_wall_counters, legal_moves, legal_walls, legal_walls_bruteforce,
    wall_closes_no_curve,
};
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

fn wall_set(moves: &[Move]) -> std::collections::HashSet<(bool, u8, u8)> {
    moves
        .iter()
        .filter_map(|m| match *m {
            Move::Wall { wc, wr, horiz } => Some((horiz, wc, wr)),
            Move::Step(_) => None,
        })
        .collect()
}

/// Place a wall anchor bit without flipping turn / decrementing counts.
fn with_wall(b: &Board, s: &State, wc: u8, wr: u8, horiz: bool) -> State {
    let mut t = *s;
    if horiz {
        t.h_walls |= 1u64 << b.hbit(wc, wr);
    } else {
        t.v_walls |= 1u64 << b.vbit(wc, wr);
    }
    t
}

/// Set-equality `legal_walls == legal_walls_bruteforce` at every position of
/// complete seeded wall-dense playouts; returns positions checked and how
/// many of them had a nonempty legal wall set.
fn check_board(w: u8, h: u8, walls: u8, min_positions: usize) -> (usize, usize) {
    let b = Board::new(w, h, walls);
    let mut rng = Lcg::new(
        0xD5_0000 ^ ((w as u64) << 24) ^ ((h as u64) << 16) ^ ((walls as u64) << 8),
    );
    let mut positions = 0usize;
    let mut wallful = 0usize;
    // Complete games (to terminal or a generous ply cap) until enough
    // positions are checked. Wall-dense: prefer wall moves 3/4 of the time.
    for _game in 0..10_000 {
        let mut s = b.initial();
        for _ply in 0..200 {
            if b.is_terminal(&s) {
                break;
            }
            let lw = legal_walls(&b, &s);
            let bf = legal_walls_bruteforce(&b, &s);
            assert_eq!(
                wall_set(&lw),
                wall_set(&bf),
                "{w}x{h} W{walls}: legal_walls != bruteforce at pawns={:?} \
                 h={:#x} v={:#x} walls_left={:?} turn={}",
                s.pawn,
                s.h_walls,
                s.v_walls,
                s.walls_left,
                s.turn
            );
            positions += 1;
            if !lw.is_empty() {
                wallful += 1;
            }

            let moves = legal_moves(&b, &s);
            if moves.is_empty() {
                break;
            }
            // Wall-dense bias: pick among walls 3/4 of the time when possible.
            let walls_only: Vec<Move> = moves
                .iter()
                .copied()
                .filter(|m| matches!(m, Move::Wall { .. }))
                .collect();
            let pick = if !walls_only.is_empty() && !rng.next().is_multiple_of(4) {
                walls_only[(rng.next() % walls_only.len() as u64) as usize]
            } else {
                moves[(rng.next() % moves.len() as u64) as usize]
            };
            s = apply(&b, &s, pick);
        }
        // Keep playing complete games until both floors are met: enough
        // positions overall AND enough positions that actually had legal
        // walls (so the fast path is genuinely exercised).
        if positions >= min_positions && wallful >= 60 {
            break;
        }
    }
    assert!(
        positions >= min_positions,
        "{w}x{h} W{walls}: only {positions} positions checked, need >= {min_positions}"
    );
    (positions, wallful)
}

/// Gate (a): set-equality over dense playouts on every board in the matrix,
/// including the even-width boards where the keystone bug lived.
#[test]
fn set_equality_dense_playouts_all_boards() {
    let boards: [(u8, u8, u8); 8] = [
        (3, 3, 1),
        (4, 3, 2),
        (4, 4, 2),
        (3, 5, 2),
        (6, 4, 1),
        (6, 5, 2),
        (8, 3, 3),
        (5, 6, 2),
    ];
    let (skips0, falls0) = dsu_wall_counters();
    for &(w, h, walls) in &boards {
        let (positions, wallful) = check_board(w, h, walls, 500);
        // Sanity: enough positions actually exercised wall generation. (On
        // tiny budgets like W1 most plies of a complete game have the walls
        // exhausted, so this is an absolute floor, not a fraction.)
        assert!(
            wallful >= 60,
            "{w}x{h} W{walls}: only {wallful}/{positions} positions had legal walls — \
             playouts not wall-dense enough to exercise the fast path"
        );
    }
    // The fast path must actually have been exercised in both directions.
    let (skips1, falls1) = dsu_wall_counters();
    assert!(
        skips1 > skips0,
        "DSU fast path never skipped a BFS across all dense playouts (is QS_DSU_WALLS=0 set?)"
    );
    assert!(
        falls1 > falls0,
        "DSU fast path never fell through to the BFS across all dense playouts"
    );
}

/// Gate (c)(i): a floating loop (no border contact) enclosing a pawn.
/// Three walls form a "U" around the 2x2 cell box (2..=3, 2..=3) on 6x6;
/// the candidate V(3,2) closes the loop. With pawn0 inside, it must be
/// EXCLUDED — and it must have fallen to the BFS (extreme posts share the
/// U's component).
#[test]
fn floating_loop_enclosing_pawn_excluded() {
    let b = Board::new(6, 6, 4);
    let s = State {
        pawn: [b.idx(2, 2), b.idx(0, 4)], // p0 INSIDE the box, goal row 5
        h_walls: (1u64 << b.hbit(2, 1)) | (1u64 << b.hbit(2, 3)),
        v_walls: 1u64 << b.vbit(1, 2),
        walls_left: [2, 2],
        turn: 0,
    };
    // The candidate closes a curve (loop through the U) -> no fast skip.
    assert!(
        !wall_closes_no_curve(&b, &s, 3, 2, false),
        "loop-closing candidate V(3,2) must fall to the BFS"
    );
    let walls = wall_set(&legal_walls(&b, &s));
    assert!(
        !walls.contains(&(false, 3, 2)),
        "V(3,2) seals pawn0 inside the loop and must be excluded; got {walls:?}"
    );
    // Independent confirmation that it really strands pawn0.
    let probed = with_wall(&b, &s, 3, 2, false);
    assert!(!b.has_path(&probed, 0), "V(3,2) should strand pawn0");
    // And the DSU result still matches brute force here.
    assert_eq!(walls, wall_set(&legal_walls_bruteforce(&b, &s)));
}

/// Gate (c)(ii): the SAME loop with no pawn inside — the closing candidate
/// seals only empty cells, so the BFS must admit it (fall-through, then
/// INCLUDED).
#[test]
fn floating_loop_not_enclosing_pawn_included() {
    let b = Board::new(6, 6, 4);
    let s = State {
        pawn: [b.idx(0, 1), b.idx(5, 4)], // both pawns OUTSIDE the box
        h_walls: (1u64 << b.hbit(2, 1)) | (1u64 << b.hbit(2, 3)),
        v_walls: 1u64 << b.vbit(1, 2),
        walls_left: [2, 2],
        turn: 0,
    };
    // Still curve-closing -> falls to the BFS...
    assert!(!wall_closes_no_curve(&b, &s, 3, 2, false));
    // ...which must INCLUDE it: no pawn is stranded.
    let walls = wall_set(&legal_walls(&b, &s));
    assert!(
        walls.contains(&(false, 3, 2)),
        "V(3,2) strands nobody (empty loop) and must be legal; got {walls:?}"
    );
    assert_eq!(walls, wall_set(&legal_walls_bruteforce(&b, &s)));
}

/// Gate (c)(iii): border peninsula on 6x5. H(0,2) touches the left border
/// post (0,3). Extending it with H(2,2) keeps ONE border contact (extreme
/// posts: peninsula/BORDER, fresh, fresh -> pairwise distinct): fast-skip OK
/// and legal. Adding the second border contact H(4,2) (right extreme post
/// (6,3) is a border post) must fall to the BFS and be excluded — it spans
/// the full width and strands pawn0.
#[test]
fn border_peninsula_one_contact_skips_two_contacts_excluded() {
    let b = Board::new(6, 5, 3);
    // One wall from the left border.
    let s1 = State {
        pawn: [b.idx(3, 0), b.idx(3, 4)],
        h_walls: 1u64 << b.hbit(0, 2),
        v_walls: 0,
        walls_left: [2, 2],
        turn: 0,
    };
    // Peninsula extension H(2,2): one border contact -> fast-skip OK.
    assert!(
        wall_closes_no_curve(&b, &s1, 2, 2, true),
        "H(2,2) extends a one-contact peninsula and must be fast-skippable"
    );
    let walls1 = wall_set(&legal_walls(&b, &s1));
    assert!(walls1.contains(&(true, 2, 2)), "H(2,2) must be legal");
    assert_eq!(walls1, wall_set(&legal_walls_bruteforce(&b, &s1)));

    // Now with the extension placed: H(4,2) would make the SECOND border
    // contact (post (6,3)) -> same BORDER component -> falls to BFS,
    // and the full-width span strands pawn0 -> excluded.
    let s2 = State {
        h_walls: s1.h_walls | (1u64 << b.hbit(2, 2)),
        ..s1
    };
    assert!(
        !wall_closes_no_curve(&b, &s2, 4, 2, true),
        "H(4,2) makes a second border contact and must fall to the BFS"
    );
    let walls2 = wall_set(&legal_walls(&b, &s2));
    assert!(
        !walls2.contains(&(true, 4, 2)),
        "H(4,2) completes a full-width barrier and must be excluded; got {walls2:?}"
    );
    assert_eq!(walls2, wall_set(&legal_walls_bruteforce(&b, &s2)));
}

/// Gate (c)(iii) complement: a border-to-border curve that does NOT strand a
/// pawn. On 5x6, V(2,0)+V(2,2) build a peninsula from the top border down
/// post-column 3; candidate V(2,4) reaches the bottom border (post (3,6)).
/// It must fall to the BFS (both extreme components are BORDER) and the BFS
/// must INCLUDE it: a full-height vertical wall separates left from right,
/// but both pawns reach their goal ROWS on their own side.
#[test]
fn border_to_border_vertical_span_falls_but_legal() {
    let b = Board::new(5, 6, 4);
    let s = State {
        pawn: [b.idx(2, 0), b.idx(2, 5)],
        h_walls: 0,
        v_walls: (1u64 << b.vbit(2, 0)) | (1u64 << b.vbit(2, 2)),
        walls_left: [2, 2],
        turn: 0,
    };
    assert!(
        !wall_closes_no_curve(&b, &s, 2, 4, false),
        "V(2,4) connects the peninsula to the bottom border and must fall to the BFS"
    );
    let walls = wall_set(&legal_walls(&b, &s));
    assert!(
        walls.contains(&(false, 2, 4)),
        "full-height vertical span strands nobody and must be legal; got {walls:?}"
    );
    assert_eq!(walls, wall_set(&legal_walls_bruteforce(&b, &s)));
}

/// Gate (c)(iv): the collinear keystone geometry (flanks sharing endpoint
/// POSTS with the gap) — exactly the configuration the deleted anchor-distance
/// heuristic got wrong. Both flank components contain border posts, so the
/// keystone's extreme posts share the BORDER component: it must fall to the
/// BFS and be excluded.
#[test]
fn collinear_keystone_falls_to_bfs_and_excluded() {
    let b = Board::new(6, 4, 5);
    let s = State {
        pawn: [b.idx(2, 1), b.idx(0, 1)],
        h_walls: (1u64 << b.hbit(0, 1)) | (1u64 << b.hbit(4, 1)),
        v_walls: 0,
        walls_left: [1, 1],
        turn: 0,
    };
    assert!(
        !wall_closes_no_curve(&b, &s, 2, 1, true),
        "keystone H(2,1) must fall to the BFS (extreme posts both reach the border)"
    );
    let walls = wall_set(&legal_walls(&b, &s));
    assert!(
        !walls.contains(&(true, 2, 1)),
        "keystone H(2,1) must be excluded; got {walls:?}"
    );
    assert_eq!(walls, wall_set(&legal_walls_bruteforce(&b, &s)));
}

/// Center-post contact: the case that mandates the THREE-post rule. On 6x6
/// the complex V(1,2)+H(1,3)+V(0,2) runs from post (1,2) to post (2,2) —
/// and (2,2) is the CENTER post of candidate H(1,1) (a perpendicular wall
/// may END at a candidate's center post; the overlap rules only forbid
/// crossing). The candidate's EXTREME posts (1,2) and (3,2) lie in different
/// components, so an extremes-only rule would wrongly fast-skip; the
/// three-post rule sees p0 ~ p1 and falls to the BFS. With pawn0 inside the
/// enclosed cells the candidate must be excluded; with pawns outside it must
/// be included.
#[test]
fn center_post_contact_requires_three_post_rule() {
    let b = Board::new(6, 6, 4);
    let complex_v = (1u64 << b.vbit(1, 2)) | (1u64 << b.vbit(0, 2));
    let complex_h = 1u64 << b.hbit(1, 3);

    // The candidate closes a loop THROUGH ITS CENTER POST -> must fall.
    let s_in = State {
        pawn: [b.idx(1, 2), b.idx(5, 5)], // p0 inside cells (1,2)/(1,3)
        h_walls: complex_h,
        v_walls: complex_v,
        walls_left: [2, 2],
        turn: 0,
    };
    assert!(
        !wall_closes_no_curve(&b, &s_in, 1, 1, true),
        "H(1,1)'s center post touches the complex: three-post rule must fall to BFS"
    );
    let walls_in = wall_set(&legal_walls(&b, &s_in));
    assert!(
        !walls_in.contains(&(true, 1, 1)),
        "H(1,1) seals pawn0 into the loop and must be excluded; got {walls_in:?}"
    );
    let probed = with_wall(&b, &s_in, 1, 1, true);
    assert!(!b.has_path(&probed, 0), "H(1,1) should strand pawn0");
    assert_eq!(walls_in, wall_set(&legal_walls_bruteforce(&b, &s_in)));

    // Same geometry, pawns outside: the loop seals only empty cells -> legal.
    let s_out = State {
        pawn: [b.idx(4, 1), b.idx(5, 5)],
        ..s_in
    };
    assert!(!wall_closes_no_curve(&b, &s_out, 1, 1, true));
    let walls_out = wall_set(&legal_walls(&b, &s_out));
    assert!(
        walls_out.contains(&(true, 1, 1)),
        "H(1,1) strands nobody here and must be legal; got {walls_out:?}"
    );
    assert_eq!(walls_out, wall_set(&legal_walls_bruteforce(&b, &s_out)));
}

/// Empty board: every candidate's three posts are fresh (or one BORDER) and
/// pairwise distinct, so everything fast-skips and matches brute force.
#[test]
fn empty_board_all_candidates_skip_and_match() {
    for &(w, h) in &[(3u8, 3u8), (6, 5), (8, 3), (5, 6)] {
        let b = Board::new(w, h, 3);
        let s = b.initial();
        for horiz in [true, false] {
            for wc in 0..w - 1 {
                for wr in 0..h - 1 {
                    assert!(
                        wall_closes_no_curve(&b, &s, wc, wr, horiz),
                        "{w}x{h}: empty-board candidate ({wc},{wr},{horiz}) should fast-skip"
                    );
                }
            }
        }
        assert_eq!(
            wall_set(&legal_walls(&b, &s)),
            wall_set(&legal_walls_bruteforce(&b, &s))
        );
    }
}
