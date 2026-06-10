//! Gates for the SHADOW legality-filter benchmark (writeup-vs-DSU counters in
//! `movegen::walls_dsu`). Instrumentation only — these tests pin (a) the
//! per-bucket self-consistency invariant `candidates == dsu_skip + dsu_fall
//! == wu_skip + wu_fall`, (b) that the tally is thread-local and drains, and
//! (c) targeted writeup-predicate decisions under the documented reading
//! (border posts on the post lattice; contact = post COINCIDENCE at >= 2 of
//! the candidate's 3 posts).
//!
//! NOTE: the tally is per-thread and `cargo test` runs each test on its own
//! thread, so draining at test start isolates each test's accounting.

use quoridor_solver::board::Board;
use quoridor_solver::movegen::{apply, legal_moves, legal_walls, take_shadow_tally};
use quoridor_solver::state::Move;

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

/// Per-bucket self-consistency over wall-dense random playouts: every
/// non-overlapping candidate is counted exactly once on the DSU side AND
/// exactly once on the writeup side, in the same bucket.
#[test]
fn shadow_tally_self_consistent_over_dense_playouts() {
    let _ = take_shadow_tally(); // isolate this test's accounting
    let boards: [(u8, u8, u8); 4] = [(5, 5, 2), (6, 5, 2), (6, 4, 2), (4, 4, 2)];
    for &(w, h, walls) in &boards {
        let b = Board::new(w, h, walls);
        let mut rng = Lcg::new(0x5AD0 ^ ((w as u64) << 16) ^ ((h as u64) << 8) ^ walls as u64);
        for _game in 0..200 {
            let mut s = b.initial();
            for _ply in 0..200 {
                if b.is_terminal(&s) {
                    break;
                }
                let _ = legal_walls(&b, &s); // tallies one call's worth
                let moves = legal_moves(&b, &s);
                if moves.is_empty() {
                    break;
                }
                // Wall-dense bias: pick among walls 3/4 of the time.
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
        }
    }

    let tally = take_shadow_tally();
    let total: u64 = tally.rows.iter().map(|r| r.candidates).sum();
    assert!(
        total > 0,
        "shadow tally empty across dense playouts (is QS_SHADOW=0 or QS_DSU_WALLS=0 set?)"
    );
    let mut nonzero_buckets = 0;
    for (bucket, r) in tally.rows.iter().enumerate() {
        assert_eq!(
            r.candidates,
            r.dsu_skip + r.dsu_fall,
            "bucket {bucket}: candidates != dsu_skip + dsu_fall"
        );
        assert_eq!(
            r.candidates,
            r.wu_skip + r.wu_fall,
            "bucket {bucket}: candidates != wu_skip + wu_fall"
        );
        if r.candidates > 0 {
            nonzero_buckets += 1;
            // Op accounting must be live wherever candidates were examined.
            assert!(r.dsu_finds > 0 && r.dsu_unions > 0, "bucket {bucket}: no op accounting");
        }
    }
    // Wall-dense playouts on W2 boards must populate several density buckets
    // (0..=4 placed walls exist on a W2 board).
    assert!(
        nonzero_buckets >= 3,
        "only {nonzero_buckets} density buckets populated — playouts not wall-dense enough"
    );
    // Drained: a second take must be empty.
    let drained = take_shadow_tally();
    assert!(
        drained.rows.iter().all(|r| r.candidates == 0),
        "take_shadow_tally did not drain the thread-local tally"
    );
}

/// Targeted writeup-predicate decisions under the documented reading, read
/// back through the tally (the predicate itself is private by design — the
/// shadow is observation-only). Empty 5x5 board, ONE call to `legal_walls`:
/// FAITHFUL writeup rule on an empty board: contacts (border posts +
/// occupied posts, counted together) max out at 1 per candidate on any board
/// wider than 2 (an H-candidate's posts sit in one interior post-row and can
/// touch the side border at most once; transposed for V). One contact = a
/// peninsula = cannot close a curve, so the writeup predicate must fire on
/// ZERO empty-board candidates — the original mis-reading ("border at >= 1
/// post OR walls at >= 2") wrongly fired on half of them.
#[test]
fn shadow_writeup_predicate_empty_board_border_rule() {
    let _ = take_shadow_tally();
    let b = Board::new(5, 5, 2);
    let s = b.initial();
    let lw = legal_walls(&b, &s);
    assert_eq!(lw.len(), 32, "empty 5x5: all 32 candidates legal");
    let tally = take_shadow_tally();
    let r = &tally.rows[0]; // 0 placed walls
    assert_eq!(r.candidates, 32);
    assert_eq!((r.dsu_skip, r.dsu_fall), (32, 0), "empty board: all fast-skip");
    assert_eq!(
        (r.wu_skip, r.wu_fall),
        (32, 0),
        "faithful writeup rule: no empty-board candidate reaches 2 contacts"
    );
}

/// FAITHFUL contact rule pinned through the tally on a 6x6 board with two
/// parallel floating H-walls at (1,1) and (1,3) (occupied posts (1..=3, 2)
/// and (1..=3, 4)). The expected wu_fall is recomputed by an INDEPENDENT
/// in-test implementation of the faithful predicate (contacts = candidate
/// posts that are border posts OR occupied posts, counted together, >= 2 =>
/// fall) over the same candidate set, so the test pins semantics rather than
/// a brittle hand count. No candidate strands a pawn here (the placed walls
/// are floating, separate components), so the legal set IS the candidate set.
#[test]
fn shadow_writeup_predicate_two_post_contact_rule() {
    use quoridor_solver::state::State;
    let _ = take_shadow_tally();
    let b = Board::new(6, 6, 4);
    let s = State {
        pawn: [b.idx(5, 0), b.idx(5, 5)],
        h_walls: (1u64 << b.hbit(1, 1)) | (1u64 << b.hbit(1, 3)),
        v_walls: 0,
        walls_left: [3, 3],
        turn: 0,
    };
    let lw = legal_walls(&b, &s);
    let tally = take_shadow_tally();
    let r = &tally.rows[2]; // 2 placed walls
    assert_eq!(r.candidates, r.wu_skip + r.wu_fall);
    assert!(r.candidates > 0);

    // Independent faithful-predicate reference over the candidate set.
    let occupied: std::collections::HashSet<(u8, u8)> = [(1u8, 2u8), (2, 2), (3, 2), (1, 4), (2, 4), (3, 4)]
        .into_iter()
        .collect();
    let mut ref_fall = 0u64;
    let mut seen = 0u64;
    for horiz in [true, false] {
        for wc in 0..b.w - 1 {
            for wr in 0..b.h - 1 {
                let posts: [(u8, u8); 3] = if horiz {
                    [(wc, wr + 1), (wc + 1, wr + 1), (wc + 2, wr + 1)]
                } else {
                    [(wc + 1, wr), (wc + 1, wr + 1), (wc + 1, wr + 2)]
                };
                if !lw.iter().any(|m| matches!(*m, Move::Wall { wc: c, wr: r2, horiz: hz } if c == wc && r2 == wr && hz == horiz)) {
                    continue; // overlap-excluded: not a candidate
                }
                seen += 1;
                let contacts = posts
                    .iter()
                    .filter(|&&(pc, pr)| {
                        let border = pc == 0 || pc == b.w || pr == 0 || pr == b.h;
                        border || occupied.contains(&(pc, pr))
                    })
                    .count();
                if contacts >= 2 {
                    ref_fall += 1;
                }
            }
        }
    }
    assert_eq!(
        seen, r.candidates,
        "every non-overlapping candidate is legal on this board (BFS excludes none)"
    );
    assert_eq!(
        r.wu_fall, ref_fall,
        "tally must match the independent faithful-predicate reference"
    );
    assert!(ref_fall > 0, "the fixture must exercise the fall path");
}
