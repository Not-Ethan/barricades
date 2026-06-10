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
/// every candidate with a border post must wu_fall; interior candidates have
/// no occupied posts (no walls placed) and must wu_skip. On 5x5 the anchors
/// are 4x4 = 16 per orientation; interior-by-posts means H-walls with
/// wc in {1} x wr in 0..4? — count instead by formula: an H-candidate's posts
/// are (wc, wr+1)..(wc+2, wr+1) with wr+1 in 1..=3 (never a border row), so
/// it is border iff wc == 0 or wc + 2 == 5, i.e. wc in {0, 3}: 8 of 16
/// H-candidates are border; by the same argument (transposed) 8 of 16
/// V-candidates. Expect wu_fall == 16, wu_skip == 16, dsu_skip == 32 (empty
/// board: everything fast-skips).
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
        (16, 16),
        "writeup border rule on posts: exactly the wc/wr in {{0, 3}} candidates touch the border"
    );
}

/// Contact rule ((b): >= 2 occupied posts) pinned through the tally on a 6x6
/// board with two parallel floating H-walls at (1,1) and (1,3), whose post
/// spans are (1,2)..(3,2) and (1,4)..(3,4). The only INTERIOR candidates
/// with >= 2 post contacts are the verticals V(0,2), V(1,2), V(2,2): their
/// posts (wc+1, 2..=4) hit both occupied post-rows 2 and 4 in one column.
/// (H-candidates keep all 3 posts in ONE post-row; the two-contact spans in
/// rows 2/4 all overlap a placed wall, so none of them is a candidate. V
/// spans other than wr=2 contain only one of rows {2,4}.) No candidate can
/// strand a pawn here (one extra wall closes no curve: the two placed walls
/// are floating, separate components), so the BFS excludes nothing and the
/// legal set IS the candidate set — letting the test enumerate candidates
/// through the public move list. Expected: wu_fall == border-post candidates
/// + 3, everything else wu_skip.
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
    let r = &tally.rows[2]; // 2 placed walls (walls budget 4 -> 2*4 - 3 - 3)
    assert_eq!(r.candidates, r.wu_skip + r.wu_fall);
    assert!(r.candidates > 0);

    // Hand count (see the doc comment): exactly three interior candidates
    // have >= 2 occupied-post contacts. Every other wu_fall must come from
    // the border-post rule (a), which the loop below recounts independently
    // over the candidate set.
    let interior_two_contact = 3u64; // V(0,2), V(1,2), V(2,2)
    let mut border = 0u64;
    let mut seen = 0u64;
    for horiz in [true, false] {
        for wc in 0..b.w - 1 {
            for wr in 0..b.h - 1 {
                // Mirror walls_dsu's candidate filter: skip overlaps.
                let posts: [(u8, u8); 3] = if horiz {
                    [(wc, wr + 1), (wc + 1, wr + 1), (wc + 2, wr + 1)]
                } else {
                    [(wc + 1, wr), (wc + 1, wr + 1), (wc + 1, wr + 2)]
                };
                // Overlap check via the public API: a candidate is one of
                // walls_dsu's candidates iff it appears in lw OR fell to the
                // BFS and was excluded — on this board nothing is excluded
                // (verify below), so lw IS the candidate set.
                if !lw.iter().any(|m| matches!(*m, Move::Wall { wc: c, wr: r2, horiz: hz } if c == wc && r2 == wr && hz == horiz)) {
                    continue;
                }
                seen += 1;
                if posts
                    .iter()
                    .any(|&(pc, pr)| pc == 0 || pc == b.w || pr == 0 || pr == b.h)
                {
                    border += 1;
                }
            }
        }
    }
    assert_eq!(
        seen, r.candidates,
        "every non-overlapping candidate is legal on this board (BFS excludes none), \
         so the legal set must equal the candidate set"
    );
    assert_eq!(
        r.wu_fall,
        border + interior_two_contact,
        "wu_fall must be exactly border-post candidates plus the three \
         interior two-contact verticals V(0,2)/V(1,2)/V(2,2)"
    );
}
