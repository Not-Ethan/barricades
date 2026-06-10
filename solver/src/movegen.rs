//! Move generation: pawn steps (with jump rules), wall placements, and `apply`.
//! Mirrors `smallboard/engine.py` exactly; the cross-language differential test
//! (`tests/diff_vs_smallboard.rs`) guards the equivalence.
//!
//! Wall legality is exact: every non-overlapping candidate wall is accepted
//! only if BOTH pawns still have a path to their goal after it is placed.
//! The authority is the two `has_path` BFS checks; `legal_walls_bruteforce`
//! runs them on every candidate and is the test reference.
//!
//! `legal_walls` additionally uses a PROVABLY SOUND fast path (default ON,
//! `QS_DSU_WALLS=0` disables): by planar duality, a set of blocked edges
//! separates the pawn grid iff the wall segments (drawn on the lattice of
//! POSTS, the `(w+1) x (h+1)` cell corners) contain a path connecting two
//! distinct border posts or a closed loop. We maintain a DSU over posts with
//! all border posts pre-merged into one BORDER component and each placed
//! wall's three posts unioned. A candidate whose three posts lie in pairwise-
//! distinct components closes no curve and therefore CANNOT disconnect
//! anything: its BFS is skipped and it is legal (given the overlap test).
//! Otherwise the candidate falls through to the BFS pair, which remains the
//! authority — the fast path is a one-sided SKIP condition and never rejects.
//!
//! History: a prior anchor-Chebyshev "floating-wall" predicate was deleted
//! after it admitted an illegal board-spanning keystone wall (collinear walls
//! sharing an endpoint POST looked "floating" to anchor distance), inverting
//! values on even-width boards (`tests/wall_legality.rs` pins it). The DSU
//! path computes touching on the posts themselves, so that geometry is
//! handled natively: the keystone's extreme posts are both in the BORDER
//! component and it falls to the BFS. Gated by set-equality vs brute force
//! (`tests/dsu_walls.rs`).

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::board::Board;
use crate::state::{Move, State};

/// `DIRS` in the same order as `smallboard/engine.py`.
const DIRS: [(i16, i16); 4] = [(0, 1), (0, -1), (1, 0), (-1, 0)];

#[inline]
fn on_board(b: &Board, c: i16, r: i16) -> bool {
    c >= 0 && r >= 0 && c < b.w as i16 && r < b.h as i16
}

/// Whether the step from `(c, r)` to orthogonally-adjacent `(c+dc, r+dr)` is
/// blocked by a wall. Thin wrapper over the Task-2 predicate so movegen and
/// BFS share the SAME blocking logic.
#[inline]
fn blocked(b: &Board, s: &State, c: i16, r: i16, dc: i16, dr: i16) -> bool {
    b.step_blocked(s, c, r, dc, dr)
}

/// Legal pawn-step destination cell indices. Mirrors `Engine.legal_steps`.
pub fn legal_steps(b: &Board, s: &State) -> Vec<u8> {
    let (mc, mr) = b.cr(s.pawn[s.turn as usize]);
    let (oc, or) = b.cr(s.pawn[(1 - s.turn) as usize]);
    let (mc, mr) = (mc as i16, mr as i16);
    let (oc, or) = (oc as i16, or as i16);

    let mut out: Vec<u8> = Vec::with_capacity(5);
    for &(dx, dy) in &DIRS {
        let (ac, ar) = (mc + dx, mr + dy);
        if !on_board(b, ac, ar) || blocked(b, s, mc, mr, dx, dy) {
            continue;
        }
        if (ac, ar) != (oc, or) {
            out.push(b.idx(ac as u8, ar as u8));
            continue;
        }
        // The adjacent cell holds the opponent: try to jump.
        let (lc, lr) = (oc + dx, or + dy);
        if on_board(b, lc, lr) && !blocked(b, s, oc, or, dx, dy) {
            out.push(b.idx(lc as u8, lr as u8));
        } else {
            // Straight jump blocked -> the two perpendicular diagonals.
            for &(px, py) in &DIRS {
                if (px, py) == (dx, dy) || (px, py) == (-dx, -dy) {
                    continue;
                }
                let (gc, gr) = (oc + px, or + py);
                if on_board(b, gc, gr) && !blocked(b, s, oc, or, px, py) {
                    out.push(b.idx(gc as u8, gr as u8));
                }
            }
        }
    }
    out
}

/// Whether placing a wall at anchor `(wc, wr)` with orientation `horiz` would
/// overlap an existing wall (i.e. is geometrically illegal). Mirrors
/// `Engine._overlaps`. Uses the boundary-guarded anchor accessors so
/// out-of-range neighbours count as absent.
fn overlaps(b: &Board, s: &State, wc: u8, wr: u8, horiz: bool) -> bool {
    let (c, r) = (wc as i16, wr as i16);
    if horiz {
        b.h_anchor(s, c, r)
            || b.h_anchor(s, c - 1, r)
            || b.h_anchor(s, c + 1, r)
            || b.v_anchor(s, c, r)
    } else {
        b.v_anchor(s, c, r)
            || b.v_anchor(s, c, r - 1)
            || b.v_anchor(s, c, r + 1)
            || b.h_anchor(s, c, r)
    }
}

/// Apply a move, returning the successor state. Mirrors `Engine.apply_move`.
pub fn apply(b: &Board, s: &State, m: Move) -> State {
    let mut t = *s;
    match m {
        Move::Step(dest) => {
            t.pawn[s.turn as usize] = dest;
        }
        Move::Wall { wc, wr, horiz } => {
            if horiz {
                t.h_walls |= 1u64 << b.hbit(wc, wr);
            } else {
                t.v_walls |= 1u64 << b.vbit(wc, wr);
            }
            t.walls_left[s.turn as usize] -= 1;
        }
    }
    t.turn = 1 - s.turn;
    t
}

/// Set the wall anchor bit for `(wc, wr, horiz)` WITHOUT flipping turn or
/// decrementing the wall count — used only to test path connectivity.
#[inline]
fn with_wall_bit(b: &Board, s: &State, wc: u8, wr: u8, horiz: bool) -> State {
    let mut t = *s;
    if horiz {
        t.h_walls |= 1u64 << b.hbit(wc, wr);
    } else {
        t.v_walls |= 1u64 << b.vbit(wc, wr);
    }
    t
}

// ---------------------------------------------------------------------------
// DSU-on-posts fast path (see module docs for the planar-duality soundness
// argument). Posts are the (w+1) x (h+1) lattice corners; post (pc, pr) has
// pc ∈ 0..=w, pr ∈ 0..=h and id `pr*(w+1)+pc`. Derived from `step_blocked`:
//   * H-wall anchor (wc, wr) blocks the north steps from cells (wc, wr) and
//     (wc+1, wr), i.e. it lies on post-row wr+1 and covers posts
//     (wc, wr+1), (wc+1, wr+1), (wc+2, wr+1).
//   * V-wall anchor (wc, wr) blocks the east steps from cells (wc, wr) and
//     (wc, wr+1), i.e. it lies on post-column wc+1 and covers posts
//     (wc+1, wr), (wc+1, wr+1), (wc+1, wr+2).
// A border post has pc ∈ {0, w} or pr ∈ {0, h}. A candidate's CENTER post is
// always interior, and the overlap rules forbid a crossing wall at the center
// — but a PERPENDICULAR existing wall may END at the candidate's center post
// (e.g. H-candidate (c, r) vs existing V(c, r+1) or V(c, r-1), neither of
// which `overlaps` checks). A curve may therefore enter the candidate through
// its center, which is why the skip test must be the THREE-post rule
// (pairwise-distinct components), not extremes-only.
// ---------------------------------------------------------------------------

/// Max posts: boards are capped at 8x8 cells -> (8+1)*(8+1) = 81 posts.
const MAX_POSTS: usize = 81;

/// The three posts covered by a wall at anchor `(wc, wr)` with orientation
/// `horiz`, in span order: [extreme, center, extreme].
#[inline]
fn wall_posts(wc: u8, wr: u8, horiz: bool) -> [(u8, u8); 3] {
    if horiz {
        [(wc, wr + 1), (wc + 1, wr + 1), (wc + 2, wr + 1)]
    } else {
        [(wc + 1, wr), (wc + 1, wr + 1), (wc + 1, wr + 2)]
    }
}

/// DSU over lattice posts with all border posts pre-merged into one BORDER
/// component and every placed wall's three posts unioned. Rebuilt per
/// `legal_walls` call (~border perimeter + 2 unions per placed wall).
struct PostDsu {
    parent: [u8; MAX_POSTS],
    /// Posts per row: `w + 1`.
    pw: u8,
}

impl PostDsu {
    #[inline]
    fn pid(&self, pc: u8, pr: u8) -> u8 {
        pr * self.pw + pc
    }

    /// Find with path halving.
    fn find(&mut self, mut x: u8) -> u8 {
        while self.parent[x as usize] != x {
            let g = self.parent[self.parent[x as usize] as usize];
            self.parent[x as usize] = g;
            x = g;
        }
        x
    }

    fn union(&mut self, a: u8, b: u8) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra != rb {
            self.parent[ra as usize] = rb;
        }
    }

    /// Build from the placed walls of `s`.
    fn build(b: &Board, s: &State) -> PostDsu {
        let pw = b.w + 1; // posts per row
        let ph = b.h + 1; // post rows
        let n = pw as usize * ph as usize;
        debug_assert!(n <= MAX_POSTS);
        let mut parent = [0u8; MAX_POSTS];
        for (i, p) in parent.iter_mut().enumerate().take(n) {
            *p = i as u8;
        }
        let mut dsu = PostDsu { parent, pw };

        // Pre-merge the border: every perimeter post joins post (0, 0).
        let border = dsu.pid(0, 0);
        for pc in 0..pw {
            let top = dsu.pid(pc, 0);
            let bot = dsu.pid(pc, ph - 1);
            dsu.union(top, border);
            dsu.union(bot, border);
        }
        for pr in 0..ph {
            let left = dsu.pid(0, pr);
            let right = dsu.pid(pw - 1, pr);
            dsu.union(left, border);
            dsu.union(right, border);
        }

        // Union each placed wall's three posts.
        let aw = b.w - 1; // anchors per row
        let mut bits = s.h_walls;
        while bits != 0 {
            let i = bits.trailing_zeros() as u8;
            bits &= bits - 1;
            let (wc, wr) = (i % aw, i / aw);
            let [p0, p1, p2] = wall_posts(wc, wr, true);
            let (a, c, e) = (dsu.pid(p0.0, p0.1), dsu.pid(p1.0, p1.1), dsu.pid(p2.0, p2.1));
            dsu.union(a, c);
            dsu.union(c, e);
        }
        let mut bits = s.v_walls;
        while bits != 0 {
            let i = bits.trailing_zeros() as u8;
            bits &= bits - 1;
            let (wc, wr) = (i % aw, i / aw);
            let [p0, p1, p2] = wall_posts(wc, wr, false);
            let (a, c, e) = (dsu.pid(p0.0, p0.1), dsu.pid(p1.0, p1.1), dsu.pid(p2.0, p2.1));
            dsu.union(a, c);
            dsu.union(c, e);
        }
        dsu
    }

    /// THREE-POST skip test: the candidate closes no curve iff the components
    /// of its three posts are pairwise distinct (untouched posts are fresh
    /// singletons; all border posts are one component). By planar duality such
    /// a wall cannot disconnect any two cells, so its connectivity BFS may be
    /// skipped. One-sided: `false` only means "fall through to the BFS".
    fn closes_no_curve(&mut self, wc: u8, wr: u8, horiz: bool) -> bool {
        let [p0, p1, p2] = wall_posts(wc, wr, horiz);
        let r0 = self.find(self.pid(p0.0, p0.1));
        let r1 = self.find(self.pid(p1.0, p1.1));
        let r2 = self.find(self.pid(p2.0, p2.1));
        r0 != r1 && r0 != r2 && r1 != r2
    }
}

/// Whether the DSU fast path is enabled (`QS_DSU_WALLS`, default ON; `=0`
/// disables for A/B). Read once per process.
fn dsu_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("QS_DSU_WALLS")
            .map(|v| v.trim() != "0")
            .unwrap_or(true)
    })
}

/// Candidates accepted by the DSU fast path without a BFS.
static DSU_SKIPS: AtomicU64 = AtomicU64::new(0);
/// Candidates that fell through to the BFS pair (curve-closing or DSU off-path).
static DSU_BFS_FALLS: AtomicU64 = AtomicU64::new(0);

/// Process-wide DSU fast-path counters: `(dsu_skips, dsu_bfs_falls)` — how
/// many non-overlapping candidates were accepted without a BFS vs fell
/// through to the BFS pair. Monotonic; printed by the `solve` CLI.
pub fn dsu_wall_counters() -> (u64, u64) {
    (
        DSU_SKIPS.load(Ordering::Relaxed),
        DSU_BFS_FALLS.load(Ordering::Relaxed),
    )
}

/// Test/diagnostic probe: would the DSU fast path skip the BFS for this
/// candidate (i.e. its three posts lie in pairwise-distinct components of the
/// post-DSU built from `s`)? Builds a fresh DSU per call; `legal_walls`
/// builds one per call and reuses it across candidates.
pub fn wall_closes_no_curve(b: &Board, s: &State, wc: u8, wr: u8, horiz: bool) -> bool {
    let mut dsu = PostDsu::build(b, s);
    dsu.closes_no_curve(wc, wr, horiz)
}

/// Legal wall placements. Mirrors `Engine.legal_walls`.
///
/// EXACT: a candidate wall is legal iff it does not overlap an existing wall
/// AND both pawns still have a path to their goal after it is placed. The
/// connectivity BFS runs on every non-overlapping candidate EXCEPT those the
/// DSU-on-posts fast path proves harmless (pairwise-distinct post components
/// close no curve, hence cannot disconnect — see module docs). Curve-closing
/// candidates always get the BFS pair, which remains the authority.
/// `QS_DSU_WALLS=0` disables the fast path entirely.
pub fn legal_walls(b: &Board, s: &State) -> Vec<Move> {
    if dsu_enabled() {
        walls_dsu(b, s)
    } else {
        walls_inner(b, s)
    }
}

/// Explicit brute-force reference, kept for tests: always runs the two-player
/// path check on every non-overlapping candidate, no fast path of any kind.
pub fn legal_walls_bruteforce(b: &Board, s: &State) -> Vec<Move> {
    walls_inner(b, s)
}

/// `legal_walls` with the DSU-on-posts fast path. Same candidate iteration
/// order as `walls_inner`, so the returned ordering is identical; only the
/// per-candidate BFS is (soundly) skipped when the wall closes no curve.
fn walls_dsu(b: &Board, s: &State) -> Vec<Move> {
    if s.walls_left[s.turn as usize] == 0 {
        return Vec::new();
    }
    let mut dsu = PostDsu::build(b, s);
    let (mut skips, mut falls) = (0u64, 0u64);
    let mut out: Vec<Move> = Vec::new();
    for &horiz in &[true, false] {
        for wc in 0..b.w - 1 {
            for wr in 0..b.h - 1 {
                if overlaps(b, s, wc, wr, horiz) {
                    continue;
                }
                if dsu.closes_no_curve(wc, wr, horiz) {
                    // Provably cannot disconnect anything: skip the BFS.
                    skips += 1;
                    out.push(Move::Wall { wc, wr, horiz });
                    continue;
                }
                // Curve-closing: the BFS pair is the authority.
                falls += 1;
                let s2 = with_wall_bit(b, s, wc, wr, horiz);
                if b.has_path(&s2, 0) && b.has_path(&s2, 1) {
                    out.push(Move::Wall { wc, wr, horiz });
                }
            }
        }
    }
    DSU_SKIPS.fetch_add(skips, Ordering::Relaxed);
    DSU_BFS_FALLS.fetch_add(falls, Ordering::Relaxed);
    out
}

fn walls_inner(b: &Board, s: &State) -> Vec<Move> {
    if s.walls_left[s.turn as usize] == 0 {
        return Vec::new();
    }
    let mut out: Vec<Move> = Vec::new();
    for &horiz in &[true, false] {
        for wc in 0..b.w - 1 {
            for wr in 0..b.h - 1 {
                if overlaps(b, s, wc, wr, horiz) {
                    continue;
                }
                // Always verify connectivity: a wall is legal only if BOTH
                // pawns retain a path to goal. No fast-path bypass.
                let s2 = with_wall_bit(b, s, wc, wr, horiz);
                if b.has_path(&s2, 0) && b.has_path(&s2, 1) {
                    out.push(Move::Wall { wc, wr, horiz });
                }
            }
        }
    }
    out
}

/// All legal moves: steps then walls (matches `Engine.legal_moves` ordering of
/// the two groups; within-group order is irrelevant for the set-equality
/// checks the tests perform).
pub fn legal_moves(b: &Board, s: &State) -> Vec<Move> {
    let mut out: Vec<Move> = legal_steps(b, s).into_iter().map(Move::Step).collect();
    out.extend(legal_walls(b, s));
    out
}

#[cfg(test)]
mod tests {
    use crate::board::Board;
    #[test]
    fn opening_steps_3x3() {
        // p0 at (1,0): can go up to (1,1), left to (0,0), right to (2,0); down is off-board.
        let b = Board::new(3, 3, 1);
        let s = b.initial();
        let mut steps: Vec<(u8, u8)> = crate::movegen::legal_steps(&b, &s)
            .iter()
            .map(|&i| b.cr(i))
            .collect();
        steps.sort();
        assert_eq!(steps, vec![(0, 0), (1, 1), (2, 0)]);
    }
}
