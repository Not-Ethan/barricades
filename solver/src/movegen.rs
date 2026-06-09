//! Move generation: pawn steps (with jump rules), wall placements (with a
//! floating-wall fast-path), and `apply`. Mirrors `smallboard/engine.py`
//! exactly; the cross-language differential test
//! (`tests/diff_vs_smallboard.rs`) and the internal soundness check
//! (`selfcheck_fast_path`) guard the equivalence.

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

/// Conservative predicate: does placing wall `(wc, wr, horiz)` require the
/// (expensive) two-player path check?
///
/// A wall can only disconnect a pawn from its goal if it (transitively) bridges
/// two opposite board borders — and a single wall cannot do that unless it
/// touches the border directly or extends/crosses an existing wall. We return
/// `true` whenever the wall touches the board boundary, OR any existing wall
/// anchor (either orientation) lies within Chebyshev distance 1 of `(wc, wr)`,
/// OR a perpendicular crossing is nearby. **When in doubt, return `true`** —
/// over-returning only costs speed, while a wrong `false` is a correctness bug
/// caught by `selfcheck_fast_path`.
pub fn needs_path_check(b: &Board, s: &State, wc: u8, wr: u8, horiz: bool) -> bool {
    let (c, r) = (wc as i16, wr as i16);

    // A horizontal wall spans cell columns [wc, wc+1] sitting between rows
    // wr and wr+1; it touches the left border when wc == 0 and the right
    // border when wc + 1 == w - 1 (i.e. wc == w - 2). A vertical wall spans
    // rows [wr, wr+1] between columns wc and wc+1; touches bottom when wr == 0
    // and top when wr == h - 2. For the conservative predicate we treat a wall
    // touching ANY border as needing the check (a wall flush against any edge
    // can be the seed of a goal-line-spanning barrier together with others, but
    // critically: any wall that can disconnect must touch a border or another
    // wall, so the union of "touches border" + "near a wall" is sufficient).
    if wc == 0 || wr == 0 || wc + 1 >= b.w - 1 || wr + 1 >= b.h - 1 {
        // Note: `wc + 1 >= w - 1` <=> `wc >= w - 2`; with anchors in 0..w-1
        // this captures the wall's far end reaching the opposite border.
        return true;
    }

    // Any existing wall anchor (either orientation) within Chebyshev distance 1
    // of (wc, wr) means this wall could touch / extend / cross it.
    for &dc in &[-1i16, 0, 1] {
        for &dr in &[-1i16, 0, 1] {
            let (nc, nr) = (c + dc, r + dr);
            if b.h_anchor(s, nc, nr) || b.v_anchor(s, nc, nr) {
                return true;
            }
        }
    }
    let _ = horiz;
    false
}

/// Legal wall placements. Mirrors `Engine.legal_walls`, but skips the path
/// check for "floating" walls that provably cannot disconnect anything.
pub fn legal_walls(b: &Board, s: &State) -> Vec<Move> {
    walls_inner(b, s, false)
}

/// `legal_walls` with `needs_path_check` forced to `true` everywhere (the
/// brute-force reference).
pub fn legal_walls_bruteforce(b: &Board, s: &State) -> Vec<Move> {
    walls_inner(b, s, true)
}

fn walls_inner(b: &Board, s: &State, force_check: bool) -> Vec<Move> {
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
                if force_check || needs_path_check(b, s, wc, wr, horiz) {
                    let s2 = with_wall_bit(b, s, wc, wr, horiz);
                    if b.has_path(&s2, 0) && b.has_path(&s2, 1) {
                        out.push(Move::Wall { wc, wr, horiz });
                    }
                } else {
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

/// Seeded random playout that, at every visited non-terminal node, asserts the
/// fast-path `legal_walls` equals `legal_walls_bruteforce` (as sets). Panics on
/// mismatch. Returns the number of nodes checked.
///
/// ~40 games of up to 30 plies, driven by a simple LCG.
pub fn selfcheck_fast_path(b: &Board, seed: u64) -> usize {
    let mut rng = Lcg::new(seed);
    let mut checked = 0usize;
    for _ in 0..40 {
        let mut s = b.initial();
        for _ in 0..30 {
            if b.is_terminal(&s) {
                break;
            }
            // Soundness: fast-path walls must equal brute-force walls.
            let fast = legal_walls(b, &s);
            let brute = legal_walls_bruteforce(b, &s);
            assert!(
                same_wall_set(&fast, &brute),
                "fast-path mismatch at {:?}: fast={:?} brute={:?}",
                s.pawn,
                fast,
                brute
            );
            checked += 1;

            // Advance via a uniformly random legal move.
            let moves = legal_moves(b, &s);
            if moves.is_empty() {
                break;
            }
            let pick = (rng.next() % moves.len() as u64) as usize;
            s = apply(b, &s, moves[pick]);
        }
    }
    checked
}

/// Whether two `Move::Wall` lists describe the same set.
fn same_wall_set(a: &[Move], c: &[Move]) -> bool {
    if a.len() != c.len() {
        return false;
    }
    a.iter().all(|m| c.contains(m))
}

/// Minimal LCG (Numerical Recipes constants) for reproducible playouts.
struct Lcg(u64);
impl Lcg {
    #[inline]
    fn new(seed: u64) -> Self {
        Lcg(seed ^ 0x9E37_79B9_7F4A_7C15)
    }
    #[inline]
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // Return the high bits, which have the best statistical quality.
        self.0 >> 16
    }
}

#[cfg(test)]
mod tests {
    use crate::board::Board;
    #[test]
    fn fast_path_matches_bruteforce_5x5() {
        let b = Board::new(5, 5, 3);
        assert!(crate::movegen::selfcheck_fast_path(&b, 123) > 1000);
    }
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
