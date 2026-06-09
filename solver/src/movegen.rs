//! Move generation: pawn steps (with jump rules), wall placements, and `apply`.
//! Mirrors `smallboard/engine.py` exactly; the cross-language differential test
//! (`tests/diff_vs_smallboard.rs`) guards the equivalence.
//!
//! Wall legality is fully exact: every non-overlapping candidate wall is
//! accepted only if BOTH pawns still have a path to their goal after it is
//! placed (the two `has_path` BFS checks). There is deliberately NO
//! "floating-wall" fast-path: a prior conservative predicate that tried to skip
//! the BFS could (in a narrow keystone configuration) admit an illegal
//! board-spanning wall that strands a pawn, inverting solver values on
//! even-width boards. The BFS on a <=49-cell board is microseconds, so we always
//! run it; speed is recovered by higher tiers (persistent memo / per-config
//! tables), never by skipping the connectivity proof.

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

/// Legal wall placements. Mirrors `Engine.legal_walls`.
///
/// EXACT: a candidate wall is legal iff it does not overlap an existing wall AND
/// both pawns still have a path to their goal after it is placed. The
/// connectivity BFS runs on EVERY non-overlapping candidate — there is no
/// fast-path that skips it. (See the module docs: the old floating-wall
/// predicate could wrongly admit a board-spanning "keystone" wall.)
pub fn legal_walls(b: &Board, s: &State) -> Vec<Move> {
    walls_inner(b, s)
}

/// Explicit brute-force reference, kept for tests. Identical behaviour to
/// `legal_walls` (both always run the two-player path check on every
/// non-overlapping candidate).
pub fn legal_walls_bruteforce(b: &Board, s: &State) -> Vec<Move> {
    walls_inner(b, s)
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
