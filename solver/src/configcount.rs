//! Exact counter of legal wall configurations on a [`Board`].
//!
//! A *wall configuration* is any geometrically-legal arrangement of walls —
//! any number of walls (0 up to the board's geometric maximum), ignoring
//! walls-per-player budgets and ignoring path/reachability constraints. This
//! matches the reference writeup's `CONFIG_TOTALS`.
//!
//! Each wall anchor `(wc, wr)` (with `wc ∈ 0..w-1`, `wr ∈ 0..h-1`) is labelled
//! with one of three states — **none**, **horizontal (H)**, or **vertical (V)**
//! — subject to the non-overlap rules mirrored from `Engine._overlaps` (see
//! [`crate::movegen`]):
//!
//! - H and V at the same anchor conflict (a cross) → each anchor holds at most
//!   one of {none, H, V} (automatically satisfied by the one-state model).
//! - Two **H** walls at horizontally-adjacent anchors conflict.
//! - Two **V** walls at vertically-adjacent anchors conflict.
//! - No other inter-anchor conflicts exist.
//!
//! So we count the labellings of every anchor with {none, H, V} such that no
//! two horizontally-adjacent anchors are both H, and no two vertically-adjacent
//! anchors are both V. The all-none labelling is the single 0-wall config.

use crate::board::Board;

/// Exact count of legal wall configurations on `b`, via recursive backtracking
/// enumeration over anchors `0..A` where `A = (w-1)*(h-1)`.
///
/// Anchors use the linear order `a = wr*(w-1) + wc`. With that order an anchor
/// `a = (wc, wr)`'s H-left-neighbour `(wc-1, wr)` has index `a-1` and its
/// V-below-neighbour `(wc, wr-1)` has index `a-(w-1)` — both already decided
/// when we reach `a`, so each branch only checks against already-placed walls.
///
/// Returns `1` for the degenerate `A == 0` case (a 1-wide or 1-tall board has
/// no anchors, hence only the empty configuration).
pub fn count_wall_configs(b: &Board) -> u64 {
    let w = b.w as i32;
    let h = b.h as i32;
    // Anchor grid is (w-1) x (h-1); degenerate boards have no anchors.
    if w <= 1 || h <= 1 {
        return 1;
    }
    let stride = (w - 1) as u32; // anchors per row
    let total = stride * ((h - 1) as u32); // A = (w-1)*(h-1)
    debug_assert!(total <= 64, "anchor count must fit a u64 bitset");
    count_from(0, total, stride, 0u64, 0u64)
}

/// Recurse from anchor `a`, with `h_set`/`v_set` the bitsets of anchors already
/// labelled H / V respectively. `total` is the anchor count `A`, `stride` is the
/// number of anchors per row (`w-1`). Returns the number of valid completions.
fn count_from(a: u32, total: u32, stride: u32, h_set: u64, v_set: u64) -> u64 {
    if a == total {
        return 1;
    }
    let wc = a % stride; // column within the anchor grid

    // Branch "none" is always legal.
    let mut count = count_from(a + 1, total, stride, h_set, v_set);

    // Branch "H": illegal only if the horizontal-left neighbour `a-1` is H.
    // The left neighbour exists iff this anchor is not in the first column.
    let h_left_is_h = wc > 0 && (h_set & (1u64 << (a - 1))) != 0;
    if !h_left_is_h {
        count += count_from(a + 1, total, stride, h_set | (1u64 << a), v_set);
    }

    // Branch "V": illegal only if the vertical-below neighbour `a-stride` is V.
    // The below neighbour exists iff this anchor is not in the first row.
    let v_below_is_v = a >= stride && (v_set & (1u64 << (a - stride))) != 0;
    if !v_below_is_v {
        count += count_from(a + 1, total, stride, h_set, v_set | (1u64 << a));
    }

    count
}

#[cfg(test)]
mod tests {
    use crate::board::Board;
    use crate::configcount::count_wall_configs;
    #[test]
    fn matches_writeup_small() {
        assert_eq!(count_wall_configs(&Board::new(2, 5, 9)), 60);
        assert_eq!(count_wall_configs(&Board::new(3, 5, 9)), 1_880);
        assert_eq!(count_wall_configs(&Board::new(4, 5, 9)), 70_944);
        assert_eq!(count_wall_configs(&Board::new(5, 5, 9)), 2_532_560);
    }
    #[test]
    fn empty_when_no_anchors() {
        assert_eq!(count_wall_configs(&Board::new(1, 5, 9)), 1); // w-1 == 0 -> no anchors
        assert_eq!(count_wall_configs(&Board::new(5, 1, 9)), 1); // h-1 == 0
    }
}
