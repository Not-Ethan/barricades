use crate::bitboard::path_exists;
use crate::coords::{on_board, DIRS4};
use crate::state::{GameState, Move};

pub fn is_blocked(s: &GameState, a: (i32, i32), b: (i32, i32)) -> bool {
    let (ax, ay) = a;
    let (bx, by) = b;
    let (dx, dy) = (bx - ax, by - ay);
    debug_assert!(dx.abs() + dy.abs() == 1, "is_blocked: cells must be orthogonally adjacent");
    if dy == 1 { return s.has_h(ax, ay) || s.has_h(ax - 1, ay); }
    if dy == -1 { return s.has_h(ax, by) || s.has_h(ax - 1, by); }
    if dx == 1 { return s.has_v(ax, ay) || s.has_v(ax, ay - 1); }
    s.has_v(bx, ay) || s.has_v(bx, ay - 1)
}

pub fn legal_steps(s: &GameState) -> Vec<(i32, i32)> {
    let me = s.pawns[s.turn as usize];
    let me = (me.0 as i32, me.1 as i32);
    let opp = s.pawns[1 - s.turn as usize];
    let opp = (opp.0 as i32, opp.1 as i32);
    let mut dests = Vec::new();
    for (dx, dy) in DIRS4 {
        let adj = (me.0 + dx, me.1 + dy);
        if !on_board(adj.0, adj.1) || is_blocked(s, me, adj) { continue; }
        if adj != opp { dests.push(adj); continue; }
        let landing = (opp.0 + dx, opp.1 + dy);
        if on_board(landing.0, landing.1) && !is_blocked(s, opp, landing) {
            dests.push(landing);
        } else {
            for (px, py) in DIRS4 {
                if (px, py) == (dx, dy) || (px, py) == (-dx, -dy) { continue; }
                let diag = (opp.0 + px, opp.1 + py);
                if on_board(diag.0, diag.1) && !is_blocked(s, opp, diag) { dests.push(diag); }
            }
        }
    }
    dests
}

fn overlaps(s: &GameState, c: i32, r: i32, orient: u8) -> bool {
    if orient == 0 {
        s.has_h(c, r) || s.has_h(c - 1, r) || s.has_h(c + 1, r) || s.has_v(c, r)
    } else {
        s.has_v(c, r) || s.has_v(c, r - 1) || s.has_v(c, r + 1) || s.has_h(c, r)
    }
}

fn with_wall(s: &GameState, c: i32, r: i32, orient: u8) -> GameState {
    let mut g = *s;
    let bp = (r * 8 + c) as u64;
    if orient == 0 { g.h_mask |= 1u64 << bp; } else { g.v_mask |= 1u64 << bp; }
    g
}

#[inline]
fn post_idx(px: i32, py: i32) -> u32 {
    (px * 10 + py) as u32
}

/// Bitset (u128, 100 posts) of all contact posts occupied by existing walls.
fn occupied_posts(s: &GameState) -> u128 {
    let mut bits = 0u128;
    let mut hm = s.h_mask;
    while hm != 0 {
        let i = hm.trailing_zeros() as i32;
        hm &= hm - 1;
        let (c, r) = (i % 8, i / 8);
        for px in [c, c + 1, c + 2] {
            bits |= 1u128 << post_idx(px, r + 1);
        }
    }
    let mut vm = s.v_mask;
    while vm != 0 {
        let i = vm.trailing_zeros() as i32;
        vm &= vm - 1;
        let (c, r) = (i % 8, i / 8);
        for py in [r, r + 1, r + 2] {
            bits |= 1u128 << post_idx(c + 1, py);
        }
    }
    bits
}

#[inline]
fn is_boundary_post(px: i32, py: i32) -> bool {
    px == 0 || px == 9 || py == 0 || py == 9
}

/// The three contact posts of a candidate wall (orient 0=H, 1=V).
fn wall_posts(c: i32, r: i32, orient: u8) -> [(i32, i32); 3] {
    if orient == 0 {
        [(c, r + 1), (c + 1, r + 1), (c + 2, r + 1)]
    } else {
        [(c + 1, r), (c + 1, r + 1), (c + 1, r + 2)]
    }
}

/// True if this candidate could possibly complete a cut (needs the path BFS).
/// Conservative: returns false (skip BFS) only when <2 of the 3 posts are anchored
/// (boundary or coincident with an existing wall's post) — provably always-legal
/// for a length-2 wall.
fn needs_path_check(occupied: u128, c: i32, r: i32, orient: u8) -> bool {
    let mut anchored = 0;
    for (px, py) in wall_posts(c, r, orient) {
        if is_boundary_post(px, py) || (occupied >> post_idx(px, py)) & 1 != 0 {
            anchored += 1;
        }
    }
    anchored >= 2
}

pub fn legal_walls(s: &GameState) -> Vec<(i32, i32, u8)> {
    if s.walls_left[s.turn as usize] == 0 {
        return Vec::new();
    }
    let occupied = occupied_posts(s);
    let mut res = Vec::new();
    for orient in [0u8, 1u8] {
        for c in 0..8 {
            for r in 0..8 {
                if overlaps(s, c, r, orient) {
                    continue;
                }
                if needs_path_check(occupied, c, r, orient) {
                    let s2 = with_wall(s, c, r, orient);
                    if path_exists(&s2, 0) && path_exists(&s2, 1) {
                        res.push((c, r, orient));
                    }
                } else {
                    res.push((c, r, orient)); // <2 anchored posts -> always legal
                }
            }
        }
    }
    res
}

pub fn legal_moves(s: &GameState) -> Vec<Move> {
    let mut out: Vec<Move> = legal_steps(s).into_iter().map(|(c, r)| Move::Step { c, r }).collect();
    for (c, r, orient) in legal_walls(s) { out.push(Move::Wall { c, r, orient }); }
    out
}
