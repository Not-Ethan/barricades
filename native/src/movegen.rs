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

pub fn legal_walls(s: &GameState) -> Vec<(i32, i32, u8)> {
    if s.walls_left[s.turn as usize] == 0 { return Vec::new(); }
    let mut res = Vec::new();
    for orient in [0u8, 1u8] {
        for c in 0..8 {
            for r in 0..8 {
                if overlaps(s, c, r, orient) { continue; }
                let s2 = with_wall(s, c, r, orient);
                if path_exists(&s2, 0) && path_exists(&s2, 1) { res.push((c, r, orient)); }
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
