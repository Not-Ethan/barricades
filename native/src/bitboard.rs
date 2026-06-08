use crate::state::GameState;

const FULL: u128 = (1u128 << 81) - 1;

#[inline]
fn bit(c: i32, r: i32) -> u128 { 1u128 << (r * 9 + c) }

fn row_mask(r: i32) -> u128 { let mut m = 0u128; for c in 0..9 { m |= bit(c, r); } m }
fn col_mask(c: i32) -> u128 { let mut m = 0u128; for r in 0..9 { m |= bit(c, r); } m }

fn can_move_masks(s: &GameState) -> (u128, u128, u128, u128) {
    let (mut bn, mut bs, mut be, mut bw) = (0u128, 0u128, 0u128, 0u128);
    let mut hm = s.h_mask;
    while hm != 0 {
        let i = hm.trailing_zeros() as i32; hm &= hm - 1;
        let (a, b) = (i % 8, i / 8);
        bn |= bit(a, b) | bit(a + 1, b);
        bs |= bit(a, b + 1) | bit(a + 1, b + 1);
    }
    let mut vm = s.v_mask;
    while vm != 0 {
        let i = vm.trailing_zeros() as i32; vm &= vm - 1;
        let (a, b) = (i % 8, i / 8);
        be |= bit(a, b) | bit(a, b + 1);
        bw |= bit(a + 1, b) | bit(a + 1, b + 1);
    }
    let can_n = FULL & !row_mask(8) & !bn;
    let can_s = FULL & !row_mask(0) & !bs;
    let can_e = FULL & !col_mask(8) & !be;
    let can_w = FULL & !col_mask(0) & !bw;
    (can_n, can_s, can_e, can_w)
}

#[inline]
fn expand(frontier: u128, m: (u128, u128, u128, u128)) -> u128 {
    let (cn, cs, ce, cw) = m;
    let n = (frontier & cn) << 9;
    let s = (frontier & cs) >> 9;
    let e = (frontier & ce) << 1;
    let w = (frontier & cw) >> 1;
    (n | s | e | w) & FULL
}

pub fn bfs_dist(s: &GameState, player: usize) -> Option<u32> {
    let (c, r) = (s.pawns[player].0 as i32, s.pawns[player].1 as i32);
    let goal = if player == 0 { row_mask(8) } else { row_mask(0) };
    let start = bit(c, r);
    if start & goal != 0 { return Some(0); }
    let masks = can_move_masks(s);
    let mut visited = start;
    let mut frontier = start;
    let mut dist = 0u32;
    while frontier != 0 {
        let nxt = expand(frontier, masks) & !visited;
        if nxt == 0 { return None; }
        dist += 1;
        if nxt & goal != 0 { return Some(dist); }
        visited |= nxt;
        frontier = nxt;
    }
    None
}

pub fn path_exists(s: &GameState, player: usize) -> bool { bfs_dist(s, player).is_some() }
