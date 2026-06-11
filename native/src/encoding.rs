use crate::state::{GameState, Move};

pub const N_ACTIONS: usize = 140;

const DIRS12: [(i32, i32); 12] = [
    (0, 1), (0, -1), (1, 0), (-1, 0),
    (0, 2), (0, -2), (2, 0), (-2, 0),
    (1, 1), (-1, 1), (1, -1), (-1, -1),
];

fn dir_index(d: (i32, i32)) -> usize {
    DIRS12.iter().position(|&x| x == d).expect("non-canonical step delta")
}

#[inline]
fn cf_cell(c: i32, r: i32, flip: bool) -> (i32, i32) { (c, if flip { 8 - r } else { r }) }

#[inline]
fn cf_wall(c: i32, r: i32, flip: bool) -> (i32, i32) { (c, if flip { 7 - r } else { r }) }

/// Write the 6x9x9 planes (row-major: plane*81 + row*9 + col) into `out` (len 486),
/// which the caller must pre-zero.
pub fn encode_planes(s: &GameState, out: &mut [f32]) {
    let flip = s.turn == 1;
    let me = s.pawns[s.turn as usize];
    let opp = s.pawns[1 - s.turn as usize];
    let mc = cf_cell(me.0 as i32, me.1 as i32, flip);
    let oc = cf_cell(opp.0 as i32, opp.1 as i32, flip);
    out[(mc.1 * 9 + mc.0) as usize] = 1.0;
    out[81 + (oc.1 * 9 + oc.0) as usize] = 1.0;
    let mut hm = s.h_mask;
    while hm != 0 {
        let i = hm.trailing_zeros() as i32; hm &= hm - 1;
        let (cc, cr) = cf_wall(i % 8, i / 8, flip);
        out[2 * 81 + (cr * 9 + cc) as usize] = 1.0;
    }
    let mut vm = s.v_mask;
    while vm != 0 {
        let i = vm.trailing_zeros() as i32; vm &= vm - 1;
        let (cc, cr) = cf_wall(i % 8, i / 8, flip);
        out[3 * 81 + (cr * 9 + cc) as usize] = 1.0;
    }
    let w_me = s.walls_left[s.turn as usize] as f32 / 10.0;
    let w_op = s.walls_left[1 - s.turn as usize] as f32 / 10.0;
    for k in 0..81 { out[4 * 81 + k] = w_me; out[5 * 81 + k] = w_op; }
}

pub fn move_to_action(m: &Move, s: &GameState) -> usize {
    let flip = s.turn == 1;
    match *m {
        Move::Step { c, r } => {
            let me = s.pawns[s.turn as usize];
            let mc = cf_cell(me.0 as i32, me.1 as i32, flip);
            let dest = cf_cell(c, r, flip);
            dir_index((dest.0 - mc.0, dest.1 - mc.1))
        }
        Move::Wall { c, r, orient } => {
            let (cc, cr) = cf_wall(c, r, flip);
            let off = if orient == 0 { 0 } else { 64 };
            (12 + off + cr * 8 + cc) as usize
        }
    }
}

pub fn action_to_move(idx: usize, s: &GameState) -> Move {
    debug_assert!(idx < N_ACTIONS, "action_to_move: idx out of range");
    let flip = s.turn == 1;
    if idx < 12 {
        let (dx, dy) = DIRS12[idx];
        let me = s.pawns[s.turn as usize];
        let mc = cf_cell(me.0 as i32, me.1 as i32, flip);
        let real = cf_cell(mc.0 + dx, mc.1 + dy, flip);
        Move::Step { c: real.0, r: real.1 }
    } else {
        let a = idx - 12;
        let orient = if a < 64 { 0u8 } else { 1u8 };
        let a = a % 64;
        let (cr, cc) = ((a / 8) as i32, (a % 8) as i32);
        let real = cf_wall(cc, cr, flip);
        Move::Wall { c: real.0, r: real.1, orient }
    }
}
