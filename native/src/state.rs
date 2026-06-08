use crate::coords::goal_row;

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct GameState {
    pub pawns: [(u8, u8); 2],
    pub h_mask: u64,
    pub v_mask: u64,
    pub walls_left: [u8; 2],
    pub turn: u8,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Move {
    Step { c: i32, r: i32 },
    Wall { c: i32, r: i32, orient: u8 },
}

impl GameState {
    #[inline]
    pub fn has_h(&self, c: i32, r: i32) -> bool {
        c >= 0 && c < 8 && r >= 0 && r < 8 && (self.h_mask >> (r * 8 + c)) & 1 != 0
    }
    #[inline]
    pub fn has_v(&self, c: i32, r: i32) -> bool {
        c >= 0 && c < 8 && r >= 0 && r < 8 && (self.v_mask >> (r * 8 + c)) & 1 != 0
    }
}

pub fn initial_state() -> GameState {
    GameState { pawns: [(4, 0), (4, 8)], h_mask: 0, v_mask: 0, walls_left: [10, 10], turn: 0 }
}

pub fn apply_move(s: &GameState, m: &Move) -> GameState {
    let mut g = *s;
    match *m {
        Move::Step { c, r } => { g.pawns[s.turn as usize] = (c as u8, r as u8); }
        Move::Wall { c, r, orient } => {
            g.walls_left[s.turn as usize] -= 1;
            let bp = (r * 8 + c) as u64;
            if orient == 0 { g.h_mask |= 1u64 << bp; } else { g.v_mask |= 1u64 << bp; }
        }
    }
    g.turn = 1 - s.turn;
    g
}

pub fn winner(s: &GameState) -> Option<usize> {
    for p in 0..2 {
        if s.pawns[p].1 as i32 == goal_row(p) { return Some(p); }
    }
    None
}

pub fn is_terminal(s: &GameState) -> bool {
    winner(s).is_some()
}
