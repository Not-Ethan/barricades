use crate::state::State;

/// Static board geometry: dimensions, walls-per-player, and precomputed masks.
///
/// Coordinates: cell `(c, r)` with `c ∈ 0..w`, `r ∈ 0..h`; bit index
/// `idx(c,r) = r*w + c`. Wall anchors `(wc, wr)` with `wc ∈ 0..w-1`,
/// `wr ∈ 0..h-1`; anchor bit index `wr*(w-1)+wc`. W,H ≤ 8 (idx < 64).
#[derive(Clone, Copy, Debug)]
pub struct Board {
    pub w: u8,
    pub h: u8,
    pub walls: u8,
    /// Bitset of every valid cell (`w*h` low bits set).
    pub full: u64,
    /// Goal-row cell masks, indexed by player. BFS targets next task.
    pub goal_mask: [u64; 2],
}

impl Board {
    pub fn new(w: u8, h: u8, walls: u8) -> Board {
        // Cell mask: all w*h low bits.
        let n = (w as u32) * (h as u32);
        let full: u64 = if n >= 64 {
            u64::MAX
        } else {
            (1u64 << n) - 1
        };

        // Goal-row masks: a full row of cells at the player's goal row.
        let row_mask = |r: u8| -> u64 {
            let mut m = 0u64;
            for c in 0..w {
                m |= 1u64 << (r * w + c);
            }
            m
        };
        let goal_mask = [row_mask(h - 1), row_mask(0)];

        Board {
            w,
            h,
            walls,
            full,
            goal_mask,
        }
    }

    /// Cell bit index from coordinates: `r*w + c`.
    #[inline]
    pub fn idx(&self, c: u8, r: u8) -> u8 {
        r * self.w + c
    }

    /// Inverse of `idx`: `(c, r) = (idx % w, idx / w)`.
    #[inline]
    pub fn cr(&self, idx: u8) -> (u8, u8) {
        (idx % self.w, idx / self.w)
    }

    /// Goal row for a player: `h-1` for player 0, `0` for player 1.
    #[inline]
    pub fn goal_row(&self, player: u8) -> u8 {
        if player == 0 { self.h - 1 } else { 0 }
    }

    /// Starting position: pawns at `(w/2, 0)` and `(w/2, h-1)`, no walls,
    /// full wall counts, player 0 to move.
    pub fn initial(&self) -> State {
        State {
            pawn: [self.idx(self.w / 2, 0), self.idx(self.w / 2, self.h - 1)],
            h_walls: 0,
            v_walls: 0,
            walls_left: [self.walls, self.walls],
            turn: 0,
        }
    }

    /// The winning player, if any: player `p` wins iff its pawn's row equals
    /// `goal_row(p)`.
    pub fn winner(&self, s: &State) -> Option<u8> {
        for p in 0u8..2 {
            let (_, r) = self.cr(s.pawn[p as usize]);
            if r == self.goal_row(p) {
                return Some(p);
            }
        }
        None
    }

    #[inline]
    pub fn is_terminal(&self, s: &State) -> bool {
        self.winner(s).is_some()
    }

    /// Horizontal wall anchor bit index: `wr*(w-1)+wc`.
    #[inline]
    pub fn hbit(&self, wc: u8, wr: u8) -> u8 {
        wr * (self.w - 1) + wc
    }

    /// Vertical wall anchor bit index: `wr*(w-1)+wc`.
    #[inline]
    pub fn vbit(&self, wc: u8, wr: u8) -> u8 {
        wr * (self.w - 1) + wc
    }

    /// Whether a horizontal wall anchor is set in `s`.
    #[inline]
    pub fn has_h(&self, s: &State, wc: u8, wr: u8) -> bool {
        s.h_walls & (1u64 << self.hbit(wc, wr)) != 0
    }

    /// Whether a vertical wall anchor is set in `s`.
    #[inline]
    pub fn has_v(&self, s: &State, wc: u8, wr: u8) -> bool {
        s.v_walls & (1u64 << self.vbit(wc, wr)) != 0
    }
}
