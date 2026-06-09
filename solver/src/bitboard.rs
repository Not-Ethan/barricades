use crate::board::Board;
use crate::state::State;

impl Board {
    /// Whether a horizontal wall anchor `(wc, wr)` exists and is set in `s`.
    /// Anchors only exist for `wc ∈ 0..w-1`, `wr ∈ 0..h-1`; out-of-range
    /// coordinates (passed as `i16` to avoid `u8` underflow) are "no wall".
    #[inline]
    pub(crate) fn h_anchor(&self, s: &State, wc: i16, wr: i16) -> bool {
        if wc < 0 || wr < 0 || wc >= (self.w as i16 - 1) || wr >= (self.h as i16 - 1) {
            return false;
        }
        self.has_h(s, wc as u8, wr as u8)
    }

    /// Whether a vertical wall anchor `(wc, wr)` exists and is set in `s`.
    #[inline]
    pub(crate) fn v_anchor(&self, s: &State, wc: i16, wr: i16) -> bool {
        if wc < 0 || wr < 0 || wc >= (self.w as i16 - 1) || wr >= (self.h as i16 - 1) {
            return false;
        }
        self.has_v(s, wc as u8, wr as u8)
    }

    /// Whether the step from `(c, r)` to the orthogonally-adjacent cell in
    /// direction `(dc, dr)` is blocked by a wall. Mirrors
    /// `smallboard/engine.py::is_blocked` with explicit boundary guards.
    pub(crate) fn step_blocked(&self, s: &State, c: i16, r: i16, dc: i16, dr: i16) -> bool {
        if dr == 1 {
            // North: (c,r)->(c,r+1)
            self.h_anchor(s, c, r) || self.h_anchor(s, c - 1, r)
        } else if dr == -1 {
            // South: (c,r)->(c,r-1)
            self.h_anchor(s, c, r - 1) || self.h_anchor(s, c - 1, r - 1)
        } else if dc == 1 {
            // East: (c,r)->(c+1,r)
            self.v_anchor(s, c, r) || self.v_anchor(s, c, r - 1)
        } else {
            // West: (c,r)->(c-1,r)
            self.v_anchor(s, c - 1, r) || self.v_anchor(s, c - 1, r - 1)
        }
    }

    /// BFS shortest distance (in steps) from `s.pawn[player]` to any cell in
    /// the player's goal row, respecting wall blocking. `Some(0)` if already
    /// on the goal row; `None` if no path exists.
    pub fn dist_to_goal(&self, s: &State, player: u8) -> Option<u32> {
        let goal = self.goal_row(player);
        let start = s.pawn[player as usize];
        let (_, sr) = self.cr(start);
        if sr == goal {
            return Some(0);
        }

        let mut seen: u64 = 1u64 << start;
        // Cell-queue BFS; boards are tiny so a Vec frontier is plenty.
        let mut frontier: Vec<u8> = vec![start];
        let mut dist: u32 = 0;
        const DIRS: [(i16, i16); 4] = [(0, 1), (0, -1), (1, 0), (-1, 0)];

        while !frontier.is_empty() {
            dist += 1;
            let mut next: Vec<u8> = Vec::new();
            for &cell in &frontier {
                let (c, r) = self.cr(cell);
                let (c, r) = (c as i16, r as i16);
                for &(dc, dr) in &DIRS {
                    let nc = c + dc;
                    let nr = r + dr;
                    if nc < 0 || nr < 0 || nc >= self.w as i16 || nr >= self.h as i16 {
                        continue;
                    }
                    if self.step_blocked(s, c, r, dc, dr) {
                        continue;
                    }
                    let nidx = self.idx(nc as u8, nr as u8);
                    let bit = 1u64 << nidx;
                    if seen & bit != 0 {
                        continue;
                    }
                    if nr as u8 == goal {
                        return Some(dist);
                    }
                    seen |= bit;
                    next.push(nidx);
                }
            }
            frontier = next;
        }
        None
    }

    /// Whether `player` still has any path to their goal row.
    #[inline]
    pub fn has_path(&self, s: &State, player: u8) -> bool {
        self.dist_to_goal(s, player).is_some()
    }
}

#[cfg(test)]
mod tests {
    use crate::board::Board;

    #[test]
    fn open_board_distance_5x5() {
        let b = Board::new(5, 5, 3);
        let s = b.initial();
        // player 0 at row 0, goal row 4 -> distance 4 on an empty board
        assert_eq!(b.dist_to_goal(&s, 0), Some(4));
        assert_eq!(b.dist_to_goal(&s, 1), Some(4));
    }

    #[test]
    fn wall_lengthens_path() {
        let b = Board::new(3, 3, 1);
        let mut s = b.initial(); // p0 at (1,0), goal row 2, dist 2
        assert_eq!(b.dist_to_goal(&s, 0), Some(2));
        s.h_walls |= 1u64 << b.hbit(0, 0); // wall between rows 0,1 at cols 0,1
        assert!(b.has_path(&s, 0)); // path still exists (around the wall)
    }
}
