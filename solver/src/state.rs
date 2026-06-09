/// A Quoridor game position.
///
/// `Copy` + `Hash` so it can key a transposition table in later tasks.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct State {
    /// Pawn cell indices, one per player (`idx = r*w + c`).
    pub pawn: [u8; 2],
    /// Horizontal wall anchor bitset (bit = `wr*(w-1)+wc`).
    pub h_walls: u64,
    /// Vertical wall anchor bitset (bit = `wr*(w-1)+wc`).
    pub v_walls: u64,
    /// Walls remaining for each player.
    pub walls_left: [u8; 2],
    /// Player to move (0 or 1).
    pub turn: u8,
}

#[cfg(test)]
mod tests {
    use crate::board::Board;

    #[test]
    fn initial_state_5x5() {
        let b = Board::new(5, 5, 3);
        let s = b.initial();
        assert_eq!(s.pawn[0], b.idx(2, 0));
        assert_eq!(s.pawn[1], b.idx(2, 4));
        assert_eq!(s.walls_left, [3, 3]);
        assert_eq!(s.turn, 0);
        assert!(!b.is_terminal(&s));
        assert!(b.winner(&s).is_none());
    }

    #[test]
    fn winner_on_goal_row() {
        let b = Board::new(3, 3, 1);
        let mut s = b.initial();
        s.pawn[0] = b.idx(1, 2); // player 0 on its goal row (h-1=2)
        assert_eq!(b.winner(&s), Some(0));
        assert!(b.is_terminal(&s));
    }
}
