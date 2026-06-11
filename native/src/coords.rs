pub const N: i32 = 9;
pub const DIRS4: [(i32, i32); 4] = [(0, 1), (0, -1), (1, 0), (-1, 0)];

#[inline]
pub fn on_board(c: i32, r: i32) -> bool {
    c >= 0 && c < N && r >= 0 && r < N
}

#[inline]
pub fn goal_row(player: usize) -> i32 {
    if player == 0 { N - 1 } else { 0 }
}
