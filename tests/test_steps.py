from core.state import GameState
from core.rules import legal_steps


def _state(p0, p1, h=(), v=(), turn=0):
    return GameState(
        pawns=(p0, p1),
        h_walls=frozenset(h),
        v_walls=frozenset(v),
        walls_left=(10, 10),
        turn=turn,
    )


def test_center_has_four_moves():
    s = _state((4, 4), (0, 0))
    assert set(legal_steps(s)) == {(4, 5), (4, 3), (5, 4), (3, 4)}


def test_corner_has_two_moves():
    s = _state((0, 0), (8, 8))
    assert set(legal_steps(s)) == {(0, 1), (1, 0)}


def test_wall_removes_a_move():
    s = _state((4, 4), (0, 0), h=[(4, 4)])
    assert (4, 5) not in legal_steps(s)
    assert (4, 3) in legal_steps(s)


def test_straight_jump_over_opponent():
    s = _state((4, 4), (4, 5))
    moves = set(legal_steps(s))
    assert (4, 6) in moves        # straight jump
    assert (4, 5) not in moves    # cannot land on opponent


def test_diagonal_jump_when_wall_behind_opponent():
    s = _state((4, 4), (4, 5), h=[(4, 5)])
    moves = set(legal_steps(s))
    assert (4, 6) not in moves            # straight jump blocked by wall
    assert (3, 5) in moves and (5, 5) in moves  # both diagonals available


def test_diagonal_jump_when_opponent_on_edge():
    s = _state((4, 7), (4, 8))
    moves = set(legal_steps(s))
    assert (3, 8) in moves and (5, 8) in moves
    assert (4, 9) not in moves
