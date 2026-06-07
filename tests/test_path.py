from core.state import GameState, initial_state
from core.rules import has_path_to_goal, shortest_path_len


def _state(p0, p1, h=(), v=()):
    return GameState(
        pawns=(p0, p1),
        h_walls=frozenset(h),
        v_walls=frozenset(v),
        walls_left=(10, 10),
        turn=0,
    )


def test_open_board_distances():
    s = initial_state()
    assert shortest_path_len(s, 0) == 8
    assert shortest_path_len(s, 1) == 8


def test_path_exists_on_open_board():
    s = initial_state()
    assert has_path_to_goal(s, 0)
    assert has_path_to_goal(s, 1)


def test_already_on_goal_is_zero():
    s = _state((4, 8), (4, 0))
    assert shortest_path_len(s, 0) == 0


def test_wall_lengthens_path():
    s = _state((4, 0), (0, 8), h=[(3, 0), (5, 0)])
    assert has_path_to_goal(s, 0)
    assert shortest_path_len(s, 0) > 8


def test_fully_walled_off_has_no_path():
    h = [(c, 0) for c in range(0, 8, 2)]  # (0,0),(2,0),(4,0),(6,0)
    h.append((7, 0))                      # covers cols 7,8
    s = _state((4, 0), (0, 8), h=h)
    assert not has_path_to_goal(s, 0)
    assert shortest_path_len(s, 0) is None
