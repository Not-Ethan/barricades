from core.state import GameState, initial_state
from core.bitboard import bfs_dist, path_exists


def _state(p0, p1, h=(), v=(), turn=0):
    return GameState((p0, p1), frozenset(h), frozenset(v), (10, 10), turn)


def test_open_board_distances():
    s = initial_state()
    assert bfs_dist(s, 0) == 8
    assert bfs_dist(s, 1) == 8


def test_already_on_goal_zero():
    assert bfs_dist(_state((4, 8), (0, 0)), 0) == 0
    assert bfs_dist(_state((0, 0), (4, 0)), 1) == 0


def test_wall_lengthens_path():
    s = _state((4, 0), (0, 8), h=[(3, 0), (5, 0)])
    assert path_exists(s, 0)
    assert bfs_dist(s, 0) > 8


def test_fully_walled_off_no_path():
    h = [(c, 0) for c in range(0, 8, 2)] + [(7, 0)]
    s = _state((4, 0), (0, 8), h=h)
    assert not path_exists(s, 0)
    assert bfs_dist(s, 0) is None


def test_vertical_walls_do_not_block_vertical_moves():
    # a column of V-walls along col 0 must not change p0's straight-up distance
    s = _state((0, 0), (8, 8), v=[(0, 0), (0, 2), (0, 4), (0, 6)])
    assert bfs_dist(s, 0) == 8


def test_equivalence_with_reference_over_random_states():
    import random
    from core.state import initial_state
    from core.rules import (
        legal_moves, apply_move, is_terminal, shortest_path_len,
        has_path_to_goal, _shortest_path_len_ref,
    )
    rng = random.Random(12345)
    checked = 0
    for game in range(60):
        s = initial_state()
        for _ in range(60):
            if is_terminal(s):
                break
            for p in (0, 1):
                # public (bitboard) path must equal the pure-Python reference
                assert shortest_path_len(s, p) == _shortest_path_len_ref(s, p)
                assert has_path_to_goal(s, p) == (_shortest_path_len_ref(s, p) is not None)
                checked += 1
            s = apply_move(s, rng.choice(legal_moves(s)))
    assert checked > 1000      # sanity: we actually exercised many states
