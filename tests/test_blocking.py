from core.state import GameState
from core.rules import is_blocked


def _state(h=(), v=()):
    return GameState(
        pawns=((4, 0), (4, 8)),
        h_walls=frozenset(h),
        v_walls=frozenset(v),
        walls_left=(10, 10),
        turn=0,
    )


def test_no_walls_nothing_blocked():
    s = _state()
    assert not is_blocked(s, (4, 4), (4, 5))   # up
    assert not is_blocked(s, (4, 4), (4, 3))   # down
    assert not is_blocked(s, (4, 4), (5, 4))   # right
    assert not is_blocked(s, (4, 4), (3, 4))   # left


def test_horizontal_wall_blocks_vertical_moves():
    s = _state(h=[(4, 4)])
    assert is_blocked(s, (4, 4), (4, 5))
    assert is_blocked(s, (4, 5), (4, 4))       # symmetric (down)
    assert is_blocked(s, (5, 4), (5, 5))
    assert not is_blocked(s, (3, 4), (3, 5))   # outside the 2-segment span
    assert not is_blocked(s, (4, 4), (5, 4))   # horizontal move unaffected


def test_vertical_wall_blocks_horizontal_moves():
    s = _state(v=[(4, 4)])
    assert is_blocked(s, (4, 4), (5, 4))
    assert is_blocked(s, (5, 4), (4, 4))       # symmetric (left)
    assert is_blocked(s, (4, 5), (5, 5))
    assert not is_blocked(s, (4, 3), (5, 3))   # outside span
    assert not is_blocked(s, (4, 4), (4, 5))   # vertical move unaffected
