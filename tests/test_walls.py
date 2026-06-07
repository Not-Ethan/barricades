from core.state import GameState, Wall, initial_state
from core.rules import legal_walls


def _state(h=(), v=(), walls_left=(10, 10), turn=0):
    return GameState(
        pawns=((4, 0), (4, 8)),
        h_walls=frozenset(h),
        v_walls=frozenset(v),
        walls_left=walls_left,
        turn=turn,
    )


def test_open_board_wall_count():
    s = initial_state()
    assert len(legal_walls(s)) == 128


def test_no_walls_left_means_none():
    s = _state(walls_left=(0, 10), turn=0)
    assert legal_walls(s) == []


def test_cannot_overlap_parallel_horizontal():
    s = _state(h=[(3, 3)])
    walls = set((w.c, w.r, w.orient) for w in legal_walls(s))
    assert (3, 3, "H") not in walls
    assert (2, 3, "H") not in walls
    assert (4, 3, "H") not in walls
    assert (3, 4, "H") in walls
    assert (3, 3, "V") not in walls


def test_cannot_overlap_parallel_vertical():
    s = _state(v=[(3, 3)])
    walls = set((w.c, w.r, w.orient) for w in legal_walls(s))
    assert (3, 3, "V") not in walls
    assert (3, 2, "V") not in walls
    assert (3, 4, "V") not in walls
    assert (4, 3, "V") in walls
    assert (3, 3, "H") not in walls


def test_wall_that_seals_a_player_is_illegal():
    h = [(0, 0), (2, 0), (4, 0)]
    s = _state(h=h)
    from core.rules import has_path_to_goal
    from core.state import GameState as GS
    for w in legal_walls(s):
        if w.orient == "H":
            s2 = GS(s.pawns, s.h_walls | {(w.c, w.r)}, s.v_walls,
                    s.walls_left, s.turn)
        else:
            s2 = GS(s.pawns, s.h_walls, s.v_walls | {(w.c, w.r)},
                    s.walls_left, s.turn)
        assert has_path_to_goal(s2, 0) and has_path_to_goal(s2, 1)


def test_legal_walls_excludes_a_specific_sealing_wall():
    from core.state import Wall
    # Walls lining both sides of column 4 (rows 0..7) so col 4 is a corridor.
    v = [(3, 0), (3, 2), (3, 4), (3, 6), (4, 0), (4, 2), (4, 4), (4, 6)]
    s = _state(v=v)  # player 0 at (4,0); climbs straight up column 4 to row 8
    walls = legal_walls(s)
    keys = set((w.c, w.r, w.orient) for w in walls)
    # H(4,5) blocks (4,5)<->(4,6); with the side walls it seals player 0's only path.
    # It does not overlap any H wall and does not cross a V wall at (4,5).
    assert (4, 5, "H") not in keys      # excluded because it seals player 0
    # A far-away wall that seals nobody must still be offered.
    assert (7, 7, "H") in keys
