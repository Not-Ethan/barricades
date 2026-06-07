from core.state import Step, Wall, GameState, initial_state, goal_row


def test_initial_state():
    s = initial_state()
    assert s.pawns == ((4, 0), (4, 8))
    assert s.walls_left == (10, 10)
    assert s.turn == 0
    assert s.h_walls == frozenset()
    assert s.v_walls == frozenset()


def test_goal_rows():
    assert goal_row(0) == 8
    assert goal_row(1) == 0


def test_state_is_immutable_and_hashable():
    s = initial_state()
    import dataclasses
    try:
        s.turn = 1
        assert False, "expected FrozenInstanceError"
    except dataclasses.FrozenInstanceError:
        pass
    assert hash(s) == hash(initial_state())


def test_move_types():
    st = Step((4, 1))
    assert st.to_cell == (4, 1)
    w = Wall(2, 3, "H")
    assert (w.c, w.r, w.orient) == (2, 3, "H")
