import random
from smallboard.engine import Engine, Step, Wall, State


def test_initial_and_basic_3x3():
    e = Engine(3, 1)
    s = e.initial_state()
    assert s.pawns == ((1, 0), (1, 2))
    assert s.walls_left == (1, 1)
    assert e.shortest_path_len(s, 0) == 2 and e.shortest_path_len(s, 1) == 2
    assert not e.is_terminal(s)
    steps = {m.to_cell for m in e.legal_moves(s) if isinstance(m, Step)}
    assert (1, 1) in steps and (0, 0) in steps and (2, 0) in steps


def test_winner_and_apply_3x3():
    e = Engine(3, 1)
    s = State(((1, 1), (0, 0)), frozenset(), frozenset(), (1, 1), 0)
    s2 = e.apply_move(s, Step((1, 2)))
    assert e.winner(s2) == 0 and e.is_terminal(s2)


def test_differential_vs_core_at_N9():
    from core.state import GameState, Step as CStep, Wall as CWall, initial_state
    from core import rules

    def to_core(s):
        return GameState(s.pawns, frozenset(s.h_walls), frozenset(s.v_walls),
                         s.walls_left, s.turn)

    def sb_mv_key(m):
        return ("step", m.to_cell[0], m.to_cell[1]) if isinstance(m, Step) \
            else ("wall", m.c, m.r, m.orient)

    def core_mv_key(m):
        return ("step", m.to_cell[0], m.to_cell[1]) if isinstance(m, CStep) \
            else ("wall", m.c, m.r, m.orient)

    e = Engine(9, 10)
    rng = random.Random(123)
    checked = 0
    for _ in range(40):
        s = e.initial_state()
        cs = initial_state()
        for _ in range(60):
            if e.is_terminal(s):
                break
            assert {sb_mv_key(m) for m in e.legal_moves(s)} == \
                   {core_mv_key(m) for m in rules.legal_moves(cs)}
            for p in (0, 1):
                assert e.shortest_path_len(s, p) == rules.shortest_path_len(cs, p)
            assert e.winner(s) == rules.winner(cs)
            sb_moves = e.legal_moves(s)
            i = rng.randrange(len(sb_moves))
            m = sb_moves[i]
            s = e.apply_move(s, m)
            cm = CStep(m.to_cell) if isinstance(m, Step) else CWall(m.c, m.r, m.orient)
            cs = rules.apply_move(cs, cm)
            checked += 1
    assert checked > 1500
