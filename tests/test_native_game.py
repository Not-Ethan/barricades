import random
import barricades_native as bn
from core.state import GameState, Step, Wall, initial_state
from core import rules


def to_native(s):
    return (tuple(s.pawns), sorted(s.h_walls), sorted(s.v_walls),
            tuple(s.walls_left), s.turn)


def mv_to_tuple(m):
    if isinstance(m, Step):
        return ("step", m.to_cell[0], m.to_cell[1])
    return ("wall", m.c, m.r, m.orient)


def test_open_board_basics():
    s = to_native(initial_state())
    assert bn.shortest_path_len(s, 0) == 8
    assert bn.shortest_path_len(s, 1) == 8
    assert bn.winner(s) is None
    assert bn.is_terminal(s) is False
    py = {mv_to_tuple(m) for m in rules.legal_moves(initial_state())}
    assert set(bn.legal_moves(s)) == py


def test_differential_over_random_games():
    rng = random.Random(12345)
    checked = 0
    for _ in range(80):
        s = initial_state()
        for _ in range(80):
            if rules.is_terminal(s):
                break
            ns = to_native(s)
            assert set(bn.legal_moves(ns)) == {mv_to_tuple(m) for m in rules.legal_moves(s)}
            for p in (0, 1):
                assert bn.shortest_path_len(ns, p) == rules.shortest_path_len(s, p)
            assert bn.winner(ns) == rules.winner(s)
            assert bn.is_terminal(ns) == rules.is_terminal(s)
            me = s.pawns[s.turn]
            for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                b = (me[0] + dx, me[1] + dy)
                if 0 <= b[0] < 9 and 0 <= b[1] < 9:
                    assert bn.is_blocked(ns, me, b) == rules.is_blocked(s, me, b)
            mv = rng.choice(rules.legal_moves(s))
            assert bn.apply_move(ns, mv_to_tuple(mv)) == to_native(rules.apply_move(s, mv))
            checked += 1
            s = rules.apply_move(s, mv)
    assert checked > 2000


def test_legal_walls_match_in_wall_dense_positions():
    # The fast-path for wall legality must match the Python full-BFS reference
    # exactly, ESPECIALLY where walls are dense and blocking is actually possible.
    # Bias the random playout toward wall moves so positions accumulate walls.
    import random
    from core.state import Step, Wall
    rng = random.Random(99)
    checked = 0
    dense = 0
    for _ in range(120):
        s = initial_state()
        for _ in range(60):
            if rules.is_terminal(s):
                break
            ns = to_native(s)
            assert set(bn.legal_moves(ns)) == {mv_to_tuple(m) for m in rules.legal_moves(s)}
            placed = len(s.h_walls) + len(s.v_walls)
            if placed >= 6:
                dense += 1
            checked += 1
            moves = rules.legal_moves(s)
            walls = [m for m in moves if isinstance(m, Wall)]
            pick = rng.choice(walls) if (walls and rng.random() < 0.75) else rng.choice(moves)
            s = rules.apply_move(s, pick)
    assert checked > 2000
    assert dense > 300
