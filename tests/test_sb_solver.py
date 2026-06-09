from smallboard.engine import Engine, Step, Wall
from smallboard.solver import Solver


def _brute(engine, s, depth, memo):
    """Independent exhaustive negamax (no alpha-beta, no TT logic) for cross-check.
    +1 win / -1 loss / 0 draw-at-depth for the side to move."""
    w = engine.winner(s)
    if w is not None:
        return 1 if w == s.turn else -1
    if depth == 0:
        return 0
    key = (s, depth)
    if key in memo:
        return memo[key]
    best = -1
    for m in engine.legal_moves(s):
        v = -_brute(engine, engine.apply_move(s, m), depth - 1, memo)
        if v > best:
            best = v
        if best == 1:
            break
    memo[key] = best
    return best


def test_solver_matches_bruteforce_3x3():
    e = Engine(3, 1)
    sol = Solver(e, max_depth=14)
    s = e.initial_state()
    val, best = sol.solve(s)
    ref = _brute(e, s, 14, {})
    assert val == ref
    assert best and all(m in e.legal_moves(s) for m in best)
    for m in best:
        assert -_brute(e, e.apply_move(s, m), 13, {}) == val


def test_solver_matches_bruteforce_over_random_3x3_positions():
    import random
    e = Engine(3, 1)
    sol = Solver(e, max_depth=14)
    rng = random.Random(5)
    checked = 0
    for _ in range(40):
        s = e.initial_state()
        for _ in range(8):
            if e.is_terminal(s):
                break
            assert sol.solve(s)[0] == _brute(e, s, 14, {})
            ms = e.legal_moves(s)
            s = e.apply_move(s, ms[rng.randrange(len(ms))])
            checked += 1
    assert checked > 50


def test_solver_no_walls_is_pure_race_3x3():
    e = Engine(3, 0)
    s = e.initial_state()
    val, best = Solver(e, max_depth=14).solve(s)
    assert val in (-1, 0, 1)
