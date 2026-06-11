import random
import barricades_native as bn
from core.state import GameState, Step
from core import rules
from tests.test_native_game import to_native, mv_to_tuple

PLY_BOUND = 36


def py_solve_race(s, depth, memo):
    """Reference: depth-bounded negamax over pawn moves. +1 win / -1 loss / 0 draw-at-bound for side to move."""
    w = rules.winner(s)
    if w is not None:
        return 1 if w == s.turn else -1
    if depth == 0:
        return 0
    key = (s.pawns, s.turn, depth)
    if key in memo:
        return memo[key]
    best = -1
    for c in rules.legal_steps(s):
        v = -py_solve_race(rules.apply_move(s, Step(c)), depth - 1, memo)
        if v > best:
            best = v
        if best == 1:
            break
    memo[key] = best
    return best


def _random_zero_wall_position(rng):
    from core.state import initial_state, Wall
    s = initial_state()
    for _ in range(120):
        if rules.is_terminal(s):
            return None
        if s.walls_left == (0, 0):
            return s
        moves = rules.legal_moves(s)
        walls = [m for m in moves if isinstance(m, Wall)]
        s = rules.apply_move(s, rng.choice(walls) if walls else rng.choice(moves))
    return None


def test_solve_race_matches_reference():
    rng = random.Random(7)
    checked = 0
    for _ in range(60):
        s = _random_zero_wall_position(rng)
        if s is None or rules.is_terminal(s):
            continue
        assert s.walls_left == (0, 0)
        val, mv = bn.solve_race(to_native(s))
        ref = py_solve_race(s, PLY_BOUND, {})
        assert val == ref, f"value mismatch {val} vs {ref} at {s}"
        assert tuple(mv) in {mv_to_tuple(m) for m in rules.legal_moves(s)}
        if val == 1:
            after = rules.apply_move(s, Step((mv[1], mv[2])))
            assert -py_solve_race(after, PLY_BOUND - 1, {}) == 1
        checked += 1
    assert checked >= 10
