from core.state import GameState, initial_state
from agents.heuristics import evaluate, WIN_SCORE


def _state(p0, p1, wl=(10, 10), turn=0, h=(), v=()):
    return GameState((p0, p1), frozenset(h), frozenset(v), wl, turn)


def test_initial_position_is_balanced():
    # symmetric start -> evaluation ~0 from either player's POV
    s = initial_state()
    assert evaluate(s, 0) == 0
    assert evaluate(s, 1) == 0


def test_closer_to_goal_is_better():
    # player 0 one step from goal row 8, player 1 still far
    s = _state((4, 7), (4, 1))
    assert evaluate(s, 0) > 0          # good for player 0
    assert evaluate(s, 1) < 0          # bad for player 1 (same position, opp POV)
    assert evaluate(s, 0) == -evaluate(s, 1)  # zero-sum symmetry


def test_winning_position_scores_win():
    # player 0 already on goal row 8
    s = _state((4, 8), (4, 1))
    assert evaluate(s, 0) >= WIN_SCORE
    assert evaluate(s, 1) <= -WIN_SCORE


def test_unreachable_goal_is_worst():
    # player 0 fully walled off (no path) is terrible for player 0
    h = [(c, 0) for c in range(0, 8, 2)] + [(7, 0)]
    s = _state((4, 0), (0, 8), h=h)
    assert evaluate(s, 0) < evaluate(initial_state(), 0)
