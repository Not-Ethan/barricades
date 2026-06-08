"""Evaluation function variants for the Quoridor barricades engines.

evaluate_tempo(state, player)
    A tempo-accurate variant of agents.heuristics.evaluate that applies a
    ±0.5 turn bonus at EVERY position (not just when a player is one step from
    goal).  This models the half-tempo advantage of the side whose turn it is.

    eval_tempo(state, p) = (opp_dist - my_dist) + wall_term + TURN
    where TURN = +0.5 if state.turn == player else -0.5

    The function is zero-sum for non-terminal positions where both players
    have a path:  evaluate_tempo(s, p) == -evaluate_tempo(s, 1-p).
    WIN_SCORE and _UNREACHABLE behaviour are identical to the base evaluate().
"""

from core.rules import shortest_path_len, winner

WIN_SCORE = 10_000
_UNREACHABLE = 1_000   # large stand-in distance when a player has no path
_TEMPO_BONUS = 0.5     # half-tempo advantage for the side to move


def evaluate_tempo(state, player: int) -> float:
    """Score the position from `player`'s point of view using a full-tempo term.

    Identical to agents.heuristics.evaluate except:
    - The tempo bonus is always ±_TEMPO_BONUS based solely on whose turn it is,
      rather than only when the mover is one step from goal.

    Positive = good for `player`.  Zero-sum when both players have paths.
    """
    w = winner(state)
    if w is not None:
        return WIN_SCORE if w == player else -WIN_SCORE

    opp = 1 - player

    d_self = shortest_path_len(state, player)
    d_opp  = shortest_path_len(state, opp)

    eff_self = _UNREACHABLE if d_self is None else d_self
    eff_opp  = _UNREACHABLE if d_opp  is None else d_opp

    # Primary term: how much closer we are to goal than the opponent.
    path_term = eff_opp - eff_self

    # Extra penalty when self specifically has no path.
    if d_self is None:
        path_term -= _UNREACHABLE

    # Secondary tie-breaker: keeping more walls is a mild advantage.
    wall_term = 0.1 * (state.walls_left[player] - state.walls_left[opp])

    # Tempo term: always ±0.5 based on whose turn it is (full-tempo model).
    tempo = _TEMPO_BONUS if state.turn == player else -_TEMPO_BONUS

    return path_term + wall_term + tempo


# --- Robustness / reciprocal-distance variant -----------------------------
# The base evaluate() (path-difference + wall economy + one-step tempo) PLUS a
# turn-AGNOSTIC reciprocal-distance term that sharpens as a pawn nears its goal.
# Being close is worth progressively more (1/(d+1)), so the engine values
# converting a lead and finishing — without the depth-parity noise that made the
# turn-dependent tempo term hurt search.
from agents.heuristics import evaluate as _base_evaluate  # noqa: E402

_RECIP_K = 4.0


def evaluate_robust(state, player: int) -> float:
    base = _base_evaluate(state, player)
    if base >= WIN_SCORE or base <= -WIN_SCORE:
        return base  # terminal — leave decisive scores untouched
    opp = 1 - player
    d_self = shortest_path_len(state, player)
    d_opp = shortest_path_len(state, opp)
    eff_self = _UNREACHABLE if d_self is None else d_self
    eff_opp = _UNREACHABLE if d_opp is None else d_opp
    recip = _RECIP_K * (1.0 / (eff_self + 1) - 1.0 / (eff_opp + 1))
    return base + recip
