from core.rules import shortest_path_len, winner

WIN_SCORE = 10_000
_UNREACHABLE = 1_000  # large stand-in distance when a player has no path
_TEMPO_BONUS = 0.5    # bonus for the side that has an immediate winning step


def _tempo(state, player):
    """Return +_TEMPO_BONUS if `player` is about to step to their goal (one step
    away and it is their turn), or -_TEMPO_BONUS if the opponent is. Otherwise 0.
    This term is zero-sum when both players have paths: _tempo(s,p)==-_tempo(s,1-p)."""
    mover = state.turn
    if shortest_path_len(state, mover) == 1:
        return _TEMPO_BONUS if mover == player else -_TEMPO_BONUS
    return 0.0


def evaluate(state, player):
    """Score the position from `player`'s point of view. Positive = good for
    `player`. Zero-sum when both players have paths: evaluate(s, p) == -evaluate(s, 1-p).

    If a player has no path to their goal it is treated as a very long distance
    (_UNREACHABLE). An additional penalty is applied when `player` specifically
    has no path (a unilateral catastrophe even if the opponent is also stuck)."""
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

    # Extra penalty when self specifically has no path (even if opp also stuck).
    if d_self is None:
        path_term -= _UNREACHABLE

    # Secondary tie-breaker: keeping more walls is a mild advantage.
    wall_term = 0.1 * (state.walls_left[player] - state.walls_left[opp])

    # Tempo term: reward having an immediate winning step available.
    tempo = _tempo(state, player)

    return path_term + wall_term + tempo
