import random

from core.state import Step
from core.rules import legal_steps, legal_moves, apply_move, shortest_path_len
from agents.base import Agent

_INF = 10_000


class GreedyAgent(Agent):
    name = "greedy"

    def __init__(self, seed=None):
        self._rng = random.Random(seed)

    def select_move(self, state):
        me = state.turn
        steps = [Step(c) for c in legal_steps(state)]
        if not steps:
            return self._rng.choice(legal_moves(state))
        best_dist = None
        best = []
        for move in steps:
            d = shortest_path_len(apply_move(state, move), me)
            d = _INF if d is None else d
            if best_dist is None or d < best_dist:
                best_dist = d
                best = [move]
            elif d == best_dist:
                best.append(move)
        return self._rng.choice(best)
