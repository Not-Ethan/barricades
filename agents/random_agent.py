import random

from core.rules import legal_moves
from agents.base import Agent


class RandomAgent(Agent):
    name = "random"

    def __init__(self, seed=None):
        self._rng = random.Random(seed)

    def select_move(self, state):
        return self._rng.choice(legal_moves(state))
