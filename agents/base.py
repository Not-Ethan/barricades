from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class Analysis:
    best_move: object                 # a Step or Wall
    value: float = 0.0                # eval from current player's POV
    candidates: list = field(default_factory=list)  # [(move, score), ...]
    stats: dict = field(default_factory=dict)        # nodes, depth, time, ...


class Agent(ABC):
    name = "agent"

    @abstractmethod
    def select_move(self, state):
        """Return a legal Move for the current player in `state`."""

    def analyze(self, state):
        """Default: wrap select_move with no extra info."""
        return Analysis(best_move=self.select_move(state))
