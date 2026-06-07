from itertools import count

from core.state import initial_state
from core.rules import legal_moves, apply_move


class Game:
    def __init__(self, game_id, controllers):
        self.id = game_id
        self.controllers = list(controllers)   # ["human" | agent-name, ...]
        # Raw controller specs (str or {name,params}); the app may override this
        # with objects carrying agent params. Defaulted so it always exists.
        self._specs = list(controllers)
        self.history = [initial_state()]

    @property
    def state(self):
        return self.history[-1]

    @property
    def move_count(self):
        return len(self.history) - 1

    def apply(self, move):
        if move not in legal_moves(self.state):
            raise ValueError(f"illegal move: {move!r}")
        self.history.append(apply_move(self.state, move))

    def undo(self):
        if len(self.history) > 1:
            self.history.pop()


class GameStore:
    def __init__(self):
        self._games = {}
        self._ids = count(1)

    def create(self, controllers):
        gid = f"g{next(self._ids)}"
        g = Game(gid, controllers)
        self._games[gid] = g
        return g

    def get(self, gid):
        if gid not in self._games:
            raise KeyError(gid)
        return self._games[gid]
