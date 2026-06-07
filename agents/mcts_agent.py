import math
import random
import time

from core.state import Step
from core.rules import (
    legal_moves, legal_steps, apply_move, is_terminal, winner, shortest_path_len,
)
from agents.base import Agent, Analysis
from agents.heuristics import evaluate

_BIG = 9999


class _Node:
    __slots__ = ("state", "parent", "move", "children", "untried", "N", "W")

    def __init__(self, state, parent=None, move=None):
        self.state = state
        self.parent = parent
        self.move = move          # move that led from parent to here
        self.children = []
        self.untried = None       # lazily filled list of legal moves
        self.N = 0
        self.W = 0.0              # total value, from ROOT player's perspective


def _greedy_step(state, rng):
    """Pick a legal step minimizing the mover's shortest path (ties by rng)."""
    mover = state.turn
    best, best_d = [], None
    for c in legal_steps(state):
        d = shortest_path_len(apply_move(state, Step(c)), mover)
        d = _BIG if d is None else d
        if best_d is None or d < best_d:
            best_d, best = d, [c]
        elif d == best_d:
            best.append(c)
    if best:
        return Step(rng.choice(best))
    return rng.choice(legal_moves(state))   # no steps (rare): any legal move


def _rollout_value(state, root_player, rng, cap):
    """Play a greedy race to terminal (or cap). Return value in [-1,1] for root."""
    s = state
    for _ in range(cap):
        if is_terminal(s):
            return 1.0 if winner(s) == root_player else -1.0
        s = apply_move(s, _greedy_step(s, rng))
    if is_terminal(s):
        return 1.0 if winner(s) == root_player else -1.0
    return max(-1.0, min(1.0, evaluate(s, root_player) / 10.0))


class MCTSAgent(Agent):
    name = "mcts"

    def __init__(self, time_budget=1.0, max_sims=100_000, c=1.4,
                 rollout_cap=40, seed=None):
        self.time_budget = time_budget
        self.max_sims = max_sims
        self.c = c
        self.rollout_cap = rollout_cap
        self._rng = random.Random(seed)

    def _uct_child(self, node, root_player):
        log_n = math.log(node.N)
        best, best_u = None, None
        for ch in node.children:
            q = ch.W / ch.N
            exploit = q if node.state.turn == root_player else -q
            u = exploit + self.c * math.sqrt(log_n / ch.N)
            if best_u is None or u > best_u:
                best_u, best = u, ch
        return best

    def _simulate(self, root, root_player):
        node = root
        # Selection + one expansion
        while not is_terminal(node.state):
            if node.untried is None:
                node.untried = legal_moves(node.state)
                self._rng.shuffle(node.untried)
            if node.untried:
                move = node.untried.pop()
                child = _Node(apply_move(node.state, move), parent=node, move=move)
                node.children.append(child)
                node = child
                break
            node = self._uct_child(node, root_player)
        # Rollout
        v = _rollout_value(node.state, root_player, self._rng, self.rollout_cap)
        # Backprop
        while node is not None:
            node.N += 1
            node.W += v
            node = node.parent

    def analyze(self, state):
        t0 = time.monotonic()
        root = _Node(state)
        root_player = state.turn
        deadline = t0 + self.time_budget
        sims = 0
        while sims < self.max_sims:
            self._simulate(root, root_player)
            sims += 1
            if time.monotonic() >= deadline:
                break
        if not root.children:                      # degenerate fallback
            return Analysis(best_move=legal_moves(state)[0], value=0.0,
                            candidates=[], stats={"sims": sims, "time_ms": 0})
        ordered = sorted(root.children, key=lambda ch: ch.N, reverse=True)
        top_n = ordered[0].N
        winners = [ch for ch in ordered if ch.N == top_n]
        best = self._rng.choice(winners)
        value = root.W / root.N if root.N else 0.0
        candidates = [(ch.move, ch.W / ch.N if ch.N else 0.0) for ch in ordered[:8]]
        return Analysis(best_move=best.move, value=value, candidates=candidates,
                        stats={"sims": sims,
                               "time_ms": int((time.monotonic() - t0) * 1000)})

    def select_move(self, state):
        return self.analyze(state).best_move
