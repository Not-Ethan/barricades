import math
import random
import time

from core.state import Step
from core.rules import (
    legal_moves, apply_move, is_terminal, winner,
)
from agents.base import Agent, Analysis
from agents.heuristics import evaluate, WIN_SCORE
from agents.movegen import relevant_moves

# Scale factor mapping heuristic evaluate() values into [-1, 1].
# typical path-difference values are 0–10 steps; WIN_SCORE is 10_000.
# We divide by SCALE so that a ~8-step advantage ≈ 1.0; terminal states
# are clamped to ±1 before dividing.
_EVAL_SCALE = 8.0


class _Node:
    __slots__ = ("state", "parent", "move", "children", "untried", "N", "W")

    def __init__(self, state, parent=None, move=None):
        self.state = state
        self.parent = parent
        self.move = move          # move that led from parent to here
        self.children = []
        self.untried = None       # lazily filled list of candidate moves
        self.N = 0
        self.W = 0.0              # total value, from ROOT player's perspective


def _heuristic_value(state, root_player):
    """Return a value in [-1, 1] for root_player using the positional heuristic.

    Terminal states return ±1 exactly.  Non-terminal states are evaluated
    with agents.heuristics.evaluate and clamped to [-1, 1] via _EVAL_SCALE.
    """
    if is_terminal(state):
        return 1.0 if winner(state) == root_player else -1.0
    raw = evaluate(state, root_player)
    # WIN_SCORE appears in evaluate() for already-terminal positions; guard.
    if raw >= WIN_SCORE:
        return 1.0
    if raw <= -WIN_SCORE:
        return -1.0
    return max(-1.0, min(1.0, raw / _EVAL_SCALE))


def _candidate_moves(state):
    """Return candidate moves for tree expansion.

    Uses relevant_moves() to exclude pointless walls (walls that don't
    lengthen the opponent's shortest path).  Falls back to legal_moves()
    only if relevant_moves() returns nothing — in practice there is always
    at least one legal step, so the fallback is a safety net.
    """
    moves = relevant_moves(state)
    if not moves:
        moves = legal_moves(state)
    return moves


class MCTSAgent(Agent):
    name = "mcts"

    def __init__(self, time_budget=1.0, max_sims=100_000, c=1.4,
                 rollout_cap=40, seed=None):
        self.time_budget = time_budget
        self.max_sims = max_sims
        self.c = c
        # rollout_cap kept for API compatibility but no longer used
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
                moves = _candidate_moves(node.state)
                # Partition: steps first, walls last. Shuffle within each group
                # so that steps are expanded before walls, allowing UCT to
                # identify good moves faster.
                steps = [m for m in moves if isinstance(m, Step)]
                walls = [m for m in moves if not isinstance(m, Step)]
                self._rng.shuffle(steps)
                self._rng.shuffle(walls)
                # Store walls first (popped last = tried last) so steps tried first.
                node.untried = walls + steps
            if node.untried:
                move = node.untried.pop()
                child = _Node(apply_move(node.state, move), parent=node, move=move)
                node.children.append(child)
                node = child
                break
            node = self._uct_child(node, root_player)
        # Leaf evaluation: heuristic value (no rollout)
        v = _heuristic_value(node.state, root_player)
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
