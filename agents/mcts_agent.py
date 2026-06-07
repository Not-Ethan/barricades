import math
import random
import time
from collections import deque

from core.state import Step
from core.coords import N, on_board
from core.rules import (
    legal_moves, legal_steps, apply_move, is_terminal, winner, shortest_path_len,
    is_blocked,
)
from agents.base import Agent, Analysis
from agents.heuristics import evaluate

_BIG = 9999

DIRS = [(0, 1), (0, -1), (1, 0), (-1, 0)]


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


def _goal_row(player):
    """Return the goal row for a player (0→row 8, 1→row 0)."""
    return N - 1 if player == 0 else 0


def _goal_dist_map(state, player):
    """BFS backwards from the goal row to all reachable cells for `player`.
    Returns dict: cell → distance to goal row (ignoring opponent pawn).
    This lets us evaluate all step-destinations with a single BFS."""
    goal = _goal_row(player)
    # Seed the BFS with all goal-row cells that are reachable (on the board).
    dist = {}
    queue = deque()
    for col in range(N):
        cell = (col, goal)
        dist[cell] = 0
        queue.append(cell)
    # BFS backwards: edge (a, b) is passable iff is_blocked(state, a, b) is False
    # and is_blocked is symmetric, so we can traverse in either direction.
    while queue:
        cell = queue.popleft()
        d = dist[cell]
        for dx, dy in DIRS:
            nxt = (cell[0] + dx, cell[1] + dy)
            if not on_board(nxt) or nxt in dist:
                continue
            if is_blocked(state, cell, nxt):
                continue
            dist[nxt] = d + 1
            queue.append(nxt)
    return dist


def _greedy_step_with_maps(state, rng, dist_maps):
    """Pick a legal step minimizing the mover's shortest path (ties by rng).
    `dist_maps` is a pre-computed dict {player: goal_dist_map} for the current
    wall configuration. During a rollout, walls don't change, so we can reuse."""
    mover = state.turn
    steps = legal_steps(state)
    if not steps:
        return rng.choice(legal_moves(state))

    dist_map = dist_maps[mover]
    best, best_d = [], None
    for c in steps:
        d = dist_map.get(c, _BIG)
        if best_d is None or d < best_d:
            best_d, best = d, [c]
        elif d == best_d:
            best.append(c)
    return Step(rng.choice(best))


def _rollout_value(state, root_player, rng, cap):
    """Play a greedy race to terminal (or cap). Return value in [-1,1] for root.
    Pre-computes distance maps once per rollout since walls don't change."""
    s = state
    # Pre-compute distance maps from each player's goal row (walls fixed).
    dist_maps = {0: _goal_dist_map(s, 0), 1: _goal_dist_map(s, 1)}
    for _ in range(cap):
        if is_terminal(s):
            return 1.0 if winner(s) == root_player else -1.0
        s = apply_move(s, _greedy_step_with_maps(s, rng, dist_maps))
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
                moves = legal_moves(node.state)
                # Partition: steps first, walls last. Shuffle within each group
                # so that steps are expanded before walls, allowing UCT to
                # identify good moves faster (walls lose tempo in greedy rollouts).
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
