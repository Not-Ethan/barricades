# Phase 4a: MCTS Agent — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** A Monte-Carlo Tree Search agent (UCT) for Quoridor with greedy "race" rollouts, registered as `mcts`, with a populated `Analysis` (visits/value/candidates) so it works in the web debug view. The first non-minimax search agent and the seam toward an AlphaZero-style net later.

**Architecture:** Pure functions/classes over the existing `core` API, self-contained in `agents/`. A tree of `_Node`s; UCT selection from the root player's fixed perspective; expansion one child at a time; greedy-race rollouts (cheap, terminate fast); value backpropagation. No changes to `core`. Reuses `agents/heuristics.evaluate` for the rollout-cap fallback.

**Tech Stack:** Python 3.11+, pytest, stdlib `math`/`time`/`random`. Depends only on `core` and `agents.base`/`agents.heuristics`.

---

## Conventions / facts this plan relies on
- `core` exports `legal_moves`, `legal_steps`, `apply_move`, `is_terminal`, `winner`, `shortest_path_len`; `Step(to_cell)`. `apply_move` switches turn. Player 0 goal row 8, player 1 goal row 0.
- `agents.base`: `Agent` (`name`, `select_move`, default `analyze`), `Analysis(best_move, value, candidates, stats)`.
- `agents.heuristics.evaluate(state, player)` — zero-sum positional score (path-difference based), already merged.
- **Registry seam:** ADD only a `"mcts"` entry to `agents/registry.py`.

## File Structure
- `agents/mcts_agent.py` — `_Node`, rollout helper, `MCTSAgent`.
- `agents/registry.py` — MODIFY: add `"mcts": MCTSAgent`.
- `tests/test_mcts.py`, `tests/test_mcts_arena.py`.

## Design notes (read before coding)
- **Value perspective:** every node stores `W` (sum of rollout values) and `N` (visits) from the **root player's fixed perspective**. A rollout that ends with the root player winning contributes `+1`; losing `-1`; a length-capped rollout contributes `clamp(evaluate(s, root_player)/10, -1, 1)`.
- **UCT (from the perspective of the player to move at the parent):** for a parent `node` choosing among children, the exploitation term for child `ch` is `q = ch.W/ch.N` if `node.state.turn == root_player` else `-(ch.W/ch.N)`, and `ucb = q + c * sqrt(ln(node.N) / ch.N)`. This is the standard sign-flip for the minimizing side.
- **Selection/expansion:** descend from root; at each non-terminal node, if it has untried moves, expand exactly one (pop a shuffled untried move, create the child, stop descending and roll out from the child); otherwise pick the best-UCT child and continue. A terminal node rolls out trivially (returns its terminal value).
- **Rollout policy = greedy race:** at each rollout ply, the side to move takes the legal **step** that minimizes its own `shortest_path_len` (ties broken by the agent's RNG); never places walls. This terminates fast (≤ ~20 plies) and gives far better signal than random rollouts. Cap the length; on cap, use the heuristic fallback value.
- **Budget:** run simulations until `time_budget` seconds elapse or `max_sims` reached (whichever first). Always run ≥1 simulation.
- **Move choice:** robust child = child with the most visits `N` (ties → RNG). `analyze` reports `value = root.W/root.N`, `candidates = top-8 children by visits with their mean value`, `stats = {"sims": n, "time_ms": t}`.
- **Stateless across moves** (no tree reuse) — consistent with the project's agent contract.

---

## Task 1: MCTS node, rollout, and agent

**Files:** Create `agents/mcts_agent.py`, test `tests/test_mcts.py`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_mcts.py
import time

from core.state import GameState, Step, initial_state
from core.rules import legal_moves
from agents.mcts_agent import MCTSAgent


def _state(p0, p1, wl=(10, 10), turn=0, h=(), v=()):
    return GameState((p0, p1), frozenset(h), frozenset(v), wl, turn)


def test_returns_legal_move():
    a = MCTSAgent(time_budget=0.3, seed=0)
    s = initial_state()
    assert a.select_move(s) in legal_moves(s)


def test_takes_immediate_win():
    # Player 0 is one step from goal (row 8) with NO walls left (only steps
    # available), and the opponent is one step from THEIR goal (row 0). Stepping
    # to (4,8) is the ONLY move that wins; any other move lets greedy-rollout
    # player 1 step to (4,0) and win. This makes the winning move uniquely +1,
    # which a greedy-rollout MCTS will reliably select (unlike a far-opponent
    # position, where many moves all win in rollout).
    a = MCTSAgent(time_budget=0.5, seed=0)
    s = _state((4, 7), (4, 1), wl=(0, 10), turn=0)
    move = a.select_move(s)
    assert isinstance(move, Step) and move.to_cell == (4, 8)


def test_analyze_populates_fields():
    a = MCTSAgent(time_budget=0.4, seed=0)
    s = initial_state()
    info = a.analyze(s)
    assert info.best_move in legal_moves(s)
    assert isinstance(info.value, (int, float))
    assert len(info.candidates) > 0
    assert info.stats["sims"] > 0


def test_respects_time_budget():
    a = MCTSAgent(time_budget=0.3, seed=0)
    s = initial_state()
    t0 = time.monotonic()
    a.select_move(s)
    assert time.monotonic() - t0 < 2.0


def test_max_sims_cap_is_honored():
    a = MCTSAgent(time_budget=60.0, max_sims=50, seed=0)
    s = initial_state()
    info = a.analyze(s)
    assert info.stats["sims"] <= 50


def test_name():
    assert MCTSAgent().name == "mcts"
```

- [ ] **Step 2: Run, verify it fails** — `pytest tests/test_mcts.py -q` → ModuleNotFoundError.

- [ ] **Step 3: Implement `agents/mcts_agent.py`** (the tests are the contract; this skeleton is correct):

```python
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
```

- [ ] **Step 4: Run, verify pass.** If `test_takes_immediate_win` is flaky, that's a real bug (the root must expand and visit the winning move) — debug, don't weaken. **Step 5: Commit** `feat: MCTS agent with UCT and greedy rollouts`.

---

## Task 2: Register mcts + strength tests

**Files:** MODIFY `agents/registry.py`, test `tests/test_mcts_arena.py`.

**Merge seam:** only ADD the import and the `"mcts"` entry; do not touch other entries.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_mcts_arena.py
from agents.registry import make_agent, available_agents
from agents.arena import run_match


def test_mcts_registered():
    assert "mcts" in available_agents()
    assert make_agent("mcts", time_budget=0.1).name == "mcts"


def test_mcts_beats_random():
    def mk_mcts(seed):
        return make_agent("mcts", time_budget=0.2, seed=seed)

    def mk_random(seed):
        return make_agent("random", seed=seed)

    wins_mcts, wins_random, draws = run_match(mk_mcts, mk_random, games=6)
    assert wins_mcts > wins_random


def test_mcts_competitive_with_greedy():
    # Deterministic (seeded) match. MCTS with greedy rollouts should be at least
    # as strong as bare greedy. If this fails, increase the MCTS budget/sims or
    # improve the rollout — do NOT weaken the assertion.
    def mk_mcts(seed):
        return make_agent("mcts", time_budget=0.3, seed=seed)

    def mk_greedy(seed):
        return make_agent("greedy", seed=seed)

    wins_mcts, wins_greedy, draws = run_match(mk_mcts, mk_greedy, games=4)
    assert wins_mcts >= wins_greedy
```

- [ ] **Step 2: Run, verify fail** (mcts not registered).

- [ ] **Step 3: Modify `agents/registry.py`** — add import + one entry:
```python
from agents.mcts_agent import MCTSAgent
# ...
_FACTORIES = {
    "random": RandomAgent,
    "greedy": GreedyAgent,
    "minimax": MinimaxAgent,
    "mcts": MCTSAgent,
}
```

- [ ] **Step 4: Run, verify pass.** The matches are seeded → deterministic (not flaky). If `test_mcts_competitive_with_greedy` fails, tune the MCTS budget in the test and/or the rollout to genuinely pass it; debug rather than lower the bar. **Step 5:** Run full suite `pytest -q`. **Step 6: Commit** `feat: register mcts agent`.

---

## Done criteria
- `pytest -q` green.
- `MCTSAgent` returns legal moves, takes immediate wins, respects time/sim budgets, and `analyze` populates value/candidates/sims.
- `mcts` beats `random` and is at least competitive with `greedy` in deterministic arena matches.
- Only `agents/registry.py` modified among pre-existing files (clean integration; `mcts` then appears automatically in `/agents` and the web UI).
