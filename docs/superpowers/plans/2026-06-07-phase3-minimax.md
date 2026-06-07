# Phase 3: Minimax Engine + Analysis — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** A real Quoridor opponent — alpha-beta search with a shortest-path-difference evaluation — plus a populated `Analysis` so the Phase 2 debug view can show eval, candidate moves, and search stats.

**Architecture:** Pure functions over the existing `core` API. A position evaluator (`heuristics.py`), a move-ordering helper, and `MinimaxAgent` doing iterative-deepening alpha-beta under a wall-clock budget. Registered as `"minimax"`. No changes to `core`. Self-contained in `agents/`.

**Tech Stack:** Python 3.11+, pytest, stdlib `time` for the budget. Depends only on `core` (`legal_moves`, `legal_steps`, `legal_walls`, `apply_move`, `is_terminal`, `winner`, `shortest_path_len`) and `agents.base`.

---

## Conventions / facts this plan relies on
- Player 0 goal row 8, player 1 goal row 0. `shortest_path_len(state, p)` = BFS distance, `None` if unreachable.
- `apply_move` switches turn. `winner(state)` ∈ {0,1,None}. `is_terminal` true when someone reached goal.
- `Agent` (in `agents/base.py`): `name`, `select_move(state)`, default `analyze(state)`. `Analysis(best_move, value=0.0, candidates=[], stats={})`.
- `legal_moves(state)` returns `Step`/`Wall` objects; `Step.to_cell`, `Wall(c,r,orient)`.

## File Structure
- `agents/heuristics.py` — `evaluate(state, player)` and `WIN_SCORE`.
- `agents/minimax_agent.py` — `MinimaxAgent`, `ordered_moves`, internal alpha-beta.
- `agents/registry.py` — MODIFY: add `"minimax": MinimaxAgent`.
- `tests/test_heuristics.py`, `tests/test_minimax.py` — tests.

---

## Task 1: Position evaluation

**Files:** Create `agents/heuristics.py`, test `tests/test_heuristics.py`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_heuristics.py
from core.state import GameState, initial_state
from agents.heuristics import evaluate, WIN_SCORE


def _state(p0, p1, wl=(10, 10), turn=0, h=(), v=()):
    return GameState((p0, p1), frozenset(h), frozenset(v), wl, turn)


def test_initial_position_is_balanced():
    # symmetric start -> evaluation ~0 from either player's POV
    s = initial_state()
    assert evaluate(s, 0) == 0
    assert evaluate(s, 1) == 0


def test_closer_to_goal_is_better():
    # player 0 one step from goal row 8, player 1 still far
    s = _state((4, 7), (4, 1))
    assert evaluate(s, 0) > 0          # good for player 0
    assert evaluate(s, 1) < 0          # bad for player 1 (same position, opp POV)
    assert evaluate(s, 0) == -evaluate(s, 1)  # zero-sum symmetry


def test_winning_position_scores_win():
    # player 0 already on goal row 8
    s = _state((4, 8), (4, 1))
    assert evaluate(s, 0) >= WIN_SCORE
    assert evaluate(s, 1) <= -WIN_SCORE


def test_unreachable_goal_is_worst():
    # player 0 fully walled off (no path) is terrible for player 0
    h = [(c, 0) for c in range(0, 8, 2)] + [(7, 0)]
    s = _state((4, 0), (0, 8), h=h)
    assert evaluate(s, 0) < evaluate(initial_state(), 0)
```

- [ ] **Step 2: Run, verify it fails** — `pytest tests/test_heuristics.py -q` → ModuleNotFoundError.

- [ ] **Step 3: Implement `agents/heuristics.py`**

```python
from core.rules import shortest_path_len, winner

WIN_SCORE = 10_000
_UNREACHABLE = 1_000  # large stand-in distance when a player has no path


def _dist(state, player):
    d = shortest_path_len(state, player)
    return _UNREACHABLE if d is None else d


def evaluate(state, player):
    """Score the position from `player`'s point of view. Positive = good for
    `player`. Zero-sum: evaluate(s, p) == -evaluate(s, 1 - p)."""
    w = winner(state)
    if w is not None:
        return WIN_SCORE if w == player else -WIN_SCORE
    opp = 1 - player
    # Primary term: how much closer we are to goal than the opponent.
    path_term = _dist(state, opp) - _dist(state, player)
    # Secondary tie-breaker: keeping more walls is a mild advantage.
    wall_term = 0.1 * (state.walls_left[player] - state.walls_left[opp])
    return path_term + wall_term
```

- [ ] **Step 4: Run, verify pass.** **Step 5: Commit** `feat: position evaluation heuristic`.

---

## Task 2: MinimaxAgent (alpha-beta + iterative deepening)

**Files:** Create `agents/minimax_agent.py`, test `tests/test_minimax.py`.

**Design notes for the implementer:**
- Search maximizes `evaluate(state, root_player)` where `root_player` is the side to move at the root. Standard negamax or min/max-by-turn; either is fine — be consistent and test it.
- **Iterative deepening:** search depth 1, then 2, … until the wall-clock budget (`time_budget` seconds, default 1.0) is exceeded; keep the best move from the last *completed* depth. Always complete depth 1 even if over budget (so a move always exists).
- **Move ordering** (`ordered_moves(state)`): try pawn steps that reduce our own shortest path first, then other steps, then walls. Good ordering makes alpha-beta prune far more.
- **Wall-candidate limiting** for tractability in pure Python: at search nodes (not the root), cap the number of wall placements considered to the `wall_cap` most "relevant" ones — walls adjacent to either pawn or lying on a pawn's current shortest path. The root considers all legal walls (so the agent can find any move) unless that is too slow; default `wall_cap=None` at root, `wall_cap=12` at deeper nodes. Document this clearly — it is a strength cap, not a correctness issue.
- Terminal/depth-0 nodes return `evaluate(state, root_player)`.
- `analyze(state)` runs the same search at the root, returning `Analysis(best_move, value=<root score>, candidates=<(move, score) for each root move, sorted desc, top ~8>, stats={"nodes": n, "depth": d, "time_ms": t})`.
- `select_move` returns `analyze(state).best_move`. Tie-break equal-scored root moves with a seeded RNG for reproducibility.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_minimax.py
import time

from core.state import GameState, Step, Wall, initial_state
from core.rules import legal_moves
from agents.minimax_agent import MinimaxAgent


def _state(p0, p1, wl=(10, 10), turn=0, h=(), v=()):
    return GameState((p0, p1), frozenset(h), frozenset(v), wl, turn)


def test_returns_legal_move():
    a = MinimaxAgent(time_budget=0.5, seed=0)
    s = initial_state()
    assert a.select_move(s) in legal_moves(s)


def test_takes_immediate_win():
    # player 0 at (4,7) can step to (4,8) and win this move.
    a = MinimaxAgent(time_budget=0.5, seed=0)
    s = _state((4, 7), (0, 0))
    move = a.select_move(s)
    assert isinstance(move, Step) and move.to_cell == (4, 8)


def test_blocks_or_races_sensibly_not_suicidal():
    # On the open board the engine should advance, not waste the turn on a
    # far-corner wall. Best move should reduce its own distance or be a wall
    # that hurts the opponent — assert it is at least not strictly worsening.
    from agents.heuristics import evaluate
    from core.rules import apply_move
    a = MinimaxAgent(time_budget=0.5, seed=0)
    s = initial_state()
    move = a.select_move(s)
    before = evaluate(s, s.turn)
    after = evaluate(apply_move(s, move), s.turn)
    assert after >= before


def test_analyze_populates_fields():
    a = MinimaxAgent(time_budget=0.5, seed=0)
    s = initial_state()
    info = a.analyze(s)
    assert info.best_move in legal_moves(s)
    assert isinstance(info.value, (int, float))
    assert len(info.candidates) > 0
    assert info.stats["nodes"] > 0
    assert info.stats["depth"] >= 1


def test_respects_time_budget():
    a = MinimaxAgent(time_budget=0.3, seed=0)
    s = initial_state()
    t0 = time.monotonic()
    a.select_move(s)
    # generous upper bound: budget + overhead for finishing the current depth
    assert time.monotonic() - t0 < 3.0


def test_name_and_params():
    assert MinimaxAgent().name == "minimax"
```

- [ ] **Step 2: Run, verify it fails.**

- [ ] **Step 3: Implement `agents/minimax_agent.py`.**
Implement per the design notes above. Suggested skeleton (fill in correctly; the tests are the contract):

```python
import random
import time

from core.state import Step, Wall
from core.rules import (
    legal_moves, legal_steps, legal_walls, apply_move, is_terminal,
    shortest_path_len,
)
from agents.base import Agent, Analysis
from agents.heuristics import evaluate, WIN_SCORE


def ordered_moves(state, wall_cap=None):
    """Steps first (those reducing our own shortest path ahead of others),
    then walls (optionally capped to the most relevant)."""
    me = state.turn
    base = shortest_path_len(state, me)
    steps = []
    for c in legal_steps(state):
        d = shortest_path_len(apply_move(state, Step(c)), me)
        steps.append((d if d is not None else 1_000, Step(c)))
    steps.sort(key=lambda t: t[0])
    ordered = [m for _, m in steps]

    walls = legal_walls(state)
    if wall_cap is not None and len(walls) > wall_cap:
        opp = 1 - me
        p0 = state.pawns[me]
        p1 = state.pawns[opp]

        def relevance(w):
            # closeness of the wall anchor to either pawn (smaller = more relevant)
            return min(abs(w.c - p1[0]) + abs(w.r - p1[1]),
                       abs(w.c - p0[0]) + abs(w.r - p0[1]))

        walls = sorted(walls, key=relevance)[:wall_cap]
    return ordered + walls


class MinimaxAgent(Agent):
    name = "minimax"

    def __init__(self, time_budget=1.0, max_depth=64, wall_cap=12, seed=None):
        self.time_budget = time_budget
        self.max_depth = max_depth
        self.wall_cap = wall_cap
        self._rng = random.Random(seed)
        self._nodes = 0
        self._deadline = 0.0

    class _Timeout(Exception):
        pass

    def _search(self, state, depth, alpha, beta, root_player):
        self._nodes += 1
        # Evaluate leaves WITHOUT a timeout check so depth-1 always completes
        # (guarantees a move exists even under a tiny budget).
        if is_terminal(state) or depth == 0:
            return evaluate(state, root_player)
        if time.monotonic() > self._deadline:
            raise MinimaxAgent._Timeout()
        maximizing = state.turn == root_player
        moves = ordered_moves(state, wall_cap=self.wall_cap)
        if maximizing:
            best = -float("inf")
            for m in moves:
                val = self._search(apply_move(state, m), depth - 1, alpha, beta, root_player)
                best = max(best, val)
                alpha = max(alpha, best)
                if alpha >= beta:
                    break
            return best
        else:
            best = float("inf")
            for m in moves:
                val = self._search(apply_move(state, m), depth - 1, alpha, beta, root_player)
                best = min(best, val)
                beta = min(beta, best)
                if alpha >= beta:
                    break
            return best

    def analyze(self, state):
        self._nodes = 0
        self._deadline = time.monotonic() + self.time_budget
        t0 = time.monotonic()
        root_player = state.turn
        root_moves = ordered_moves(state, wall_cap=None)  # all moves at root
        best_move = root_moves[0]
        best_scores = {}
        completed_depth = 0
        for depth in range(1, self.max_depth + 1):
            try:
                scores = {}
                for m in root_moves:
                    scores[m] = self._search(apply_move(state, m), depth - 1,
                                             -float("inf"), float("inf"), root_player)
                best_scores = scores
                completed_depth = depth
                # early exit: a forced win found
                if max(scores.values()) >= WIN_SCORE:
                    break
            except MinimaxAgent._Timeout:
                break
        # choose best (random tie-break)
        best_val = max(best_scores.values())
        winners = [m for m, v in best_scores.items() if v == best_val]
        best_move = self._rng.choice(winners)
        candidates = sorted(best_scores.items(), key=lambda kv: kv[1], reverse=True)[:8]
        return Analysis(
            best_move=best_move,
            value=best_val,
            candidates=[(m, v) for m, v in candidates],
            stats={"nodes": self._nodes, "depth": completed_depth,
                   "time_ms": int((time.monotonic() - t0) * 1000)},
        )

    def select_move(self, state):
        return self.analyze(state).best_move
```

- [ ] **Step 4: Run, verify pass.** If `test_takes_immediate_win` or budget tests are flaky, debug per superpowers:systematic-debugging — do NOT weaken assertions. **Step 5: Commit** `feat: MinimaxAgent with iterative-deepening alpha-beta`.

---

## Task 3: Register minimax + prove it beats greedy

**Files:** MODIFY `agents/registry.py`, test `tests/test_minimax_arena.py`.

**IMPORTANT (merge seam):** Only ADD a line to `_FACTORIES`. Do not reorder or touch other entries.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_minimax_arena.py
from agents.registry import make_agent, available_agents
from agents.arena import run_match


def test_minimax_registered():
    assert "minimax" in available_agents()
    assert make_agent("minimax", time_budget=0.2).name == "minimax"


def test_minimax_beats_greedy():
    def mk_mm(seed):
        return make_agent("minimax", time_budget=0.2, seed=seed)

    def mk_greedy(seed):
        return make_agent("greedy", seed=seed)

    wins_mm, wins_greedy, draws = run_match(mk_mm, mk_greedy, games=4)
    assert wins_mm > wins_greedy
```

- [ ] **Step 2: Run, verify fail** (minimax not registered).

- [ ] **Step 3: Modify `agents/registry.py`** — add the import and one entry:
```python
from agents.minimax_agent import MinimaxAgent
# ...
_FACTORIES = {
    "random": RandomAgent,
    "greedy": GreedyAgent,
    "minimax": MinimaxAgent,
}
```

- [ ] **Step 4: Run, verify pass.** (Keep `time_budget` small so the match runs quickly.) If minimax does not beat greedy, that is a real engine bug — debug the search/eval, do not lower the bar. **Step 5:** Run full suite `pytest -q`. **Step 6: Commit** `feat: register minimax agent`.

---

## Done criteria
- `pytest -q` green.
- `MinimaxAgent` returns legal moves, takes immediate wins, respects its time budget, and `analyze` returns a populated `Analysis` (value, candidates, nodes/depth/time stats).
- `minimax` beats `greedy` head-to-head in the arena.
- Only `agents/registry.py` was modified outside new files (clean seam for parallel Phase 2).
