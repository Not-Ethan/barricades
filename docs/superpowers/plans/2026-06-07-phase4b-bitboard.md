# Phase 4b: Bitboard Core Optimization — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Speed up the engine's hot path — the BFS used by `shortest_path_len` / `has_path_to_goal` (which dominates `legal_walls`, all heuristics, minimax, and MCTS rollouts) — by reimplementing it as a bitwise flood-fill, **behind the frozen `core` public API**. No behavior changes: all 114 existing tests must still pass, plus a new equivalence test proving the fast path matches the reference exactly.

**Architecture:** A new `core/bitboard.py` holds an 81-bit board representation (Python int), precomputed direction masks, per-state wall-passability masks, and layered flood-fill BFS. `core/rules.py` keeps the **exact same public signatures and `GameState` shape** (frozensets, fields unchanged) — only the *bodies* of `shortest_path_len` and `has_path_to_goal` are rewired to call the bitboard BFS. The previous pure-Python BFS is kept as a private reference (`_shortest_path_len_ref`) used only by the equivalence test and benchmark. Nothing in `agents/`, `server/`, or `web/` changes.

**Tech Stack:** Python 3.14, pytest, stdlib only (pure-int bitboards — no numpy needed).

---

## Conventions / facts this relies on
- Cells `(col, row)`, 0..8. Bit index `= row * 9 + col`. Board is 81 bits in a Python int.
- Player 0 goal row 8, player 1 goal row 0.
- Wall blocking (must match `is_blocked` EXACTLY):
  - H-wall anchor `(a,b)` blocks NORTH from cells `(a,b)` and `(a+1,b)`; blocks SOUTH from `(a,b+1)` and `(a+1,b+1)`.
  - V-wall anchor `(a,b)` blocks EAST from cells `(a,b)` and `(a,b+1)`; blocks WEST from `(a+1,b)` and `(a+1,b+1)`.
- BFS ignores the opponent pawn (walls only) — same as the current `shortest_path_len`.

## File Structure
- `core/bitboard.py` — NEW: bit helpers, masks, `bfs_dist(state, player)`, `path_exists(state, player)`.
- `core/rules.py` — MODIFY: rewire `shortest_path_len`/`has_path_to_goal` to the bitboard path; keep a `_shortest_path_len_ref` private reference (the current body).
- `tests/test_bitboard.py` — NEW: equivalence vs reference + direct unit cases.
- `scripts/bench_core.py` — NEW: informational benchmark (reference vs bitboard; legal_walls throughput).

---

## Task 1: Bitboard primitives + flood-fill BFS

**Files:** Create `core/bitboard.py`, test `tests/test_bitboard.py`.

- [ ] **Step 1: Write the failing test** (unit cases first; equivalence test added in Task 2)

```python
# tests/test_bitboard.py
from core.state import GameState, initial_state
from core.bitboard import bfs_dist, path_exists


def _state(p0, p1, h=(), v=(), turn=0):
    return GameState((p0, p1), frozenset(h), frozenset(v), (10, 10), turn)


def test_open_board_distances():
    s = initial_state()
    assert bfs_dist(s, 0) == 8
    assert bfs_dist(s, 1) == 8


def test_already_on_goal_zero():
    assert bfs_dist(_state((4, 8), (0, 0)), 0) == 0
    assert bfs_dist(_state((0, 0), (4, 0)), 1) == 0


def test_wall_lengthens_path():
    s = _state((4, 0), (0, 8), h=[(3, 0), (5, 0)])
    assert path_exists(s, 0)
    assert bfs_dist(s, 0) > 8


def test_fully_walled_off_no_path():
    h = [(c, 0) for c in range(0, 8, 2)] + [(7, 0)]
    s = _state((4, 0), (0, 8), h=h)
    assert not path_exists(s, 0)
    assert bfs_dist(s, 0) is None


def test_vertical_walls_do_not_block_vertical_moves():
    # a column of V-walls along col 0 must not change p0's straight-up distance
    s = _state((0, 0), (8, 8), v=[(0, 0), (0, 2), (0, 4), (0, 6)])
    assert bfs_dist(s, 0) == 8
```

- [ ] **Step 2: Run, verify fail** — `pytest tests/test_bitboard.py -q` → ModuleNotFoundError.

- [ ] **Step 3: Implement `core/bitboard.py`**

```python
"""Bitwise flood-fill BFS over the 9x9 Quoridor board.

Bit index = row * 9 + col. Movement is computed with per-direction
"can move from here" masks derived from the wall sets, matching
core.rules.is_blocked exactly. BFS ignores the opponent pawn.
"""

N = 9
FULL = (1 << (N * N)) - 1


def _bit(c, r):
    return 1 << (r * N + c)


# Static per-row / per-col masks.
_ROW = [sum(_bit(c, r) for c in range(N)) for r in range(N)]
_COL = [sum(_bit(c, r) for r in range(N)) for c in range(N)]

_GOAL_ROW_MASK = {0: _ROW[N - 1], 1: _ROW[0]}

# Source cells from which a step in a given direction is impossible due to the
# board edge (independent of walls).
_EDGE_N = _ROW[N - 1]        # cannot go north from top row
_EDGE_S = _ROW[0]            # cannot go south from bottom row
_EDGE_E = _COL[N - 1]        # cannot go east from right col
_EDGE_W = _COL[0]            # cannot go west from left col


def _can_move_masks(state):
    """Return (canN, canS, canE, canW): for each direction, the set of source
    cells from which that one-step move is allowed (edges + walls)."""
    blockN = blockS = blockE = blockW = 0
    for (a, b) in state.h_walls:
        blockN |= _bit(a, b) | _bit(a + 1, b)            # N from (a,b),(a+1,b)
        blockS |= _bit(a, b + 1) | _bit(a + 1, b + 1)    # S from (a,b+1),(a+1,b+1)
    for (a, b) in state.v_walls:
        blockE |= _bit(a, b) | _bit(a, b + 1)            # E from (a,b),(a,b+1)
        blockW |= _bit(a + 1, b) | _bit(a + 1, b + 1)    # W from (a+1,b),(a+1,b+1)
    canN = FULL & ~_EDGE_N & ~blockN
    canS = FULL & ~_EDGE_S & ~blockS
    canE = FULL & ~_EDGE_E & ~blockE
    canW = FULL & ~_EDGE_W & ~blockW
    return canN, canS, canE, canW


def _expand(frontier, masks):
    canN, canS, canE, canW = masks
    n = (frontier & canN) << N
    s = (frontier & canS) >> N
    e = (frontier & canE) << 1
    w = (frontier & canW) >> 1
    return (n | s | e | w) & FULL


def bfs_dist(state, player):
    """Shortest path length (in steps) from the player's pawn to its goal row,
    ignoring the opponent. None if unreachable. Matches shortest_path_len."""
    c, r = state.pawns[player]
    goal = _GOAL_ROW_MASK[player]
    start = _bit(c, r)
    if start & goal:
        return 0
    masks = _can_move_masks(state)
    visited = start
    frontier = start
    dist = 0
    while frontier:
        nxt = _expand(frontier, masks) & ~visited
        if not nxt:
            return None
        dist += 1
        if nxt & goal:
            return dist
        visited |= nxt
        frontier = nxt
    return None


def path_exists(state, player):
    return bfs_dist(state, player) is not None
```

- [ ] **Step 4: Run, verify pass.** **Step 5: Commit** `feat: bitwise flood-fill BFS for the board`.

---

## Task 2: Rewire core + equivalence test (the safety net)

**Files:** MODIFY `core/rules.py`, test `tests/test_bitboard.py` (add equivalence).

- [ ] **Step 1: Add the equivalence test** to `tests/test_bitboard.py`

```python
def test_equivalence_with_reference_over_random_states():
    import random
    from core.state import initial_state
    from core.rules import (
        legal_moves, apply_move, is_terminal, shortest_path_len,
        has_path_to_goal, _shortest_path_len_ref,
    )
    rng = random.Random(12345)
    checked = 0
    for game in range(60):
        s = initial_state()
        for _ in range(60):
            if is_terminal(s):
                break
            for p in (0, 1):
                # public (bitboard) path must equal the pure-Python reference
                assert shortest_path_len(s, p) == _shortest_path_len_ref(s, p)
                assert has_path_to_goal(s, p) == (_shortest_path_len_ref(s, p) is not None)
                checked += 1
            s = apply_move(s, rng.choice(legal_moves(s)))
    assert checked > 1000      # sanity: we actually exercised many states
```

- [ ] **Step 2: Run, verify it fails** — `_shortest_path_len_ref` does not exist yet → ImportError.

- [ ] **Step 3: Modify `core/rules.py`.** Rename the CURRENT `shortest_path_len` body to a private reference, and make the public functions use the bitboard path.

Replace the existing:
```python
def shortest_path_len(state, player):
    """BFS distance ... """
    start = state.pawns[player]
    ...
    return None


def has_path_to_goal(state, player):
    return shortest_path_len(state, player) is not None
```
with:
```python
def _shortest_path_len_ref(state, player):
    """Pure-Python reference BFS (kept for equivalence testing/benchmarks)."""
    start = state.pawns[player]
    target = goal_row(player)
    if start[1] == target:
        return 0
    seen = {start}
    queue = deque([(start, 0)])
    while queue:
        cell, dist = queue.popleft()
        for dx, dy in DIRS:
            nxt = (cell[0] + dx, cell[1] + dy)
            if not on_board(nxt) or nxt in seen:
                continue
            if is_blocked(state, cell, nxt):
                continue
            if nxt[1] == target:
                return dist + 1
            seen.add(nxt)
            queue.append((nxt, dist + 1))
    return None


def shortest_path_len(state, player):
    """BFS distance from player's pawn to its goal row, ignoring the opponent.
    Returns None if no path exists. (Bitboard flood-fill; equivalent to the
    pure-Python reference, verified in tests/test_bitboard.py.)"""
    from core.bitboard import bfs_dist
    return bfs_dist(state, player)


def has_path_to_goal(state, player):
    from core.bitboard import path_exists
    return path_exists(state, player)
```
(Keep the `deque` import and `DIRS` — `_shortest_path_len_ref` still uses them. Do not change any other function. `is_blocked`, `legal_steps`, `legal_walls`, `legal_moves`, `apply_move`, etc. stay exactly as-is — `legal_walls` automatically gets faster because it calls `has_path_to_goal`.)

- [ ] **Step 4: Run the equivalence test + the full core/agent suites:**
  - `pytest tests/test_bitboard.py -q` → all pass (incl. equivalence over >1000 checks).
  - `pytest tests/ -q -k "not server and not az and not mcts and not minimax"` → fast core/agent tests still green.
  - Then the FULL suite `pytest -q` → all 114 + new tests pass.
  If the equivalence test finds ANY mismatch, the bitboard math is wrong — fix `core/bitboard.py` (do not weaken the assertion). The reference is the ground truth.
- [ ] **Step 5: Commit** `perf: route shortest-path through bitboard BFS behind frozen API`.

---

## Task 3: Benchmark (informational)

**Files:** Create `scripts/bench_core.py`.

- [ ] **Step 1: Implement `scripts/bench_core.py`**

```python
"""Benchmark the bitboard BFS vs the pure-Python reference. Informational.
Usage: python scripts/bench_core.py"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
from core.state import initial_state
from core.rules import (
    legal_moves, apply_move, is_terminal, legal_walls,
    shortest_path_len, _shortest_path_len_ref,
)
from core.bitboard import bfs_dist


def sample_states(n=300, seed=7):
    rng = random.Random(seed)
    states, s = [], initial_state()
    while len(states) < n:
        if is_terminal(s):
            s = initial_state()
            continue
        states.append(s)
        s = apply_move(s, rng.choice(legal_moves(s)))
    return states


def time_it(fn, states, reps=20):
    t0 = time.monotonic()
    for _ in range(reps):
        for s in states:
            fn(s, 0)
            fn(s, 1)
    return time.monotonic() - t0


def main():
    states = sample_states()
    ref = time_it(_shortest_path_len_ref, states)
    fast = time_it(bfs_dist, states)
    print(f"shortest-path over {len(states)} states x20 reps:")
    print(f"  reference (pure-Python BFS): {ref:.3f}s")
    print(f"  bitboard flood-fill:         {fast:.3f}s")
    print(f"  speedup: {ref / fast:.2f}x")
    # legal_walls throughput (path-check dominated)
    t0 = time.monotonic()
    for s in states[:60]:
        legal_walls(s)
    print(f"legal_walls over 60 states: {time.monotonic() - t0:.3f}s "
          f"(now uses bitboard has_path)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it** (`python scripts/bench_core.py`) and confirm it prints a speedup > 1x. **Step 3: Commit** `chore: core BFS benchmark script`.

---

## Done criteria
- `pytest -q` fully green (all 114 prior tests + new bitboard tests, including equivalence over >1000 checks).
- `shortest_path_len` / `has_path_to_goal` are bitboard-backed; `GameState` shape and all public signatures unchanged; `agents/`, `server/`, `web/` untouched.
- `scripts/bench_core.py` shows a measurable speedup of the BFS hot path.
- The pure-Python reference remains as `_shortest_path_len_ref` for ongoing equivalence testing.

## Note
This optimizes the dominant hot path (BFS, which drives `legal_walls` and every search heuristic) with the lowest-risk change: same `GameState`, same API, a permanent equivalence oracle. `legal_steps`/`is_blocked` are left as-is (cheap per call); they can be bit-accelerated later behind the same API if profiling warrants.
