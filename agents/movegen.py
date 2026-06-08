"""relevant_walls / relevant_moves — move-generation pruning primitive.

A wall is *relevant* iff placing it strictly increases the opponent's
shortest-path distance to their goal row.  Walls that cannot possibly
lengthen that path are pruned early, reducing the branching factor at
every search node.

Implementation strategy: path-adjacent filtering (v1)
------------------------------------------------------
1.  Run a BFS from the opponent's pawn and recover one shortest path.
2.  Restrict candidate walls to those whose 2-segment span touches at
    least one cell on that path (the "path-adjacent" set).
3.  For each candidate, apply the wall and recheck the opponent's BFS
    distance; keep only those that strictly increase it.

Step 2 is free (set intersection); step 3 requires at most
~path-length * 4 BFS calls instead of ~128.  In practice this is
20-40 candidate walls on an open board.

This file does NOT modify core/, existing agents, registry, or
strength.py.
"""

from collections import deque

from core.coords import N
from core.rules import legal_steps, legal_walls, apply_move, shortest_path_len, is_blocked
from core.state import Step, Wall

# Type alias for clarity (a Move is a Step or Wall).
Move = object

_DIRS = [(0, 1), (0, -1), (1, 0), (-1, 0)]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _shortest_path_cells(state, player):
    """Return the set of cells on *one* shortest path from *player*'s pawn to
    its goal row, or an empty set if no path exists.

    Uses a standard BFS with parent tracking; backtracks from the first
    goal-row cell reached to recover the path.
    """
    from core.state import goal_row as _goal_row
    goal = _goal_row(player)
    start = state.pawns[player]

    if start[1] == goal:
        return {start}

    # BFS – parent map tracks one predecessor per cell (sufficient for a
    # single shortest path).
    parent = {start: None}
    queue = deque([(start, 0)])
    goal_cell = None
    goal_dist = None

    while queue:
        cell, dist = queue.popleft()
        # Once we've exhausted the goal-distance layer, stop.
        if goal_dist is not None and dist > goal_dist:
            break
        for dx, dy in _DIRS:
            nxt = (cell[0] + dx, cell[1] + dy)
            if not (0 <= nxt[0] < N and 0 <= nxt[1] < N):
                continue
            if nxt in parent:
                continue
            if is_blocked(state, cell, nxt):
                continue
            parent[nxt] = cell
            if nxt[1] == goal:
                if goal_cell is None:
                    goal_cell = nxt
                    goal_dist = dist + 1
                # Do NOT break – drain the rest of this BFS layer so that
                # parent entries for all goal-row cells are recorded (though
                # we only backtrack one path).
            elif goal_dist is None:
                queue.append((nxt, dist + 1))

    if goal_cell is None:
        return set()

    # Backtrack from the first goal cell to the start.
    path: set = set()
    cell = goal_cell
    while cell is not None:
        path.add(cell)
        cell = parent[cell]
    return path


def _wall_footprint(wall: Wall):
    """The four board cells adjacent to (and potentially blocked by) *wall*.

    An H-wall at (c, r) blocks the horizontal edge between rows r and r+1
    for columns c and c+1.  The four neighbouring cells are therefore:
        (c, r), (c+1, r), (c, r+1), (c+1, r+1).

    A V-wall at (c, r) blocks the vertical edge between columns c and c+1
    for rows r and r+1.  Same four cells by symmetry.
    """
    c, r = wall.c, wall.r
    return {(c, r), (c + 1, r), (c, r + 1), (c + 1, r + 1)}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def relevant_walls(state) -> list:
    """Return the subset of legal_walls(state) that strictly increases the
    opponent's shortest-path distance to their goal row.

    A wall that does not lengthen the opponent's optimal path is "pointless"
    and is excluded.  The returned list is always a subset of legal_walls.

    Performance: path-adjacent pre-filtering reduces BFS calls from ~128
    (brute force) to ~20-40, giving ~0.2–1 ms per call in pure Python.
    """
    opp = 1 - state.turn
    base_dist = shortest_path_len(state, opp)
    if base_dist is None:
        # Opponent has no path (shouldn't happen in a legal game, but be safe).
        return []

    # Step 1: recover one shortest path for the opponent.
    path_cells = _shortest_path_cells(state, opp)
    if not path_cells:
        return []

    # Step 2: collect only candidate walls whose footprint overlaps the path.
    candidates = [
        w for w in legal_walls(state)
        if _wall_footprint(w) & path_cells
    ]

    # Step 3: keep only those that genuinely increase the opponent's distance.
    result = []
    for w in candidates:
        s2 = apply_move(state, w)
        new_dist = shortest_path_len(s2, opp)
        # new_dist should never be None here (legal_walls guarantees both
        # players retain a path), but guard defensively.
        if new_dist is not None and new_dist > base_dist:
            result.append(w)
    return result


def relevant_moves(state) -> list:
    """All legal step moves plus the relevant (path-lengthening) wall moves.

    Equivalent to legal_moves(state) but with pointless walls pruned.
    """
    return [Step(c) for c in legal_steps(state)] + relevant_walls(state)
