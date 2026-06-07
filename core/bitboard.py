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
