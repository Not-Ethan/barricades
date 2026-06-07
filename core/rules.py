from collections import deque

from core.coords import N, on_board
from core.state import Step, Wall, GameState, goal_row

DIRS = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # up, down, right, left


def is_blocked(state, a, b):
    """True if the edge between adjacent cells a and b is blocked by a wall."""
    (ax, ay), (bx, by) = a, b
    dx, dy = bx - ax, by - ay
    if dy == 1:    # a -> up
        return (ax, ay) in state.h_walls or (ax - 1, ay) in state.h_walls
    if dy == -1:   # a -> down (edge sits at row by)
        return (ax, by) in state.h_walls or (ax - 1, by) in state.h_walls
    if dx == 1:    # a -> right
        return (ax, ay) in state.v_walls or (ax, ay - 1) in state.v_walls
    if dx == -1:   # a -> left (edge sits at col bx)
        return (bx, ay) in state.v_walls or (bx, ay - 1) in state.v_walls
    raise ValueError(f"cells {a} and {b} are not orthogonally adjacent")


def legal_steps(state):
    """Destination cells the current player's pawn may step to (incl. jumps)."""
    me = state.pawns[state.turn]
    opp = state.pawns[1 - state.turn]
    dests = []
    for dx, dy in DIRS:
        adj = (me[0] + dx, me[1] + dy)
        if not on_board(adj) or is_blocked(state, me, adj):
            continue
        if adj != opp:
            dests.append(adj)
            continue
        # Opponent is adjacent in this direction: attempt a jump.
        landing = (opp[0] + dx, opp[1] + dy)
        if on_board(landing) and not is_blocked(state, opp, landing):
            dests.append(landing)  # straight jump
        else:
            # Straight jump blocked: diagonal jumps perpendicular to dir.
            for px, py in DIRS:
                if (px, py) == (dx, dy) or (px, py) == (-dx, -dy):
                    continue  # skip same/opposite axis
                diag = (opp[0] + px, opp[1] + py)
                if on_board(diag) and not is_blocked(state, opp, diag):
                    dests.append(diag)
    return dests


def shortest_path_len(state, player):
    """BFS distance from player's pawn to its goal row, ignoring the opponent.
    Returns None if no path exists."""
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


def has_path_to_goal(state, player):
    return shortest_path_len(state, player) is not None


def _with_wall(state, wall):
    """Return a copy of state with `wall` added (no turn change, no decrement)."""
    if wall.orient == "H":
        return GameState(state.pawns, state.h_walls | {(wall.c, wall.r)},
                         state.v_walls, state.walls_left, state.turn)
    return GameState(state.pawns, state.h_walls,
                     state.v_walls | {(wall.c, wall.r)},
                     state.walls_left, state.turn)


def _overlaps(state, wall):
    """True if `wall` overlaps or crosses an existing wall."""
    c, r = wall.c, wall.r
    if wall.orient == "H":
        return ((c, r) in state.h_walls
                or (c - 1, r) in state.h_walls
                or (c + 1, r) in state.h_walls
                or (c, r) in state.v_walls)        # crossing
    return ((c, r) in state.v_walls
            or (c, r - 1) in state.v_walls
            or (c, r + 1) in state.v_walls
            or (c, r) in state.h_walls)            # crossing


def legal_walls(state):
    """All legal wall placements for the current player."""
    if state.walls_left[state.turn] <= 0:
        return []
    result = []
    for orient in ("H", "V"):
        for c in range(N - 1):
            for r in range(N - 1):
                w = Wall(c, r, orient)
                if _overlaps(state, w):
                    continue
                s2 = _with_wall(state, w)
                if has_path_to_goal(s2, 0) and has_path_to_goal(s2, 1):
                    result.append(w)
    return result


def legal_moves(state):
    """All legal moves (Steps and Walls) for the current player."""
    return [Step(c) for c in legal_steps(state)] + legal_walls(state)


def apply_move(state, move):
    """Return the new state after applying `move`. Assumes the move is legal."""
    if isinstance(move, Step):
        pawns = list(state.pawns)
        pawns[state.turn] = move.to_cell
        return GameState(tuple(pawns), state.h_walls, state.v_walls,
                         state.walls_left, 1 - state.turn)
    # Wall
    left = list(state.walls_left)
    left[state.turn] -= 1
    if move.orient == "H":
        h = state.h_walls | {(move.c, move.r)}
        v = state.v_walls
    else:
        h = state.h_walls
        v = state.v_walls | {(move.c, move.r)}
    return GameState(state.pawns, h, v, tuple(left), 1 - state.turn)


def is_terminal(state):
    return winner(state) is not None


def winner(state):
    for p in (0, 1):
        if state.pawns[p][1] == goal_row(p):
            return p
    return None
