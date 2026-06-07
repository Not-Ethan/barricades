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
