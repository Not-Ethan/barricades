from collections import deque

from core.coords import N, on_board
from core.state import Step, Wall, GameState, goal_row


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
