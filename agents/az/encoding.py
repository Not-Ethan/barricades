import numpy as np

from core.state import Step, Wall
from core.rules import legal_moves

N = 9
N_PLANES = 6
N_ACTIONS = 140

# 12 pawn directions (dx, dy) in canonical frame (N = +row).
_DIRS = [
    (0, 1), (0, -1), (1, 0), (-1, 0),       # 0..3 steps N S E W
    (0, 2), (0, -2), (2, 0), (-2, 0),       # 4..7 straight jumps
    (1, 1), (-1, 1), (1, -1), (-1, -1),     # 8..11 diagonal jumps
]
_DIR_INDEX = {d: i for i, d in enumerate(_DIRS)}


def canonical_flip(state):
    return state.turn == 1


def _cf_cell(cell, flip):
    c, r = cell
    return (c, (N - 1 - r) if flip else r)


def _cf_wall_anchor(c, r, flip):
    return (c, (N - 2 - r) if flip else r)   # (c, 7-r) under flip


def encode_planes(state):
    flip = canonical_flip(state)
    me = state.pawns[state.turn]
    opp = state.pawns[1 - state.turn]
    planes = np.zeros((N_PLANES, N, N), dtype=np.float32)
    mc = _cf_cell(me, flip)
    oc = _cf_cell(opp, flip)
    planes[0, mc[1], mc[0]] = 1.0          # [plane, row, col]
    planes[1, oc[1], oc[0]] = 1.0
    for (c, r) in state.h_walls:
        cc, cr = _cf_wall_anchor(c, r, flip)
        planes[2, cr, cc] = 1.0
    for (c, r) in state.v_walls:
        cc, cr = _cf_wall_anchor(c, r, flip)
        planes[3, cr, cc] = 1.0
    planes[4, :, :] = state.walls_left[state.turn] / 10.0
    planes[5, :, :] = state.walls_left[1 - state.turn] / 10.0
    return planes


def move_to_action(move, state):
    flip = canonical_flip(state)
    if isinstance(move, Step):
        me = _cf_cell(state.pawns[state.turn], flip)
        dest = _cf_cell(move.to_cell, flip)
        d = (dest[0] - me[0], dest[1] - me[1])
        return _DIR_INDEX[d]
    # Wall
    cc, cr = _cf_wall_anchor(move.c, move.r, flip)
    off = 0 if move.orient == "H" else 64
    return 12 + off + cr * 8 + cc


def action_to_move(idx, state):
    """Inverse of move_to_action, returning the REAL move."""
    flip = canonical_flip(state)
    if idx < 12:
        dx, dy = _DIRS[idx]
        me = _cf_cell(state.pawns[state.turn], flip)
        cdest = (me[0] + dx, me[1] + dy)
        real_dest = _cf_cell(cdest, flip)   # flip is its own inverse
        return Step(real_dest)
    a = idx - 12
    orient = "H" if a < 64 else "V"
    a %= 64
    cr, cc = divmod(a, 8)
    real_c, real_r = _cf_wall_anchor(cc, cr, flip)   # flip is its own inverse
    return Wall(real_c, real_r, orient)


def legal_action_mask(state):
    mask = np.zeros(N_ACTIONS, dtype=np.float32)
    for m in legal_moves(state):
        mask[move_to_action(m, state)] = 1.0
    return mask
