import numpy as np
from smallboard.engine import Step, Wall

_DIRS = [
    (0, 1), (0, -1), (1, 0), (-1, 0),
    (0, 2), (0, -2), (2, 0), (-2, 0),
    (1, 1), (-1, 1), (1, -1), (-1, -1),
]
_DIR_INDEX = {d: i for i, d in enumerate(_DIRS)}


class Encoder:
    """6xNxN planes + (12 + 2*(N-1)^2) canonical actions, current-player-relative."""

    def __init__(self, engine):
        self.N = engine.N
        self.W = engine.W
        self.anchors = self.N - 1
        self.n_actions = 12 + 2 * self.anchors ** 2

    def _flip(self, s):
        return s.turn == 1

    def _cf_cell(self, cell, flip):
        c, r = cell
        return (c, (self.N - 1 - r) if flip else r)

    def _cf_wall(self, c, r, flip):
        return (c, (self.N - 2 - r) if flip else r)

    def encode_planes(self, s):
        N = self.N
        flip = self._flip(s)
        me = s.pawns[s.turn]
        opp = s.pawns[1 - s.turn]
        planes = np.zeros((6, N, N), dtype=np.float32)
        mc = self._cf_cell(me, flip)
        oc = self._cf_cell(opp, flip)
        planes[0, mc[1], mc[0]] = 1.0
        planes[1, oc[1], oc[0]] = 1.0
        for (c, r) in s.h_walls:
            cc, cr = self._cf_wall(c, r, flip)
            planes[2, cr, cc] = 1.0
        for (c, r) in s.v_walls:
            cc, cr = self._cf_wall(c, r, flip)
            planes[3, cr, cc] = 1.0
        planes[4, :, :] = s.walls_left[s.turn] / max(1, self.W)
        planes[5, :, :] = s.walls_left[1 - s.turn] / max(1, self.W)
        return planes

    def move_to_action(self, m, s):
        flip = self._flip(s)
        if isinstance(m, Step):
            me = self._cf_cell(s.pawns[s.turn], flip)
            dest = self._cf_cell(m.to_cell, flip)
            return _DIR_INDEX[(dest[0] - me[0], dest[1] - me[1])]
        cc, cr = self._cf_wall(m.c, m.r, flip)
        off = 0 if m.orient == "H" else self.anchors ** 2
        return 12 + off + cr * self.anchors + cc

    def action_to_move(self, idx, s):
        flip = self._flip(s)
        if idx < 12:
            dx, dy = _DIRS[idx]
            me = self._cf_cell(s.pawns[s.turn], flip)
            real = self._cf_cell((me[0] + dx, me[1] + dy), flip)
            return Step(real)
        a = idx - 12
        orient = "H" if a < self.anchors ** 2 else "V"
        a %= self.anchors ** 2
        cr, cc = divmod(a, self.anchors)
        real_c, real_r = self._cf_wall(cc, cr, flip)
        return Wall(real_c, real_r, orient)

    def legal_action_mask(self, s, engine):
        mask = np.zeros(self.n_actions, dtype=np.float32)
        for m in engine.legal_moves(s):
            mask[self.move_to_action(m, s)] = 1.0
        return mask
