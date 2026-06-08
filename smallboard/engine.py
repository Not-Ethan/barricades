from collections import deque
from dataclasses import dataclass

DIRS = [(0, 1), (0, -1), (1, 0), (-1, 0)]


@dataclass(frozen=True)
class Step:
    to_cell: tuple


@dataclass(frozen=True)
class Wall:
    c: int
    r: int
    orient: str


@dataclass(frozen=True)
class State:
    pawns: tuple
    h_walls: frozenset
    v_walls: frozenset
    walls_left: tuple
    turn: int


class Engine:
    """Quoridor rules parameterized by board size N and walls-per-player W."""

    def __init__(self, N, W):
        self.N = N
        self.W = W

    def initial_state(self):
        N = self.N
        return State(((N // 2, 0), (N // 2, N - 1)), frozenset(), frozenset(),
                     (self.W, self.W), 0)

    def goal_row(self, p):
        return self.N - 1 if p == 0 else 0

    def on_board(self, c, r):
        return 0 <= c < self.N and 0 <= r < self.N

    def is_blocked(self, s, a, b):
        (ax, ay), (bx, by) = a, b
        dx, dy = bx - ax, by - ay
        if dy == 1:
            return (ax, ay) in s.h_walls or (ax - 1, ay) in s.h_walls
        if dy == -1:
            return (ax, by) in s.h_walls or (ax - 1, by) in s.h_walls
        if dx == 1:
            return (ax, ay) in s.v_walls or (ax, ay - 1) in s.v_walls
        return (bx, ay) in s.v_walls or (bx, ay - 1) in s.v_walls

    def legal_steps(self, s):
        me = s.pawns[s.turn]
        opp = s.pawns[1 - s.turn]
        dests = []
        for dx, dy in DIRS:
            adj = (me[0] + dx, me[1] + dy)
            if not self.on_board(*adj) or self.is_blocked(s, me, adj):
                continue
            if adj != opp:
                dests.append(adj)
                continue
            landing = (opp[0] + dx, opp[1] + dy)
            if self.on_board(*landing) and not self.is_blocked(s, opp, landing):
                dests.append(landing)
            else:
                for px, py in DIRS:
                    if (px, py) == (dx, dy) or (px, py) == (-dx, -dy):
                        continue
                    diag = (opp[0] + px, opp[1] + py)
                    if self.on_board(*diag) and not self.is_blocked(s, opp, diag):
                        dests.append(diag)
        return dests

    def shortest_path_len(self, s, p):
        start = s.pawns[p]
        target = self.goal_row(p)
        if start[1] == target:
            return 0
        seen = {start}
        q = deque([(start, 0)])
        while q:
            cell, d = q.popleft()
            for dx, dy in DIRS:
                nxt = (cell[0] + dx, cell[1] + dy)
                if not self.on_board(*nxt) or nxt in seen:
                    continue
                if self.is_blocked(s, cell, nxt):
                    continue
                if nxt[1] == target:
                    return d + 1
                seen.add(nxt)
                q.append((nxt, d + 1))
        return None

    def has_path(self, s, p):
        return self.shortest_path_len(s, p) is not None

    def _with_wall(self, s, w):
        if w.orient == "H":
            return State(s.pawns, s.h_walls | {(w.c, w.r)}, s.v_walls,
                         s.walls_left, s.turn)
        return State(s.pawns, s.h_walls, s.v_walls | {(w.c, w.r)},
                     s.walls_left, s.turn)

    def _overlaps(self, s, w):
        c, r = w.c, w.r
        if w.orient == "H":
            return ((c, r) in s.h_walls or (c - 1, r) in s.h_walls
                    or (c + 1, r) in s.h_walls or (c, r) in s.v_walls)
        return ((c, r) in s.v_walls or (c, r - 1) in s.v_walls
                or (c, r + 1) in s.v_walls or (c, r) in s.h_walls)

    def legal_walls(self, s):
        if s.walls_left[s.turn] <= 0:
            return []
        out = []
        for orient in ("H", "V"):
            for c in range(self.N - 1):
                for r in range(self.N - 1):
                    w = Wall(c, r, orient)
                    if self._overlaps(s, w):
                        continue
                    s2 = self._with_wall(s, w)
                    if self.has_path(s2, 0) and self.has_path(s2, 1):
                        out.append(w)
        return out

    def legal_moves(self, s):
        return [Step(c) for c in self.legal_steps(s)] + self.legal_walls(s)

    def apply_move(self, s, m):
        if isinstance(m, Step):
            pawns = list(s.pawns)
            pawns[s.turn] = m.to_cell
            return State(tuple(pawns), s.h_walls, s.v_walls, s.walls_left,
                         1 - s.turn)
        left = list(s.walls_left)
        left[s.turn] -= 1
        if m.orient == "H":
            return State(s.pawns, s.h_walls | {(m.c, m.r)}, s.v_walls,
                         tuple(left), 1 - s.turn)
        return State(s.pawns, s.h_walls, s.v_walls | {(m.c, m.r)},
                     tuple(left), 1 - s.turn)

    def winner(self, s):
        for p in (0, 1):
            if s.pawns[p][1] == self.goal_row(p):
                return p
        return None

    def is_terminal(self, s):
        return self.winner(s) is not None
