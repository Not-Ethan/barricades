from dataclasses import dataclass
from core.coords import N


@dataclass(frozen=True)
class Step:
    to_cell: tuple  # (col, row)


@dataclass(frozen=True)
class Wall:
    c: int          # anchor col, 0..N-2
    r: int          # anchor row, 0..N-2
    orient: str     # "H" or "V"


# A Move is a Step or a Wall.

@dataclass(frozen=True)
class GameState:
    pawns: tuple            # ((c, r), (c, r)) for players 0 and 1
    h_walls: frozenset      # set of (c, r) horizontal wall anchors
    v_walls: frozenset      # set of (c, r) vertical wall anchors
    walls_left: tuple       # (n0, n1)
    turn: int               # 0 or 1


def goal_row(player):
    return N - 1 if player == 0 else 0


def initial_state():
    mid = N // 2
    return GameState(
        pawns=((mid, 0), (mid, N - 1)),
        h_walls=frozenset(),
        v_walls=frozenset(),
        walls_left=(10, 10),
        turn=0,
    )
