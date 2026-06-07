import numpy as np

from core.state import GameState, Step, Wall, initial_state
from core.rules import legal_moves
from agents.az.encoding import (
    N_ACTIONS, N_PLANES, encode_planes, move_to_action, action_to_move,
    legal_action_mask, canonical_flip,
)


def _state(p0, p1, wl=(10, 10), turn=0, h=(), v=()):
    return GameState((p0, p1), frozenset(h), frozenset(v), wl, turn)


def test_constants():
    assert N_ACTIONS == 140
    assert N_PLANES == 6


def test_planes_shape_and_pawns_player0():
    s = initial_state()
    planes = encode_planes(s)
    assert planes.shape == (N_PLANES, 9, 9)
    # turn 0: no flip. my pawn (p0) at (4,0); opp (p1) at (4,8).
    assert planes[0, 0, 4] == 1 and planes[0].sum() == 1     # my pawn plane (row,col)
    assert planes[1, 8, 4] == 1 and planes[1].sum() == 1     # opp pawn plane


def test_planes_canonicalized_for_player1():
    # turn 1: flip. my pawn = p1 at (4,8) -> canonical (4,0). opp p0 (4,0) -> (4,8).
    s = _state((4, 0), (4, 8), turn=1)
    planes = encode_planes(s)
    assert planes[0, 0, 4] == 1     # my pawn canonically at bottom (row 0)
    assert planes[1, 8, 4] == 1     # opp canonically at top


def test_wall_planes_and_canonical_flip():
    assert canonical_flip(_state((4, 0), (4, 8), turn=0)) is False
    assert canonical_flip(_state((4, 0), (4, 8), turn=1)) is True
    # H wall anchor (2,3), turn 0 -> plane2 at [3,2]
    s = _state((4, 0), (4, 8), h=[(2, 3)], turn=0)
    p = encode_planes(s)
    assert p[2, 3, 2] == 1 and p[2].sum() == 1
    # same wall, turn 1 (flip) -> canonical anchor (2, 7-3)=(2,4) -> plane2 at [4,2]
    s2 = _state((4, 0), (4, 8), h=[(2, 3)], turn=1)
    p2 = encode_planes(s2)
    assert p2[2, 4, 2] == 1


def test_walls_remaining_planes():
    s = _state((4, 0), (4, 8), wl=(7, 10), turn=0)
    p = encode_planes(s)
    assert np.allclose(p[4], 0.7)     # my walls remaining / 10
    assert np.allclose(p[5], 1.0)     # opp walls remaining / 10


def test_step_action_roundtrip_no_flip():
    s = initial_state()           # p0 at (4,0), turn 0
    mv = Step((4, 1))             # north step
    idx = move_to_action(mv, s)
    assert idx == 0               # N
    assert action_to_move(idx, s) == mv


def test_step_action_roundtrip_with_flip():
    s = _state((4, 0), (4, 8), turn=1)   # p1 at (4,8) to move, flip
    mv = Step((4, 7))                     # p1 steps toward its goal (row 0): real delta (0,-1)
    idx = move_to_action(mv, s)
    assert idx == 0                       # canonical N (advancing upward)
    assert action_to_move(idx, s) == mv   # inverse returns the REAL move


def test_wall_action_roundtrip_no_flip():
    s = initial_state()
    for w in [Wall(0, 0, "H"), Wall(7, 7, "H"), Wall(3, 5, "V"), Wall(0, 0, "V")]:
        idx = move_to_action(w, s)
        assert 12 <= idx < 140
        assert action_to_move(idx, s) == w


def test_wall_action_roundtrip_with_flip():
    s = _state((4, 0), (4, 8), turn=1)
    w = Wall(2, 3, "H")
    idx = move_to_action(w, s)
    # canonical anchor (2, 7-3)=(2,4): idx = 12 + 0 + 4*8 + 2 = 46
    assert idx == 46
    assert action_to_move(idx, s) == w     # inverse maps back to real (2,3,H)


def test_legal_mask_matches_legal_moves():
    for s in [initial_state(), _state((4, 4), (4, 5), turn=0),
              _state((4, 0), (4, 8), h=[(2, 3)], turn=1)]:
        mask = legal_action_mask(s)
        assert mask.shape == (N_ACTIONS,)
        assert mask.sum() == len(legal_moves(s))
        # every legal move's action index is unmasked, and bijective (no collisions)
        idxs = [move_to_action(m, s) for m in legal_moves(s)]
        assert len(set(idxs)) == len(idxs)        # bijective on legal moves
        for i in idxs:
            assert mask[i] == 1
