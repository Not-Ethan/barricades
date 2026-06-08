import random
import numpy as np
from core.state import GameState, Step, Wall, initial_state
from core import rules
from agents.az import encoding as enc
from agents.az.train import mirror_planes, LR_PERM, augment_lr


def lr_mirror_state(s):
    pawns = tuple((8 - c, r) for (c, r) in s.pawns)
    h = frozenset((7 - c, r) for (c, r) in s.h_walls)
    v = frozenset((7 - c, r) for (c, r) in s.v_walls)
    return GameState(pawns, h, v, s.walls_left, s.turn)


def lr_mirror_move(m):
    if isinstance(m, Step):
        return Step((8 - m.to_cell[0], m.to_cell[1]))
    return Wall(7 - m.c, m.r, m.orient)


def test_perm_is_involution():
    assert np.array_equal(LR_PERM[LR_PERM], np.arange(140))


def test_planes_mirror_commutes_with_encoding():
    rng = random.Random(5)
    checked = 0
    for _ in range(40):
        s = initial_state()
        for _ in range(60):
            if rules.is_terminal(s):
                break
            got = mirror_planes(enc.encode_planes(s))
            want = enc.encode_planes(lr_mirror_state(s))
            assert np.array_equal(got, want), f"plane mirror mismatch at {s}"
            for m in rules.legal_moves(s):
                a = enc.move_to_action(m, s)
                a_mir = enc.move_to_action(lr_mirror_move(m), lr_mirror_state(s))
                assert LR_PERM[a] == a_mir
            checked += 1
            s = rules.apply_move(s, rng.choice(rules.legal_moves(s)))
    assert checked > 1500


def test_augment_lr_doubles_and_preserves_z_feats():
    s = initial_state()
    planes = enc.encode_planes(s)
    pi = np.zeros(140, dtype=np.float32); pi[enc.move_to_action(Step((4, 1)), s)] = 1.0
    ex = [(planes, pi, 1.0, np.array([3.0, 5, 5, 7], dtype=np.float32))]
    out = augment_lr(ex)
    assert len(out) == 2
    orig_a = int(np.argmax(out[0][1])); mir_a = int(np.argmax(out[1][1]))
    assert LR_PERM[orig_a] == mir_a
    assert out[1][2] == 1.0
    assert np.array_equal(np.asarray(out[1][3]), np.asarray(out[0][3]))
