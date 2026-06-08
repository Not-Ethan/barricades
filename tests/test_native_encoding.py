import random
import numpy as np
import barricades_native as bn
from core.state import initial_state
from core import rules
from agents.az import encoding as enc
from tests.test_native_game import to_native, mv_to_tuple


def test_planes_match_initial():
    s = initial_state()
    got = bn.encode_planes(to_native(s))
    want = enc.encode_planes(s)
    assert got.shape == (6, 9, 9)
    assert got.dtype == np.float32
    assert np.array_equal(got, want)


def test_encoding_differential_over_random_games():
    rng = random.Random(7)
    checked = 0
    for _ in range(60):
        s = initial_state()
        for _ in range(80):
            if rules.is_terminal(s):
                break
            ns = to_native(s)
            assert np.array_equal(bn.encode_planes(ns), enc.encode_planes(s))
            for m in rules.legal_moves(s):
                mt = mv_to_tuple(m)
                idx = bn.move_to_action(mt, ns)
                assert idx == enc.move_to_action(m, s)
                assert bn.move_to_action(bn.action_to_move(idx, ns), ns) == idx
            checked += 1
            s = rules.apply_move(s, rng.choice(rules.legal_moves(s)))
    assert checked > 1500
