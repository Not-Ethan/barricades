import numpy as np
import random
from smallboard.engine import Engine, Step, Wall
from smallboard.encoding import Encoder


def test_action_count_and_roundtrip_3x3():
    e = Engine(3, 1)
    enc = Encoder(e)
    assert enc.n_actions == 12 + 2 * (3 - 1) ** 2   # 20
    s = e.initial_state()
    planes = enc.encode_planes(s)
    assert planes.shape == (6, 3, 3) and planes.dtype == np.float32
    for m in e.legal_moves(s):
        a = enc.move_to_action(m, s)
        assert 0 <= a < enc.n_actions
        assert enc.move_to_action(enc.action_to_move(a, s), s) == a


def test_roundtrip_over_random_games_5x5():
    e = Engine(5, 3)
    enc = Encoder(e)
    rng = random.Random(9)
    checked = 0
    for _ in range(30):
        s = e.initial_state()
        for _ in range(40):
            if e.is_terminal(s):
                break
            assert enc.encode_planes(s).shape == (6, 5, 5)
            for m in e.legal_moves(s):
                a = enc.move_to_action(m, s)
                assert enc.move_to_action(enc.action_to_move(a, s), s) == a
            ms = e.legal_moves(s)
            s = e.apply_move(s, ms[rng.randrange(len(ms))])
            checked += 1
    assert checked > 500
