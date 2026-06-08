import random
import numpy as np
import barricades_native as bn
from core.state import GameState, Step, initial_state
from core import rules
from tests.test_native_game import to_native, mv_to_tuple


def _state(p0, p1, wl=(10, 10), turn=0, h=(), v=()):
    return GameState((p0, p1), frozenset(h), frozenset(v), wl, turn)


def test_tree_returns_legal_move_heuristic():
    t = bn.Tree(to_native(initial_state()), 1.5, 0)
    mv = t.run_heuristic(120)
    assert mv in {mv_to_tuple(m) for m in rules.legal_moves(initial_state())}


def test_tree_takes_immediate_win():
    # opponent placed off the goal rows so the root is not already terminal
    # (player 1's goal is row 0, so (0, 0) would mean p1 has already won)
    s = _state((4, 7), (0, 4))
    t = bn.Tree(to_native(s), 1.5, 0)
    mv = t.run_heuristic(160)
    assert mv == ("step", 4, 8)


def test_prepare_receive_protocol_runs():
    s = initial_state()
    t = bn.Tree(to_native(s), 1.5, 1)
    evals, guard = 0, 0
    while evals < 64 and guard < 512:
        guard += 1
        planes = t.prepare_leaf()
        if planes is None:
            continue
        policy = np.full(140, 1.0 / 140, dtype=np.float32)
        t.receive(policy, 0.0)
        evals += 1
    mv, pi = t.best_move(0.0)
    assert mv in {mv_to_tuple(m) for m in rules.legal_moves(s)}
    pi = np.asarray(pi, dtype=np.float32)
    assert pi.shape == (140,)
    assert abs(float(pi.sum()) - 1.0) < 1e-4


def test_native_agent_beats_random():
    from agents.native_agent import NativeMctsAgent
    from agents.random_agent import RandomAgent
    wins = 0
    for g in range(20):
        a, b = NativeMctsAgent(sims=120, seed=g), RandomAgent(seed=1000 + g)
        players = (a, b) if g % 2 == 0 else (b, a)
        s = initial_state()
        for _ in range(300):
            if rules.is_terminal(s):
                break
            s = rules.apply_move(s, players[s.turn].select_move(s))
        w = rules.winner(s)
        if (w == 0 and g % 2 == 0) or (w == 1 and g % 2 == 1):
            wins += 1
    assert wins >= 16
