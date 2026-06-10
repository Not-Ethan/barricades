import numpy as np
import torch

from agents.az.model import QuoridorNet
from agents.native_agent import _to_native
from core.state import initial_state
from scripts.selfplay_pool import run_selfplay_pool, _features


def _net(seed=0):
    torch.manual_seed(seed)
    return QuoridorNet(channels=32, blocks=3).eval()


def test_features_initial_state():
    # symmetric start: both pawns 8 rows from goal, 10 walls each -> [0, 10, 10, 0]
    assert _features(_to_native(initial_state())) == [0.0, 10.0, 10.0, 0.0]


def test_pure_selfplay_wellformed():
    ex, st = run_selfplay_pool(total_games=8, n_games=8, sims=20, device="cpu",
                               learner_net=_net(), opponent_nets=None, seed=1, max_plies=40)
    assert st["games"] == 8 and len(ex) == st["examples"] > 0
    planes, pi, z, feats = ex[0]
    assert planes.shape == (6, 9, 9) and pi.shape == (140,) and feats.shape == (4,)
    assert all(e[2] in (-1.0, 0.0, 1.0) for e in ex)        # z is a game outcome
    assert all(e[3][3] >= 1.0 for e in ex)                  # plies_to_end >= 1


def test_pool_records_only_learner_side():
    learner, opp = _net(0), _net(7)
    ex_self, _ = run_selfplay_pool(total_games=12, n_games=12, sims=20, device="cpu",
                                   learner_net=learner, opponent_nets=None, seed=2, max_plies=40)
    ex_pool, _ = run_selfplay_pool(total_games=12, n_games=12, sims=20, device="cpu",
                                   learner_net=learner, opponent_nets=[opp], pool_frac=1.0,
                                   seed=2, max_plies=40)
    # learner-vs-pool games record only the learner's ~half of plies
    assert 0 < len(ex_pool) < 0.7 * len(ex_self)


def test_engine_equivalence_vs_native():
    """Pure self-play through this driver must statistically match the native pool
    with the same net -> the pool arms vary only the opponent, not the engine."""
    from scripts.selfplay_native import run_selfplay
    net = _net(3)
    n = 32
    ex_pool, _ = run_selfplay_pool(total_games=n, n_games=n, sims=40, device="cpu",
                                   learner_net=net, opponent_nets=None, seed=5, max_plies=80)
    ex_nat, _ = run_selfplay(total_games=n, n_games=n, sims=40, device="cpu",
                             net=net, seed=5, max_plies=80)
    gl_pool, gl_nat = len(ex_pool) / n, len(ex_nat) / n
    assert abs(gl_pool - gl_nat) / gl_nat < 0.20, (gl_pool, gl_nat)
