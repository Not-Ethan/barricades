import numpy as np

from agents.az.diagnostics import policy_diagnostics, WALL_ACTION_START


def _ex(pi):
    """Minimal example tuple; only pi (index 1) is read by diagnostics."""
    return (np.zeros((6, 9, 9), np.float32), np.asarray(pi, np.float32), 0.0,
            np.zeros(4, np.float32))


def test_empty():
    d = policy_diagnostics([])
    assert d == dict(n=0, wall_mass=0.0, wall_argmax_rate=0.0, entropy=0.0)


def test_all_pawn_moves_zero_wall_mass():
    # all mass on action 0 (a pawn move)
    pi = np.zeros(140); pi[0] = 1.0
    d = policy_diagnostics([_ex(pi), _ex(pi)])
    assert d["n"] == 2
    assert d["wall_mass"] == 0.0
    assert d["wall_argmax_rate"] == 0.0
    assert d["entropy"] == 0.0  # deterministic policy -> zero entropy


def test_all_wall_full_mass_and_argmax():
    pi = np.zeros(140); pi[WALL_ACTION_START] = 1.0  # a wall action
    d = policy_diagnostics([_ex(pi)])
    assert d["wall_mass"] == 1.0
    assert d["wall_argmax_rate"] == 1.0


def test_half_wall_mass_and_unnormalized_input():
    # unnormalized counts: 5 on a pawn move, 5 on a wall move -> 0.5 wall mass
    pi = np.zeros(140); pi[1] = 5.0; pi[WALL_ACTION_START + 3] = 5.0
    d = policy_diagnostics([_ex(pi)])
    assert abs(d["wall_mass"] - 0.5) < 1e-9
    # argmax tie -> numpy picks first (the pawn move at index 1), so not a wall
    assert d["wall_argmax_rate"] == 0.0


def test_entropy_uniform():
    pi = np.ones(140) / 140.0
    d = policy_diagnostics([_ex(pi)])
    assert abs(d["entropy"] - np.log(140)) < 1e-3
