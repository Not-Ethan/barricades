import numpy as np
import barricades_native as bn


def _uniform_eval(planes):
    b = planes.shape[0]
    policy = np.full((b, 140), 1.0 / 140, dtype=np.float32)
    value = np.zeros(b, dtype=np.float32)
    return policy, value


def test_pool_produces_wellformed_examples_and_drains_all_games():
    pool = bn.SelfPlayPool(
        n_games=8, total_games=8, sims=16, c_puct=1.5, seed=0,
        dirichlet_alpha=0.5, dirichlet_eps=0.25, temp_moves=4, max_plies=120,
    )
    examples = []
    guard = 0
    while pool.games_remaining() > 0 and guard < 200_000:
        guard += 1
        planes = pool.step()
        if planes is None:
            continue
        planes = np.asarray(planes, dtype=np.float32)
        assert planes.ndim == 4 and planes.shape[1:] == (6, 9, 9)
        assert planes.shape[0] >= 1
        policy, value = _uniform_eval(planes)
        pool.feed(policy, value)
        examples.extend(pool.drain())
    assert pool.games_remaining() == 0
    assert len(examples) > 0
    for planes, pi, z, feats in examples:
        planes = np.asarray(planes, dtype=np.float32)
        pi = np.asarray(pi, dtype=np.float32)
        feats = np.asarray(feats, dtype=np.float32)
        assert planes.shape == (6, 9, 9)
        assert pi.shape == (140,)
        assert abs(float(pi.sum()) - 1.0) < 1e-4
        assert z in (-1.0, 0.0, 1.0)
        assert feats.shape == (4,)  # path_diff, walls_left_own, walls_left_opp, plies_to_end


def test_driver_drains_all_examples_no_loss():
    # Regression: the driver must drain examples even on a step()->None and after
    # the loop, or the last finalized game's examples are silently lost.
    # max_plies=12 is below the ~15-ply minimum to reach a goal, so every game
    # caps at exactly 12 plies => deterministic example count.
    from scripts.selfplay_native import run_selfplay
    examples, st = run_selfplay(total_games=4, n_games=4, sims=8, device="cpu",
                                max_plies=12)
    assert st["examples"] == len(examples)
    assert len(examples) == 4 * 12  # no loss across 4 games

    # A single game (the step that finalizes it returns None) must be fully drained.
    ex1, _ = run_selfplay(total_games=1, n_games=1, sims=8, device="cpu",
                          max_plies=12)
    assert len(ex1) == 12


def test_feed_length_mismatch_raises():
    import numpy as np
    import pytest
    pool = bn.SelfPlayPool(n_games=4, total_games=4, sims=8, seed=0)
    planes = None
    while planes is None:
        planes = pool.step()
    m = np.asarray(planes).shape[0]
    # wrong policy row count -> clean ValueError, not a panic
    with pytest.raises(ValueError):
        pool.feed(np.full((m + 1, 140), 1.0 / 140, np.float32), np.zeros(m, np.float32))


def test_capped_games_are_draws_with_plies_to_end():
    import numpy as np
    # max_plies=12 is below the ~15-ply minimum to reach a goal, so every game
    # caps -> winner None -> all z==0, and feats[3] (plies_to_end) in [1, 12].
    pool = bn.SelfPlayPool(n_games=4, total_games=4, sims=8, seed=0,
                           max_plies=12, temp_moves=4)
    examples = []
    guard = 0
    while pool.games_remaining() > 0 and guard < 200_000:
        guard += 1
        planes = pool.step()
        if planes is not None:
            b = np.asarray(planes).shape[0]
            pool.feed(np.full((b, 140), 1.0 / 140, np.float32), np.zeros(b, np.float32))
        examples.extend(pool.drain())
    examples.extend(pool.drain())
    assert len(examples) == 4 * 12
    for _planes, _pi, z, feats in examples:
        assert z == 0.0
        p = float(np.asarray(feats)[3])
        assert 1.0 <= p <= 12.0
