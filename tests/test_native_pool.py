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
