def test_async_driver_drains_all_examples_no_loss():
    # max_plies=12 < ~15-ply min to reach a goal, so every game caps -> exactly
    # 12 examples/game. The 2-pool pipeline must drain every game with no loss
    # or duplication.
    from scripts.selfplay_native import run_selfplay
    ex, st = run_selfplay(total_games=8, n_games=4, sims=8, device="cpu", max_plies=12)
    assert st["examples"] == len(ex)
    assert len(ex) == 8 * 12
    for planes, pi, z, feats in ex:
        import numpy as np
        assert np.asarray(planes).shape == (6, 9, 9)
        assert abs(float(np.asarray(pi).sum()) - 1.0) < 1e-4
        assert z in (-1.0, 0.0, 1.0)
        assert np.asarray(feats).shape == (4,)
