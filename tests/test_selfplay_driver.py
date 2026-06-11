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


def test_path_diff_sign_aligns_with_outcome():
    # End-to-end: feats[0]=path_diff and z are both mover-relative, so a position
    # whose mover eventually WON should have a higher mean path_diff (closer to its
    # goal) than one whose mover LOST. Guards the encode/features/finalize
    # perspective convention across the Rust<->Python boundary against regressions.
    import numpy as np
    from scripts.selfplay_native import run_selfplay
    ex, _ = run_selfplay(total_games=12, n_games=8, sims=8, device="cpu",
                         max_plies=200, seed=0)
    wins = [float(np.asarray(f)[0]) for _p, _pi, z, f in ex if z == 1.0]
    losses = [float(np.asarray(f)[0]) for _p, _pi, z, f in ex if z == -1.0]
    assert len(wins) > 20 and len(losses) > 20, (len(wins), len(losses))
    assert sum(wins) / len(wins) > sum(losses) / len(losses)
