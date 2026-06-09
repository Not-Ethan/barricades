from smallboard.validate import (theoretical_result, optimal_move_agreement,
                                  az_vs_solver, train_small_az, value_accuracy)


def test_theoretical_result_3x3():
    val, side = theoretical_result(3, 1)
    assert val in (-1, 0, 1)
    assert side in ("p0", "p1", "draw")


def test_train_and_metrics_run_3x3():
    # tiny training, then the metrics execute and return sane numbers.
    net, eng, enc = train_small_az(N=3, W=1, iterations=2, games=6, sims=20,
                                   epochs=2, seed=0)
    agree = optimal_move_agreement(net, eng, enc, n_positions=20, seed=1)
    assert 0.0 <= agree <= 1.0
    res = az_vs_solver(net, eng, enc, games=4, az_sims=40, seed=2)
    assert set(res) >= {"az_as_winner_winrate", "az_never_loses_won"}
    assert 0.0 <= res["az_as_winner_winrate"] <= 1.0
    vacc = value_accuracy(net, eng, enc, n_positions=20, seed=3)
    assert -1.0 <= vacc <= 1.0
