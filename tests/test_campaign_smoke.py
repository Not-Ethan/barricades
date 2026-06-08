import os
import tempfile
from scripts.campaign import run_campaign, anneal_lambda


def test_anneal_lambda_schedule():
    # 0 -> 1 over the first 60% of iterations, then clamped at 1.0
    assert anneal_lambda(0, 5) == 0.0
    assert anneal_lambda(5, 5) == 1.0
    vals = [anneal_lambda(i, 5) for i in range(5)]
    assert vals == sorted(vals)          # monotonic non-decreasing
    assert all(0.0 <= v <= 1.0 for v in vals)


def test_campaign_runs_and_records_wellformed_history():
    with tempfile.TemporaryDirectory() as d:
        net, hist = run_campaign(iterations=2, games_per_iter=4, n_games=4, sims=8,
                                 max_plies=20, epochs=2, device="cpu", eval_games=4,
                                 eval_opponent="random", out_dir=d, log=lambda *_: None)
        assert len(hist) == 2
        for rec in hist:
            assert set(rec) >= {"it", "lam", "loss", "mean_game_len",
                                "games_per_sec", "winrate", "eval_opponent"}
            assert rec["loss"] == rec["loss"]            # not NaN
            assert rec["mean_game_len"] > 0
            assert 0.0 <= rec["winrate"] <= 1.0
        assert hist[0]["lam"] <= hist[1]["lam"]          # lambda anneals up
        assert os.path.exists(os.path.join(d, "campaign_final.pt"))
        assert os.path.exists(os.path.join(d, "campaign_it0.pt"))
