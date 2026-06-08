"""AZ campaign: iterate self-play (async native pool) -> dense targets (annealed
lambda) -> train (3-head) -> checkpoint -> quick win-rate vs a baseline.

Baseline defaults to GREEDY: random is a weak yardstick for Quoridor (it barely
uses walls), so "beat random" mostly means "the net learned to walk to its goal".
Greedy races the optimal shortest path, so beating it requires real wall tactics.

Usage: python scripts/campaign.py [iterations] [games_per_iter] [sims] [device]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from agents.az.model import QuoridorNet
from agents.az.train import form_dense_targets, train_minibatched, save_checkpoint
from scripts.selfplay_native import run_selfplay


def anneal_lambda(it, iterations, warmup_frac=0.6):
    """0 -> 1 linearly over the first warmup_frac of iterations, then 1."""
    w = max(1, int(iterations * warmup_frac))
    return min(1.0, it / w)


def _make_opponent(opponent, seed):
    if opponent == "random":
        from agents.random_agent import RandomAgent
        return RandomAgent(seed=seed)
    if opponent == "greedy":
        from agents.greedy_agent import GreedyAgent
        return GreedyAgent(seed=seed)
    if opponent == "minimax":
        from agents.minimax_agent import MinimaxAgent
        return MinimaxAgent(time_budget=0.1, seed=seed)
    raise ValueError(f"unknown opponent: {opponent}")


def winrate_vs(net, device, opponent="greedy", sims=60, games=10):
    """Win fraction of the net-driven MCTS vs a baseline (random|greedy|minimax),
    alternating colors. Greedy/minimax are stronger yardsticks than random for
    Quoridor (they force the net to actually use walls, not just advance)."""
    from agents.native_agent import NativeMctsAgent
    from core.state import initial_state
    from core.rules import apply_move, is_terminal, winner

    net.eval()

    def eval_fn(planes):
        x = torch.from_numpy(np.asarray(planes)).to(device)
        with torch.no_grad():
            out = net(x)
            pol = torch.softmax(out[0], dim=1).cpu().numpy()
            val = out[1].squeeze(1).cpu().numpy()
        return pol, val

    wins = 0
    for g in range(games):
        a = NativeMctsAgent(sims=sims, seed=g, eval_fn=eval_fn)
        b = _make_opponent(opponent, seed=1000 + g)
        players = (a, b) if g % 2 == 0 else (b, a)
        s = initial_state()
        for _ in range(400):
            if is_terminal(s):
                break
            s = apply_move(s, players[s.turn].select_move(s))
        w = winner(s)
        if (w == 0 and g % 2 == 0) or (w == 1 and g % 2 == 1):
            wins += 1
    return wins / games


def run_campaign(iterations=5, games_per_iter=256, n_games=256, sims=100,
                 max_plies=80, epochs=4, lr=1e-3, device="mps", seed=0,
                 channels=32, blocks=3, init_ckpt=None, out_dir="models",
                 eval_games=10, eval_opponent="greedy", log=print):
    # Tip: keep n_games < games_per_iter so the pool refills and MPS batches stay
    # full (n_games == games_per_iter means no refill -> the batch decays in the
    # tail as games finish, cutting throughput).
    net = QuoridorNet(channels=channels, blocks=blocks)
    if init_ckpt and os.path.exists(init_ckpt):
        net.load_state_dict(torch.load(init_ckpt, map_location="cpu"), strict=False)
    net = net.to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    os.makedirs(out_dir, exist_ok=True)
    history = []
    for it in range(iterations):
        lam = anneal_lambda(it, iterations)
        examples, st = run_selfplay(total_games=games_per_iter, n_games=n_games,
                                    sims=sims, device=device, net=net, seed=seed + it * 2,
                                    max_plies=max_plies)
        # Build targets on CPU (a whole iteration's examples won't fit in GPU mem);
        # train_minibatched moves minibatches to `device`.
        batch = form_dense_targets(examples, lam=lam, device="cpu")
        loss = train_minibatched(net, opt, batch, epochs=epochs, device=device)
        wr = winrate_vs(net, device, opponent=eval_opponent, games=eval_games)
        rec = dict(it=it, lam=round(lam, 3), loss=round(loss, 4),
                   mean_game_len=round(st["examples"] / max(1, st["games"]), 1),
                   games_per_sec=round(st["games_per_sec"], 2),
                   winrate=wr, eval_opponent=eval_opponent)
        history.append(rec)
        log(rec)
        save_checkpoint(net, os.path.join(out_dir, f"campaign_it{it}.pt"))
    save_checkpoint(net, os.path.join(out_dir, "campaign_final.pt"))
    return net, history


if __name__ == "__main__":
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    gpi = int(sys.argv[2]) if len(sys.argv) > 2 else 256
    sims = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    device = sys.argv[4] if len(sys.argv) > 4 else "mps"
    n_games = min(256, gpi)   # refill keeps MPS batches full when gpi is large
    _, hist = run_campaign(iterations=iters, games_per_iter=gpi, n_games=n_games,
                           sims=sims, device=device)
    print("game length:", [h["mean_game_len"] for h in hist])
    print("winrate vs %s:" % (hist[0]["eval_opponent"] if hist else "?"),
          [h["winrate"] for h in hist])
