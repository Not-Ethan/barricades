"""Strength ladder: a net checkpoint's win-rate vs greedy + a minimax ladder.
The scalable reference ("our Stockfish") for tracking AZ strength.

Usage: python scripts/eval_ladder.py [checkpoint] [games_per_rung] [az_sims]
"""
import os
import sys
import math
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from concurrent.futures import ProcessPoolExecutor
from core.state import initial_state
from core.rules import apply_move, is_terminal, winner

CKPT = sys.argv[1] if len(sys.argv) > 1 else "models/campaign10k/campaign_final.pt"
GAMES = int(sys.argv[2]) if len(sys.argv) > 2 else 20
AZ_SIMS = int(sys.argv[3]) if len(sys.argv) > 3 else 100

RUNGS = ["greedy", "minimax-d1", "minimax-d2", "minimax-d3", "minimax-t0.25"]

_NET = None


def _load_net():
    global _NET
    if _NET is None:
        from agents.az.model import QuoridorNet
        n = QuoridorNet(32, 3)
        n.load_state_dict(torch.load(CKPT, map_location="cpu"), strict=False)
        _NET = n.eval()
    return _NET


def make_opponent(rung, seed):
    if rung == "greedy":
        from agents.greedy_agent import GreedyAgent
        return GreedyAgent(seed=seed)
    from agents.minimax_agent import MinimaxAgent
    if rung.startswith("minimax-d"):
        return MinimaxAgent(max_depth=int(rung.split("d")[1]), time_budget=10.0, seed=seed)
    if rung.startswith("minimax-t"):
        return MinimaxAgent(time_budget=float(rung.split("t")[1]), seed=seed)
    raise ValueError(rung)


def make_az(seed):
    from agents.native_agent import NativeMctsAgent
    net = _load_net()

    def eval_fn(planes):
        x = torch.from_numpy(np.asarray(planes))           # CPU in workers
        with torch.no_grad():
            out = net(x)
            return torch.softmax(out[0], 1).numpy(), out[1].squeeze(1).numpy()
    return NativeMctsAgent(sims=AZ_SIMS, seed=seed, eval_fn=eval_fn)


def play_one(args):
    rung, i = args
    az, opp = make_az(i), make_opponent(rung, 5000 + i)
    players = (az, opp) if i % 2 == 0 else (opp, az)
    s = initial_state()
    for _ in range(400):
        if is_terminal(s):
            break
        s = apply_move(s, players[s.turn].select_move(s))
    w = winner(s)
    return 1 if ((w == 0 and i % 2 == 0) or (w == 1 and i % 2 == 1)) else 0


def main():
    if not os.path.exists(CKPT):
        print(f"checkpoint missing: {CKPT}"); return
    print(f"ckpt={CKPT} games/rung={GAMES} az_sims={AZ_SIMS}")
    for rung in RUNGS:
        t0 = time.time()
        with ProcessPoolExecutor() as ex:
            wins = sum(ex.map(play_one, [(rung, i) for i in range(GAMES)]))
        se = 100 * math.sqrt(0.25 / GAMES)
        print(f"  AZ vs {rung:14s}: {wins}/{GAMES} = {100*wins/GAMES:5.1f}% (±{2*se:.0f}%)  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
