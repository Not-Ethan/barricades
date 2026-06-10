"""Evaluate a distilled net (as an MCTS agent) vs minimax at several depths.
Uses the native-backed minimax (fast d3/d4/d5). Parallel across games with
OMP_NUM_THREADS pinned (set it in the environment) to avoid thread oversubscription.

Usage: python scripts/distill_eval.py --ckpt models/distill_d3.pt --games 16 --sims 100 --depths 3,4,5
"""
import argparse
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from core.state import initial_state
from core.rules import apply_move, is_terminal, winner
from agents.native_agent import NativeMctsAgent
from agents.minimax_agent import MinimaxAgent
from agents.az.model import QuoridorNet

_CKPT = None
_SIMS = 100
_NET = None


def _net():
    global _NET
    if _NET is None:
        n = QuoridorNet(32, 3)
        n.load_state_dict(torch.load(_CKPT, map_location="cpu"), strict=False)
        _NET = n.eval()
    return _NET


def _az(seed):
    net = _net()

    def eval_fn(planes):
        x = torch.from_numpy(np.asarray(planes))           # CPU in workers
        with torch.no_grad():
            out = net(x)
            return torch.softmax(out[0], 1).numpy(), out[1].squeeze(1).numpy()
    return NativeMctsAgent(sims=_SIMS, seed=seed, eval_fn=eval_fn)


def play_one(args):
    depth, i = args
    az = _az(i)
    opp = MinimaxAgent(max_depth=depth, time_budget=10.0, wall_cap=12, seed=2000 + i)
    players = (az, opp) if i % 2 == 0 else (opp, az)
    s = initial_state()
    for _ in range(200):
        if is_terminal(s):
            break
        s = apply_move(s, players[s.turn].select_move(s))
    w = winner(s)
    return 1 if ((w == 0 and i % 2 == 0) or (w == 1 and i % 2 == 1)) else 0


def main():
    global _CKPT, _SIMS
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="models/distill_d3.pt")
    ap.add_argument("--games", type=int, default=16)
    ap.add_argument("--sims", type=int, default=100)
    ap.add_argument("--depths", default="3,4,5")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()
    _CKPT, _SIMS = a.ckpt, a.sims
    print(f"distilled net (MCTS sims={a.sims}) vs minimax, {a.games} games/rung:")
    for d in [int(x) for x in a.depths.split(",")]:
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            wins = sum(ex.map(play_one, [(d, i) for i in range(a.games)]))
        se = 100 * math.sqrt(0.25 / a.games)
        print(f"  vs minimax-d{d}: {wins}/{a.games} = {100 * wins / a.games:5.1f}% (+-{2 * se:.0f}%)", flush=True)


if __name__ == "__main__":
    main()
