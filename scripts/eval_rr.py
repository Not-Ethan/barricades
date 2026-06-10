"""Round-robin strength evaluation with Elo. Robust to the non-transitivity of the
heuristic agents: plays a pool round-robin (color-balanced, no-progress adjudication)
and fits Bradley-Terry/Elo ratings.

Agent specs: random | greedy | dN (minimax depth N) | net:<ckpt>:<ch>:<bl>:<sims>

Usage:
  python scripts/eval_rr.py --agents random greedy d1 d2 d3 net:models/distill_d3_w.pt:64:5:100 \
      --games 16
"""
import argparse
import itertools
import math
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from core.state import initial_state
from core.rules import apply_move, is_terminal, winner, shortest_path_len
from agents.native_agent import NativeMctsAgent
from agents.minimax_agent import MinimaxAgent
from agents.greedy_agent import GreedyAgent
from agents.random_agent import RandomAgent
from agents.az.model import QuoridorNet

_NET_CACHE = {}


def make_agent(spec, seed):
    if spec == "random":
        return RandomAgent(seed=seed)
    if spec == "greedy":
        return GreedyAgent(seed=seed)
    if spec[0] == "d" and spec[1:].isdigit():
        return MinimaxAgent(max_depth=int(spec[1:]), time_budget=20, wall_cap=12, seed=seed)
    if spec.startswith("net:"):
        _, path, ch, bl, sims = spec.split(":")
        key = (path, ch, bl)
        if key not in _NET_CACHE:
            n = QuoridorNet(int(ch), int(bl))
            n.load_state_dict(torch.load(path, map_location="cpu"), strict=False)
            _NET_CACHE[key] = n.eval()
        net = _NET_CACHE[key]

        def eval_fn(planes):
            x = torch.from_numpy(np.asarray(planes))
            with torch.no_grad():
                o = net(x)
                return torch.softmax(o[0], 1).numpy(), o[1].squeeze(1).numpy()
        return NativeMctsAgent(sims=int(sims), seed=seed, eval_fn=eval_fn)
    raise ValueError(f"unknown agent spec: {spec}")


def play_game(args):
    """Returns (specA, specB, A_score) where A_score in {1, 0.5, 0}."""
    specA, specB, seed, cap, a_first = args
    a = make_agent(specA, seed)
    b = make_agent(specB, seed + 777)
    pl = (a, b) if a_first else (b, a)
    a_idx = 0 if a_first else 1
    s = initial_state()
    for _ in range(cap):
        if is_terminal(s):
            break
        s = apply_move(s, pl[s.turn].select_move(s))
    w = winner(s)
    if w is None:                            # no-progress: closer-to-goal wins
        d0 = shortest_path_len(s, 0)
        d1 = shortest_path_len(s, 1)
        d0 = 999 if d0 is None else d0
        d1 = 999 if d1 is None else d1
        if d0 < d1:
            w = 0
        elif d1 < d0:
            w = 1
        else:
            return (specA, specB, 0.5)
    return (specA, specB, 1.0 if w == a_idx else 0.0)


def fit_elo(points, n_between, agents, iters=3000):
    """Bradley-Terry MM fit. points[i]=total score of i; n_between[(i,j)] symmetric."""
    gamma = {a: 1.0 for a in agents}
    for _ in range(iters):
        for i in agents:
            denom = sum(n_between.get(frozenset((i, j)), 0) / (gamma[i] + gamma[j])
                        for j in agents if j != i)
            if denom > 0:
                gamma[i] = max(points[i], 1e-9) / denom
        gm = math.exp(sum(math.log(max(gamma[a], 1e-12)) for a in agents) / len(agents))
        for a in agents:
            gamma[a] /= gm
    return {a: 400.0 * math.log10(max(gamma[a], 1e-12)) for a in agents}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents", nargs="+",
                    default=["random", "greedy", "d1", "d2", "d3", "net:models/distill_d3_w.pt:64:5:100"])
    ap.add_argument("--games", type=int, default=16, help="games per pair (colors alternated)")
    ap.add_argument("--cap", type=int, default=160)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()

    agents = a.agents
    tasks = []
    for A, B in itertools.combinations(agents, 2):
        for g in range(a.games):
            tasks.append((A, B, g, a.cap, g % 2 == 0))   # alternate who is player 0

    pair_score = defaultdict(float)    # (A,B) -> A's total points vs B
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for A, B, sc in ex.map(play_game, tasks, chunksize=2):
            pair_score[(A, B)] += sc

    points = defaultdict(float)
    n_between = {}
    print("=== win matrix (row's score vs column, /games) ===")
    short = {s: (s.split(":")[0] + ":" + os.path.basename(s.split(":")[1]) if s.startswith("net:") else s) for s in agents}
    hdr = "".join(f"{short[c][:10]:>11}" for c in agents)
    print(f"{'':<12}{hdr}")
    for A in agents:
        row = ""
        for B in agents:
            if A == B:
                row += f"{'—':>11}"
                continue
            sA = pair_score[(A, B)] if (A, B) in pair_score else (a.games - pair_score[(B, A)])
            row += f"{sA / a.games:>11.2f}"
        print(f"{short[A][:12]:<12}{row}")
    for A, B in itertools.combinations(agents, 2):
        sA = pair_score[(A, B)]
        points[A] += sA
        points[B] += a.games - sA
        n_between[frozenset((A, B))] = a.games

    elo = fit_elo(points, n_between, agents)
    base = min(elo.values())
    print("\n=== Elo (anchored, min=0) | avg score vs field ===")
    n_field = (len(agents) - 1) * a.games
    for ag in sorted(agents, key=lambda x: -elo[x]):
        print(f"  {short[ag]:<28} Elo {elo[ag] - base:6.0f} | {100 * points[ag] / n_field:5.1f}%")


if __name__ == "__main__":
    main()
