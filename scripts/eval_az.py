"""Evaluate the bootstrapped AZ net vs the engine pool. Parallel games.

Usage: python scripts/eval_az.py [games] [sims]
"""
import os
import sys
import time
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from concurrent.futures import ProcessPoolExecutor
from core.state import initial_state
from core.rules import apply_move, is_terminal, winner

CKPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "models", "az_bootstrap.pt")
SIMS = 120
OPP_TB = 0.1


def make(spec, seed):
    if spec == "az":
        from agents.az.agent import AZAgent
        return AZAgent(checkpoint=CKPT, sims=SIMS, seed=seed)
    if spec == "random":
        from agents.random_agent import RandomAgent
        return RandomAgent(seed=seed)
    if spec == "greedy":
        from agents.greedy_agent import GreedyAgent
        return GreedyAgent(seed=seed)
    if spec == "minimax":
        from agents.minimax_agent import MinimaxAgent
        return MinimaxAgent(time_budget=OPP_TB, seed=seed)
    if spec == "mcts":
        from agents.mcts_agent import MCTSAgent
        return MCTSAgent(time_budget=OPP_TB, seed=seed)
    raise ValueError(spec)


def play_one(args):
    a, b, i, mp = args
    if i % 2 == 0:
        p0, p1, a0 = make(a, i), make(b, 9000 + i), True
    else:
        p0, p1, a0 = make(b, 9000 + i), make(a, i), False
    s = initial_state()
    ag = (p0, p1)
    for _ in range(mp):
        if is_terminal(s):
            break
        s = apply_move(s, ag[s.turn].select_move(s))
    w = winner(s)
    if w is None:
        return "draw"
    return "a" if ((a0 and w == 0) or (not a0 and w == 1)) else "b"


def match(opp, games):
    args = [("az", opp, i, 300) for i in range(games)]
    res = {"a": 0, "b": 0, "draw": 0}
    t0 = time.time()
    with ProcessPoolExecutor() as ex:
        for r in ex.map(play_one, args):
            res[r] += 1
    se = 100 * math.sqrt(0.25 / games)
    print(f"AZ(bootstrap, {SIMS} sims) vs {opp}: AZ={res['a']} {opp}={res['b']} "
          f"draw={res['draw']} -> AZ winrate {100*res['a']/games:.1f}% (±{2*se:.1f}%)"
          f"  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    G = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    if len(sys.argv) > 2:
        SIMS = int(sys.argv[2])
    if not os.path.exists(CKPT):
        print(f"checkpoint missing: {CKPT}"); sys.exit(1)
    print(f"cores={os.cpu_count()} games/opp={G} az_sims={SIMS} opp_budget={OPP_TB}s", flush=True)
    for opp in ["random", "greedy", "minimax", "mcts"]:
        match(opp, G)
