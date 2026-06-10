"""Per-move timing for MinimaxAgent, python vs native backend, at fixed depths.
Usage: python scripts/bench_minimax.py"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.state import initial_state
from core.rules import apply_move, is_terminal
from agents.minimax_agent import MinimaxAgent
from agents.greedy_agent import GreedyAgent


def positions():
    pos = [initial_state()]
    g, s = GreedyAgent(seed=0), initial_state()
    for i in range(14):
        if is_terminal(s):
            break
        s = apply_move(s, g.select_move(s))
        if i in (4, 8, 12):
            pos.append(s)
    return pos


def main():
    pos = positions()
    print(f"positions={len(pos)}")
    res = {}
    for backend in ("python", "native"):
        for D in (1, 2, 3, 4):
            ts = []
            for s in pos:
                a = MinimaxAgent(max_depth=D, time_budget=300.0, wall_cap=12, backend=backend)
                t0 = time.perf_counter()
                a.analyze(s)
                ts.append((time.perf_counter() - t0) * 1000)
            res[(backend, D)] = sum(ts) / len(ts)
            print(f"  {backend:6s} d{D}: {res[(backend, D)]:9.1f} ms/move avg", flush=True)
    print("--- speedup (python/native) ---")
    for D in (1, 2, 3, 4):
        print(f"  d{D}: {res[('python', D)] / res[('native', D)]:.1f}x")


if __name__ == "__main__":
    main()
