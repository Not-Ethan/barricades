"""Benchmark async native self-play throughput; sweep sims/cap and project the
100k wall-clock. The decision gate before any campaign run. Carryover is on
(SelfPlayPool default).

Usage: python scripts/bench_selfplay.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from agents.az.model import QuoridorNet
from scripts.selfplay_native import run_selfplay


def main():
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device={dev}")
    configs = [
        ("sims=100 cap=80", dict(sims=100, max_plies=80)),
        ("sims=50  cap=80", dict(sims=50, max_plies=80)),
        ("sims=50  cap=60", dict(sims=50, max_plies=60)),
    ]
    net = QuoridorNet(32, 3).to(dev).eval()
    run_selfplay(total_games=16, n_games=16, sims=50, device=dev, net=net, max_plies=60)  # warmup
    best = None
    for label, kw in configs:
        _, st = run_selfplay(total_games=512, n_games=256, device=dev, net=net, **kw)
        gps = st["games_per_sec"]
        proj = 100_000 / gps / 3600.0
        flag = "  <-- mean_batch<128 (MPS underfed)" if st["mean_batch"] < 128 else ""
        print(f"  {label}: games/s={gps:.1f} mean_batch={st['mean_batch']:.0f} "
              f"examples/game={st['examples']/st['games']:.0f} -> 100k={proj:.2f}h{flag}")
        if best is None or proj < best[1]:
            best = (label, proj)
    print(f"\nBEST: {best[0]} -> {best[1]:.2f}h")
    if best[1] <= 2.0:
        print("GATE PASSED: a config projects <=2h. Cleared to run the campaign.")
    else:
        print("GATE: best projection > 2h. Levers: carryover (verify on), lower sims, "
              "lower cap, or the dense-reward shortening (Workstream B/C) which cuts "
              "game length over iterations.")


if __name__ == "__main__":
    main()
