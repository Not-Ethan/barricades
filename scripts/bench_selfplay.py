"""Benchmark native batched-MPS self-play throughput and project the 100k run.

Usage: python scripts/bench_selfplay.py [sims]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from scripts.selfplay_native import run_selfplay


def main():
    sims = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device={dev} sims={sims}")
    # warm up MPS / caches with a tiny run
    run_selfplay(total_games=16, n_games=16, sims=sims, device=dev)
    # measured run
    _, st = run_selfplay(total_games=512, n_games=256, sims=sims, device=dev)
    gps = st["games_per_sec"]
    proj = 100_000 / gps / 3600.0
    print(f"  games/sec={gps:.1f}  mean_batch={st['mean_batch']:.0f}  "
          f"batches={st['batches']}  examples={st['examples']}")
    print(f"  ==> projected 100k games: {proj:.2f} hours")
    if st["mean_batch"] < 128:
        print("  WARNING: mean batch < 128 -- MPS underfed; raise n_games.")
    if proj > 2.0:
        print("  GATE: projection > 2h. Lever options before the 100k run:")
        print("   - lower sims, raise n_games, or add subtree carryover (see spec).")
    else:
        print("  GATE PASSED: projection <= 2h. Cleared to launch the 100k campaign.")


if __name__ == "__main__":
    main()
