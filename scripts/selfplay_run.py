"""Self-play training from a distilled warm-start (the original break-the-plateau run,
but seeded with a wall-competent net instead of cold).

Usage: python scripts/selfplay_run.py --init models/distill_d3_w.pt --out models/selfplay_warm \
         --iters 25 --channels 64 --blocks 5 --reward outcome --device mps
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.campaign import run_campaign


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="models/distill_d3_w.pt")
    ap.add_argument("--out", default="models/selfplay_warm")
    ap.add_argument("--iters", type=int, default=25)
    ap.add_argument("--gpi", type=int, default=1000)
    ap.add_argument("--n-games", type=int, default=512)
    ap.add_argument("--sims", type=int, default=100)
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=5)
    ap.add_argument("--reward", default="outcome", choices=["dense", "outcome"])
    ap.add_argument("--device", default="mps")
    a = ap.parse_args()
    print(f"=== self-play from {a.init} | {a.channels}ch/{a.blocks}b | reward={a.reward} "
          f"| {a.iters}it x {a.gpi} -> {a.out} ===", flush=True)
    run_campaign(iterations=a.iters, games_per_iter=a.gpi, n_games=a.n_games, sims=a.sims,
                 max_plies=80, device=a.device, channels=a.channels, blocks=a.blocks,
                 init_ckpt=a.init, reward_mode=a.reward, warmup_frac=0.8,
                 eval_opponent="greedy", eval_games=10, eval_every=5, out_dir=a.out)
    print("=== self-play done ===", flush=True)


if __name__ == "__main__":
    main()
