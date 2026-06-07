"""Smoke-train an AZ net and save a checkpoint. Usage:
    python scripts/train_az.py --iterations 3 --games 4 --sims 60
"""
import argparse
import os
import sys

# Make the repo root importable when run as `python scripts/train_az.py`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.az.model import QuoridorNet
from agents.az.train import run_training, save_checkpoint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=3)
    ap.add_argument("--games", type=int, default=4)
    ap.add_argument("--sims", type=int, default=60)
    ap.add_argument("--channels", type=int, default=32)
    ap.add_argument("--blocks", type=int, default=3)
    ap.add_argument("--out", default="models/az_smoke.pt")
    args = ap.parse_args()
    net = QuoridorNet(channels=args.channels, blocks=args.blocks)
    hist = run_training(net, iterations=args.iterations,
                        games_per_iter=args.games, sims=args.sims)
    save_checkpoint(net, args.out)
    print(f"saved {args.out}; loss history: {hist}")


if __name__ == "__main__":
    main()
