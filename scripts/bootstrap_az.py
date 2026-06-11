"""Supervised bootstrap: generate teacher games, train QuoridorNet on them.

Usage (quick smoke test):
    python scripts/bootstrap_az.py --games 4 --budget 0.02 --epochs 2

Full run (run by the controller, not manually):
    python scripts/bootstrap_az.py --games 300 --budget 0.05 --epochs 6
"""
import argparse
import os
import sys
import random
from concurrent.futures import ProcessPoolExecutor

# Make the repo root importable when run as `python scripts/bootstrap_az.py`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from agents.az.bootstrap import worker
from agents.az.model import QuoridorNet
from agents.az.train import examples_to_batch, train_step, save_checkpoint


def _build_args_list(n_games, budget, matchup, temp_moves, max_plies,
                     explore_eps, base_seed):
    """Return a list of (spec0, spec1, seed, temp_moves, max_plies, explore_eps)
    tuples, one per game."""
    mm_spec = lambda b: {"engine": "minimax", "params": {"time_budget": b}}
    mc_spec = lambda b: {"engine": "mcts", "params": {"time_budget": b}}

    rng = random.Random(base_seed)
    args_list = []
    for i in range(n_games):
        seed = rng.randrange(1 << 30)
        if matchup == "minimax":
            s0, s1 = mm_spec(budget), mm_spec(budget)
        elif matchup == "mcts":
            s0, s1 = mc_spec(budget), mc_spec(budget)
        else:  # "mixed": alternate and shuffle assignments
            if i % 4 < 2:
                s0, s1 = mm_spec(budget), mc_spec(budget)
            else:
                s0, s1 = mc_spec(budget), mm_spec(budget)
        args_list.append((s0, s1, seed, temp_moves, max_plies, explore_eps))
    return args_list


def main():
    ap = argparse.ArgumentParser(
        description="Bootstrap QuoridorNet by imitating teacher engines."
    )
    ap.add_argument("--games", type=int, default=300,
                    help="Number of teacher games to generate (default: 300)")
    ap.add_argument("--budget", type=float, default=0.05,
                    help="Time budget per move per teacher engine (default: 0.05s)")
    ap.add_argument("--epochs", type=int, default=6,
                    help="Training epochs over the collected data (default: 6)")
    ap.add_argument("--channels", type=int, default=32,
                    help="QuoridorNet channel width (default: 32)")
    ap.add_argument("--blocks", type=int, default=3,
                    help="QuoridorNet residual blocks (default: 3)")
    ap.add_argument("--matchup", default="mixed",
                    choices=["mixed", "minimax", "mcts"],
                    help="Which teacher engines to use (default: mixed)")
    ap.add_argument("--out", default="models/az_bootstrap.pt",
                    help="Output checkpoint path (default: models/az_bootstrap.pt)")
    ap.add_argument("--batch", type=int, default=256,
                    help="Minibatch size for training (default: 256)")
    ap.add_argument("--seed", type=int, default=0,
                    help="Master RNG seed (default: 0)")
    ap.add_argument("--workers", type=int, default=None,
                    help="Parallel data-gen workers (default: os.cpu_count())")
    args = ap.parse_args()

    print(f"Bootstrap config: games={args.games}, budget={args.budget}s, "
          f"matchup={args.matchup}, epochs={args.epochs}, "
          f"channels={args.channels}, blocks={args.blocks}")

    # ---- Data generation ----
    args_list = _build_args_list(
        n_games=args.games,
        budget=args.budget,
        matchup=args.matchup,
        temp_moves=12,
        max_plies=200,
        explore_eps=0.15,
        base_seed=args.seed,
    )

    all_examples = []
    max_workers = args.workers  # None => os.cpu_count()
    print(f"Generating {args.games} teacher games "
          f"(parallel workers={max_workers or 'auto'})...")
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        for game_examples in pool.map(worker, args_list):
            all_examples.extend(game_examples)

    n_games_done = args.games
    n_positions = len(all_examples)
    print(f"Generated {n_games_done} games, {n_positions} positions.")

    if n_positions == 0:
        print("ERROR: No training examples generated. Exiting.")
        sys.exit(1)

    # ---- Training ----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {device}")

    net = QuoridorNet(channels=args.channels, blocks=args.blocks)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    rng = random.Random(args.seed + 1)
    indices = list(range(n_positions))

    for epoch in range(1, args.epochs + 1):
        rng.shuffle(indices)
        epoch_losses = []
        for start in range(0, n_positions, args.batch):
            batch_idx = indices[start: start + args.batch]
            if len(batch_idx) < 2:
                continue  # skip tiny tail batches
            mini = [all_examples[i] for i in batch_idx]
            batch = examples_to_batch(mini, device=device)
            loss = train_step(net, optimizer, batch)
            epoch_losses.append(loss)
        mean_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else float("nan")
        print(f"  Epoch {epoch}/{args.epochs}: mean_loss={mean_loss:.4f}")

    # ---- Save checkpoint ----
    out_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        args.out,
    )
    save_checkpoint(net, out_path)
    print(f"Checkpoint saved to {out_path}")


if __name__ == "__main__":
    main()
