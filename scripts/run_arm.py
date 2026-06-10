"""Launch one Phase-1 ablation arm.

Usage: python scripts/run_arm.py <A|B|C|D> [device]

2x2 factorial: reward {dense, outcome} x opponent {self, pool}.
  A: dense   + self   (control - reproduce the race plateau at scale)
  B: outcome + self   (drop dense - isolate the reward bias)
  C: dense   + pool   (isolate self-play co-adaptation)
  D: outcome + pool   (combined best-bet)

Each arm writes checkpoints + history.json to models/arm_<X>/.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.campaign import run_campaign

ARMS = {
    "A": dict(reward_mode="dense",   opponent="self", seed=0),
    "B": dict(reward_mode="outcome", opponent="self", seed=1000),
    "C": dict(reward_mode="dense",   opponent="pool", seed=2000),
    "D": dict(reward_mode="outcome", opponent="pool", seed=3000),
}

# Phase-1 budget: 25 iters x 1000 games = 25k/arm (enough to see escape-or-not;
# the plateau historically appeared by iter ~13). n_games < games_per_iter keeps
# the self-play batch full. Minimax eval every 5th iter (expensive); the cheap
# per-iter diagnostics (game_len, wall_mass, entropy) are the primary signal.
COMMON = dict(iterations=25, games_per_iter=1000, n_games=512, sims=100,
              max_plies=80, warmup_frac=0.8, eval_opponent="minimax",
              eval_games=6, eval_every=5)


def main():
    if len(sys.argv) < 2 or sys.argv[1].upper() not in ARMS:
        sys.exit("usage: run_arm.py <A|B|C|D> [device]")
    arm = sys.argv[1].upper()
    device = sys.argv[2] if len(sys.argv) > 2 else "cuda"
    cfg = ARMS[arm]
    out_dir = f"models/arm_{arm}"
    print(f"=== ARM {arm}: reward={cfg['reward_mode']} opponent={cfg['opponent']} "
          f"device={device} -> {out_dir} ===", flush=True)
    run_campaign(device=device, out_dir=out_dir, reward_mode=cfg["reward_mode"],
                 opponent=cfg["opponent"], seed=cfg["seed"], **COMMON)
    print(f"=== ARM {arm} DONE ===", flush=True)


if __name__ == "__main__":
    main()
