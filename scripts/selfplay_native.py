"""Batched-MPS self-play driver: Rust SelfPlayPool <-> PyTorch net on MPS.

Usage: python scripts/selfplay_native.py [total_games] [n_games] [sims] [device]
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

import barricades_native as bn
from agents.az.model import QuoridorNet


def run_selfplay(total_games=256, n_games=256, sims=100, device="mps",
                 channels=32, blocks=3, ckpt=None, seed=0):
    net = QuoridorNet(channels=channels, blocks=blocks)
    if ckpt and os.path.exists(ckpt):
        net.load_state_dict(torch.load(ckpt, map_location="cpu")["model"])
    net = net.to(device).eval()

    pool = bn.SelfPlayPool(n_games=n_games, total_games=total_games, sims=sims,
                           seed=seed)
    examples, batches, batch_pos = [], 0, 0
    t0 = time.perf_counter()
    while pool.games_remaining() > 0:
        planes = pool.step()
        if planes is None:
            continue
        x = torch.from_numpy(np.asarray(planes)).to(device)
        with torch.no_grad():
            logits, value = net(x)
            policy = torch.softmax(logits, dim=1).cpu().numpy()
            value = value.squeeze(1).cpu().numpy()
        pool.feed(np.ascontiguousarray(policy, dtype=np.float32),
                  np.ascontiguousarray(value, dtype=np.float32))
        examples.extend(pool.drain())
        batches += 1
        batch_pos += x.shape[0]
    dt = time.perf_counter() - t0
    return examples, dict(games=total_games, seconds=dt, batches=batches,
                          mean_batch=batch_pos / max(batches, 1),
                          games_per_sec=total_games / dt,
                          examples=len(examples))


if __name__ == "__main__":
    total = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    ngames = int(sys.argv[2]) if len(sys.argv) > 2 else 256
    sims = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    device = sys.argv[4] if len(sys.argv) > 4 else "mps"
    _, stats = run_selfplay(total, ngames, sims, device)
    print(stats)
