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


def run_selfplay(total_games=512, n_games=256, sims=100, device="mps",
                 channels=32, blocks=3, ckpt=None, net=None, seed=0,
                 max_plies=200, temp_moves=10):
    """Async batched self-play: two SelfPlayPools ping-pong so each pool's Rust
    step() (CPU, GIL released) overlaps the other's in-flight GPU forward. MPS
    runs net(x) asynchronously; only .cpu() blocks, so the overlap is automatic.
    Returns (examples, stats). `examples` are (planes(6,9,9), pi(140), z, feats(4)).
    """
    if net is None:
        net = QuoridorNet(channels=channels, blocks=blocks)
        if ckpt and os.path.exists(ckpt):
            net.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=False)
        net = net.to(device)
    net.eval()

    half = max(1, n_games // 2)
    g0 = total_games // 2
    pools = [
        bn.SelfPlayPool(n_games=half, total_games=g0, sims=sims, seed=seed,
                        max_plies=max_plies, temp_moves=temp_moves),
        bn.SelfPlayPool(n_games=max(1, n_games - half), total_games=total_games - g0,
                        sims=sims, seed=seed + 1, max_plies=max_plies, temp_moves=temp_moves),
    ]

    def forward(planes):
        x = torch.from_numpy(np.asarray(planes)).to(device)
        with torch.no_grad():
            out = net(x)                         # 2- or 3-tuple; index positionally
            logits, value = out[0], out[1]
            pol = torch.softmax(logits, dim=1)
            val = value.squeeze(1)
        return x.shape[0], pol, val              # GPU tensors; not synced yet

    examples, batches, batch_pos = [], 0, 0
    inflight = None                              # (pool_idx, pol_gpu, val_gpu) or None
    nxt = 0
    t0 = time.perf_counter()

    def any_work():
        return inflight is not None or any(p.games_remaining() > 0 for p in pools)

    while any_work():
        # pick a pool to step (CPU) that is NOT the in-flight one and still has games
        step_idx = None
        for cand in (nxt, 1 - nxt):
            if (inflight is None or cand != inflight[0]) and pools[cand].games_remaining() > 0:
                step_idx = cand
                break
        planes = pools[step_idx].step() if step_idx is not None else None  # overlaps in-flight GPU
        # sync + feed the in-flight forward (GPU already worked during the step above)
        if inflight is not None:
            b, pol_g, val_g = inflight
            pool = pools[b]
            pool.feed(np.ascontiguousarray(pol_g.cpu().numpy(), dtype=np.float32),
                      np.ascontiguousarray(val_g.cpu().numpy(), dtype=np.float32))
            examples.extend(pool.drain())
            inflight = None
        # submit the freshly-stepped pool's forward (async)
        if planes is not None:
            m, pol_g, val_g = forward(planes)
            inflight = (step_idx, pol_g, val_g)
            batches += 1
            batch_pos += m
            nxt = 1 - step_idx

    for p in pools:                              # final drains (example-loss guard)
        examples.extend(p.drain())
    dt = time.perf_counter() - t0
    return examples, dict(games=total_games, seconds=dt, batches=batches,
                          mean_batch=batch_pos / max(batches, 1),
                          games_per_sec=total_games / dt, examples=len(examples))


if __name__ == "__main__":
    total = int(sys.argv[1]) if len(sys.argv) > 1 else 512
    ngames = int(sys.argv[2]) if len(sys.argv) > 2 else 256
    sims = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    device = sys.argv[4] if len(sys.argv) > 4 else "mps"
    _, stats = run_selfplay(total, ngames, sims, device)
    print(stats)
