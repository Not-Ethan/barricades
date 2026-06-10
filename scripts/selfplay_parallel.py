"""Parallel self-play campaign. The native pool's Python feed loop is ~1 core, so we
shard each iteration's generation across K worker processes (each single-threaded:
RAYON/OMP=1) generating games/K on CPU, pool the examples, then train once on the main
device (MPS/CUDA). ~K x faster generation on a multi-core box.

Run with the thread caps so workers don't oversubscribe:
  RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/selfplay_parallel.py \
      --init models/distill_d3_w.pt --out models/selfplay_par --iters 25 \
      --channels 64 --blocks 5 --reward outcome --device mps --workers 8
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from agents.az.model import QuoridorNet
from agents.az.train import augment_lr, form_dense_targets, train_minibatched, save_checkpoint
from agents.az.diagnostics import policy_diagnostics
from scripts.campaign import anneal_lambda, winrate_vs


def _gen_worker(args):
    """Generate n_total games with the net at ckpt (CPU). Returns example list."""
    ckpt, ch, bl, n_total, n_games, sims, seed, max_plies = args
    import torch as _t
    from agents.az.model import QuoridorNet as _Q
    from scripts.selfplay_native import run_selfplay
    net = _Q(ch, bl)
    net.load_state_dict(_t.load(ckpt, map_location="cpu"), strict=False)
    net.eval()
    ex, _ = run_selfplay(total_games=n_total, n_games=n_games, sims=sims, device="cpu",
                         net=net, seed=seed, max_plies=max_plies)
    return ex


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="models/distill_d3_w.pt")
    ap.add_argument("--out", default="models/selfplay_par")
    ap.add_argument("--iters", type=int, default=25)
    ap.add_argument("--gpi", type=int, default=1000)
    ap.add_argument("--n-games", type=int, default=128)
    ap.add_argument("--sims", type=int, default=100)
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=5)
    ap.add_argument("--reward", default="outcome", choices=["dense", "outcome"])
    ap.add_argument("--device", default="mps")
    ap.add_argument("--workers", type=int, default=max(2, (os.cpu_count() or 4) - 1))
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--max-plies", type=int, default=80)
    a = ap.parse_args()

    net = QuoridorNet(a.channels, a.blocks)
    if a.init and os.path.exists(a.init):
        net.load_state_dict(torch.load(a.init, map_location="cpu"), strict=False)
    net = net.to(a.device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    os.makedirs(a.out, exist_ok=True)
    tmp_ckpt = os.path.join(a.out, "_current.pt")
    per_worker = max(1, a.gpi // a.workers)
    total_games = per_worker * a.workers
    print(f"=== parallel self-play from {a.init} | {a.channels}ch/{a.blocks}b | reward={a.reward} "
          f"| {a.workers} workers x {per_worker} games | device={a.device} -> {a.out} ===", flush=True)

    history = []
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for it in range(a.iters):
            lam = 1.0 if a.reward == "outcome" else anneal_lambda(it, a.iters, 0.8)
            torch.save(net.state_dict(), tmp_ckpt)        # workers load the current net
            t0 = time.time()
            tasks = [(tmp_ckpt, a.channels, a.blocks, per_worker, a.n_games, a.sims,
                      it * 10000 + w * 131 + 1, a.max_plies) for w in range(a.workers)]
            examples = []
            for chunk in pool.map(_gen_worker, tasks):
                examples.extend(chunk)
            gen_s = time.time() - t0

            diag = policy_diagnostics(examples)
            batch = form_dense_targets(augment_lr(examples), lam=lam, device="cpu")
            loss = train_minibatched(net, opt, batch, epochs=a.epochs, device=a.device)
            wr = (winrate_vs(net, a.device, opponent="greedy", games=10)
                  if (it % a.eval_every == 0 or it == a.iters - 1) else None)
            rec = dict(it=it, lam=round(lam, 3), loss=round(loss, 4),
                       mean_game_len=round(len(examples) / max(1, total_games), 1),
                       games_per_sec=round(total_games / gen_s, 2),
                       wall_mass=diag["wall_mass"], wall_argmax=diag["wall_argmax_rate"],
                       entropy=diag["entropy"], winrate=wr)
            history.append(rec)
            print(rec, flush=True)
            save_checkpoint(net, os.path.join(a.out, f"campaign_it{it}.pt"))
            with open(os.path.join(a.out, "history.json"), "w") as f:
                json.dump(history, f, indent=2)
    save_checkpoint(net, os.path.join(a.out, "campaign_final.pt"))
    print("=== parallel self-play done ===", flush=True)


if __name__ == "__main__":
    main()
