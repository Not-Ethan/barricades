"""Distill a QuoridorNet from minimax data (scripts/distill_gen.py).

Supervised imitation: policy head learns the expert (minimax) move, value head the
game outcome, aux dist head path_diff. Walls are the MINORITY class (~24% of expert
moves) and plain CE under-fits them, so we reweight the policy loss by class
(inverse-frequency by default): the net pays equal attention to the positions where
the EXPERT chose a wall and learns *that specific* wall — it is NOT a reward for
placing walls (the value target is purely the game outcome; pointless walls lose).

Usage: python scripts/distill_train.py --data data/distill_d3.npz --out models/distill_d3_w.pt \
         --channels 64 --blocks 5 --epochs 25 --wall-weight auto
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F

from agents.az.model import QuoridorNet
from agents.az.train import augment_lr, form_dense_targets, save_checkpoint

WALL_ACTION_START = 12


def load_examples(path):
    d = np.load(path)
    planes, pi, z, feats = d["planes"], d["pi"], d["z"], d["feats"]
    return [(planes[i], pi[i], float(z[i]), feats[i]) for i in range(len(z))]


def _resolve_device(want):
    if want == "mps" and torch.backends.mps.is_available():
        return "mps"
    if want == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def train_epoch(net, opt, planes, pi, v_t, d_t, w, device, batch_size, seed, beta):
    n = planes.shape[0]
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    net.train()
    losses = []
    for i in range(0, n, batch_size):
        idx = perm[i:i + batch_size]
        p = planes[idx].to(device)
        pj = pi[idx].to(device)
        vt = v_t[idx].to(device)
        dt = d_t[idx].to(device)
        wi = w[idx].to(device)
        logits, value, dist = net(p)
        logp = F.log_softmax(logits, dim=1)
        ce = -(pj * logp).sum(dim=1)                 # per-example policy CE
        policy_loss = (wi * ce).sum() / wi.sum()     # class-weighted mean
        value_loss = F.mse_loss(value, vt)
        dist_loss = F.mse_loss(dist, dt)
        loss = policy_loss + value_loss + beta * dist_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    return sum(losses) / max(1, len(losses))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/distill_d3.npz")
    ap.add_argument("--out", default="models/distill_d3_w.pt")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--wall-weight", default="auto",
                    help="'auto' = inverse-frequency (classes balanced), or a float")
    a = ap.parse_args()

    device = _resolve_device(a.device)
    examples = load_examples(a.data)
    examples = augment_lr(examples)
    planes, pi, v_t, d_t = form_dense_targets(examples, lam=1.0, device="cpu")

    is_wall = (pi.argmax(dim=1) >= WALL_ACTION_START)
    wf = float(is_wall.float().mean())
    if a.wall_weight == "auto":
        ww = (1.0 - wf) / wf                          # balance the two classes
    else:
        ww = float(a.wall_weight)
    w = torch.where(is_wall, torch.tensor(ww), torch.tensor(1.0))
    print(f"examples={planes.shape[0]} | expert wall frac={wf:.3f} | wall_weight={ww:.2f} "
          f"| net={a.channels}ch/{a.blocks}b | device={device}", flush=True)

    net = QuoridorNet(channels=a.channels, blocks=a.blocks).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    for ep in range(a.epochs):
        loss = train_epoch(net, opt, planes, pi, v_t, d_t, w, device, a.batch_size,
                           seed=ep, beta=1.0)
        print(f"epoch {ep + 1}/{a.epochs}: loss {loss:.4f}", flush=True)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    save_checkpoint(net, a.out)
    print(f"saved distilled net -> {a.out}")


if __name__ == "__main__":
    main()
