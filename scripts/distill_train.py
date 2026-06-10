"""Distill a QuoridorNet from minimax data produced by scripts/distill_gen.py.

Supervised: policy head learns the expert (minimax) move (one-hot CE), value head
learns the game outcome (pure-outcome target, lam=1), aux dist head learns path_diff.
Reuses the existing AZ trainer.

Usage: python scripts/distill_train.py --data data/distill_d3.npz --out models/distill_d3.pt
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from agents.az.model import QuoridorNet
from agents.az.train import augment_lr, form_dense_targets, train_minibatched, save_checkpoint


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/distill_d3.npz")
    ap.add_argument("--out", default="models/distill_d3.pt")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--channels", type=int, default=32)
    ap.add_argument("--blocks", type=int, default=3)
    a = ap.parse_args()

    device = _resolve_device(a.device)
    examples = load_examples(a.data)
    print(f"loaded {len(examples)} positions; expert wall-move frac="
          f"{float((np.stack([e[1] for e in examples]).argmax(1) >= 12).mean()):.3f}")
    examples = augment_lr(examples)
    print(f"after LR augment: {len(examples)} | device={device}", flush=True)

    batch = form_dense_targets(examples, lam=1.0, device="cpu")   # lam=1 -> pure-outcome value
    net = QuoridorNet(channels=a.channels, blocks=a.blocks).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    for ep in range(a.epochs):
        loss = train_minibatched(net, opt, batch, epochs=1, device=device, seed=ep)
        print(f"epoch {ep + 1}/{a.epochs}: loss {loss:.4f}", flush=True)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    save_checkpoint(net, a.out)
    print(f"saved distilled net -> {a.out}")


if __name__ == "__main__":
    main()
