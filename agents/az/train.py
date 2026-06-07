import numpy as np
import torch
import torch.nn.functional as F


def examples_to_batch(examples, device="cpu"):
    planes = torch.from_numpy(np.stack([e[0] for e in examples])).to(device)
    pi = torch.from_numpy(np.stack([e[1] for e in examples])).to(device)
    z = torch.tensor([e[2] for e in examples], dtype=torch.float32,
                     device=device).unsqueeze(1)
    return planes, pi, z


def train_step(net, optimizer, batch):
    net.train()
    planes, target_pi, target_z = batch
    logits, value = net(planes)
    logp = F.log_softmax(logits, dim=1)
    policy_loss = -(target_pi * logp).sum(dim=1).mean()
    value_loss = F.mse_loss(value, target_z)
    loss = policy_loss + value_loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss.item())


def run_training(net, iterations=3, games_per_iter=4, sims=60, epochs=4,
                 lr=1e-3, seed=0, log=print):
    """Self-play + train loop. Returns list of per-iteration mean losses."""
    import random
    from agents.az.model import NetWrapper
    from agents.az.selfplay import play_selfplay_game
    rng = random.Random(seed)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    wrap = NetWrapper(net)
    history = []
    for it in range(iterations):
        examples = []
        for g in range(games_per_iter):
            examples += play_selfplay_game(wrap, sims=sims, seed=rng.randrange(1 << 30))
        batch = examples_to_batch(examples)
        losses = [train_step(net, opt, batch) for _ in range(epochs)]
        history.append(sum(losses) / len(losses))
        log(f"iter {it+1}/{iterations}: examples={len(examples)} "
            f"loss={history[-1]:.4f}")
    return history


def save_checkpoint(net, path):
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(net.state_dict(), path)


def load_checkpoint(net, path):
    net.load_state_dict(torch.load(path, map_location="cpu"))
    return net
