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
    logits, value, _dist = net(planes)
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


def form_dense_targets(examples, lam, gamma=0.99, scale=5.0, dist_norm=10.0,
                       device="cpu"):
    """examples: (planes(6,9,9), pi(140), z, feats=[path_diff, wl_own, wl_opp, plies_to_end]).
    v_target = lam*(z*gamma**plies_to_end) + (1-lam)*tanh(path_diff/scale).
    dist_target = path_diff/dist_norm. Returns (planes, pi, v_target, dist_target) tensors."""
    planes = torch.from_numpy(np.stack([e[0] for e in examples])).to(device)
    pi = torch.from_numpy(np.stack([e[1] for e in examples])).to(device)
    z = np.array([e[2] for e in examples], dtype=np.float32)
    feats = np.stack([np.asarray(e[3], dtype=np.float32) for e in examples])  # (N,4)
    path_diff = feats[:, 0]
    plies = feats[:, 3]
    shaped = z * (gamma ** plies)
    potential = np.tanh(path_diff / scale)
    v_target = lam * shaped + (1.0 - lam) * potential
    dist_target = path_diff / dist_norm
    v_t = torch.from_numpy(v_target.astype(np.float32)).unsqueeze(1).to(device)
    d_t = torch.from_numpy(dist_target.astype(np.float32)).unsqueeze(1).to(device)
    return planes, pi, v_t, d_t


def train_step_dense(net, optimizer, batch, beta=1.0):
    """3-head train step: policy CE + value MSE + beta * distance MSE."""
    net.train()
    planes, target_pi, target_v, target_d = batch
    logits, value, dist = net(planes)
    logp = F.log_softmax(logits, dim=1)
    policy_loss = -(target_pi * logp).sum(dim=1).mean()
    value_loss = F.mse_loss(value, target_v)
    dist_loss = F.mse_loss(dist, target_d)
    loss = policy_loss + value_loss + beta * dist_loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss.item())


def train_minibatched(net, optimizer, batch, epochs=4, batch_size=2048, beta=1.0,
                      device="cpu", seed=0):
    """Minibatched SGD over a full (planes, pi, v_target, dist_target) batch.

    The full batch is kept where `form_dense_targets` built it (pass device="cpu"
    so a whole iteration's examples don't all sit in GPU memory); each minibatch is
    moved to `device` for the forward/backward. This bounds activation memory to
    `batch_size` rows — full-batch training on a 1000-game iteration (~80k rows)
    would OOM the GPU. Returns the mean per-minibatch loss over all steps.
    """
    planes, pi, v_t, d_t = batch
    n = planes.shape[0]
    g = torch.Generator().manual_seed(seed)
    losses = []
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            mb = (planes[idx].to(device), pi[idx].to(device),
                  v_t[idx].to(device), d_t[idx].to(device))
            losses.append(train_step_dense(net, optimizer, mb, beta=beta))
    return sum(losses) / max(1, len(losses))
