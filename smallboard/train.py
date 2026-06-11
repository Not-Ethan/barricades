import numpy as np
import torch
import torch.nn.functional as F


def form_targets(examples, n_actions, lam, gamma=0.99, scale=4.0):
    """examples: (planes, pi_vec, z, path_diff, plies_to_end).
    v_target = lam*(z*gamma**plies) + (1-lam)*tanh(path_diff/scale)."""
    planes = torch.from_numpy(np.stack([e[0] for e in examples]))
    pi = torch.from_numpy(np.stack([e[1] for e in examples]))
    z = np.array([e[2] for e in examples], dtype=np.float32)
    path_diff = np.array([e[3] for e in examples], dtype=np.float32)
    plies = np.array([e[4] for e in examples], dtype=np.float32)
    v = lam * (z * gamma ** plies) + (1.0 - lam) * np.tanh(path_diff / scale)
    v_t = torch.from_numpy(v.astype(np.float32)).unsqueeze(1)
    return planes, pi, v_t


def train_step(net, optimizer, batch):
    net.train()
    planes, target_pi, target_v = batch
    logits, value = net(planes)
    logp = F.log_softmax(logits, dim=1)
    policy_loss = -(target_pi * logp).sum(dim=1).mean()
    value_loss = F.mse_loss(value, target_v)
    loss = policy_loss + value_loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss.item())
