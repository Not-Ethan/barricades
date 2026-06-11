import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class _ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.b1 = nn.BatchNorm2d(ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.b2 = nn.BatchNorm2d(ch)

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return F.relu(x + y)


class SmallNet(nn.Module):
    """Small CNN over NxN: policy (n_actions) + value (tanh) heads."""

    def __init__(self, n_actions, channels=16, blocks=2):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(6, channels, 3, padding=1),
                                  nn.BatchNorm2d(channels), nn.ReLU())
        self.body = nn.Sequential(*[_ResBlock(channels) for _ in range(blocks)])
        self.p_conv = nn.Sequential(nn.Conv2d(channels, 2, 1),
                                    nn.BatchNorm2d(2), nn.ReLU())
        self.p_fc = nn.LazyLinear(n_actions)
        self.v_conv = nn.Sequential(nn.Conv2d(channels, 1, 1),
                                    nn.BatchNorm2d(1), nn.ReLU())
        self.v_fc1 = nn.LazyLinear(32)
        self.v_fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.body(self.stem(x))
        p = self.p_fc(self.p_conv(x).flatten(1))
        v = self.v_conv(x).flatten(1)
        v = torch.tanh(self.v_fc2(F.relu(self.v_fc1(v))))
        return p, v


class NetWrapper:
    """Predicts (priors over legal moves, value) for a state."""

    def __init__(self, net, engine, encoder, device="cpu"):
        self.net = net.to(device)
        self.e = engine
        self.enc = encoder
        self.device = device

    def predict(self, s):
        self.net.eval()
        planes = self.enc.encode_planes(s)
        x = torch.from_numpy(planes).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, value = self.net(x)
        logits = logits[0].cpu().numpy()
        legal = self.e.legal_moves(s)
        idxs = np.array([self.enc.move_to_action(m, s) for m in legal])
        sel = logits[idxs]
        sel = sel - sel.max()
        exp = np.exp(sel)
        probs = exp / exp.sum()
        return {m: float(p) for m, p in zip(legal, probs)}, float(value.item())
