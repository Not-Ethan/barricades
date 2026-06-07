import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.rules import legal_moves
from agents.az.encoding import N_ACTIONS, N_PLANES, encode_planes, move_to_action


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


class QuoridorNet(nn.Module):
    def __init__(self, channels=32, blocks=3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(N_PLANES, channels, 3, padding=1),
            nn.BatchNorm2d(channels), nn.ReLU())
        self.body = nn.Sequential(*[_ResBlock(channels) for _ in range(blocks)])
        self.p_conv = nn.Sequential(nn.Conv2d(channels, 2, 1),
                                    nn.BatchNorm2d(2), nn.ReLU())
        self.p_fc = nn.Linear(2 * 9 * 9, N_ACTIONS)
        self.v_conv = nn.Sequential(nn.Conv2d(channels, 1, 1),
                                    nn.BatchNorm2d(1), nn.ReLU())
        self.v_fc1 = nn.Linear(9 * 9, 64)
        self.v_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.body(self.stem(x))
        p = self.p_fc(self.p_conv(x).flatten(1))
        v = self.v_conv(x).flatten(1)
        v = torch.tanh(self.v_fc2(F.relu(self.v_fc1(v))))
        return p, v


class NetWrapper:
    """Holds a net; predicts (priors over legal real moves, value) for a state."""

    def __init__(self, net, device="cpu"):
        self.net = net.to(device)
        self.device = device

    def predict(self, state):
        self.net.eval()
        planes = encode_planes(state)
        x = torch.from_numpy(planes).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, value = self.net(x)
        logits = logits[0].cpu().numpy()
        legal = legal_moves(state)
        idxs = np.array([move_to_action(m, state) for m in legal])
        sel = logits[idxs]
        sel = sel - sel.max()
        exp = np.exp(sel)
        probs = exp / exp.sum()
        priors = {m: float(p) for m, p in zip(legal, probs)}
        return priors, float(value.item())
