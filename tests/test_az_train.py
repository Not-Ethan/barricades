import numpy as np
import torch

from agents.az.model import QuoridorNet, NetWrapper
from agents.az.selfplay import play_selfplay_game
from agents.az.train import train_step, examples_to_batch
from agents.az.encoding import N_ACTIONS, N_PLANES


def test_selfplay_produces_examples():
    wrap = NetWrapper(QuoridorNet(channels=16, blocks=1))
    ex = play_selfplay_game(wrap, sims=10, temp_moves=4, seed=0, max_plies=60)
    assert len(ex) > 0
    planes, pi, z = ex[0]
    assert planes.shape == (N_PLANES, 9, 9)
    assert pi.shape == (N_ACTIONS,) and abs(pi.sum() - 1.0) < 1e-5
    assert z in (-1.0, 0.0, 1.0)


def test_train_step_overfits_tiny_batch():
    # Net should be able to drive loss DOWN on a fixed tiny batch (learning works).
    torch.manual_seed(0)
    net = QuoridorNet(channels=16, blocks=1)
    wrap = NetWrapper(net)
    ex = play_selfplay_game(wrap, sims=8, temp_moves=2, seed=1, max_plies=40)[:8]
    batch = examples_to_batch(ex)
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    first = train_step(net, opt, batch)
    for _ in range(30):
        last = train_step(net, opt, batch)
    assert last < first * 0.8      # loss dropped meaningfully
    assert np.isfinite(last)
