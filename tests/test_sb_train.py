import numpy as np
import torch
from smallboard.engine import Engine
from smallboard.encoding import Encoder
from smallboard.model import SmallNet, NetWrapper
from smallboard.selfplay import play_game
from smallboard.train import form_targets, train_step


def test_selfplay_produces_wellformed_examples_3x3():
    e = Engine(3, 1)
    enc = Encoder(e)
    wrap = NetWrapper(SmallNet(enc.n_actions, channels=8, blocks=1), e, enc)
    ex = play_game(e, enc, wrap, sims=20, seed=0, max_plies=40)
    assert len(ex) > 0
    for planes, pi, z, pathdiff, plies in ex:
        assert planes.shape == (6, 3, 3)
        assert pi.shape == (enc.n_actions,)
        assert abs(float(pi.sum()) - 1.0) < 1e-4
        assert z in (-1.0, 0.0, 1.0)


def test_train_step_reduces_loss():
    e = Engine(3, 1)
    enc = Encoder(e)
    net = SmallNet(enc.n_actions, channels=8, blocks=1)
    net(torch.zeros(1, 6, 3, 3))   # init LazyLinear
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    wrap = NetWrapper(net, e, enc)
    ex = []
    for g in range(6):
        ex += play_game(e, enc, wrap, sims=20, seed=g, max_plies=40)
    batch = form_targets(ex, enc.n_actions, lam=0.5)
    first = train_step(net, opt, batch)
    for _ in range(20):
        last = train_step(net, opt, batch)
    assert last < first
