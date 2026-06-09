import numpy as np
import torch
from smallboard.engine import Engine, Step, State
from smallboard.encoding import Encoder
from smallboard.model import SmallNet, NetWrapper
from smallboard.mcts import PUCTSearch


def test_net_shapes():
    e = Engine(5, 3)
    enc = Encoder(e)
    net = SmallNet(enc.n_actions, channels=16, blocks=2)
    p, v = net(torch.zeros(2, 6, 5, 5))
    assert p.shape == (2, enc.n_actions) and v.shape == (2, 1)


def test_mcts_returns_legal_move_3x3():
    e = Engine(3, 1)
    enc = Encoder(e)
    wrap = NetWrapper(SmallNet(enc.n_actions, channels=8, blocks=1), e, enc)
    mv, pi, _ = PUCTSearch(wrap, sims=40, seed=0).run(e.initial_state())
    assert mv in e.legal_moves(e.initial_state())
    assert abs(sum(pi.values()) - 1.0) < 1e-5


def test_mcts_takes_immediate_win_3x3():
    e = Engine(3, 1)
    enc = Encoder(e)
    wrap = NetWrapper(SmallNet(enc.n_actions, channels=8, blocks=1), e, enc)
    # p0 at (1,1) is one step from its goal row 2; p1 at (0,2) is NOT on its goal
    # (row 0), so this is non-terminal and p0 has a genuine immediate win.
    s = State(((1, 1), (0, 2)), frozenset(), frozenset(), (1, 1), 0)
    assert not e.is_terminal(s)            # sanity: the fixture is a live position
    mv, _, _ = PUCTSearch(wrap, sims=80, seed=0).run(s)
    assert isinstance(mv, Step) and mv.to_cell == (1, 2)
