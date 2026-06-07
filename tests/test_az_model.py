import torch

from core.state import initial_state
from core.rules import legal_moves
from agents.az.encoding import N_ACTIONS, N_PLANES
from agents.az.model import QuoridorNet, NetWrapper


def test_forward_shapes():
    net = QuoridorNet(channels=16, blocks=2)
    x = torch.zeros(4, N_PLANES, 9, 9)
    logits, value = net(x)
    assert logits.shape == (4, N_ACTIONS)
    assert value.shape == (4, 1)
    assert torch.all(value <= 1) and torch.all(value >= -1)


def test_predict_returns_legal_priors_and_value():
    net = QuoridorNet(channels=16, blocks=2)
    wrap = NetWrapper(net)
    s = initial_state()
    priors, value = wrap.predict(s)
    legal = set(legal_moves(s))
    assert set(priors.keys()) == legal           # priors only over legal moves
    assert abs(sum(priors.values()) - 1.0) < 1e-4  # normalized
    assert all(p >= 0 for p in priors.values())
    assert -1.0 <= value <= 1.0


def test_predict_is_deterministic_in_eval():
    net = QuoridorNet(channels=16, blocks=2)
    wrap = NetWrapper(net)
    s = initial_state()
    p1, v1 = wrap.predict(s)
    p2, v2 = wrap.predict(s)
    assert v1 == v2 and p1 == p2
