import torch
import numpy as np
from agents.az.model import QuoridorNet


def test_three_heads_shapes():
    net = QuoridorNet(32, 3)
    p, v, d = net(torch.zeros(3, 6, 9, 9))
    assert p.shape == (3, 140)
    assert v.shape == (3, 1)
    assert d.shape == (3, 1)


def test_loads_two_head_checkpoint_strict_false():
    net = QuoridorNet(32, 3)
    # a 2-head state_dict = current dict minus the new distance-head params
    two_head = {k: t for k, t in net.state_dict().items() if not k.startswith("d_")}
    fresh = QuoridorNet(32, 3)
    missing, unexpected = fresh.load_state_dict(two_head, strict=False)
    assert not unexpected
    assert missing and all(k.startswith("d_") for k in missing)


def test_loads_real_bootstrap_checkpoint():
    import os
    ckpt = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "models", "az_bootstrap.pt")
    if not os.path.exists(ckpt):
        return  # bootstrap ckpt optional in CI
    net = QuoridorNet(32, 3)
    missing, unexpected = net.load_state_dict(torch.load(ckpt, map_location="cpu"),
                                              strict=False)
    assert all(k.startswith("d_") for k in missing)
