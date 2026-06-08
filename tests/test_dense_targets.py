import numpy as np
import torch
from agents.az.train import form_dense_targets, train_step_dense
from agents.az.model import QuoridorNet


def _ex(z, plies, path_diff):
    planes = np.zeros((6, 9, 9), dtype=np.float32)
    pi = np.full(140, 1.0 / 140, dtype=np.float32)
    feats = np.array([path_diff, 5.0, 5.0, plies], dtype=np.float32)
    return (planes, pi, float(z), feats)


def test_value_target_blend_and_discount():
    ex = [_ex(z=1.0, plies=4, path_diff=5.0)]
    # lam=1 -> pure discounted outcome z*gamma^plies
    _, _, v1, d1 = form_dense_targets(ex, lam=1.0, gamma=0.99, scale=5.0, dist_norm=10.0)
    assert abs(float(v1[0, 0]) - (1.0 * 0.99 ** 4)) < 1e-5
    assert abs(float(d1[0, 0]) - (5.0 / 10.0)) < 1e-6
    # lam=0 -> pure potential tanh(path_diff/scale)
    _, _, v0, _ = form_dense_targets(ex, lam=0.0, gamma=0.99, scale=5.0, dist_norm=10.0)
    assert abs(float(v0[0, 0]) - np.tanh(5.0 / 5.0)) < 1e-5
    # lam=0.5 -> average of the two
    _, _, vh, _ = form_dense_targets(ex, lam=0.5, gamma=0.99, scale=5.0, dist_norm=10.0)
    expect = 0.5 * (1.0 * 0.99 ** 4) + 0.5 * np.tanh(1.0)
    assert abs(float(vh[0, 0]) - expect) < 1e-5


def test_capped_draw_still_has_signal():
    # z=0 (draw) but path_diff>0 -> potential term keeps a positive value target
    ex = [_ex(z=0.0, plies=10, path_diff=4.0)]
    _, _, v, _ = form_dense_targets(ex, lam=0.5, gamma=0.99, scale=5.0, dist_norm=10.0)
    assert float(v[0, 0]) > 0.0


def test_train_step_reduces_loss():
    net = QuoridorNet(16, 2)
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    ex = [_ex(1.0, 6, 5.0), _ex(-1.0, 8, -4.0), _ex(0.0, 12, 1.0)] * 8
    batch = form_dense_targets(ex, lam=0.5, gamma=0.99, scale=5.0, dist_norm=10.0)
    first = train_step_dense(net, opt, batch, beta=1.0)
    for _ in range(15):
        last = train_step_dense(net, opt, batch, beta=1.0)
    assert last < first
