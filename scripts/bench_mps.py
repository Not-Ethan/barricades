"""Benchmark QuoridorNet on CPU vs MPS (Apple GPU): inference at various batch
sizes, training throughput, and correctness. Tells us whether batching on MPS
makes GPU-accelerated self-play/training worthwhile.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from agents.az.model import QuoridorNet, N_PLANES, N_ACTIONS


def sync(dev):
    if dev == "mps":
        torch.mps.synchronize()
    elif dev == "cuda":
        torch.cuda.synchronize()


def bench_forward(net, dev, bs, iters=30, warmup=8):
    net = net.to(dev).eval()
    x = torch.randn(bs, N_PLANES, 9, 9, device=dev)
    with torch.no_grad():
        for _ in range(warmup):
            net(x)
        sync(dev)
        t0 = time.perf_counter()
        for _ in range(iters):
            net(x)
        sync(dev)
    return (time.perf_counter() - t0) / iters  # sec/batch


def bench_train(net, dev, bs, iters=15, warmup=4):
    net = net.to(dev).train()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    x = torch.randn(bs, N_PLANES, 9, 9, device=dev)
    pi = torch.softmax(torch.randn(bs, N_ACTIONS, device=dev), dim=1)
    z = torch.randn(bs, 1, device=dev)

    def step():
        logits, val = net(x)
        loss = -(pi * F.log_softmax(logits, 1)).sum(1).mean() + F.mse_loss(val, z)
        opt.zero_grad(); loss.backward(); opt.step()

    for _ in range(warmup):
        step()
    sync(dev)
    t0 = time.perf_counter()
    for _ in range(iters):
        step()
    sync(dev)
    return (time.perf_counter() - t0) / iters


def correctness(net):
    net_cpu = net.to("cpu").eval()
    x = torch.randn(16, N_PLANES, 9, 9)
    with torch.no_grad():
        pc, vc = net_cpu(x)
        net_mps = net.to("mps").eval()
        pm, vm = net_mps(x.to("mps"))
    pm, vm = pm.cpu(), vm.cpu()
    return (pc - pm).abs().max().item(), (vc - vm).abs().max().item()


def main():
    print(f"torch {torch.__version__}  mps_available={torch.backends.mps.is_available()}"
          f"  mps_built={torch.backends.mps.is_built()}  cpu_threads={torch.get_num_threads()}")
    if not torch.backends.mps.is_available():
        print("MPS not available."); return

    for ch, bl, tag in [(32, 3, "current 32ch/3blk"), (64, 5, "bigger 64ch/5blk")]:
        print(f"\n=== {tag} ===")
        net = QuoridorNet(channels=ch, blocks=bl)
        dmax, vmax = correctness(QuoridorNet(channels=ch, blocks=bl))
        print(f"  MPS vs CPU output max-abs-diff: policy={dmax:.2e} value={vmax:.2e}")
        print(f"  {'batch':>6} | {'CPU pos/s':>11} | {'MPS pos/s':>11} | {'speedup':>7}")
        for bs in [1, 8, 32, 128, 512, 2048, 8192]:
            cpu_s = bench_forward(net, "cpu", bs)
            mps_s = bench_forward(net, "mps", bs)
            cps, mps_ps = bs / cpu_s, bs / mps_s
            print(f"  {bs:>6} | {cps:>11,.0f} | {mps_ps:>11,.0f} | {mps_ps/cps:>6.1f}x")
        # training throughput
        for bs in [256, 1024]:
            ct = bench_train(net, "cpu", bs)
            mt = bench_train(net, "mps", bs)
            print(f"  TRAIN bs={bs}: CPU {bs/ct:,.0f} pos/s  MPS {bs/mt:,.0f} pos/s "
                  f"({(bs/mt)/(bs/ct):.1f}x)")


if __name__ == "__main__":
    main()
