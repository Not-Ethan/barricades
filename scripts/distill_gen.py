"""Generate distillation data: minimax self-play games (random openings for
diversity), labelling every minimax-decided position with the expert move
(one-hot policy) + game outcome (value) + path-diff feats. Saves an .npz that
scripts/distill_train.py consumes.

Usage: python scripts/distill_gen.py --games 2500 --depth 3 --out data/distill_d3.npz [--workers N]
"""
import argparse
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import barricades_native as bn
from core.state import initial_state, Step
from core.rules import apply_move, is_terminal, winner
from agents.minimax_agent import MinimaxAgent
from agents.native_agent import _to_native, _from_tuple

N_ACTIONS = 140
_INF_DIST = 1000


def _move_to_tuple(m):
    if isinstance(m, Step):
        return ("step", m.to_cell[0], m.to_cell[1])
    return ("wall", m.c, m.r, m.orient)


def _features(nt):
    turn, wl = nt[4], nt[3]
    d_self = bn.shortest_path_len(nt, turn)
    d_opp = bn.shortest_path_len(nt, 1 - turn)
    d_self = _INF_DIST if d_self is None else d_self
    d_opp = _INF_DIST if d_opp is None else d_opp
    return [float(d_opp - d_self), float(wl[turn]), float(wl[1 - turn]), 0.0]


def gen_game(args):
    seed, depth, k_lo, k_hi, max_plies = args
    rng = random.Random(seed)
    mm = MinimaxAgent(max_depth=depth, time_budget=1e9, wall_cap=12)
    gs = initial_state()
    # Random opening (diversity): a few uniformly-random legal plies.
    for _ in range(rng.randint(k_lo, k_hi)):
        if is_terminal(gs):
            break
        gs = apply_move(gs, _from_tuple(rng.choice(bn.legal_moves(_to_native(gs)))))
    # Expert (minimax) plays out; record every decided position.
    records = []
    plies = 0
    while not is_terminal(gs) and plies < max_plies:
        nt = _to_native(gs)
        move = mm.select_move(gs)
        action = bn.move_to_action(_move_to_tuple(move), nt)
        planes = np.asarray(bn.encode_planes(nt), dtype=np.float32)
        records.append((planes, action, gs.turn, _features(nt)))
        gs = apply_move(gs, move)
        plies += 1
    w = winner(gs)            # None if capped (draw)
    n = len(records)
    out = []
    for k, (planes, action, mover, feats) in enumerate(records):
        z = 0.0 if w is None else (1.0 if w == mover else -1.0)
        pi = np.zeros(N_ACTIONS, dtype=np.float32)
        pi[action] = 1.0
        f = list(feats)
        f[3] = float(n - k)   # plies_to_end
        out.append((planes, pi, np.float32(z), np.asarray(f, dtype=np.float32)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=2500)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--out", default="data/distill_d3.npz")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--k-open-lo", type=int, default=4)
    ap.add_argument("--k-open-hi", type=int, default=12)
    ap.add_argument("--max-plies", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    tasks = [(a.seed + i, a.depth, a.k_open_lo, a.k_open_hi, a.max_plies) for i in range(a.games)]
    planes_l, pi_l, z_l, feats_l = [], [], [], []
    done = 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for game in ex.map(gen_game, tasks, chunksize=4):
            for planes, pi, z, feats in game:
                planes_l.append(planes); pi_l.append(pi); z_l.append(z); feats_l.append(feats)
            done += 1
            if done % 100 == 0:
                print(f"  {done}/{a.games} games, {len(planes_l)} positions", flush=True)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    np.savez_compressed(a.out,
                        planes=np.stack(planes_l), pi=np.stack(pi_l),
                        z=np.asarray(z_l, dtype=np.float32), feats=np.stack(feats_l))
    print(f"saved {len(planes_l)} positions from {a.games} games -> {a.out}")


if __name__ == "__main__":
    main()
