"""Two-net self-play over the native Tree API, for the opponent-pool arms (C, D).

The native `SelfPlayPool` drives every game with ONE net (canonical, mover's-POV
encoding), so it cannot put a DIFFERENT net on each side of a game. An opponent
pool needs exactly that: the learner on one side, a frozen past-checkpoint net on
the other. This driver replicates the native self-play loop in Python over per-game
`bn.Tree` objects and routes each game's leaf evaluations to the net controlling
that game's current root mover (an AZ agent searches the whole move with its own
net, modelling the opponent with its own eval -> route by root mover, not leaf
depth).

Faithful to native semantics (subtree carryover, temp_moves temperature, Dirichlet
root noise on first root expansion, sims accounting, the `features()` tuple and z
convention from native/src/selfplay.rs). Validated against the native pool in
tests/test_selfplay_pool.py. Slower than the rayon pool (Python per-slot loop), but
the box is latency-bound and idle, so we run pool arms as their own processes.

Only the LEARNER's positions become training examples. A `pool_frac` fraction of
games are learner-vs-pool; the rest are ordinary self-play (learner on both sides,
both sides recorded).
"""
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

import barricades_native as bn
from agents.native_agent import _to_native
from core.state import initial_state

WALL_DIST_INF = 1000  # native bfs_dist().unwrap_or(1000)


def _forward(net, planes, device):
    """planes: (B,6,9,9) -> (policy (B,140) softmaxed, value (B,))."""
    x = torch.from_numpy(np.asarray(planes)).to(device)
    with torch.no_grad():
        out = net(x)
        pol = torch.softmax(out[0], dim=1).cpu().numpy()
        val = out[1].squeeze(1).cpu().numpy()
    return pol, val


def _features(state):
    """Match native selfplay.rs features(): [d_opp - d_self, wl_mover, wl_opp, 0].
    feats[3] (plies_to_end) is filled at finalize."""
    turn = state[4]
    wl = state[3]
    d_self = bn.shortest_path_len(state, turn)
    d_opp = bn.shortest_path_len(state, 1 - turn)
    d_self = WALL_DIST_INF if d_self is None else d_self
    d_opp = WALL_DIST_INF if d_opp is None else d_opp
    return [float(d_opp - d_self), float(wl[turn]), float(wl[1 - turn]), 0.0]


class _Slot:
    __slots__ = ("tree", "state", "ply", "sims_done", "phase", "records",
                 "learner_sides", "opp_net", "pending", "active")

    def __init__(self):
        self.active = False


class _PoolDriver:
    def __init__(self, total_games, n_games, sims, device, learner_net, opponent_nets,
                 pool_frac, seed, max_plies, temp_moves, c_puct, dir_alpha, dir_eps):
        self.total = total_games
        self.sims = sims
        self.device = device
        self.learner = learner_net
        self.opps = list(opponent_nets or [])
        self.pool_frac = pool_frac if self.opps else 0.0
        self.max_plies = max_plies
        self.temp_moves = temp_moves
        self.c_puct = c_puct
        self.dir_alpha = dir_alpha
        self.dir_eps = dir_eps
        self.rng = random.Random(seed)
        self.next_seed = seed
        self.launched = 0
        self.finished = 0
        self.out = []
        self.init_native = _to_native(initial_state())
        self.slots = [_Slot() for _ in range(min(n_games, total_games))]
        for s in self.slots:
            self._start(s)

    # --- per-slot lifecycle -------------------------------------------------
    def _assign(self):
        """Return (learner_sides set, opp_net or None)."""
        if self.opps and self.rng.random() < self.pool_frac:
            side = self.rng.randint(0, 1)
            return {side}, self.opps[self.rng.randrange(len(self.opps))]
        return {0, 1}, None  # ordinary self-play: learner on both sides

    def _start(self, slot):
        if self.launched >= self.total:
            slot.active = False
            return
        seed = self.next_seed
        self.next_seed += 1
        slot.tree = bn.Tree(self.init_native, self.c_puct, seed)
        slot.state = self.init_native
        slot.ply = 0
        slot.sims_done = 0
        slot.phase = "await"
        slot.records = []
        slot.learner_sides, slot.opp_net = self._assign()
        slot.pending = None
        slot.active = True
        self.launched += 1

    def _eval_net(self, slot):
        return self.learner if slot.state[4] in slot.learner_sides else slot.opp_net

    def _commit(self, slot):
        temp = 1.0 if slot.ply < self.temp_moves else 0.0
        mv, pi = slot.tree.best_move(temp)
        pre = slot.state
        if pre[4] in slot.learner_sides:
            planes = np.asarray(bn.encode_planes(pre), dtype=np.float32)
            slot.records.append((planes, np.asarray(pi, dtype=np.float32), pre[4],
                                 _features(pre)))
        nxt = bn.apply_move(pre, mv)
        slot.state = nxt
        slot.ply += 1
        if bn.is_terminal(nxt):
            self._finalize(slot, bn.winner(nxt)); return
        if slot.ply >= self.max_plies:
            self._finalize(slot, None); return  # cap -> draw
        slot.tree.advance(mv)  # subtree carryover
        slot.sims_done = min(slot.tree.root_visits(), self.sims)
        # Native guards this with root_expanded(), which isn't exposed to Python.
        # apply_root_noise is a documented no-op on an unexpanded root (the feed
        # trigger then handles noise) and is idempotent, so call unconditionally.
        if self.dir_alpha > 0.0:
            slot.tree.apply_root_noise(self.dir_alpha, self.dir_eps)
        slot.phase = "await"

    def _finalize(self, slot, winner):
        recs = slot.records
        n = len(recs)
        for k, (planes, pi, player, feats) in enumerate(recs):
            z = 0.0 if winner is None else (1.0 if winner == player else -1.0)
            f = list(feats)
            f[3] = float(n - k)  # plies_to_end, matching native finalize
            self.out.append((planes, pi, z, np.asarray(f, dtype=np.float32)))
        self.finished += 1
        self._start(slot)  # refill (or deactivate if launched == total)

    # --- main loop ----------------------------------------------------------
    def run(self):
        t0 = time.perf_counter()
        while self.finished < self.total:
            # 1) commit moves for slots that finished their search last tick
            for s in self.slots:
                if s.active and s.phase == "ready":
                    self._commit(s)
            # 2) advance each searching slot to its next leaf needing eval
            pending = []
            for s in self.slots:
                if not s.active or s.phase != "await":
                    continue
                while True:
                    planes = s.tree.prepare_leaf()
                    if planes is None:           # terminal leaf: free sim, no eval
                        s.sims_done += 1
                        if s.sims_done >= self.sims:
                            s.phase = "ready"
                            break
                    else:
                        s.pending = np.asarray(planes, dtype=np.float32)
                        pending.append(s)
                        break
            # 3) evaluate pending leaves, grouped by the net that owns each game
            if pending:
                groups = {}
                for s in pending:
                    net = self._eval_net(s)
                    groups.setdefault(id(net), (net, []))[1].append(s)
                for _gid, (net, slots) in groups.items():
                    batch = np.stack([s.pending for s in slots])
                    pol, val = _forward(net, batch, self.device)
                    for row, s in enumerate(slots):
                        s.tree.receive(np.ascontiguousarray(pol[row], dtype=np.float32),
                                       float(val[row]))
                        if s.sims_done == 0 and self.dir_alpha > 0.0:
                            s.tree.apply_root_noise(self.dir_alpha, self.dir_eps)
                        s.sims_done += 1
                        if s.sims_done >= self.sims:
                            s.phase = "ready"
                        s.pending = None
        dt = time.perf_counter() - t0
        stats = dict(games=self.total, seconds=dt,
                     games_per_sec=self.total / dt if dt > 0 else 0.0,
                     examples=len(self.out))
        return self.out, stats


def run_selfplay_pool(total_games=512, n_games=256, sims=100, device="cuda",
                      learner_net=None, opponent_nets=None, pool_frac=0.5, seed=0,
                      max_plies=200, temp_moves=10, c_puct=1.5,
                      dir_alpha=0.5, dir_eps=0.25):
    """Self-play with an opponent pool. Returns (examples, stats) with the same
    example shape as scripts.selfplay_native.run_selfplay:
    (planes(6,9,9), pi(140), z, feats(4)).

    learner_net: the net being trained (required; eval mode set by caller).
    opponent_nets: list of frozen nets; if empty/None -> pure self-play.
    pool_frac: fraction of games that are learner-vs-pool (rest are self-play).
    """
    if learner_net is None:
        raise ValueError("run_selfplay_pool requires a learner_net")
    learner_net.eval()
    for op in (opponent_nets or []):
        op.eval()
    driver = _PoolDriver(total_games, n_games, sims, device, learner_net,
                         opponent_nets, pool_frac, seed, max_plies, temp_moves,
                         c_puct, dir_alpha, dir_eps)
    return driver.run()
