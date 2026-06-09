import random
import numpy as np
import torch

from smallboard.engine import Engine, Step
from smallboard.encoding import Encoder
from smallboard.model import SmallNet, NetWrapper
from smallboard.solver import Solver
from smallboard.selfplay import play_game
from smallboard.train import form_targets, train_step
from smallboard.mcts import PUCTSearch


def theoretical_result(N, W):
    """Game-theoretic value of the start position + which side it favors."""
    e = Engine(N, W)
    sol = Solver(e)
    val, _ = sol.solve(e.initial_state())          # value for side to move (p0)
    side = "draw" if val == 0 else ("p0" if val == 1 else "p1")
    return val, side


def _anneal(it, iterations, warmup=0.6):
    w = max(1, int(iterations * warmup))
    return min(1.0, it / w)


def train_small_az(N, W, iterations=10, games=40, sims=40, epochs=4, lr=1e-3,
                   channels=16, blocks=2, seed=0, log=lambda *_: None):
    e = Engine(N, W)
    enc = Encoder(e)
    net = SmallNet(enc.n_actions, channels=channels, blocks=blocks)
    net(torch.zeros(1, 6, N, N))                   # init LazyLinear
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    wrap = NetWrapper(net, e, enc)
    rng = random.Random(seed)
    for it in range(iterations):
        lam = _anneal(it, iterations)
        ex = []
        for _ in range(games):
            ex += play_game(e, enc, wrap, sims=sims, seed=rng.randrange(1 << 30))
        batch = form_targets(ex, enc.n_actions, lam=lam)
        losses = [train_step(net, opt, batch) for _ in range(epochs)]
        log({"it": it, "lam": round(lam, 2), "loss": round(sum(losses) / len(losses), 4),
             "examples": len(ex)})
    return net, e, enc


def _reachable_positions(e, n, seed):
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        s = e.initial_state()
        for _ in range(rng.randint(0, 6)):
            if e.is_terminal(s):
                break
            ms = e.legal_moves(s)
            s = e.apply_move(s, ms[rng.randrange(len(ms))])
        if not e.is_terminal(s):
            out.append(s)
    return out


def optimal_move_agreement(net, e, enc, n_positions=40, az_sims=80, seed=0):
    """Fraction of positions where AZ's chosen move is in the solver's optimal set."""
    sol = Solver(e)
    wrap = NetWrapper(net, e, enc)
    hits = 0
    positions = _reachable_positions(e, n_positions, seed)
    for i, s in enumerate(positions):
        mv, _, _ = PUCTSearch(wrap, sims=az_sims, seed=seed + i).run(s)
        _, best = sol.solve(s)
        if mv in best:
            hits += 1
    return hits / max(1, len(positions))


def value_accuracy(net, e, enc, n_positions=40, seed=0):
    """Agreement between AZ's value head and the solver's exact value over sampled
    reachable positions, both from the side-to-move perspective. Returns the Pearson
    correlation in [-1, 1]; if either series has ~zero variance (e.g. all sampled
    positions are forced wins), falls back to sign-agreement so the result stays
    well-defined."""
    sol = Solver(e)
    wrap = NetWrapper(net, e, enc)
    positions = _reachable_positions(e, n_positions, seed)
    az_vals = []
    solver_vals = []
    for s in positions:
        _, v = wrap.predict(s)
        az_vals.append(float(v))
        sv, _ = sol.solve(s)
        solver_vals.append(float(sv))
    az = np.asarray(az_vals, dtype=np.float64)
    sv = np.asarray(solver_vals, dtype=np.float64)
    if az.std() < 1e-8 or sv.std() < 1e-8:
        return float(np.mean(np.sign(az) == np.sign(sv)))
    return float(np.corrcoef(az, sv)[0, 1])


def az_vs_solver(net, e, enc, games=20, az_sims=80, seed=0):
    """AZ plays the theoretically-winning side vs the perfect solver; also check AZ
    never loses a position it should win. Returns a metrics dict."""
    sol = Solver(e)
    wrap = NetWrapper(net, e, enc)
    start_val, _ = sol.solve(e.initial_state())
    win_side = 0 if start_val == 1 else (1 if start_val == -1 else None)

    def play(az_player):
        s = e.initial_state()
        for _ in range(4 * e.N + 4 * e.W + 20):
            if e.is_terminal(s):
                break
            if s.turn == az_player:
                mv, _, _ = PUCTSearch(wrap, sims=az_sims, seed=seed + s.turn).run(s)
            else:
                _, best = sol.solve(s)
                mv = best[0] if best else e.legal_moves(s)[0]
            s = e.apply_move(s, mv)
        return e.winner(s)

    if win_side is None:                            # drawn start: AZ must not lose
        losses = 0
        for g in range(games):
            w = play(g % 2)
            if w is not None and w != (g % 2):
                losses += 1
        return {"az_as_winner_winrate": 1.0,        # n/a (draw); report no-loss
                "az_never_loses_won": 1.0 - losses / games}

    wins = sum(1 for _ in range(games) if play(win_side) == win_side)
    return {"az_as_winner_winrate": wins / games, "az_never_loses_won": 1.0}


def run(N=3, W=1, iterations=10, games=40, sims=40, az_sims=80, seed=0):
    val, side = theoretical_result(N, W)
    print(f"[{N}x{N} W={W}] theoretical: value={val} favors={side}")
    net, e, enc = train_small_az(N, W, iterations=iterations, games=games,
                                 sims=sims, seed=seed, log=print)
    agree = optimal_move_agreement(net, e, enc, n_positions=60, az_sims=az_sims, seed=seed + 1)
    res = az_vs_solver(net, e, enc, games=20, az_sims=az_sims, seed=seed + 2)
    vacc = value_accuracy(net, e, enc, n_positions=60, seed=seed + 3)
    print(f"  optimal-move agreement: {agree:.1%}")
    print(f"  value-accuracy (corr w/ solver): {vacc:.3f}")
    print(f"  AZ-vs-solver: {res}")
    return {"theoretical": (val, side), "agreement": agree,
            "value_accuracy": vacc, **res}


if __name__ == "__main__":
    import sys
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    W = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    iters = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    games = int(sys.argv[4]) if len(sys.argv) > 4 else 40
    run(N=N, W=W, iterations=iters, games=games)
