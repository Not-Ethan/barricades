import random
import numpy as np
from smallboard.mcts import PUCTSearch

_UNREACH = 1000


def play_game(engine, encoder, wrap, sims=40, temp_moves=6, seed=None,
              max_plies=80, dirichlet_alpha=0.6):
    """One self-play game. Returns examples
    (planes, pi_vec, z, path_diff, plies_to_end) per move."""
    rng = random.Random(seed)
    s = engine.initial_state()
    history = []
    ply = 0
    while not engine.is_terminal(s) and ply < max_plies:
        search = PUCTSearch(wrap, sims=sims, seed=rng.randrange(1 << 30),
                            dirichlet_alpha=dirichlet_alpha)
        _, pi, _ = search.run(s)
        pi_vec = np.zeros(encoder.n_actions, dtype=np.float32)
        for m, p in pi.items():
            pi_vec[encoder.move_to_action(m, s)] = p
        d_self = engine.shortest_path_len(s, s.turn)
        d_opp = engine.shortest_path_len(s, 1 - s.turn)
        path_diff = ((d_opp if d_opp is not None else _UNREACH)
                     - (d_self if d_self is not None else _UNREACH))
        history.append((encoder.encode_planes(s), pi_vec, s.turn, float(path_diff)))
        moves = list(pi.keys())
        probs = np.array([pi[m] for m in moves])
        if ply < temp_moves:
            choice = rng.choices(moves, weights=probs)[0]
        else:
            choice = moves[int(np.argmax(probs))]
        s = engine.apply_move(s, choice)
        ply += 1
    w = engine.winner(s)
    n = len(history)
    out = []
    for k, (planes, pi_vec, player, path_diff) in enumerate(history):
        z = 0.0 if w is None else (1.0 if w == player else -1.0)
        out.append((planes, pi_vec, z, path_diff, float(n - k)))
    return out
