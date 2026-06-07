import random

import numpy as np

from core.rules import is_terminal, winner, apply_move
from core.state import initial_state
from agents.az.encoding import N_ACTIONS, encode_planes, move_to_action
from agents.az.mcts_nn import PUCTSearch


def play_selfplay_game(net_wrapper, sims=80, temp_moves=10, seed=None,
                       max_plies=200, dirichlet_alpha=0.5):
    rng = random.Random(seed)
    state = initial_state()
    history = []     # (planes, pi_vector, player_to_move)
    ply = 0
    while not is_terminal(state) and ply < max_plies:
        search = PUCTSearch(net_wrapper, sims=sims, seed=rng.randrange(1 << 30),
                            dirichlet_alpha=dirichlet_alpha)
        _, pi, _ = search.run(state)
        pi_vec = np.zeros(N_ACTIONS, dtype=np.float32)
        for m, p in pi.items():
            pi_vec[move_to_action(m, state)] = p
        history.append((encode_planes(state), pi_vec, state.turn))
        # sample a move from pi (temperature 1 early, then greedy)
        moves = list(pi.keys())
        probs = np.array([pi[m] for m in moves])
        if ply < temp_moves:
            choice = rng.choices(moves, weights=probs)[0]
        else:
            choice = moves[int(np.argmax(probs))]
        state = apply_move(state, choice)
        ply += 1
    w = winner(state)              # None if capped
    examples = []
    for planes, pi_vec, player in history:
        if w is None:
            z = 0.0
        else:
            z = 1.0 if w == player else -1.0
        examples.append((planes, pi_vec, z))
    return examples
