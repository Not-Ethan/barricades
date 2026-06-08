"""Parallel new-vs-old engine tournament (games run across all CPU cores).

Usage: python scripts/tournament.py [games] [time_budget]
Default: 200 games per matchup at 0.1s/move.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from concurrent.futures import ProcessPoolExecutor

from core.state import initial_state
from core.rules import apply_move, is_terminal, winner
from agents.minimax_agent import MinimaxAgent
from agents.mcts_agent import MCTSAgent
from agents.greedy_agent import GreedyAgent
from agents.movegen import probable_moves


def make(spec, seed, tb):
    if spec == "mm_base":
        return MinimaxAgent(time_budget=tb, seed=seed)
    if spec == "mm_prob":
        return MinimaxAgent(time_budget=tb, seed=seed, candidate_moves=probable_moves)
    if spec == "mc_base":
        return MCTSAgent(time_budget=tb, seed=seed)
    if spec == "mc_prob":
        return MCTSAgent(time_budget=tb, seed=seed, candidate_moves=probable_moves)
    if spec == "greedy":
        return GreedyAgent(seed=seed)
    raise ValueError(spec)


def play_one(args):
    a_spec, b_spec, i, tb, max_plies = args
    # Alternate which spec is player 0 across games.
    if i % 2 == 0:
        p0, p1, a_is_p0 = make(a_spec, i, tb), make(b_spec, 9000 + i, tb), True
    else:
        p0, p1, a_is_p0 = make(b_spec, 9000 + i, tb), make(a_spec, i, tb), False
    s = initial_state()
    agents = (p0, p1)
    for _ in range(max_plies):
        if is_terminal(s):
            break
        s = apply_move(s, agents[s.turn].select_move(s))
    w = winner(s)
    if w is None:
        return "draw"
    a_won = (a_is_p0 and w == 0) or (not a_is_p0 and w == 1)
    return "a" if a_won else "b"


def match(label, a_spec, b_spec, games, tb, max_plies=300):
    args = [(a_spec, b_spec, i, tb, max_plies) for i in range(games)]
    res = {"a": 0, "b": 0, "draw": 0}
    t0 = time.time()
    with ProcessPoolExecutor() as ex:
        for r in ex.map(play_one, args):
            res[r] += 1
    wr = 100 * res["a"] / games
    print(f"{label} @ {tb}s, {games}g: A={res['a']} B={res['b']} draw={res['draw']}"
          f"  -> A winrate {wr:.1f}%  ({time.time()-t0:.0f}s)", flush=True)
    return res


if __name__ == "__main__":
    G = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    TB = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1
    print(f"cores={os.cpu_count()}  games/matchup={G}  budget={TB}s", flush=True)
    match("minimax PROBABLE vs minimax BASELINE", "mm_prob", "mm_base", G, TB)
    match("mcts PROBABLE vs mcts BASELINE", "mc_prob", "mc_base", G, TB)
    match("minimax PROBABLE vs mcts PROBABLE", "mm_prob", "mc_prob", G, TB)
    match("minimax PROBABLE vs greedy", "mm_prob", "greedy", G, TB)
    match("mcts PROBABLE vs greedy", "mc_prob", "greedy", G, TB)
