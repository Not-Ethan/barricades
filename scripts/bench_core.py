"""Benchmark the bitboard BFS vs the pure-Python reference. Informational.
Usage: python scripts/bench_core.py"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
from core.state import initial_state
from core.rules import (
    legal_moves, apply_move, is_terminal, legal_walls,
    shortest_path_len, _shortest_path_len_ref,
)
from core.bitboard import bfs_dist


def sample_states(n=300, seed=7):
    rng = random.Random(seed)
    states, s = [], initial_state()
    while len(states) < n:
        if is_terminal(s):
            s = initial_state()
            continue
        states.append(s)
        s = apply_move(s, rng.choice(legal_moves(s)))
    return states


def time_it(fn, states, reps=20):
    t0 = time.monotonic()
    for _ in range(reps):
        for s in states:
            fn(s, 0)
            fn(s, 1)
    return time.monotonic() - t0


def main():
    states = sample_states()
    ref = time_it(_shortest_path_len_ref, states)
    fast = time_it(bfs_dist, states)
    print(f"shortest-path over {len(states)} states x20 reps:")
    print(f"  reference (pure-Python BFS): {ref:.3f}s")
    print(f"  bitboard flood-fill:         {fast:.3f}s")
    print(f"  speedup: {ref / fast:.2f}x")
    # legal_walls throughput (path-check dominated)
    t0 = time.monotonic()
    for s in states[:60]:
        legal_walls(s)
    print(f"legal_walls over 60 states: {time.monotonic() - t0:.3f}s "
          f"(now uses bitboard has_path)")


if __name__ == "__main__":
    main()
