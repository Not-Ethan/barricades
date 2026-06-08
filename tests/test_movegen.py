"""Tests for agents/movegen.py — relevant_walls / relevant_moves.

Correctness properties verified:
  1. relevant_walls(state) ⊆ legal_walls(state) always.
  2. Every wall in relevant_walls(state) strictly increases the opponent's
     shortest-path distance.
  3. At least one relevant wall exists on the initial board.
  4. A wall far from the opponent that doesn't change its distance is NOT
     in relevant_walls.
  5. relevant_moves = [Step(c) for c in legal_steps(state)] + relevant_walls.
  6. Property test over ~30 random reachable states.

Performance sanity: timing across ~50 random states is also measured.
"""

import random
import time

import pytest

from core.state import initial_state, Wall, GameState, Step
from core.rules import (
    legal_walls, legal_steps, legal_moves, apply_move,
    shortest_path_len, is_terminal,
)
from agents.movegen import relevant_walls, relevant_moves


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_states(n: int, seed: int = 42):
    """Generate *n* distinct non-terminal states by random play."""
    rng = random.Random(seed)
    states = []
    state = initial_state()
    seen = set()
    attempts = 0
    while len(states) < n and attempts < n * 100:
        attempts += 1
        if is_terminal(state):
            state = initial_state()
        moves = legal_moves(state)
        state = apply_move(state, rng.choice(moves))
        key = (state.pawns, state.h_walls, state.v_walls, state.turn)
        if key not in seen:
            seen.add(key)
            states.append(state)
    return states


def _opp_dist(state):
    """Opponent's shortest-path distance in *state*."""
    return shortest_path_len(state, 1 - state.turn)


def _opp_dist_after_wall(state, wall):
    """Opponent distance after placing *wall* from *state*."""
    s2 = apply_move(state, wall)
    return shortest_path_len(s2, 1 - state.turn)


# ---------------------------------------------------------------------------
# Test 1: relevant_walls is a subset of legal_walls
# ---------------------------------------------------------------------------

def test_relevant_walls_subset_of_legal_walls_initial():
    """Every wall in relevant_walls must be a legal wall."""
    s = initial_state()
    legal = set(legal_walls(s))
    for w in relevant_walls(s):
        assert w in legal, f"Wall {w} in relevant_walls but not in legal_walls"


# ---------------------------------------------------------------------------
# Test 2: Every returned wall genuinely increases the opponent's distance
# ---------------------------------------------------------------------------

def test_every_relevant_wall_increases_opp_distance_initial():
    """Core correctness: each relevant wall strictly lengthens the opponent's path."""
    s = initial_state()
    base = _opp_dist(s)
    assert base is not None, "Opponent has no path on initial board (unexpected)"

    rw = relevant_walls(s)
    assert rw, "Expected at least one relevant wall on the initial board"

    for w in rw:
        after = _opp_dist_after_wall(s, w)
        assert after is not None, f"Wall {w} cut off opponent path (illegal?)"
        assert after > base, (
            f"Wall {w} did NOT increase opp dist: before={base} after={after}"
        )


# ---------------------------------------------------------------------------
# Test 3: At least one relevant wall exists on the initial board
# ---------------------------------------------------------------------------

def test_at_least_one_relevant_wall_exists():
    """A wall directly in front of player 1's pawn (at (4,8)) should be relevant."""
    s = initial_state()
    rw = relevant_walls(s)
    assert len(rw) >= 1, "Expected at least one relevant wall on the initial board"

    # Also verify the specific wall Wall(3,7,'H') is relevant:
    # it blocks col 4 from crossing row 7-8, forcing player 1 (at (4,8)) to detour.
    w = Wall(3, 7, "H")
    assert w in rw, f"Wall(3,7,H) should be relevant (blocks player 1's direct path)"


# ---------------------------------------------------------------------------
# Test 4: Walls far from the opponent are NOT in relevant_walls
# ---------------------------------------------------------------------------

def test_irrelevant_wall_excluded():
    """Wall(0,0,'H') — far from player 1 at (4,8) — must not appear in relevant_walls."""
    s = initial_state()
    # Sanity: the wall is legal on the initial board.
    assert Wall(0, 0, "H") in legal_walls(s), "Wall(0,0,H) should be legal"

    # Sanity: it does NOT change the opponent's distance.
    assert _opp_dist_after_wall(s, Wall(0, 0, "H")) == _opp_dist(s), (
        "Wall(0,0,H) unexpectedly changed opponent's distance"
    )

    rw = set(relevant_walls(s))
    assert Wall(0, 0, "H") not in rw, "Wall(0,0,H) should be EXCLUDED from relevant_walls"

    # Check a few more clearly-irrelevant walls.
    for args in [(7, 7, "H"), (0, 6, "V"), (7, 0, "V")]:
        w = Wall(*args)
        assert w in legal_walls(s), f"Wall{args} expected to be legal"
        assert w not in rw, f"Wall{args} should be excluded from relevant_walls"


# ---------------------------------------------------------------------------
# Test 5: relevant_moves = all legal steps + relevant_walls
# ---------------------------------------------------------------------------

def test_relevant_moves_composition_initial():
    """relevant_moves = [Step(c) for c in legal_steps] + relevant_walls."""
    s = initial_state()
    steps_expected = [Step(c) for c in legal_steps(s)]
    walls_expected = relevant_walls(s)
    combined = steps_expected + walls_expected

    rm = relevant_moves(s)

    # Step count matches.
    rm_steps = [m for m in rm if isinstance(m, Step)]
    assert len(rm_steps) == len(steps_expected), (
        f"Step count mismatch: {len(rm_steps)} vs {len(steps_expected)}"
    )

    # Walls in relevant_moves exactly match relevant_walls.
    rm_walls = [m for m in rm if isinstance(m, Wall)]
    assert set(rm_walls) == set(walls_expected), (
        "Wall set in relevant_moves differs from relevant_walls(state)"
    )

    # Total count matches combined list.
    assert len(rm) == len(combined)


def test_relevant_moves_step_count_equals_legal_steps():
    """The step part of relevant_moves equals len(legal_steps(state))."""
    s = initial_state()
    rm = relevant_moves(s)
    step_count = sum(1 for m in rm if isinstance(m, Step))
    assert step_count == len(legal_steps(s))


# ---------------------------------------------------------------------------
# Test 6: Property test — 30 random reachable states
# ---------------------------------------------------------------------------

def test_property_random_states():
    """For ~30 random states: every wall in relevant_walls is legal and distance-increasing."""
    states = _random_states(30, seed=7)
    assert len(states) >= 20, f"Too few random states generated: {len(states)}"

    for s in states:
        opp = 1 - s.turn
        base = shortest_path_len(s, opp)
        if base is None:
            continue  # degenerate (shouldn't happen in reachable non-terminal states)

        legal = set(legal_walls(s))
        rw = relevant_walls(s)

        for w in rw:
            # Must be a legal wall.
            assert w in legal, f"Wall {w} in relevant_walls but NOT in legal_walls (state={s})"

            # Must strictly increase opponent's distance.
            s2 = apply_move(s, w)
            new_dist = shortest_path_len(s2, opp)
            assert new_dist is not None, f"Wall {w} cut off opponent path"
            assert new_dist > base, (
                f"Wall {w} did NOT increase opp dist: before={base} after={new_dist}"
            )

        # relevant_walls must be a subset of legal_walls.
        assert set(rw) <= legal, "relevant_walls is not a subset of legal_walls"


# ---------------------------------------------------------------------------
# Performance sanity: ~50 random states, measure per-call time
# ---------------------------------------------------------------------------

def test_performance_sanity():
    """Rough timing: relevant_moves should complete in reasonable time per call."""
    states = _random_states(50, seed=99)
    assert len(states) >= 30, f"Need at least 30 states for timing, got {len(states)}"

    t0 = time.perf_counter()
    for s in states:
        relevant_moves(s)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    per_call_ms = elapsed_ms / len(states)
    # Allow generous budget: <10 ms per call in pure Python is fine.
    assert per_call_ms < 10.0, (
        f"relevant_moves too slow: {per_call_ms:.2f} ms/call "
        f"(total {elapsed_ms:.1f} ms over {len(states)} states)"
    )
    # Print for informational purposes (captured by pytest -s or -v).
    print(
        f"\n[perf] relevant_moves: {per_call_ms:.3f} ms/call "
        f"over {len(states)} states ({elapsed_ms:.1f} ms total)"
    )
