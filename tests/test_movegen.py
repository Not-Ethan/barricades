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
from agents.movegen import relevant_walls, relevant_moves, probable_walls, probable_moves


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


# ===========================================================================
# Tests for probable_walls / probable_moves
# ===========================================================================

# ---------------------------------------------------------------------------
# PW-1: probable_walls(s) ⊆ legal_walls(s) always
# ---------------------------------------------------------------------------

def test_probable_walls_subset_of_legal_initial():
    """Every wall in probable_walls must be a legal wall."""
    s = initial_state()
    legal = set(legal_walls(s))
    for w in probable_walls(s):
        assert w in legal, f"Wall {w} in probable_walls but not in legal_walls"


# ---------------------------------------------------------------------------
# PW-2: Concrete included / excluded examples on the initial board
#
# Initial board: pawns at (4,0) [player 0] and (4,8) [player 1].
# Near-pawn (Chebyshev 2) examples (anchor within 2 of (4,0) or (4,8)):
#   - Wall(3,1,'H') — anchor (3,1): Chebyshev to (4,0) = max(1,1) = 1 ≤ 2. INCLUDED.
#   - Wall(4,6,'V') — anchor (4,6): Chebyshev to (4,8) = max(0,2) = 2 ≤ 2. INCLUDED.
# Edge wall examples:
#   - Wall(0,3,'H') — c==0. INCLUDED.
#   - Wall(7,5,'V') — c==7. INCLUDED.
# Dead-center wall far from both pawns (no existing walls):
#   - Anchor (4,4): Chebyshev to (4,0) = 4, Chebyshev to (4,8) = 4.
#     Not at edge (4 != 0 and 4 != 7), no existing walls.
#     → Wall(4,4,'H') and Wall(4,4,'V') should be EXCLUDED on initial board.
# ---------------------------------------------------------------------------

def test_probable_walls_includes_near_pawn_initial():
    """Walls within Chebyshev 2 of a pawn should be in probable_walls."""
    s = initial_state()
    pw = set(probable_walls(s))
    legal = set(legal_walls(s))

    # Near player 0's pawn at (4,0): anchor (3,1), Cheb dist = max(|3-4|,|1-0|)=1.
    w_near_p0 = Wall(3, 1, "H")
    if w_near_p0 in legal:
        assert w_near_p0 in pw, f"{w_near_p0} should be in probable_walls (near pawn 0)"

    # Near player 1's pawn at (4,8): anchor (4,6), Cheb dist = max(|4-4|,|6-8|)=2.
    w_near_p1 = Wall(4, 6, "V")
    if w_near_p1 in legal:
        assert w_near_p1 in pw, f"{w_near_p1} should be in probable_walls (near pawn 1)"


def test_probable_walls_includes_edge_walls_initial():
    """Walls at board edges (c==0, c==7, r==0, r==7) should be in probable_walls."""
    s = initial_state()
    pw = set(probable_walls(s))
    legal = set(legal_walls(s))

    for w_args in [(0, 3, "H"), (7, 5, "V"), (2, 0, "H"), (3, 7, "V")]:
        w = Wall(*w_args)
        if w in legal:
            assert w in pw, f"{w} should be in probable_walls (edge wall)"


def test_probable_walls_excludes_dead_center_initial():
    """Anchor (4,4) is 4 steps from both pawns, not at an edge, no walls nearby.

    On the initial board Wall(4,4,'H') and Wall(4,4,'V') should be EXCLUDED.
    Chebyshev distance to pawn (4,0) = max(0,4)=4 > 2.
    Chebyshev distance to pawn (4,8) = max(0,4)=4 > 2.
    No existing walls, so near-existing-wall rule doesn't apply.
    Not at edge (4 != 0 and 4 != 7 for both c and r).
    """
    s = initial_state()
    pw = set(probable_walls(s))
    legal = set(legal_walls(s))

    # Verify the anchor is genuinely in the middle (safety check).
    assert Wall(4, 4, "H") in legal, "Wall(4,4,H) should be legal on initial board"
    assert Wall(4, 4, "V") in legal, "Wall(4,4,V) should be legal on initial board"

    assert Wall(4, 4, "H") not in pw, (
        "Wall(4,4,H) should be EXCLUDED from probable_walls on initial board "
        "(anchor (4,4): Cheb to (4,0)=4>2, Cheb to (4,8)=4>2, not edge, no existing walls)"
    )
    assert Wall(4, 4, "V") not in pw, (
        "Wall(4,4,V) should be EXCLUDED from probable_walls on initial board "
        "(anchor (4,4): Cheb to (4,0)=4>2, Cheb to (4,8)=4>2, not edge, no existing walls)"
    )


# ---------------------------------------------------------------------------
# PW-3: After placing a wall, a Chebyshev-1 neighbor becomes probable
#        (near-existing-wall rule) even if far from pawns.
# ---------------------------------------------------------------------------

def test_probable_walls_near_existing_wall():
    """After placing Wall(4,4,'H'), a Chebyshev-1 anchor should appear in probable_walls."""
    s = initial_state()
    # Place a wall at dead-center anchor (4,4) by building a state directly.
    # We do this by applying the wall move from initial state (p0 has walls).
    s2 = apply_move(s, Wall(4, 4, "H"))
    # Now the existing wall anchor is (4,4). A wall at anchor (5,4) has
    # Chebyshev distance max(|5-4|,|4-4|)=1 ≤ 1 from (4,4). It should be probable.
    pw2 = set(probable_walls(s2))
    legal2 = set(legal_walls(s2))
    w_neighbor = Wall(5, 4, "H")
    if w_neighbor in legal2:
        assert w_neighbor in pw2, (
            f"{w_neighbor} should be in probable_walls after placing Wall(4,4,H) "
            "(Chebyshev-1 from that anchor)"
        )

    # Also check that the dead-center anchor (4,3) with Cheb-1 from (4,4) is probable.
    w_neighbor2 = Wall(4, 3, "V")
    if w_neighbor2 in legal2:
        assert w_neighbor2 in pw2, (
            f"{w_neighbor2} should be in probable_walls after placing Wall(4,4,H)"
        )


# ---------------------------------------------------------------------------
# PW-4: probable_moves includes ALL legal steps
# ---------------------------------------------------------------------------

def test_probable_moves_includes_all_legal_steps():
    """probable_moves must include every legal step (no steps are pruned)."""
    s = initial_state()
    pm = probable_moves(s)
    pm_steps = {m.to_cell for m in pm if isinstance(m, Step)}
    expected_steps = set(legal_steps(s))
    assert pm_steps == expected_steps, (
        f"probable_moves missing steps. Expected {expected_steps}, got {pm_steps}"
    )


def test_probable_moves_steps_equal_legal_steps_random():
    """On several random states probable_moves has exactly the legal steps."""
    states = _random_states(20, seed=17)
    for s in states:
        pm = probable_moves(s)
        pm_steps = {m.to_cell for m in pm if isinstance(m, Step)}
        expected_steps = set(legal_steps(s))
        assert pm_steps == expected_steps


# ---------------------------------------------------------------------------
# PW-5: Property test over ~30 random reachable states
# ---------------------------------------------------------------------------

def test_probable_walls_property_random():
    """For ~30 random states: probable_walls ⊆ legal_walls; no crash."""
    states = _random_states(30, seed=31)
    assert len(states) >= 20, f"Too few random states: {len(states)}"
    for s in states:
        legal = set(legal_walls(s))
        pw = probable_walls(s)
        # Subset requirement.
        for w in pw:
            assert w in legal, (
                f"Wall {w} in probable_walls but NOT in legal_walls"
            )
        # probable_walls is always a subset.
        assert set(pw) <= legal


# ---------------------------------------------------------------------------
# PW-6: Performance sanity for probable_moves
# ---------------------------------------------------------------------------

def test_probable_moves_performance():
    """probable_moves should be fast; measure ms/call and typical #probable_walls."""
    states = _random_states(50, seed=77)
    assert len(states) >= 30

    wall_counts = []
    t0 = time.perf_counter()
    for s in states:
        pm = probable_moves(s)
        wall_counts.append(sum(1 for m in pm if isinstance(m, Wall)))
    elapsed_ms = (time.perf_counter() - t0) * 1000

    per_call_ms = elapsed_ms / len(states)
    avg_walls = sum(wall_counts) / len(wall_counts)

    # Should be comfortably under 10 ms/call.
    assert per_call_ms < 10.0, (
        f"probable_moves too slow: {per_call_ms:.2f} ms/call"
    )
    print(
        f"\n[perf] probable_moves: {per_call_ms:.3f} ms/call over {len(states)} states; "
        f"avg probable_walls={avg_walls:.1f}"
    )


# ===========================================================================
# Tests for candidate_moves parameter in MinimaxAgent and MCTSAgent
# ===========================================================================

def test_minimax_with_probable_moves_returns_legal():
    """MinimaxAgent(candidate_moves=probable_moves) returns a legal move."""
    from core.rules import legal_moves
    from agents.minimax_agent import MinimaxAgent
    a = MinimaxAgent(time_budget=0.3, seed=0, candidate_moves=probable_moves)
    s = initial_state()
    move = a.select_move(s)
    assert move in legal_moves(s), f"MinimaxAgent returned illegal move: {move}"


def test_mcts_with_probable_moves_returns_legal():
    """MCTSAgent(candidate_moves=probable_moves) returns a legal move."""
    from core.rules import legal_moves
    from agents.mcts_agent import MCTSAgent
    a = MCTSAgent(time_budget=0.3, seed=0, candidate_moves=probable_moves)
    s = initial_state()
    move = a.select_move(s)
    assert move in legal_moves(s), f"MCTSAgent returned illegal move: {move}"


def test_minimax_default_unchanged():
    """Default MinimaxAgent (no candidate_moves) still behaves as before."""
    from core.rules import legal_moves
    from agents.minimax_agent import MinimaxAgent
    a = MinimaxAgent(time_budget=0.3, seed=0)
    s = initial_state()
    move = a.select_move(s)
    assert move in legal_moves(s)
    # Default agent should still prefer a step on the open board.
    from core.state import Step
    assert isinstance(move, Step), "Default MinimaxAgent should step on open board"


def test_mcts_default_unchanged():
    """Default MCTSAgent (no candidate_moves) still behaves as before."""
    from core.rules import legal_moves
    from agents.mcts_agent import MCTSAgent
    a = MCTSAgent(time_budget=0.3, seed=0)
    s = initial_state()
    move = a.select_move(s)
    assert move in legal_moves(s)


def test_minimax_candidate_moves_used_at_root_and_internal():
    """MinimaxAgent with candidate_moves uses it at root and search nodes.

    Confirm it runs to completion and produces a legal move for a few states.
    """
    from core.rules import legal_moves
    from agents.minimax_agent import MinimaxAgent
    states = _random_states(5, seed=55)
    a = MinimaxAgent(time_budget=0.2, seed=0, candidate_moves=probable_moves)
    for s in states:
        move = a.select_move(s)
        assert move in legal_moves(s)


def test_mcts_candidate_moves_used_throughout():
    """MCTSAgent with candidate_moves uses it for expansion on several states."""
    from core.rules import legal_moves
    from agents.mcts_agent import MCTSAgent
    states = _random_states(5, seed=66)
    a = MCTSAgent(time_budget=0.2, seed=0, candidate_moves=probable_moves)
    for s in states:
        move = a.select_move(s)
        assert move in legal_moves(s)
