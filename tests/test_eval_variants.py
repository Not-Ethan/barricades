"""Tests for agents/eval_variants.py and the eval_fn parameter on both agents.

Model being tested
------------------
White (player 0): 3 steps from goal (goal = row 8)
Black (player 1): 5 steps from goal (goal = row 0)
White to move (turn = 0)

Before any move by white:
    evaluate_tempo(state, 0) = (5 - 3) + 0 + 0.5 = 2.5   [white to move → +0.5 tempo]

For a step toward goal (dist 3 → 2), turn flips to 1:
    evaluate_tempo(state_after, 0) = (5 - 2) + 0 - 0.5 = 2.5  → net = 0

For a wall that adds Δ to black's path, walls_left[0] -1, turn flips:
    evaluate_tempo = (5+Δ - 3) + 0.1*(-1) - 0.5 = 2 + Δ - 0.6  → net = Δ - 1.1 ≈ Δ - 1

So net change ≈ Δ_opp − 1 (to within ±0.15 due to wall_term of 0.1).

Design
------
- Useless wall (Δ=0): net ≈ -1
- Wall with Δ=1:      net ≈  0
- Wall with Δ=2:      net ≈ +1

We use walls available from this simple open-board position and verify the
relationship net ≈ Δ_opp - 1 for each, rather than hard-coding a specific Δ.
Additionally we check the zero-sum property and agent acceptance of eval_fn.
"""

import pytest

from core.state import GameState, Step, Wall
from core.rules import apply_move, shortest_path_len
from agents.eval_variants import evaluate_tempo
from agents.heuristics import evaluate as evaluate_base


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _state(p0, p1, wl=(10, 10), turn=0, h=(), v=()):
    return GameState((p0, p1), frozenset(h), frozenset(v), wl, turn)


# ---------------------------------------------------------------------------
# Board geometry:
#   Player 0 goal row = 8 (moves up, higher y).
#   Player 1 goal row = 0 (moves down, lower y).
#   White at (4, 5): shortest_path_len = 3  (steps to rows 6, 7, 8)
#   Black at (3, 5): shortest_path_len = 5  (steps to rows 4, 3, 2, 1, 0)
# ---------------------------------------------------------------------------

WHITE = (4, 5)   # player 0, 3 steps from row 8
BLACK = (3, 5)   # player 1, 5 steps from row 0

# Useless wall: far-corner, nowhere near any path.
WALL_USELESS = Wall(7, 7, "V")
# Wall that lengthens black's path by 1 step (verified below in the test):
# H wall at (2,4) blocks down from (3,5)→(3,4), adding 1 step.
WALL_DELTA1 = Wall(2, 4, "H")


class TestEvaluateTempo:
    """Direct model-validation tests for evaluate_tempo."""

    def test_step_toward_goal_net_zero(self):
        """Stepping toward goal should yield net change = 0 (baseline race).

        Before (white to move): eval = (5-3) + 0 + 0.5 = 2.5
        After step (4,5)→(4,6), dist 3→2, turn flips:
            eval = (5-2) + 0 - 0.5 = 2.5
        net = 0.
        """
        state = _state(WHITE, BLACK)
        before = evaluate_tempo(state, 0)

        # White steps from (4,5) toward goal (row 8).
        state_after = apply_move(state, Step((4, 6)))
        after = evaluate_tempo(state_after, 0)

        net = after - before
        assert net == pytest.approx(0.0, abs=1e-9), (
            f"Step toward goal should have net = 0.0, got {net:.6f}"
        )

    def test_relationship_net_equals_delta_opp_minus_one(self):
        """Key model property: net change after a wall = Δopp_dist - 1 (approx).

        For any wall placed by white:
            net ≈ Δopp - 1
        (the -1 comes from the tempo flip: +0.5 → -0.5, i.e. -1 net tempo change,
        plus a small wall_term contribution of -0.1 = net within ±0.15 of Δ-1)

        This is tested for three cases:
          - Useless wall (Δ=0) → net ≈ -1
          - Wall adding 1 step (Δ=1) → net ≈  0
        """
        state = _state(WHITE, BLACK)
        before = evaluate_tempo(state, 0)

        for wall in (WALL_USELESS, WALL_DELTA1):
            d_opp_before = shortest_path_len(state, 1)
            state_after = apply_move(state, wall)
            d_opp_after = shortest_path_len(state_after, 1)
            delta_opp = (d_opp_after or 0) - (d_opp_before or 0)

            after = evaluate_tempo(state_after, 0)
            net = after - before

            expected = delta_opp - 1.0
            assert net == pytest.approx(expected, abs=0.15), (
                f"Wall {wall} (Δopp={delta_opp}): net={net:.3f}, "
                f"expected ≈ {expected:.1f}"
            )

    def test_useless_wall_net_approximately_minus_one(self):
        """Placing a useless wall (Δopp=0) gives net ≈ -1.

        Before: (5-3) + 0 + 0.5 = 2.5 (white to move)
        After useless wall: (5-3) + 0.1*(-1) - 0.5 = 1.4
        net = 1.4 - 2.5 = -1.1 ≈ -1.
        """
        state = _state(WHITE, BLACK)
        before = evaluate_tempo(state, 0)

        d_opp_before = shortest_path_len(state, 1)
        state_after = apply_move(state, WALL_USELESS)
        d_opp_after = shortest_path_len(state_after, 1)
        assert d_opp_after == d_opp_before, (
            f"WALL_USELESS should not change black's path; "
            f"before={d_opp_before}, after={d_opp_after}"
        )

        after = evaluate_tempo(state_after, 0)
        net = after - before
        assert net == pytest.approx(-1.0, abs=0.15), (
            f"Useless wall net ≈ -1, got {net:.3f}"
        )

    def test_delta1_wall_net_approximately_zero(self):
        """Placing a wall that adds 1 to black's path gives net ≈ 0.

        Before: 2.5. After: (6-3) + 0.1*(-1) - 0.5 = 2.4. net = -0.1 ≈ 0.
        """
        state = _state(WHITE, BLACK)
        before = evaluate_tempo(state, 0)

        d_opp_before = shortest_path_len(state, 1)
        state_after = apply_move(state, WALL_DELTA1)
        d_opp_after = shortest_path_len(state_after, 1)
        delta = d_opp_after - d_opp_before
        assert delta == 1, (
            f"WALL_DELTA1 should add 1 to black's path; got Δ={delta}"
        )

        after = evaluate_tempo(state_after, 0)
        net = after - before
        assert net == pytest.approx(0.0, abs=0.15), (
            f"Wall adding 1 to opp path: net ≈ 0, got {net:.3f}"
        )

    def test_zero_sum_nonterminal(self):
        """evaluate_tempo(s, 0) + evaluate_tempo(s, 1) == 0 for non-terminal states."""
        states = [
            _state((4, 4), (4, 5)),
            _state((2, 3), (6, 5), turn=1),
            _state((0, 0), (8, 8), wl=(5, 7)),
            _state(WHITE, BLACK),
            _state(WHITE, BLACK, turn=1),
        ]
        for s in states:
            v0 = evaluate_tempo(s, 0)
            v1 = evaluate_tempo(s, 1)
            assert v0 + v1 == pytest.approx(0.0, abs=1e-9), (
                f"Not zero-sum: evaluate_tempo(s,0)={v0}, "
                f"evaluate_tempo(s,1)={v1}"
            )

    def test_terminal_win_max_score(self):
        """Terminal state returns ±WIN_SCORE (same as base evaluate)."""
        from agents.eval_variants import WIN_SCORE
        # Player 0 wins when their pawn is at goal row 8.
        s_p0_wins = _state((4, 8), (4, 4))
        assert evaluate_tempo(s_p0_wins, 0) == WIN_SCORE
        assert evaluate_tempo(s_p0_wins, 1) == -WIN_SCORE

        # Player 1 wins when their pawn is at goal row 0.
        s_p1_wins = _state((4, 4), (4, 0))
        assert evaluate_tempo(s_p1_wins, 0) == -WIN_SCORE
        assert evaluate_tempo(s_p1_wins, 1) == WIN_SCORE

    def test_tempo_differs_from_base_on_non_one_step(self):
        """evaluate_tempo differs from base evaluate when mover is NOT one step from goal.

        The base evaluate's _tempo is 0 when neither player is one step away.
        evaluate_tempo always has a ±0.5 term. So they differ.
        """
        state = _state(WHITE, BLACK)  # white 3 steps from goal, not 1
        base_val = evaluate_base(state, 0)
        tempo_val = evaluate_tempo(state, 0)
        # The base evaluate has _tempo=0 here; evaluate_tempo has +0.5.
        assert base_val != pytest.approx(tempo_val, abs=1e-9), (
            "evaluate_tempo should differ from base evaluate at 3 steps from goal"
        )
        assert tempo_val == pytest.approx(base_val + 0.5, abs=1e-9), (
            f"evaluate_tempo should be base + 0.5 when white to move and not one-step; "
            f"base={base_val}, tempo={tempo_val}"
        )

    def test_tempo_sign_flips_with_turn(self):
        """Tempo bonus is +0.5 when it is player's turn, -0.5 otherwise."""
        # White to move.
        state_white_turn = _state(WHITE, BLACK, turn=0)
        val_white_turn = evaluate_tempo(state_white_turn, 0)
        # Black to move.
        state_black_turn = _state(WHITE, BLACK, turn=1)
        val_black_turn = evaluate_tempo(state_black_turn, 0)
        # Difference should be exactly 1.0 (from -0.5 to +0.5).
        assert val_white_turn - val_black_turn == pytest.approx(1.0, abs=1e-9)


class TestAgentsAcceptEvalFn:
    """Tests that agents accept eval_fn and that default behaviour is unchanged."""

    def test_minimax_accepts_eval_fn(self):
        """MinimaxAgent should accept eval_fn without error."""
        from agents.minimax_agent import MinimaxAgent
        from core.rules import legal_moves
        state = _state((4, 7), (4, 1))
        agent = MinimaxAgent(time_budget=0.2, seed=0, eval_fn=evaluate_tempo)
        move = agent.select_move(state)
        assert move in legal_moves(state)

    def test_mcts_accepts_eval_fn(self):
        """MCTSAgent should accept eval_fn without error."""
        from agents.mcts_agent import MCTSAgent
        from core.rules import legal_moves
        state = _state((4, 7), (4, 1))
        agent = MCTSAgent(time_budget=0.2, seed=0, eval_fn=evaluate_tempo)
        move = agent.select_move(state)
        assert move in legal_moves(state)

    def test_minimax_default_unchanged(self):
        """MinimaxAgent() with no eval_fn should produce identical results to
        MinimaxAgent(eval_fn=evaluate_base) on a walls-only position."""
        from agents.minimax_agent import MinimaxAgent
        from agents.heuristics import evaluate
        state = _state((4, 7), (4, 1), wl=(0, 0))  # no walls, only steps

        a1 = MinimaxAgent(time_budget=0.3, seed=42)
        a2 = MinimaxAgent(time_budget=0.3, seed=42, eval_fn=evaluate)
        assert a1.select_move(state) == a2.select_move(state)

    def test_mcts_default_unchanged(self):
        """MCTSAgent() with no eval_fn should produce identical results to
        MCTSAgent(eval_fn=evaluate_base) on a walls-only position."""
        from agents.mcts_agent import MCTSAgent
        from agents.heuristics import evaluate
        state = _state((4, 7), (4, 1), wl=(0, 0))  # no walls, only steps

        a1 = MCTSAgent(time_budget=0.3, seed=42)
        a2 = MCTSAgent(time_budget=0.3, seed=42, eval_fn=evaluate)
        assert a1.select_move(state) == a2.select_move(state)

    def test_minimax_tempo_takes_immediate_win(self):
        """MinimaxAgent with eval_fn=evaluate_tempo still takes immediate win."""
        from agents.minimax_agent import MinimaxAgent
        state = _state((4, 7), (0, 0))
        agent = MinimaxAgent(time_budget=0.3, seed=0, eval_fn=evaluate_tempo)
        move = agent.select_move(state)
        assert isinstance(move, Step) and move.to_cell == (4, 8)

    def test_mcts_tempo_takes_immediate_win(self):
        """MCTSAgent with eval_fn=evaluate_tempo still takes immediate win."""
        from agents.mcts_agent import MCTSAgent
        state = _state((4, 7), (4, 1), wl=(0, 10))
        agent = MCTSAgent(time_budget=0.5, seed=0, eval_fn=evaluate_tempo)
        move = agent.select_move(state)
        assert isinstance(move, Step) and move.to_cell == (4, 8)
