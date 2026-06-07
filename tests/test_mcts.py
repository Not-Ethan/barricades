import time

from core.state import GameState, Step, initial_state
from core.rules import legal_moves
from agents.mcts_agent import MCTSAgent


def _state(p0, p1, wl=(10, 10), turn=0, h=(), v=()):
    return GameState((p0, p1), frozenset(h), frozenset(v), wl, turn)


def test_returns_legal_move():
    a = MCTSAgent(time_budget=0.3, seed=0)
    s = initial_state()
    assert a.select_move(s) in legal_moves(s)


def test_takes_immediate_win():
    # Player 0 is one step from goal (row 8) with NO walls left (only steps
    # available), and the opponent is one step from THEIR goal (row 0). Stepping
    # to (4,8) is the ONLY move that wins; any other move lets greedy-rollout
    # player 1 step to (4,0) and win. This makes the winning move uniquely +1,
    # which a greedy-rollout MCTS will reliably select (unlike a far-opponent
    # position, where many moves all win in rollout).
    a = MCTSAgent(time_budget=0.5, seed=0)
    s = _state((4, 7), (4, 1), wl=(0, 10), turn=0)
    move = a.select_move(s)
    assert isinstance(move, Step) and move.to_cell == (4, 8)


def test_analyze_populates_fields():
    a = MCTSAgent(time_budget=0.4, seed=0)
    s = initial_state()
    info = a.analyze(s)
    assert info.best_move in legal_moves(s)
    assert isinstance(info.value, (int, float))
    assert len(info.candidates) > 0
    assert info.stats["sims"] > 0


def test_respects_time_budget():
    a = MCTSAgent(time_budget=0.3, seed=0)
    s = initial_state()
    t0 = time.monotonic()
    a.select_move(s)
    assert time.monotonic() - t0 < 2.0


def test_max_sims_cap_is_honored():
    a = MCTSAgent(time_budget=60.0, max_sims=50, seed=0)
    s = initial_state()
    info = a.analyze(s)
    assert info.stats["sims"] <= 50


def test_name():
    assert MCTSAgent().name == "mcts"
