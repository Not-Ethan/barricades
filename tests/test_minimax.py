import time

from core.state import GameState, Step, Wall, initial_state
from core.rules import legal_moves
from agents.minimax_agent import MinimaxAgent


def _state(p0, p1, wl=(10, 10), turn=0, h=(), v=()):
    return GameState((p0, p1), frozenset(h), frozenset(v), wl, turn)


def test_returns_legal_move():
    a = MinimaxAgent(time_budget=0.5, seed=0)
    s = initial_state()
    assert a.select_move(s) in legal_moves(s)


def test_takes_immediate_win():
    # player 0 at (4,7) can step to (4,8) and win this move.
    a = MinimaxAgent(time_budget=0.5, seed=0)
    s = _state((4, 7), (0, 0))
    move = a.select_move(s)
    assert isinstance(move, Step) and move.to_cell == (4, 8)


def test_blocks_or_races_sensibly_not_suicidal():
    # On the open board the engine should advance, not waste the turn on a
    # far-corner wall. Best move should reduce its own distance or be a wall
    # that hurts the opponent — assert it is at least not strictly worsening.
    from agents.heuristics import evaluate
    from core.rules import apply_move
    a = MinimaxAgent(time_budget=0.5, seed=0)
    s = initial_state()
    move = a.select_move(s)
    before = evaluate(s, s.turn)
    after = evaluate(apply_move(s, move), s.turn)
    assert after >= before


def test_analyze_populates_fields():
    a = MinimaxAgent(time_budget=0.5, seed=0)
    s = initial_state()
    info = a.analyze(s)
    assert info.best_move in legal_moves(s)
    assert isinstance(info.value, (int, float))
    assert len(info.candidates) > 0
    assert info.stats["nodes"] > 0
    assert info.stats["depth"] >= 1


def test_respects_time_budget():
    a = MinimaxAgent(time_budget=0.3, seed=0)
    s = initial_state()
    t0 = time.monotonic()
    a.select_move(s)
    # generous upper bound: budget + overhead for finishing the current depth
    assert time.monotonic() - t0 < 3.0


def test_name_and_params():
    assert MinimaxAgent().name == "minimax"
