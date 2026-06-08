import time

from core.state import GameState, Step, initial_state
from core.rules import legal_moves
from agents.mcts_agent import MCTSAgent
from agents.greedy_agent import GreedyAgent
from agents.strength import play_recorded_game, wasted_wall_rate


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


def test_low_wasted_wall_rate():
    """New MCTS (relevant-move candidates) should place almost no pointless walls.

    Because only relevant_moves() are candidates — walls that strictly increase
    the opponent's shortest path — the wasted_wall_rate should be near 0.
    Assert ≤ 0.15 to allow a small margin for edge cases.
    """
    mcts = MCTSAgent(time_budget=0.1, seed=42)
    greedy = GreedyAgent(seed=7)

    all_records = []
    for game_seed in range(5):
        mcts_i = MCTSAgent(time_budget=0.1, seed=game_seed)
        greedy_i = GreedyAgent(seed=1000 + game_seed)
        result = play_recorded_game(mcts_i, greedy_i)
        all_records.extend(result["records"])

    # Only measure walls placed by MCTS (player 0 in these games).
    wwr = wasted_wall_rate(all_records, player=0)
    # If MCTS placed no walls at all, wwr is None — that's also acceptable.
    if wwr is not None:
        assert wwr <= 0.15, f"wasted_wall_rate={wwr:.3f} exceeds 0.15"
