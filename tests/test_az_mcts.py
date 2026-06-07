from core.state import GameState, Step, initial_state
from core.rules import legal_moves
from agents.az.model import QuoridorNet, NetWrapper
from agents.az.mcts_nn import PUCTSearch


def _wrap():
    return NetWrapper(QuoridorNet(channels=16, blocks=2))


def test_returns_legal_move_and_policy():
    search = PUCTSearch(_wrap(), sims=40, seed=0)
    s = initial_state()
    move, pi, info = search.run(s)
    assert move in legal_moves(s)
    assert abs(sum(pi.values()) - 1.0) < 1e-6      # visit-count policy normalized
    assert set(pi.keys()) <= set(legal_moves(s))
    assert info["sims"] >= 1


def test_finds_immediate_win_with_enough_sims():
    # zero walls, opp one step away: stepping to (4,8) is the only win.
    s = GameState(((4, 7), (4, 1)), frozenset(), frozenset(), (0, 10), 0)
    search = PUCTSearch(_wrap(), sims=200, seed=0)
    move, pi, info = search.run(s)
    assert isinstance(move, Step) and move.to_cell == (4, 8)
