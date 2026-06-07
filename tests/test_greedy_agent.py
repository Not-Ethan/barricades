from core.state import GameState, initial_state
from core.rules import legal_moves
from agents.greedy_agent import GreedyAgent


def test_greedy_returns_legal_move():
    agent = GreedyAgent(seed=0)
    s = initial_state()
    assert agent.select_move(s) in legal_moves(s)


def test_greedy_steps_toward_goal_on_open_board():
    agent = GreedyAgent(seed=0)
    s = GameState(((4, 0), (4, 8)), frozenset(), frozenset(), (10, 10), 0)
    move = agent.select_move(s)
    from core.state import Step
    assert isinstance(move, Step)
    assert move.to_cell[1] == 1   # moved one row closer to goal row 8


def test_greedy_has_name():
    assert GreedyAgent().name == "greedy"
