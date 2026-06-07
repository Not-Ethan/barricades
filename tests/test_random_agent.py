import random

from core.state import initial_state
from core.rules import legal_moves, apply_move, is_terminal
from agents.random_agent import RandomAgent


def test_random_agent_returns_legal_moves():
    agent = RandomAgent(seed=1)
    s = initial_state()
    for _ in range(100):
        if is_terminal(s):
            break
        move = agent.select_move(s)
        assert move in legal_moves(s)
        s = apply_move(s, move)


def test_random_agent_is_seeded_deterministic():
    s = initial_state()
    a1 = RandomAgent(seed=42)
    a2 = RandomAgent(seed=42)
    assert a1.select_move(s) == a2.select_move(s)


def test_agent_has_name():
    assert RandomAgent().name == "random"
