from core.state import initial_state
from core.rules import legal_moves, apply_move, is_terminal
from agents.registry import make_agent, available_agents
from agents.az.agent import AZAgent


def test_az_registered():
    assert "az" in available_agents()


def test_az_plays_legal_full_game():
    # small net + low sims keeps this fast; untrained is fine, must be legal
    a = AZAgent(sims=16, channels=16, blocks=1, seed=0)
    b = AZAgent(sims=16, channels=16, blocks=1, seed=1)
    s = initial_state()
    for _ in range(300):
        if is_terminal(s):
            break
        agent = a if s.turn == 0 else b
        mv = agent.select_move(s)
        assert mv in legal_moves(s)
        s = apply_move(s, mv)


def test_az_analyze_populated():
    a = AZAgent(sims=16, channels=16, blocks=1, seed=0)
    info = a.analyze(initial_state())
    assert info.best_move in legal_moves(initial_state())
    assert -1.0 <= info.value <= 1.0
    assert len(info.candidates) > 0
    assert info.stats["sims"] >= 1
