from agents.registry import make_agent, available_agents
from agents.arena import run_match


def test_minimax_registered():
    assert "minimax" in available_agents()
    assert make_agent("minimax", time_budget=0.2).name == "minimax"


def test_minimax_beats_greedy():
    def mk_mm(seed):
        return make_agent("minimax", time_budget=0.2, seed=seed)

    def mk_greedy(seed):
        return make_agent("greedy", seed=seed)

    wins_mm, wins_greedy, draws = run_match(mk_mm, mk_greedy, games=4)
    assert wins_mm > wins_greedy
