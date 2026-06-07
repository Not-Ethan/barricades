from agents.registry import make_agent, available_agents
from agents.arena import run_match


def test_mcts_registered():
    assert "mcts" in available_agents()
    assert make_agent("mcts", time_budget=0.1).name == "mcts"


def test_mcts_beats_random():
    def mk_mcts(seed):
        return make_agent("mcts", time_budget=0.2, seed=seed)

    def mk_random(seed):
        return make_agent("random", seed=seed)

    wins_mcts, wins_random, draws = run_match(mk_mcts, mk_random, games=6)
    assert wins_mcts > wins_random


def test_mcts_competitive_with_greedy():
    # Deterministic (seeded) match. MCTS with greedy rollouts should be at least
    # as strong as bare greedy. If this fails, increase the MCTS budget/sims or
    # improve the rollout — do NOT weaken the assertion.
    def mk_mcts(seed):
        return make_agent("mcts", time_budget=0.3, seed=seed)

    def mk_greedy(seed):
        return make_agent("greedy", seed=seed)

    wins_mcts, wins_greedy, draws = run_match(mk_mcts, mk_greedy, games=4)
    assert wins_mcts >= wins_greedy
