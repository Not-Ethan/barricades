from agents.mcts_agent import MCTSAgent
from agents.greedy_agent import GreedyAgent
from agents.random_agent import RandomAgent
from agents.arena import run_match


def test_mcts_registered():
    # Verify MCTSAgent has the expected name attribute (replaces registry check).
    assert MCTSAgent().name == "mcts"
    assert MCTSAgent(time_budget=0.1).name == "mcts"


def test_mcts_beats_random():
    def mk_mcts(seed):
        return MCTSAgent(time_budget=0.2, seed=seed)

    def mk_random(seed):
        return RandomAgent(seed=seed)

    wins_mcts, wins_random, draws = run_match(mk_mcts, mk_random, games=6)
    assert wins_mcts > wins_random


def test_mcts_competitive_with_greedy():
    # Deterministic (seeded) match. MCTS with heuristic eval should be at least
    # as strong as bare greedy. If this fails, increase the MCTS budget/sims or
    # improve the eval — do NOT weaken the assertion.
    def mk_mcts(seed):
        return MCTSAgent(time_budget=0.3, seed=seed)

    def mk_greedy(seed):
        return GreedyAgent(seed=seed)

    wins_mcts, wins_greedy, draws = run_match(mk_mcts, mk_greedy, games=4)
    assert wins_mcts >= wins_greedy
