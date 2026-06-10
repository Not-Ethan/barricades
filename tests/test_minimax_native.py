import pytest

from core.state import initial_state
from core.rules import apply_move, is_terminal
from agents.minimax_agent import MinimaxAgent
from agents.greedy_agent import GreedyAgent
from agents.random_agent import RandomAgent


def _positions():
    """Opening + midgame positions (random rollouts put walls on the board)."""
    pos = [initial_state()]
    for agent, steps in ((GreedyAgent(seed=1), (3, 6, 9, 12)),
                         (RandomAgent(seed=2), (4, 8, 12)),
                         (RandomAgent(seed=5), (6, 11))):
        s = initial_state()
        for i in range(max(steps) + 1):
            if is_terminal(s):
                break
            s = apply_move(s, agent.select_move(s))
            if i in steps and not is_terminal(s):
                pos.append(s)
    return pos


def test_native_backend_active():
    a = MinimaxAgent(backend="native", wall_cap=12)
    assert a.backend == "native", "native backend inactive (barricades_native not built?)"


def test_custom_eval_forces_python():
    a = MinimaxAgent(backend="native", eval_fn=lambda s, p: 0.0)
    assert a.backend == "python"


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_native_matches_python_root_scores(depth):
    py = MinimaxAgent(backend="python", wall_cap=12)
    nat = MinimaxAgent(backend="native", wall_cap=12)
    for s in _positions():
        ps = py._root_scores(s, depth)
        ns = nat._root_scores(s, depth)
        assert set(ps) == set(ns), (depth, set(ps) ^ set(ns))
        for k in ps:
            assert ps[k] == ns[k], (depth, k, ps[k], ns[k])
