from agents.random_agent import RandomAgent
from agents.greedy_agent import GreedyAgent
from agents.minimax_agent import MinimaxAgent

_FACTORIES = {
    "random": RandomAgent,
    "greedy": GreedyAgent,
    "minimax": MinimaxAgent,
}


def available_agents():
    return sorted(_FACTORIES)


def make_agent(name, **kwargs):
    if name not in _FACTORIES:
        raise ValueError(f"unknown agent: {name!r}; have {available_agents()}")
    return _FACTORIES[name](**kwargs)
