from agents.registry import make_agent
from agents.arena import play_game, run_match


def test_make_agent():
    assert make_agent("random").name == "random"
    assert make_agent("greedy").name == "greedy"


def test_play_game_returns_player_or_none():
    a = make_agent("random", seed=1)
    b = make_agent("random", seed=2)
    result = play_game(a, b, max_plies=2000)
    assert result in (0, 1, None)


def test_greedy_beats_random_in_match():
    def mk_greedy(seed):
        return make_agent("greedy", seed=seed)

    def mk_random(seed):
        return make_agent("random", seed=seed)

    wins_greedy, wins_random, draws = run_match(mk_greedy, mk_random, games=10)
    assert wins_greedy > wins_random
