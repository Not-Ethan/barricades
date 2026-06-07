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


def test_play_game_reports_winner_on_final_ply():
    from core.state import GameState
    # player 0 one step from its goal row (8); greedy will step up and win.
    near = GameState(((4, 7), (4, 1)), frozenset(), frozenset(), (10, 10), 0)
    result = play_game(make_agent("greedy", seed=0),
                       make_agent("random", seed=0),
                       max_plies=1, state=near)
    assert result == 0


def test_run_match_accounting_adds_up():
    def mk_g(seed):
        return make_agent("greedy", seed=seed)

    def mk_r(seed):
        return make_agent("random", seed=seed)

    wins_a, wins_b, draws = run_match(mk_g, mk_r, games=6)
    assert wins_a + wins_b + draws == 6


def test_make_agent_unknown_raises():
    import pytest
    with pytest.raises(ValueError):
        make_agent("nonexistent")
