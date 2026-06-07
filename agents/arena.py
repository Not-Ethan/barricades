from core.state import initial_state
from core.rules import apply_move, is_terminal, winner


def play_game(agent0, agent1, max_plies=2000, state=None):
    """Play one game. Returns winning player (0/1) or None if move cap hit.
    Starts from `state` if given, else the initial position."""
    agents = (agent0, agent1)
    if state is None:
        state = initial_state()
    for _ in range(max_plies):
        if is_terminal(state):
            return winner(state)
        move = agents[state.turn].select_move(state)
        state = apply_move(state, move)
    return winner(state) if is_terminal(state) else None


def run_match(make_a, make_b, games=10, max_plies=2000):
    """Play `games` games between two agent factories, alternating who starts.
    `make_a`/`make_b` are callables taking a `seed` kwarg. Returns
    (wins_a, wins_b, draws)."""
    wins_a = wins_b = draws = 0
    for g in range(games):
        a = make_a(seed=g)
        b = make_b(seed=1000 + g)
        if g % 2 == 0:
            result = play_game(a, b, max_plies)      # A is player 0
            a_won, b_won = result == 0, result == 1
        else:
            result = play_game(b, a, max_plies)      # A is player 1
            a_won, b_won = result == 1, result == 0
        if a_won:
            wins_a += 1
        elif b_won:
            wins_b += 1
        else:
            draws += 1
    return wins_a, wins_b, draws
