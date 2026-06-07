import random

from core.state import Step, Wall, initial_state
from core.rules import (
    legal_moves, apply_move, is_terminal, winner, legal_steps, legal_walls,
)


def test_legal_moves_is_steps_plus_walls():
    s = initial_state()
    moves = legal_moves(s)
    steps = [m for m in moves if isinstance(m, Step)]
    walls = [m for m in moves if isinstance(m, Wall)]
    assert len(steps) == len(legal_steps(s))
    assert len(walls) == len(legal_walls(s))


def test_apply_step_moves_pawn_and_switches_turn():
    s = initial_state()
    s2 = apply_move(s, Step((4, 1)))
    assert s2.pawns[0] == (4, 1)
    assert s2.turn == 1
    assert s.pawns[0] == (4, 0)   # original unchanged (immutability)


def test_apply_wall_records_and_decrements():
    s = initial_state()
    s2 = apply_move(s, Wall(3, 3, "H"))
    assert (3, 3) in s2.h_walls
    assert s2.walls_left == (9, 10)
    assert s2.turn == 1


def test_terminal_and_winner():
    s = initial_state()
    assert not is_terminal(s)
    assert winner(s) is None
    from core.state import GameState
    almost = GameState(((4, 7), (4, 1)), frozenset(), frozenset(), (10, 10), 0)
    won = apply_move(almost, Step((4, 8)))
    assert is_terminal(won)
    assert winner(won) == 0


def test_random_playout_keeps_invariants():
    rng = random.Random(0)
    s = initial_state()
    for _ in range(200):
        if is_terminal(s):
            break
        moves = legal_moves(s)
        assert moves, "a non-terminal state must have legal moves"
        s = apply_move(s, rng.choice(moves))
        from core.coords import on_board
        assert all(on_board(p) for p in s.pawns)
        assert 0 <= s.walls_left[0] <= 10 and 0 <= s.walls_left[1] <= 10
        assert len(s.h_walls) + len(s.v_walls) == 20 - sum(s.walls_left)
        from core.rules import has_path_to_goal
        assert has_path_to_goal(s, 0) and has_path_to_goal(s, 1)
