import numpy as np
import barricades_native as bn
from core.state import GameState, Step
from core import rules
from tests.test_native_game import to_native


def _state(p0, p1, wl, turn=0, h=(), v=()):
    return GameState((p0, p1), frozenset(h), frozenset(v), wl, turn)


def test_native_agent_plays_solver_move_at_zero_walls():
    from agents.native_agent import NativeMctsAgent
    s = _state((4, 6), (0, 1), (0, 0), turn=0)   # both out of walls, p0 clearly winning
    val, mv = bn.solve_race(to_native(s))
    agent = NativeMctsAgent(sims=50, seed=0)     # heuristic mode is fine
    chosen = agent.select_move(s)
    assert isinstance(chosen, Step) and chosen.to_cell == (mv[1], mv[2])


def test_carryover_pool_with_endgame_solve_smoke():
    pool = bn.SelfPlayPool(n_games=4, total_games=4, sims=12, seed=0,
                           max_plies=120, temp_moves=4, endgame_solve=True)
    examples = []
    guard = 0
    while pool.games_remaining() > 0 and guard < 500_000:
        guard += 1
        planes = pool.step()
        if planes is not None:
            b = np.asarray(planes).shape[0]
            pool.feed(np.full((b, 140), 1.0 / 140, np.float32), np.zeros(b, np.float32))
        examples.extend(pool.drain())
    examples.extend(pool.drain())
    assert pool.games_remaining() == 0
    assert len(examples) > 0
    for _p, pi, z, _f in examples:
        assert abs(float(np.asarray(pi).sum()) - 1.0) < 1e-4
        assert z in (-1.0, 0.0, 1.0)
    assert pool.games_solved() >= 1   # at least one game truncated via the solver
