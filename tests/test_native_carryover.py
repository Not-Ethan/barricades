import numpy as np
import barricades_native as bn
from core.state import GameState, initial_state
from core import rules
from tests.test_native_game import to_native, mv_to_tuple


def _drive_net(tree, evals):
    done = 0
    guard = 0
    while done < evals and guard < evals * 8 + 64:
        guard += 1
        planes = tree.prepare_leaf()
        if planes is None:
            continue
        tree.receive(np.full(140, 1.0 / 140, dtype=np.float32), 0.0)
        done += 1


def _mk(mv):
    from core.state import Step, Wall
    return Step((mv[1], mv[2])) if mv[0] == "step" else Wall(mv[1], mv[2], mv[3])


def test_advance_preserves_subtree_and_stays_usable():
    s = initial_state()
    t = bn.Tree(to_native(s), 1.5, 0)
    _drive_net(t, 120)
    mv, _ = t.best_move(0.0)
    before = t.root_visits()
    t.advance(mv)
    after = t.root_visits()
    # the new root is the chosen child: retained visits >=1 and <= the old root's
    assert 1 <= after <= before
    # the re-rooted tree is usable: more search yields a legal move for the new state
    _drive_net(t, 40)
    mv2, pi2 = t.best_move(0.0)
    legal = {mv_to_tuple(m) for m in rules.legal_moves(rules.apply_move(s, _mk(mv)))}
    assert mv2 in legal
    assert abs(float(np.asarray(pi2).sum()) - 1.0) < 1e-4


def test_carryover_pool_smoke():
    pool = bn.SelfPlayPool(n_games=4, total_games=4, sims=16, seed=0,
                           max_plies=20, temp_moves=4, carryover=True)
    examples = []
    guard = 0
    while pool.games_remaining() > 0 and guard < 200_000:
        guard += 1
        planes = pool.step()
        if planes is not None:
            b = np.asarray(planes).shape[0]
            pool.feed(np.full((b, 140), 1.0 / 140, np.float32), np.zeros(b, np.float32))
        examples.extend(pool.drain())
    examples.extend(pool.drain())
    assert pool.games_remaining() == 0
    assert len(examples) == 4 * 20  # capped games; carryover doesn't change move count
    for _p, pi, z, _f in examples:
        assert abs(float(np.asarray(pi).sum()) - 1.0) < 1e-4
        assert z in (-1.0, 0.0, 1.0)
