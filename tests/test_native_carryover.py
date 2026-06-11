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


def _state(p0, p1, turn=0, h=(), v=()):
    return GameState((p0, p1), frozenset(h), frozenset(v), (10, 10), turn)


def test_advance_negates_value_perspective():
    # Player 0 is clearly winning: p0 at (4,6) -> 2 steps from its goal row 8;
    # p1 at (0,7) -> 7 steps from its goal row 0. With p0 to move, the root value
    # (p0 perspective) must be POSITIVE. After p0's move we re-root to a p1-to-move
    # position where p1 is badly losing, so the carried root value must be NEGATIVE
    # -- which is ONLY true if advance() negated the retained subtree's W (the
    # perspective flip). A missing/wrong negation leaves it positive and fails here.
    s = _state((4, 6), (0, 7), turn=0)
    t = bn.Tree(to_native(s), 1.5, 0)
    t.run_heuristic(300)
    v_p0 = t.root_value()
    assert v_p0 > 0.0, f"p0 near goal should value root > 0, got {v_p0}"
    mv, _ = t.best_move(0.0)
    t.advance(mv)
    # read immediately after advance: reflects the carried (negated) W, before new sims
    v_p1 = t.root_value()
    assert v_p1 < 0.0, f"after re-root to losing p1, carried value must be < 0, got {v_p1}"


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
