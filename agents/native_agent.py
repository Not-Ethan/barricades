"""MCTS agent backed by the Rust core (barricades_native.Tree).

Default mode uses the Rust-internal heuristic eval (no neural net) — a fast,
self-contained bot. Pass an `eval_fn` mapping a batch of planes (B,6,9,9) float32
to (policy (B,140), value (B,)) to drive it with a network instead.
"""
import numpy as np
import barricades_native as bn
from agents.base import Agent


def _to_native(state):
    return (tuple(state.pawns), sorted(state.h_walls), sorted(state.v_walls),
            tuple(state.walls_left), state.turn)


def _from_tuple(mv):
    from core.state import Step, Wall
    if mv[0] == "step":
        return Step((mv[1], mv[2]))
    return Wall(mv[1], mv[2], mv[3])


class NativeMctsAgent(Agent):
    name = "native_mcts"

    def __init__(self, sims=160, c_puct=1.5, seed=None, eval_fn=None):
        self.sims = sims
        self.c_puct = c_puct
        self.seed = 0 if seed is None else int(seed)
        self.eval_fn = eval_fn

    def select_move(self, state):
        from core.rules import is_terminal
        if state.walls_left == (0, 0) and not is_terminal(state):
            _val, mv = bn.solve_race(_to_native(state))
            return _from_tuple(mv)
        tree = bn.Tree(_to_native(state), self.c_puct, self.seed)
        if self.eval_fn is None:
            return _from_tuple(tree.run_heuristic(self.sims))
        evals, guard = 0, 0
        while evals < self.sims and guard < self.sims * 8 + 64:
            guard += 1
            planes = tree.prepare_leaf()
            if planes is None:
                continue
            policy, value = self.eval_fn(np.asarray(planes)[None])
            tree.receive(np.ascontiguousarray(policy[0], dtype=np.float32), float(value[0]))
            evals += 1
        mv, _ = tree.best_move(0.0)
        return _from_tuple(mv)
