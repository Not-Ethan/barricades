import random
import time

from core.state import Step, Wall
from core.rules import (
    legal_steps, legal_walls, apply_move, is_terminal,
    shortest_path_len,
)
from agents.base import Agent, Analysis
from agents.heuristics import evaluate, WIN_SCORE

try:
    import barricades_native as _bn
except ImportError:                          # pure-Python environments
    _bn = None

from agents.native_agent import _to_native, _from_tuple

_UNREACHABLE = 1_000     # mirrors agents.heuristics
_TEMPO_BONUS = 0.5


def ordered_moves(state, wall_cap=None):
    """Steps first (those reducing our own shortest path ahead of others),
    then walls (optionally capped to the most relevant).

    wall_cap=None means all legal walls are included (used at the root so the
    agent can consider every move). At deeper search nodes wall_cap limits walls
    to the most positionally-relevant ones for tractability in pure Python.
    """
    me = state.turn
    steps = []
    for c in legal_steps(state):
        # Evaluate distance AFTER stepping (for sorting; ignore turn switch).
        new_state = apply_move(state, Step(c))
        # shortest_path_len uses the *new* state; me is still the same player index.
        d = shortest_path_len(new_state, me)
        steps.append((d if d is not None else 1_000, Step(c)))
    steps.sort(key=lambda t: t[0])
    ordered = [m for _, m in steps]

    walls = legal_walls(state)
    if wall_cap is not None and len(walls) > wall_cap:
        opp = 1 - me
        p0 = state.pawns[me]
        p1 = state.pawns[opp]

        def relevance(w):
            # Closeness of the wall anchor to either pawn (smaller = more relevant),
            # with (c, r, orient) as a canonical tiebreak so the capped set is
            # deterministic AND matches the native backend exactly (the cap is the
            # only thing that changes the minimax value, so the sets must agree).
            return (min(abs(w.c - p1[0]) + abs(w.r - p1[1]),
                        abs(w.c - p0[0]) + abs(w.r - p0[1])),
                    w.c, w.r, w.orient)

        walls = sorted(walls, key=relevance)[:wall_cap]
    return ordered + walls


# --- native-backed equivalents (operate on the Rust state tuple) -------------
# Tuple layout: (pawns=((c,r),(c,r)), h_walls, v_walls, walls_left=(n0,n1), turn).

def _eval_native(nt, player):
    """Mirror of agents.heuristics.evaluate on a native state tuple, using the Rust
    engine for winner + shortest-path. Arithmetic identical to evaluate()."""
    w = _bn.winner(nt)
    if w is not None:
        return WIN_SCORE if w == player else -WIN_SCORE
    opp = 1 - player
    d_self = _bn.shortest_path_len(nt, player)
    d_opp = _bn.shortest_path_len(nt, opp)
    eff_self = _UNREACHABLE if d_self is None else d_self
    eff_opp = _UNREACHABLE if d_opp is None else d_opp
    path_term = eff_opp - eff_self
    if d_self is None:
        path_term -= _UNREACHABLE
    wall_term = 0.1 * (nt[3][player] - nt[3][opp])
    mover = nt[4]
    tempo = 0.0
    if _bn.shortest_path_len(nt, mover) == 1:
        tempo = _TEMPO_BONUS if mover == player else -_TEMPO_BONUS
    return path_term + wall_term + tempo


def _native_ordered_moves(nt, wall_cap):
    """Native mirror of ordered_moves. Steps sorted by resulting own shortest path;
    walls capped by the SAME (relevance, c, r, orient) canonical key as the Python
    path, so the capped wall set is identical."""
    me = nt[4]
    steps = []
    walls = []
    for m in _bn.legal_moves(nt):
        if m[0] == "step":
            d = _bn.shortest_path_len(_bn.apply_move(nt, m), me)
            steps.append((d if d is not None else 1_000, m))
        else:
            walls.append(m)
    steps.sort(key=lambda t: t[0])
    ordered = [m for _, m in steps]
    if wall_cap is not None and len(walls) > wall_cap:
        p0 = nt[0][me]
        p1 = nt[0][1 - me]

        def relevance(m):                    # m = ("wall", c, r, orient)
            return (min(abs(m[1] - p1[0]) + abs(m[2] - p1[1]),
                        abs(m[1] - p0[0]) + abs(m[2] - p0[1])),
                    m[1], m[2], m[3])

        walls = sorted(walls, key=relevance)[:wall_cap]
    return ordered + walls


def _native_move_key(m):
    return ("S", m[1], m[2]) if m[0] == "step" else ("W", m[1], m[2], m[3])


def _py_move_key(m):
    if isinstance(m, Step):
        return ("S", m.to_cell[0], m.to_cell[1])
    return ("W", m.c, m.r, m.orient)


class MinimaxAgent(Agent):
    """Alpha-beta minimax with iterative deepening and a wall-clock time budget.

    Move ordering: pawn steps that shrink our own shortest path come first,
    then other steps, then walls. At non-root search nodes only the `wall_cap`
    most positionally-relevant wall placements are considered (strength cap,
    not a correctness issue).

    backend:
        "native" (default) reroutes move generation, shortest-path and the eval
        through the Rust engine (`barricades_native`) for a large per-node speedup.
        It is value-preserving: it returns the same minimax value as the Python
        search at any depth (the wall-cap tiebreak is canonical in both). Falls
        back to "python" automatically when the native module is unavailable or
        when a custom `eval_fn`/`candidate_moves` is supplied (which native can't
        replicate).

    eval_fn:
        Optional ``eval_fn(state, player) -> float``. Default ``None`` uses
        ``agents.heuristics.evaluate``. Supplying one forces the Python backend.
    candidate_moves:
        Optional ``state -> list[Move]`` (Python-backend only).
    """

    name = "minimax"

    def __init__(self, time_budget=1.0, max_depth=64, wall_cap=12, seed=None,
                 eval_fn=None, candidate_moves=None, backend="native"):
        self.time_budget = time_budget
        self.max_depth = max_depth
        self.wall_cap = wall_cap
        self._rng = random.Random(seed)
        self._nodes = 0
        self._deadline = 0.0
        self._eval = eval_fn if eval_fn is not None else evaluate
        self._candidate_moves = candidate_moves
        # Native backend only when no custom eval/movegen and the module is built.
        self._use_native = (backend == "native" and eval_fn is None
                            and candidate_moves is None and _bn is not None)
        self.backend = "native" if self._use_native else "python"

    class _Timeout(Exception):
        pass

    def _get_moves(self, state, at_root=False):
        """Ordered move list (Python backend). Honours a custom candidate_moves;
        otherwise the ordered_moves heuristic with wall_cap at internal nodes."""
        if self._candidate_moves is not None:
            raw = self._candidate_moves(state)
            me = state.turn
            steps = []
            walls = []
            for m in raw:
                if isinstance(m, Step):
                    s2 = apply_move(state, m)
                    d = shortest_path_len(s2, me)
                    steps.append((d if d is not None else 1_000, m))
                else:
                    walls.append(m)
            steps.sort(key=lambda t: t[0])
            return [m for _, m in steps] + walls
        cap = None if at_root else self.wall_cap
        return ordered_moves(state, wall_cap=cap)

    # --- Python backend search ----------------------------------------------
    def _search(self, state, depth, alpha, beta, root_player):
        self._nodes += 1
        # Evaluate leaves WITHOUT a prior timeout check so that depth-1 always
        # completes (guarantees a move exists even under a tiny budget).
        if is_terminal(state) or depth == 0:
            return self._eval(state, root_player)
        if time.monotonic() > self._deadline:
            raise MinimaxAgent._Timeout()
        maximizing = state.turn == root_player
        moves = self._get_moves(state, at_root=False)
        if maximizing:
            best = -float("inf")
            for m in moves:
                val = self._search(apply_move(state, m), depth - 1, alpha, beta, root_player)
                best = max(best, val)
                alpha = max(alpha, best)
                if alpha >= beta:
                    break
            return best
        else:
            best = float("inf")
            for m in moves:
                val = self._search(apply_move(state, m), depth - 1, alpha, beta, root_player)
                best = min(best, val)
                beta = min(beta, best)
                if alpha >= beta:
                    break
            return best

    # --- native backend search (same alpha-beta on the Rust state tuple) -----
    def _search_native(self, nt, depth, alpha, beta, root_player):
        self._nodes += 1
        if _bn.is_terminal(nt) or depth == 0:
            return _eval_native(nt, root_player)
        if time.monotonic() > self._deadline:
            raise MinimaxAgent._Timeout()
        maximizing = nt[4] == root_player
        moves = _native_ordered_moves(nt, self.wall_cap)   # internal node: cap walls
        if maximizing:
            best = -float("inf")
            for m in moves:
                val = self._search_native(_bn.apply_move(nt, m), depth - 1, alpha, beta, root_player)
                best = max(best, val)
                alpha = max(alpha, best)
                if alpha >= beta:
                    break
            return best
        else:
            best = float("inf")
            for m in moves:
                val = self._search_native(_bn.apply_move(nt, m), depth - 1, alpha, beta, root_player)
                best = min(best, val)
                beta = min(beta, best)
                if alpha >= beta:
                    break
            return best

    def _root_scores(self, state, depth):
        """{canonical_move_key: minimax value} for every root move at a fixed depth,
        via this agent's backend. Gates native==python value-preservation."""
        self._deadline = float("inf")
        if self._use_native:
            nt = _to_native(state)
            rp = nt[4]
            return {_native_move_key(m): self._search_native(
                        _bn.apply_move(nt, m), depth - 1, -float("inf"), float("inf"), rp)
                    for m in _native_ordered_moves(nt, None)}
        rp = state.turn
        return {_py_move_key(m): self._search(
                    apply_move(state, m), depth - 1, -float("inf"), float("inf"), rp)
                for m in self._get_moves(state, at_root=True)}

    def analyze(self, state):
        return self._analyze_native(state) if self._use_native else self._analyze_python(state)

    def _analyze_python(self, state):
        """Iterative-deepening alpha-beta (Python backend). Returns an Analysis with
        the best move, its score, the top-8 root candidates, and search stats."""
        self._nodes = 0
        self._deadline = time.monotonic() + self.time_budget
        t0 = time.monotonic()
        root_player = state.turn
        root_moves = self._get_moves(state, at_root=True)  # all moves at root
        best_scores = {m: 0.0 for m in root_moves}
        completed_depth = 0
        for depth in range(1, self.max_depth + 1):
            try:
                scores = {}
                for m in root_moves:
                    scores[m] = self._search(
                        apply_move(state, m), depth - 1,
                        -float("inf"), float("inf"), root_player,
                    )
                best_scores = scores
                completed_depth = depth
                if max(scores.values()) >= WIN_SCORE:   # forced win found
                    break
            except MinimaxAgent._Timeout:
                break
        # Position-seeded random tie-break over equal-value moves (preserves
        # strength: genuinely useful walls the shallow eval merely ties with a step
        # still get played, but deterministically per position).
        best_val = max(best_scores.values())
        winners = [m for m, v in best_scores.items() if v == best_val]
        best_move = random.Random(hash(state)).choice(winners)
        candidates = sorted(best_scores.items(), key=lambda kv: kv[1], reverse=True)[:8]
        return Analysis(
            best_move=best_move,
            value=best_val,
            candidates=[(m, v) for m, v in candidates],
            stats={
                "nodes": self._nodes,
                "depth": completed_depth,
                "time_ms": int((time.monotonic() - t0) * 1000),
            },
        )

    def _analyze_native(self, state):
        """Iterative-deepening alpha-beta on the Rust state tuple. Same control flow
        as _analyze_python; converts the chosen move back to a Step/Wall."""
        nt = _to_native(state)
        self._nodes = 0
        self._deadline = time.monotonic() + self.time_budget
        t0 = time.monotonic()
        root_player = nt[4]
        root_moves = _native_ordered_moves(nt, None)       # all walls at root
        best_scores = {m: 0.0 for m in root_moves}
        completed_depth = 0
        for depth in range(1, self.max_depth + 1):
            try:
                scores = {}
                for m in root_moves:
                    scores[m] = self._search_native(
                        _bn.apply_move(nt, m), depth - 1,
                        -float("inf"), float("inf"), root_player,
                    )
                best_scores = scores
                completed_depth = depth
                if max(scores.values()) >= WIN_SCORE:
                    break
            except MinimaxAgent._Timeout:
                break
        best_val = max(best_scores.values())
        winners = [m for m, v in best_scores.items() if v == best_val]
        best_native = random.Random(hash(state)).choice(winners)
        candidates = sorted(best_scores.items(), key=lambda kv: kv[1], reverse=True)[:8]
        return Analysis(
            best_move=_from_tuple(best_native),
            value=best_val,
            candidates=[(_from_tuple(m), v) for m, v in candidates],
            stats={
                "nodes": self._nodes,
                "depth": completed_depth,
                "time_ms": int((time.monotonic() - t0) * 1000),
            },
        )

    def select_move(self, state):
        return self.analyze(state).best_move
