import random
import time

from core.state import Step
from core.rules import (
    legal_steps, legal_walls, apply_move, is_terminal,
    shortest_path_len,
)
from agents.base import Agent, Analysis
from agents.heuristics import evaluate, WIN_SCORE


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
            # Closeness of the wall anchor to either pawn (smaller = more relevant).
            return min(abs(w.c - p1[0]) + abs(w.r - p1[1]),
                       abs(w.c - p0[0]) + abs(w.r - p0[1]))

        walls = sorted(walls, key=relevance)[:wall_cap]
    return ordered + walls


class MinimaxAgent(Agent):
    """Alpha-beta minimax with iterative deepening and a wall-clock time budget.

    Move ordering: pawn steps that shrink our own shortest path come first,
    then other steps, then walls. At non-root search nodes only the `wall_cap`
    most positionally-relevant wall placements are considered (strength cap,
    not a correctness issue).

    Parameters
    ----------
    eval_fn:
        Optional evaluation function with signature ``eval_fn(state, player) -> float``.
        Defaults to ``None``, which uses the standard ``agents.heuristics.evaluate``.
        This parameter is backward-compatible: omitting it preserves existing behaviour.
    candidate_moves:
        Optional callable ``state -> list[Move]``.  When provided, the search uses
        it for BOTH root and internal nodes instead of the default ``ordered_moves``
        with wall-cap pruning.  Steps are still ordered first (ascending resulting
        own shortest-path); walls follow in the order returned by the callable.
        When ``None`` (default), behaviour is EXACTLY as before (``ordered_moves``
        with wall_cap).
    """

    name = "minimax"

    def __init__(self, time_budget=1.0, max_depth=64, wall_cap=12, seed=None,
                 eval_fn=None, candidate_moves=None):
        self.time_budget = time_budget
        self.max_depth = max_depth
        self.wall_cap = wall_cap
        self._rng = random.Random(seed)
        self._nodes = 0
        self._deadline = 0.0
        # Use the provided eval function or fall back to the default heuristic.
        self._eval = eval_fn if eval_fn is not None else evaluate
        # Optional custom candidate-move generator.
        self._candidate_moves = candidate_moves

    class _Timeout(Exception):
        pass

    def _get_moves(self, state, at_root=False):
        """Return an ordered move list for the given state.

        When ``self._candidate_moves`` is set, delegates to it and re-applies
        the steps-first ordering (steps sorted by resulting own shortest path
        ascending, then walls in the order returned by the callable).

        When ``None``, falls back to the existing ``ordered_moves`` behaviour
        (wall_cap applied at internal nodes, ignored at root so all walls are
        considered there).
        """
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
        # Default: use the existing ordered_moves logic.
        cap = None if at_root else self.wall_cap
        return ordered_moves(state, wall_cap=cap)

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

    def analyze(self, state):
        """Run iterative-deepening alpha-beta from `state`. Returns an Analysis
        with the best move, its score, the top-8 root candidates, and search stats."""
        self._nodes = 0
        self._deadline = time.monotonic() + self.time_budget
        t0 = time.monotonic()
        root_player = state.turn
        root_moves = self._get_moves(state, at_root=True)  # all moves at root
        best_move = root_moves[0]
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
                # Early exit: a forced win was found.
                if max(scores.values()) >= WIN_SCORE:
                    break
            except MinimaxAgent._Timeout:
                break
        # Choose the best move. Among moves of EQUAL search value, break ties
        # DETERMINISTICALLY and sensibly rather than at random: prefer a step
        # that advances our pawn (smaller resulting shortest path), prefer steps
        # over walls (never burn a wall that merely ties a step), then fall back
        # to a stable order. A random tie-break caused visible oscillation and
        # marginal "random" walls in real play, because the server builds a
        # fresh, randomly-seeded agent every move and so flipped tied choices
        # turn-to-turn.
        best_val = max(best_scores.values())
        winners = [m for m, v in best_scores.items() if v == best_val]
        best_move = min(winners, key=lambda m: self._tiebreak_key(state, m))
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

    def _tiebreak_key(self, state, move):
        """Deterministic preference among equally-scored moves: advancing steps
        first (by resulting own shortest path), then walls, then a stable order.
        Conserves walls (a wall is only chosen when it strictly beats every step)
        and never oscillates (the choice depends only on the position)."""
        me = state.turn
        if isinstance(move, Step):
            d = shortest_path_len(apply_move(state, move), me)
            return (0, 10_000 if d is None else d, repr(move))
        return (1, 0, repr(move))

    def select_move(self, state):
        return self.analyze(state).best_move
