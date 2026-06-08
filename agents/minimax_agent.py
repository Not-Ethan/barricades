import random
import time

from core.state import Step
from core.rules import (
    legal_steps, legal_walls, apply_move, is_terminal,
    shortest_path_len,
)
from agents.base import Agent, Analysis
from agents.heuristics import evaluate, WIN_SCORE
from agents.movegen import relevant_moves, relevant_walls

# Transposition table entry flags.
_EXACT = 0
_LOWER = 1  # alpha cut-off (we found a lower bound — score >= stored)
_UPPER = 2  # beta cut-off (we found an upper bound — score <= stored)


def _ordered_moves(state):
    """Return moves ordered for good alpha-beta pruning.

    Steps that shrink the mover's own shortest path are returned first
    (ascending distance after the step), then wall moves that lengthen
    the opponent's path.  Only relevant walls are included — ones that
    don't lengthen the opponent's path are skipped entirely.

    Falls back to all legal steps if relevant_moves returns nothing (safety
    guard; in practice there is always at least one step available).
    """
    me = state.turn
    rm = relevant_moves(state)
    if not rm:
        # Fallback: all legal steps (should never happen in a legal game).
        from core.rules import legal_moves
        rm = legal_moves(state)

    steps = []
    walls = []
    for m in rm:
        if isinstance(m, Step):
            new_state = apply_move(state, m)
            d = shortest_path_len(new_state, me)
            steps.append((d if d is not None else 1_000, m))
        else:
            walls.append(m)

    steps.sort(key=lambda t: t[0])
    return [m for _, m in steps] + walls


class MinimaxAgent(Agent):
    """Alpha-beta minimax with iterative deepening, a transposition table,
    and relevant-move pruning for a far smaller branching factor.

    Move generation: only relevant moves are considered at every node —
    steps that reduce the mover's own shortest path come first (good
    move ordering for alpha-beta), then walls that strictly lengthen the
    opponent's shortest path (relevant_walls).  This cuts the branching
    factor from ~130 to ~10-20, allowing much deeper search within the
    same time budget.

    Transposition table: a per-search dict keyed by (state, depth_remaining)
    stores (value, flag) with flag ∈ {_EXACT, _LOWER, _UPPER} for alpha-beta
    bounds.  The TT is cleared at the start of each analyze() call.
    """

    name = "minimax"

    def __init__(self, time_budget=1.0, max_depth=64, seed=None,
                 # wall_cap kept for API compatibility but ignored (relevant_walls
                 # already prunes junk walls precisely).
                 wall_cap=None):
        self.time_budget = time_budget
        self.max_depth = max_depth
        self._rng = random.Random(seed)
        self._nodes = 0
        self._deadline = 0.0
        self._tt: dict = {}  # transposition table — cleared per analyze()

    class _Timeout(Exception):
        pass

    def _search(self, state, depth, alpha, beta, root_player):
        self._nodes += 1

        # --- Leaf evaluation (must come before timeout so depth-1 completes) ---
        if is_terminal(state) or depth == 0:
            return evaluate(state, root_player)

        # --- Timeout check (after leaf so the very first depth always finishes) ---
        if time.monotonic() > self._deadline:
            raise MinimaxAgent._Timeout()

        # --- Transposition table probe ---
        tt_key = (state, depth)
        tt_entry = self._tt.get(tt_key)
        if tt_entry is not None:
            tt_val, tt_flag = tt_entry
            if tt_flag == _EXACT:
                return tt_val
            elif tt_flag == _LOWER:
                alpha = max(alpha, tt_val)
            elif tt_flag == _UPPER:
                beta = min(beta, tt_val)
            if alpha >= beta:
                return tt_val

        orig_alpha = alpha
        maximizing = state.turn == root_player
        moves = _ordered_moves(state)

        if maximizing:
            best = -float("inf")
            for m in moves:
                val = self._search(apply_move(state, m), depth - 1, alpha, beta, root_player)
                if val > best:
                    best = val
                alpha = max(alpha, best)
                if alpha >= beta:
                    break
        else:
            best = float("inf")
            for m in moves:
                val = self._search(apply_move(state, m), depth - 1, alpha, beta, root_player)
                if val < best:
                    best = val
                beta = min(beta, best)
                if alpha >= beta:
                    break

        # --- Transposition table store ---
        if best <= orig_alpha:
            flag = _UPPER
        elif best >= beta:
            flag = _LOWER
        else:
            flag = _EXACT
        self._tt[tt_key] = (best, flag)

        return best

    def analyze(self, state):
        """Run iterative-deepening alpha-beta from `state`. Returns an Analysis
        with the best move, its score, the top-8 root candidates, and search stats."""
        self._nodes = 0
        self._tt = {}  # fresh TT per search
        self._deadline = time.monotonic() + self.time_budget
        t0 = time.monotonic()
        root_player = state.turn

        # Root candidates: use relevant_moves (same pruning as internal nodes).
        root_moves = _ordered_moves(state)
        if not root_moves:
            from core.rules import legal_moves
            root_moves = legal_moves(state)

        best_move = root_moves[0]
        best_scores = {m: 0.0 for m in root_moves}
        completed_depth = 0

        for depth in range(1, self.max_depth + 1):
            try:
                scores = {}
                # Search root moves in order (already ordered by _ordered_moves).
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

        # Choose the best move; break ties randomly for reproducibility.
        best_val = max(best_scores.values())
        winners = [m for m, v in best_scores.items() if v == best_val]
        best_move = self._rng.choice(winners)
        candidates = sorted(best_scores.items(), key=lambda kv: kv[1], reverse=True)[:8]
        return Analysis(
            best_move=best_move,
            value=best_val,
            candidates=[(m, v) for m, v in candidates],
            stats={
                "nodes": self._nodes,
                "depth": completed_depth,
                "time_ms": int((time.monotonic() - t0) * 1000),
                "tt_size": len(self._tt),
            },
        )

    def select_move(self, state):
        return self.analyze(state).best_move
