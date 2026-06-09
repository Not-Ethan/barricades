class Solver:
    """Exact full-game solver: depth-bounded negamax + alpha-beta + a transposition
    table keyed on (state, depth). Returns the game-theoretic value for the side to
    move (+1 win / 0 draw-at-bound / -1 loss) and the set of optimal moves.

    Draw handling: walls only accumulate (finite budget), so non-termination comes
    from pawn cycling; the depth bound makes the search total and scores unresolved
    lines as draws. max_depth must be large enough that any forced win is found.
    """

    def __init__(self, engine, max_depth=None):
        self.e = engine
        if max_depth is None:
            max_depth = 4 * engine.N + 2 * engine.W + 6
        self.max_depth = max_depth
        self._tt = {}

    def _ordered_moves(self, s):
        # move ordering by resulting shortest-path advantage for the mover
        mover = s.turn
        scored = []
        for m in self.e.legal_moves(s):
            s2 = self.e.apply_move(s, m)
            d_self = self.e.shortest_path_len(s2, mover)
            d_opp = self.e.shortest_path_len(s2, 1 - mover)
            big = 10 * self.e.N
            score = (d_opp if d_opp is not None else big) - \
                    (d_self if d_self is not None else big)
            scored.append((score, m))
        scored.sort(key=lambda x: -x[0])
        return [m for _, m in scored]

    def _negamax(self, s, depth, alpha, beta):
        w = self.e.winner(s)
        if w is not None:
            return 1 if w == s.turn else -1
        if depth == 0:
            return 0
        key = (s, depth)
        cached = self._tt.get(key)
        if cached is not None:
            return cached
        best = -2
        for m in self._ordered_moves(s):
            v = -self._negamax(self.e.apply_move(s, m), depth - 1, -beta, -alpha)
            if v > best:
                best = v
            if best > alpha:
                alpha = best
            if alpha >= beta:
                break
        if best == -2:
            best = -1
        self._tt[key] = best
        return best

    def solve(self, s):
        """Returns (value, [optimal moves])."""
        best_val = -2
        vals = {}
        for m in self.e.legal_moves(s):
            v = -self._negamax(self.e.apply_move(s, m), self.max_depth - 1, -2, 2)
            vals[m] = v
            if v > best_val:
                best_val = v
        if best_val == -2:
            return -1, []
        best = [m for m, v in vals.items() if v == best_val]
        return best_val, best
