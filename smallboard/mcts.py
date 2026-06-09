import math
import random


class _Node:
    __slots__ = ("state", "parent", "move", "prior", "children", "N", "W", "expanded")

    def __init__(self, state, parent=None, move=None, prior=0.0):
        self.state = state
        self.parent = parent
        self.move = move
        self.prior = prior
        self.children = []
        self.N = 0
        self.W = 0.0
        self.expanded = False


class PUCTSearch:
    def __init__(self, wrap, sims=80, c_puct=1.5, seed=None,
                 dirichlet_alpha=None, dirichlet_eps=0.25):
        self.w = wrap
        self.e = wrap.e
        self.sims = sims
        self.c_puct = c_puct
        self._rng = random.Random(seed)
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps

    def _expand(self, node, root_player):
        priors, value = self.w.predict(node.state)
        for m, p in priors.items():
            node.children.append(
                _Node(self.e.apply_move(node.state, m), node, m, p))
        node.expanded = True
        return value if node.state.turn == root_player else -value

    def _select(self, node, root_player):
        sqrt_n = math.sqrt(node.N)
        best, best_score = None, None
        for ch in node.children:
            q = (ch.W / ch.N) if ch.N else 0.0
            q = q if node.state.turn == root_player else -q
            u = self.c_puct * ch.prior * sqrt_n / (1 + ch.N)
            score = q + u
            if best_score is None or score > best_score:
                best_score, best = score, ch
        return best

    def run(self, state):
        root = _Node(state)
        root_player = state.turn
        self._expand(root, root_player)
        root.N = 1
        if self.dirichlet_alpha and root.children:
            noise = self._dirichlet(len(root.children))
            for ch, nz in zip(root.children, noise):
                ch.prior = (1 - self.dirichlet_eps) * ch.prior + self.dirichlet_eps * nz
        for _ in range(self.sims):
            node = root
            while node.expanded and not self.e.is_terminal(node.state):
                node = self._select(node, root_player)
            if self.e.is_terminal(node.state):
                w = self.e.winner(node.state)
                v = 1.0 if w == root_player else -1.0
            else:
                v = self._expand(node, root_player)
            while node is not None:
                node.N += 1
                node.W += v
                node = node.parent
        if not root.children:
            return None, {}, {"value": 0.0}
        total = sum(ch.N for ch in root.children)
        pi = {ch.move: ch.N / total for ch in root.children} if total else {}
        top = max(ch.N for ch in root.children)
        best = self._rng.choice([ch for ch in root.children if ch.N == top])
        return best.move, pi, {"value": root.W / root.N if root.N else 0.0}

    def _dirichlet(self, k):
        gs = [self._rng.gammavariate(self.dirichlet_alpha, 1.0) for _ in range(k)]
        tot = sum(gs) or 1.0
        return [g / tot for g in gs]
