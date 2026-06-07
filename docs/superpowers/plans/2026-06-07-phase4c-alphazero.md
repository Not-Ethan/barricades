# Phase 4c: AlphaZero-lite — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** A working AlphaZero-style self-play learning stack for Quoridor: state/action encoding with canonicalization, a small PyTorch policy+value net, NN-guided PUCT MCTS, a self-play data generator, a training loop, and an `AZAgent` registered as `az`. Plus a `scripts/train_az.py` that runs a short proof-of-concept training and saves a checkpoint. **Strength is not the goal — correctness of the full learning loop is.** Real strength needs far more compute than a CPU smoke-run.

**Architecture:** New subpackage `agents/az/`. Pure encoding layer (no torch) → torch model → PUCT MCTS that calls the net for priors+value (no rollouts) → self-play → training → agent. All search happens over REAL game states/moves; the net wrapper hides all canonicalization. Depends on `core`, `agents.base`, `torch`, `numpy`.

**Tech Stack:** Python 3.14, PyTorch 2.12 (CPU), numpy, pytest. (torch wheel confirmed available for 3.14.)

---

## Conventions / core facts
- `core`: `legal_moves`, `legal_steps`, `apply_move`, `is_terminal`, `winner`, `GameState`, `Step`, `Wall`, `goal_row`, `initial_state`, `is_blocked`. Cells `(col,row)` 0..8. Player 0 goal row 8, player 1 goal row 0. Walls anchors `(c,r)` 0..7, orient "H"/"V".
- `agents.base`: `Agent`, `Analysis`. Registry seam: ADD only `"az"` to `agents/registry.py`.

## File Structure
```
agents/az/
  __init__.py
  encoding.py     # planes, action<->move, canonicalization, legal mask  (NO torch)
  model.py        # QuoridorNet (torch) + NetWrapper.predict(state)->(priors, value)
  mcts_nn.py      # PUCT MCTS using NetWrapper
  selfplay.py     # play_selfplay_game -> training examples
  train.py        # loss, train_step, run_training (+ checkpoint save/load)
  agent.py        # AZAgent(Agent)
agents/registry.py # +"az"
scripts/train_az.py # CLI smoke-train
models/             # checkpoints (gitignored)
tests/test_az_encoding.py
tests/test_az_model.py
tests/test_az_mcts.py
tests/test_az_train.py
```
Add to `.gitignore`: `models/`. Add deps to `pyproject.toml`: `torch`, `numpy`.

---

## THE CANONICAL FRAME (read carefully — everything depends on it)

The net always sees the position from the **side-to-move's** perspective, oriented so that player advances toward **increasing row** (goal = row 8).

- `flip = (state.turn == 1)`. If `flip`, vertically mirror everything: cell `(c, r) -> (c, 8 - r)`.
- Under vertical mirror: a pawn at `(c,r)` -> `(c, 8-r)`. A wall anchor `(c,r)` (either orientation) -> `(c, 7 - r)` (because a wall in the gap above row `r` becomes the gap above row `7-r`). Orientation is **unchanged** (vertical mirror preserves H/V).
- "Me" = `state.pawns[state.turn]`, "opp" = `state.pawns[1-state.turn]`. After canonicalization, "me" is the upward-advancing pawn.

### Action space (140 actions, canonical frame)
- **Pawn moves: 12 direction indices** relative to my pawn (N = +row toward goal):
  - steps: `0=(0,+1)N  1=(0,-1)S  2=(+1,0)E  3=(-1,0)W`
  - straight jumps: `4=(0,+2)  5=(0,-2)  6=(+2,0)  7=(-2,0)`
  - diagonal jumps: `8=(+1,+1)  9=(-1,+1)  10=(+1,-1)  11=(-1,-1)`
- **Wall moves: 128** = `12 + orient_off + r*8 + c`, where `orient_off = 0` for H, `64` for V; `c,r in 0..7`.
- Total `N_ACTIONS = 140`.

### Mapping a REAL move to a canonical action index
- For a `Step(to_cell)`: in canonical frame my pawn is `cf(my_pawn)` and dest is `cf(to_cell)` where `cf` applies the flip if needed. The delta is `(dx, dy) = cf(to_cell) - cf(my_pawn)`. Note flipping negates `dy`. Look up `(dx,dy)` in the 12-direction table -> index 0..11.
- For a `Wall(c,r,orient)`: canonical anchor is `(c, 7-r)` if flip else `(c,r)`; index `= 12 + (0 if H else 64) + r'*8 + c` with `r'` the canonical row.

The net wrapper uses this to gather logits for each legal real move; the inverse (`action_to_move`) is provided for tests/robustness but the agent never needs it.

---

## Task 1: Encoding (pure, no torch) — THE FOUNDATION

**Files:** Create `agents/az/__init__.py` (empty), `agents/az/encoding.py`, test `tests/test_az_encoding.py`.

This task is correctness-critical; the tests are extensive on purpose.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_az_encoding.py
import numpy as np

from core.state import GameState, Step, Wall, initial_state
from core.rules import legal_moves
from agents.az.encoding import (
    N_ACTIONS, N_PLANES, encode_planes, move_to_action, action_to_move,
    legal_action_mask, canonical_flip,
)


def _state(p0, p1, wl=(10, 10), turn=0, h=(), v=()):
    return GameState((p0, p1), frozenset(h), frozenset(v), wl, turn)


def test_constants():
    assert N_ACTIONS == 140
    assert N_PLANES == 6


def test_planes_shape_and_pawns_player0():
    s = initial_state()
    planes = encode_planes(s)
    assert planes.shape == (N_PLANES, 9, 9)
    # turn 0: no flip. my pawn (p0) at (4,0); opp (p1) at (4,8).
    assert planes[0, 0, 4] == 1 and planes[0].sum() == 1     # my pawn plane (row,col)
    assert planes[1, 8, 4] == 1 and planes[1].sum() == 1     # opp pawn plane


def test_planes_canonicalized_for_player1():
    # turn 1: flip. my pawn = p1 at (4,8) -> canonical (4,0). opp p0 (4,0) -> (4,8).
    s = _state((4, 0), (4, 8), turn=1)
    planes = encode_planes(s)
    assert planes[0, 0, 4] == 1     # my pawn canonically at bottom (row 0)
    assert planes[1, 8, 4] == 1     # opp canonically at top


def test_wall_planes_and_canonical_flip():
    assert canonical_flip(_state((4, 0), (4, 8), turn=0)) is False
    assert canonical_flip(_state((4, 0), (4, 8), turn=1)) is True
    # H wall anchor (2,3), turn 0 -> plane2 at [3,2]
    s = _state((4, 0), (4, 8), h=[(2, 3)], turn=0)
    p = encode_planes(s)
    assert p[2, 3, 2] == 1 and p[2].sum() == 1
    # same wall, turn 1 (flip) -> canonical anchor (2, 7-3)=(2,4) -> plane2 at [4,2]
    s2 = _state((4, 0), (4, 8), h=[(2, 3)], turn=1)
    p2 = encode_planes(s2)
    assert p2[2, 4, 2] == 1


def test_walls_remaining_planes():
    s = _state((4, 0), (4, 8), wl=(7, 10), turn=0)
    p = encode_planes(s)
    assert np.allclose(p[4], 0.7)     # my walls remaining / 10
    assert np.allclose(p[5], 1.0)     # opp walls remaining / 10


def test_step_action_roundtrip_no_flip():
    s = initial_state()           # p0 at (4,0), turn 0
    mv = Step((4, 1))             # north step
    idx = move_to_action(mv, s)
    assert idx == 0               # N
    assert action_to_move(idx, s) == mv


def test_step_action_roundtrip_with_flip():
    s = _state((4, 0), (4, 8), turn=1)   # p1 at (4,8) to move, flip
    mv = Step((4, 7))                     # p1 steps toward its goal (row 0): real delta (0,-1)
    idx = move_to_action(mv, s)
    assert idx == 0                       # canonical N (advancing upward)
    assert action_to_move(idx, s) == mv   # inverse returns the REAL move


def test_wall_action_roundtrip_no_flip():
    s = initial_state()
    for w in [Wall(0, 0, "H"), Wall(7, 7, "H"), Wall(3, 5, "V"), Wall(0, 0, "V")]:
        idx = move_to_action(w, s)
        assert 12 <= idx < 140
        assert action_to_move(idx, s) == w


def test_wall_action_roundtrip_with_flip():
    s = _state((4, 0), (4, 8), turn=1)
    w = Wall(2, 3, "H")
    idx = move_to_action(w, s)
    # canonical anchor (2, 7-3)=(2,4): idx = 12 + 0 + 4*8 + 2 = 46
    assert idx == 46
    assert action_to_move(idx, s) == w     # inverse maps back to real (2,3,H)


def test_legal_mask_matches_legal_moves():
    for s in [initial_state(), _state((4, 4), (4, 5), turn=0),
              _state((4, 0), (4, 8), h=[(2, 3)], turn=1)]:
        mask = legal_action_mask(s)
        assert mask.shape == (N_ACTIONS,)
        assert mask.sum() == len(legal_moves(s))
        # every legal move's action index is unmasked, and bijective (no collisions)
        idxs = [move_to_action(m, s) for m in legal_moves(s)]
        assert len(set(idxs)) == len(idxs)        # bijective on legal moves
        for i in idxs:
            assert mask[i] == 1
```

- [ ] **Step 2: Run, verify fail.**

- [ ] **Step 3: Implement `agents/az/encoding.py`**

```python
import numpy as np

from core.state import Step, Wall
from core.rules import legal_moves

N = 9
N_PLANES = 6
N_ACTIONS = 140

# 12 pawn directions (dx, dy) in canonical frame (N = +row).
_DIRS = [
    (0, 1), (0, -1), (1, 0), (-1, 0),       # 0..3 steps N S E W
    (0, 2), (0, -2), (2, 0), (-2, 0),       # 4..7 straight jumps
    (1, 1), (-1, 1), (1, -1), (-1, -1),     # 8..11 diagonal jumps
]
_DIR_INDEX = {d: i for i, d in enumerate(_DIRS)}


def canonical_flip(state):
    return state.turn == 1


def _cf_cell(cell, flip):
    c, r = cell
    return (c, (N - 1 - r) if flip else r)


def _cf_wall_anchor(c, r, flip):
    return (c, (N - 2 - r) if flip else r)   # (c, 7-r) under flip


def encode_planes(state):
    flip = canonical_flip(state)
    me = state.pawns[state.turn]
    opp = state.pawns[1 - state.turn]
    planes = np.zeros((N_PLANES, N, N), dtype=np.float32)
    mc = _cf_cell(me, flip)
    oc = _cf_cell(opp, flip)
    planes[0, mc[1], mc[0]] = 1.0          # [plane, row, col]
    planes[1, oc[1], oc[0]] = 1.0
    for (c, r) in state.h_walls:
        cc, cr = _cf_wall_anchor(c, r, flip)
        planes[2, cr, cc] = 1.0
    for (c, r) in state.v_walls:
        cc, cr = _cf_wall_anchor(c, r, flip)
        planes[3, cr, cc] = 1.0
    planes[4, :, :] = state.walls_left[state.turn] / 10.0
    planes[5, :, :] = state.walls_left[1 - state.turn] / 10.0
    return planes


def move_to_action(move, state):
    flip = canonical_flip(state)
    if isinstance(move, Step):
        me = _cf_cell(state.pawns[state.turn], flip)
        dest = _cf_cell(move.to_cell, flip)
        d = (dest[0] - me[0], dest[1] - me[1])
        return _DIR_INDEX[d]
    # Wall
    cc, cr = _cf_wall_anchor(move.c, move.r, flip)
    off = 0 if move.orient == "H" else 64
    return 12 + off + cr * 8 + cc


def action_to_move(idx, state):
    """Inverse of move_to_action, returning the REAL move."""
    flip = canonical_flip(state)
    if idx < 12:
        dx, dy = _DIRS[idx]
        me = _cf_cell(state.pawns[state.turn], flip)
        cdest = (me[0] + dx, me[1] + dy)
        real_dest = _cf_cell(cdest, flip)   # flip is its own inverse
        return Step(real_dest)
    a = idx - 12
    orient = "H" if a < 64 else "V"
    a %= 64
    cr, cc = divmod(a, 8)
    real_c, real_r = _cf_wall_anchor(cc, cr, flip)   # flip is its own inverse
    return Wall(real_c, real_r, orient)


def legal_action_mask(state):
    mask = np.zeros(N_ACTIONS, dtype=np.float32)
    for m in legal_moves(state):
        mask[move_to_action(m, state)] = 1.0
    return mask
```

- [ ] **Step 4: Run, verify pass.** **Step 5: Commit** `feat: AZ state/action encoding with canonicalization`.

---

## Task 2: Network + wrapper

**Files:** Create `agents/az/model.py`, test `tests/test_az_model.py`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_az_model.py
import torch

from core.state import initial_state
from core.rules import legal_moves
from agents.az.encoding import N_ACTIONS, N_PLANES
from agents.az.model import QuoridorNet, NetWrapper


def test_forward_shapes():
    net = QuoridorNet(channels=16, blocks=2)
    x = torch.zeros(4, N_PLANES, 9, 9)
    logits, value = net(x)
    assert logits.shape == (4, N_ACTIONS)
    assert value.shape == (4, 1)
    assert torch.all(value <= 1) and torch.all(value >= -1)


def test_predict_returns_legal_priors_and_value():
    net = QuoridorNet(channels=16, blocks=2)
    wrap = NetWrapper(net)
    s = initial_state()
    priors, value = wrap.predict(s)
    legal = set(legal_moves(s))
    assert set(priors.keys()) == legal           # priors only over legal moves
    assert abs(sum(priors.values()) - 1.0) < 1e-4  # normalized
    assert all(p >= 0 for p in priors.values())
    assert -1.0 <= value <= 1.0


def test_predict_is_deterministic_in_eval():
    net = QuoridorNet(channels=16, blocks=2)
    wrap = NetWrapper(net)
    s = initial_state()
    p1, v1 = wrap.predict(s)
    p2, v2 = wrap.predict(s)
    assert v1 == v2 and p1 == p2
```

- [ ] **Step 2: Run, verify fail.**

- [ ] **Step 3: Implement `agents/az/model.py`**

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.rules import legal_moves
from agents.az.encoding import N_ACTIONS, N_PLANES, encode_planes, move_to_action


class _ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.b1 = nn.BatchNorm2d(ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.b2 = nn.BatchNorm2d(ch)

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return F.relu(x + y)


class QuoridorNet(nn.Module):
    def __init__(self, channels=32, blocks=3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(N_PLANES, channels, 3, padding=1),
            nn.BatchNorm2d(channels), nn.ReLU())
        self.body = nn.Sequential(*[_ResBlock(channels) for _ in range(blocks)])
        self.p_conv = nn.Sequential(nn.Conv2d(channels, 2, 1),
                                    nn.BatchNorm2d(2), nn.ReLU())
        self.p_fc = nn.Linear(2 * 9 * 9, N_ACTIONS)
        self.v_conv = nn.Sequential(nn.Conv2d(channels, 1, 1),
                                    nn.BatchNorm2d(1), nn.ReLU())
        self.v_fc1 = nn.Linear(9 * 9, 64)
        self.v_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.body(self.stem(x))
        p = self.p_fc(self.p_conv(x).flatten(1))
        v = self.v_conv(x).flatten(1)
        v = torch.tanh(self.v_fc2(F.relu(self.v_fc1(v))))
        return p, v


class NetWrapper:
    """Holds a net; predicts (priors over legal real moves, value) for a state."""

    def __init__(self, net, device="cpu"):
        self.net = net.to(device)
        self.device = device

    def predict(self, state):
        self.net.eval()
        planes = encode_planes(state)
        x = torch.from_numpy(planes).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, value = self.net(x)
        logits = logits[0].cpu().numpy()
        legal = legal_moves(state)
        idxs = np.array([move_to_action(m, state) for m in legal])
        sel = logits[idxs]
        sel = sel - sel.max()
        exp = np.exp(sel)
        probs = exp / exp.sum()
        priors = {m: float(p) for m, p in zip(legal, probs)}
        return priors, float(value.item())
```

- [ ] **Step 4: Run, verify pass.** **Step 5: Commit** `feat: QuoridorNet policy+value model and predict wrapper`.

---

## Task 3: PUCT MCTS using the net

**Files:** Create `agents/az/mcts_nn.py`, test `tests/test_az_mcts.py`.

Design: standard AlphaZero PUCT. Each node expanded once by a net call that sets children priors and returns the node's value (side-to-move perspective; converted to root perspective for backprop). `U(a) = c_puct * P(a) * sqrt(N_parent) / (1 + N(a))`. Robust child = max visits.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_az_mcts.py
from core.state import GameState, Step, initial_state
from core.rules import legal_moves
from agents.az.model import QuoridorNet, NetWrapper
from agents.az.mcts_nn import PUCTSearch


def _wrap():
    return NetWrapper(QuoridorNet(channels=16, blocks=2))


def test_returns_legal_move_and_policy():
    search = PUCTSearch(_wrap(), sims=40, seed=0)
    s = initial_state()
    move, pi, info = search.run(s)
    assert move in legal_moves(s)
    assert abs(sum(pi.values()) - 1.0) < 1e-6      # visit-count policy normalized
    assert set(pi.keys()) <= set(legal_moves(s))
    assert info["sims"] >= 1


def test_finds_immediate_win_with_enough_sims():
    # zero walls, opp one step away: stepping to (4,8) is the only win.
    s = GameState(((4, 7), (4, 1)), frozenset(), frozenset(), (0, 10), 0)
    search = PUCTSearch(_wrap(), sims=200, seed=0)
    move, pi, info = search.run(s)
    assert isinstance(move, Step) and move.to_cell == (4, 8)
```

NOTE on `test_finds_immediate_win_with_enough_sims`: with an untrained net the priors are ~uniform, but the SEARCH still discovers the win because the terminal child returns a true +1 value and PUCT will re-select it. Branching here is small (≤4 steps, 0 walls). If this is flaky, raise sims; do not weaken.

- [ ] **Step 2: Run, verify fail.**

- [ ] **Step 3: Implement `agents/az/mcts_nn.py`**

```python
import math
import random

from core.rules import apply_move, is_terminal, winner
from agents.az.encoding import move_to_action


class _Node:
    __slots__ = ("state", "parent", "move", "prior", "children", "N", "W", "expanded")

    def __init__(self, state, parent=None, move=None, prior=0.0):
        self.state = state
        self.parent = parent
        self.move = move
        self.prior = prior
        self.children = []
        self.N = 0
        self.W = 0.0          # from ROOT player's perspective
        self.expanded = False


class PUCTSearch:
    def __init__(self, net_wrapper, sims=160, c_puct=1.5, seed=None,
                 dirichlet_alpha=None, dirichlet_eps=0.25):
        self.net = net_wrapper
        self.sims = sims
        self.c_puct = c_puct
        self._rng = random.Random(seed)
        self.dirichlet_alpha = dirichlet_alpha     # set for self-play root noise
        self.dirichlet_eps = dirichlet_eps

    def _expand(self, node, root_player):
        priors, value = self.net.predict(node.state)
        for m, p in priors.items():
            node.children.append(
                _Node(apply_move(node.state, m), node, m, p))
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
        # expand root, optionally add Dirichlet noise to root priors
        self._expand(root, root_player)
        root.N = 1
        if self.dirichlet_alpha and root.children:
            noise = self._rng_dirichlet(len(root.children))
            for ch, nz in zip(root.children, noise):
                ch.prior = (1 - self.dirichlet_eps) * ch.prior + self.dirichlet_eps * nz
        for _ in range(self.sims):
            node = root
            while node.expanded and not is_terminal(node.state):
                node = self._select(node, root_player)
            if is_terminal(node.state):
                w = winner(node.state)
                v = 1.0 if w == root_player else -1.0
            else:
                v = self._expand(node, root_player)
            while node is not None:
                node.N += 1
                node.W += v
                node = node.parent
        # visit-count policy + robust child
        total = sum(ch.N for ch in root.children)
        pi = {ch.move: ch.N / total for ch in root.children} if total else {}
        top = max(ch.N for ch in root.children)
        best = self._rng.choice([ch for ch in root.children if ch.N == top])
        info = {"sims": self.sims, "value": root.W / root.N if root.N else 0.0,
                "visits": {ch.move: ch.N for ch in root.children}}
        return best.move, pi, info

    def _rng_dirichlet(self, k):
        # simple Dirichlet sample via gammas (stdlib random has gammavariate)
        gs = [self._rng.gammavariate(self.dirichlet_alpha, 1.0) for _ in range(k)]
        tot = sum(gs) or 1.0
        return [g / tot for g in gs]
```

- [ ] **Step 4: Run, verify pass.** **Step 5: Commit** `feat: PUCT MCTS guided by the net`.

---

## Task 4: Self-play + training

**Files:** Create `agents/az/selfplay.py`, `agents/az/train.py`, test `tests/test_az_train.py`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_az_train.py
import numpy as np
import torch

from agents.az.model import QuoridorNet, NetWrapper
from agents.az.selfplay import play_selfplay_game
from agents.az.train import train_step, examples_to_batch
from agents.az.encoding import N_ACTIONS, N_PLANES


def test_selfplay_produces_examples():
    wrap = NetWrapper(QuoridorNet(channels=16, blocks=1))
    ex = play_selfplay_game(wrap, sims=10, temp_moves=4, seed=0, max_plies=60)
    assert len(ex) > 0
    planes, pi, z = ex[0]
    assert planes.shape == (N_PLANES, 9, 9)
    assert pi.shape == (N_ACTIONS,) and abs(pi.sum() - 1.0) < 1e-5
    assert z in (-1.0, 0.0, 1.0)


def test_train_step_overfits_tiny_batch():
    # Net should be able to drive loss DOWN on a fixed tiny batch (learning works).
    torch.manual_seed(0)
    net = QuoridorNet(channels=16, blocks=1)
    wrap = NetWrapper(net)
    ex = play_selfplay_game(wrap, sims=8, temp_moves=2, seed=1, max_plies=40)[:8]
    batch = examples_to_batch(ex)
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    first = train_step(net, opt, batch)
    for _ in range(30):
        last = train_step(net, opt, batch)
    assert last < first * 0.8      # loss dropped meaningfully
    assert np.isfinite(last)
```

- [ ] **Step 2: Run, verify fail.**

- [ ] **Step 3: Implement `agents/az/selfplay.py`**

```python
import random

import numpy as np

from core.rules import is_terminal, winner, apply_move
from core.state import initial_state
from agents.az.encoding import N_ACTIONS, encode_planes, move_to_action
from agents.az.mcts_nn import PUCTSearch


def play_selfplay_game(net_wrapper, sims=80, temp_moves=10, seed=None,
                       max_plies=200, dirichlet_alpha=0.5):
    rng = random.Random(seed)
    state = initial_state()
    history = []     # (planes, pi_vector, player_to_move)
    ply = 0
    while not is_terminal(state) and ply < max_plies:
        search = PUCTSearch(net_wrapper, sims=sims, seed=rng.randrange(1 << 30),
                            dirichlet_alpha=dirichlet_alpha)
        _, pi, _ = search.run(state)
        pi_vec = np.zeros(N_ACTIONS, dtype=np.float32)
        for m, p in pi.items():
            pi_vec[move_to_action(m, state)] = p
        history.append((encode_planes(state), pi_vec, state.turn))
        # sample a move from pi (temperature 1 early, then greedy)
        moves = list(pi.keys())
        probs = np.array([pi[m] for m in moves])
        if ply < temp_moves:
            choice = rng.choices(moves, weights=probs)[0]
        else:
            choice = moves[int(np.argmax(probs))]
        state = apply_move(state, choice)
        ply += 1
    w = winner(state)              # None if capped
    examples = []
    for planes, pi_vec, player in history:
        if w is None:
            z = 0.0
        else:
            z = 1.0 if w == player else -1.0
        examples.append((planes, pi_vec, z))
    return examples
```

- [ ] **Step 4: Implement `agents/az/train.py`**

```python
import numpy as np
import torch
import torch.nn.functional as F


def examples_to_batch(examples, device="cpu"):
    planes = torch.from_numpy(np.stack([e[0] for e in examples])).to(device)
    pi = torch.from_numpy(np.stack([e[1] for e in examples])).to(device)
    z = torch.tensor([e[2] for e in examples], dtype=torch.float32,
                     device=device).unsqueeze(1)
    return planes, pi, z


def train_step(net, optimizer, batch):
    net.train()
    planes, target_pi, target_z = batch
    logits, value = net(planes)
    logp = F.log_softmax(logits, dim=1)
    policy_loss = -(target_pi * logp).sum(dim=1).mean()
    value_loss = F.mse_loss(value, target_z)
    loss = policy_loss + value_loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss.item())


def run_training(net, iterations=3, games_per_iter=4, sims=60, epochs=4,
                 lr=1e-3, seed=0, log=print):
    """Self-play + train loop. Returns list of per-iteration mean losses."""
    import random
    from agents.az.model import NetWrapper
    from agents.az.selfplay import play_selfplay_game
    rng = random.Random(seed)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    wrap = NetWrapper(net)
    history = []
    for it in range(iterations):
        examples = []
        for g in range(games_per_iter):
            examples += play_selfplay_game(wrap, sims=sims, seed=rng.randrange(1 << 30))
        batch = examples_to_batch(examples)
        losses = [train_step(net, opt, batch) for _ in range(epochs)]
        history.append(sum(losses) / len(losses))
        log(f"iter {it+1}/{iterations}: examples={len(examples)} "
            f"loss={history[-1]:.4f}")
    return history


def save_checkpoint(net, path):
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(net.state_dict(), path)


def load_checkpoint(net, path):
    net.load_state_dict(torch.load(path, map_location="cpu"))
    return net
```

- [ ] **Step 5: Run, verify pass.** `test_train_step_overfits_tiny_batch` proves learning works. **Step 6: Commit** `feat: AZ self-play and training loop`.

---

## Task 5: AZAgent + registry + CLI

**Files:** Create `agents/az/agent.py`, `scripts/train_az.py`, MODIFY `agents/registry.py` and `pyproject.toml` (+`.gitignore`), test (extend `tests/test_az_mcts.py` or new `tests/test_az_agent.py`).

- [ ] **Step 1: Write the failing test** (`tests/test_az_agent.py`)

```python
from core.state import initial_state
from core.rules import legal_moves, apply_move, is_terminal
from agents.registry import make_agent, available_agents
from agents.az.agent import AZAgent


def test_az_registered():
    assert "az" in available_agents()


def test_az_plays_legal_full_game():
    # small net + low sims keeps this fast; untrained is fine, must be legal
    a = AZAgent(sims=16, channels=16, blocks=1, seed=0)
    b = AZAgent(sims=16, channels=16, blocks=1, seed=1)
    s = initial_state()
    for _ in range(300):
        if is_terminal(s):
            break
        agent = a if s.turn == 0 else b
        mv = agent.select_move(s)
        assert mv in legal_moves(s)
        s = apply_move(s, mv)


def test_az_analyze_populated():
    a = AZAgent(sims=16, channels=16, blocks=1, seed=0)
    info = a.analyze(initial_state())
    assert info.best_move in legal_moves(initial_state())
    assert -1.0 <= info.value <= 1.0
    assert len(info.candidates) > 0
    assert info.stats["sims"] >= 1
```

- [ ] **Step 2: Run, verify fail.**

- [ ] **Step 3: Implement `agents/az/agent.py`**

```python
import os

from agents.base import Agent, Analysis
from agents.az.model import QuoridorNet, NetWrapper
from agents.az.mcts_nn import PUCTSearch

DEFAULT_CKPT = os.path.join(os.path.dirname(__file__), "..", "..",
                            "models", "az_smoke.pt")


class AZAgent(Agent):
    name = "az"

    def __init__(self, checkpoint=None, sims=120, c_puct=1.5,
                 channels=32, blocks=3, seed=None):
        net = QuoridorNet(channels=channels, blocks=blocks)
        path = checkpoint or DEFAULT_CKPT
        if os.path.exists(path):
            import torch
            try:
                net.load_state_dict(torch.load(path, map_location="cpu"))
            except Exception:
                pass    # shape mismatch / corrupt -> use fresh net
        self._wrap = NetWrapper(net)
        self._sims = sims
        self._c_puct = c_puct
        self._seed = seed

    def analyze(self, state):
        search = PUCTSearch(self._wrap, sims=self._sims, c_puct=self._c_puct,
                            seed=self._seed)
        move, _, info = search.run(state)
        cands = sorted(info["visits"].items(), key=lambda kv: kv[1], reverse=True)
        return Analysis(best_move=move, value=info["value"],
                        candidates=[(m, float(n)) for m, n in cands[:8]],
                        stats={"sims": info["sims"]})

    def select_move(self, state):
        return self.analyze(state).best_move
```
(NOTE: if a default checkpoint exists but was trained with different `channels/blocks`, the load raises and we fall back to a fresh net. The registry default below uses `channels=32, blocks=3`, matching the CLI smoke-train.)

- [ ] **Step 4: Modify `agents/registry.py`** — add import + `"az": AZAgent`. Modify `pyproject.toml` to add `torch`, `numpy` to dependencies. Add `models/` to `.gitignore`.

- [ ] **Step 5: Implement `scripts/train_az.py`**

```python
"""Smoke-train an AZ net and save a checkpoint. Usage:
    python scripts/train_az.py --iterations 3 --games 4 --sims 60
"""
import argparse

from agents.az.model import QuoridorNet
from agents.az.train import run_training, save_checkpoint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=3)
    ap.add_argument("--games", type=int, default=4)
    ap.add_argument("--sims", type=int, default=60)
    ap.add_argument("--channels", type=int, default=32)
    ap.add_argument("--blocks", type=int, default=3)
    ap.add_argument("--out", default="models/az_smoke.pt")
    args = ap.parse_args()
    net = QuoridorNet(channels=args.channels, blocks=args.blocks)
    hist = run_training(net, iterations=args.iterations,
                        games_per_iter=args.games, sims=args.sims)
    save_checkpoint(net, args.out)
    print(f"saved {args.out}; loss history: {hist}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run, verify pass.** Run full suite `pytest -q`. **Step 7: Commit** `feat: AZAgent, registry entry, and training CLI`.

---

## Task 6: Smoke-train run (proof of concept) — controller-run, not a unit test

After Tasks 1-5 pass and are merged, the CONTROLLER (not a subagent) runs a short real training and documents it:

- [ ] Run `python scripts/train_az.py --iterations 3 --games 4 --sims 60` and capture the loss history (should be finite and generally decreasing).
- [ ] Confirm the checkpoint saved to `models/az_smoke.pt`.
- [ ] Run an informational arena: `az` (with checkpoint) vs `random`, ~6 games. Report the score. **Do not assert a win** — a CPU smoke-train likely won't beat random reliably; the point is the pipeline runs end to end.
- [ ] Document the result and the scaling note (real strength needs orders of magnitude more self-play/compute, ideally GPU) in the final summary.

---

## Done criteria
- `pytest -q` green (encoding round-trips, model shapes, PUCT legal + finds forced win, self-play example shapes, training overfits a tiny batch, AZAgent plays legal full games).
- `az` registered and appears in `/agents` and the web UI (untrained net plays legal but weak).
- `scripts/train_az.py` runs the loop and saves a checkpoint; loss decreases.
- Only `agents/registry.py`, `pyproject.toml`, `.gitignore` modified among pre-existing files.

## Honest scope note
This is the AlphaZero *machinery*, correct and runnable, not a strong engine. A meaningful policy needs large-scale self-play (thousands+ games, hundreds of sims/move, many training iterations) — impractical on CPU in pure-Python-driven self-play. The value here is the working, inspectable learning loop you can scale or port to GPU.
