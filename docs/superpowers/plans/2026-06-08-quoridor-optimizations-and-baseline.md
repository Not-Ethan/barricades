# Quoridor-Specific Optimizations + Strength Baseline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Speed up self-play (floating-wall legality fast-path), double training data (L-R symmetry augmentation), add exact endgame values (`(0,0)` race solver), and add a real strength baseline (minimax ladder) — preparation only, no training campaign launched.

**Architecture:** Three optimizations to the existing `barricades_native` Rust core + Python training, each gated by the existing correctness oracles (differential vs Python `core`, commutation vs the encoder), plus a parallel eval harness. The endgame solver is one Rust component used in three places (MCTS leaf, self-play truncation, inference).

**Tech Stack:** Rust + PyO3 0.28 (`native/src/`), PyTorch on MPS, Python 3.14. Branch `az-bootstrap`.

**Spec:** `docs/superpowers/specs/2026-06-08-quoridor-optimizations-and-baseline-design.md`

---

## Conventions

**Build + test** (from repo root `/Users/Ethan_1/barricades`). Rust changes need a rebuild:
```bash
source .venv/bin/activate && maturin develop -m native/Cargo.toml -q && python -m pytest <file> -q
```
Python-only tasks skip `maturin develop`. The differential oracle: the Python `core` (`core.rules`, `core.bitboard`) is ground truth for Rust game logic. State/move tuples: `((c0,r0),(c1,r1), h_list, v_list, (n0,n1), turn)`; `("step",c,r)` / `("wall",c,r,"H"/"V")`.

**Current state:** suite ~238 green. `barricades_native` exposes game primitives, `encode_planes`/`move_to_action`/`action_to_move` (140 actions: 0–11 steps, 12–75 H-walls `12+cr*8+cc`, 76–139 V-walls `76+cr*8+cc`), `Tree` (MCTS, with `advance`/`root_visits`), `SelfPlayPool` (`carryover` default true). `agents/az/train.py` has `form_dense_targets`/`train_minibatched`. `agents/native_agent.py` has `NativeMctsAgent(sims,c_puct,seed,eval_fn)`. `scripts/eval_az.py` is the parallel match harness pattern.

---

## Task 1 (Unit 1): Floating-wall legality fast-path

Skip the path-existence BFS for candidate walls that touch the barrier (existing walls ∪ board boundary) at <2 of their 3 contact posts — such walls are always routable-around, hence legal. Gate on the differential oracle, with the fuzz extended to wall-dense positions.

**Files:**
- Modify: `native/src/movegen.rs`
- Test: `tests/test_native_game.py` (extend the differential fuzz)

- [ ] **Step 1: Extend the differential test for wall-dense positions** — append to `tests/test_native_game.py`:

```python
def test_legal_walls_match_in_wall_dense_positions():
    # The fast-path for wall legality must match the Python full-BFS reference
    # exactly, ESPECIALLY where walls are dense and blocking is actually possible.
    # Bias the random playout toward wall moves so positions accumulate walls.
    import random
    from core.state import Step, Wall
    rng = random.Random(99)
    checked = 0
    dense = 0
    for _ in range(120):
        s = initial_state()
        for _ in range(60):
            if rules.is_terminal(s):
                break
            ns = to_native(s)
            assert set(bn.legal_moves(ns)) == {mv_to_tuple(m) for m in rules.legal_moves(s)}
            placed = len(s.h_walls) + len(s.v_walls)
            if placed >= 6:
                dense += 1
            checked += 1
            moves = rules.legal_moves(s)
            walls = [m for m in moves if isinstance(m, Wall)]
            # 75% wall moves when available -> drive toward dense, blocking-prone states
            pick = rng.choice(walls) if (walls and rng.random() < 0.75) else rng.choice(moves)
            s = rules.apply_move(s, pick)
    assert checked > 2000
    assert dense > 300   # we actually exercised many wall-dense positions
```

- [ ] **Step 2: Run it against the current (correct, slow) `legal_walls`** to confirm the test passes BEFORE optimizing (it should — current code is the reference behavior):
```bash
source .venv/bin/activate && python -m pytest tests/test_native_game.py::test_legal_walls_match_in_wall_dense_positions -q
```
Expected: PASS (the un-optimized `legal_walls` already matches Python). This locks in the behavior the optimization must preserve.

- [ ] **Step 3: Add the contact-post predicate + fast-path to `native/src/movegen.rs`**

Add this helper and rewrite `legal_walls` to use it. Contact-post model: posts are at integer `(px,py)`, `px,py ∈ 0..=9`. A **horizontal** wall anchor `(c,r)` has contact posts `(c, r+1), (c+1, r+1), (c+2, r+1)`. A **vertical** wall anchor `(c,r)` has contact posts `(c+1, r), (c+1, r+1), (c+1, r+2)`. A post is *anchored* if it lies on the board boundary (`px==0||px==9||py==0||py==9`) or coincides with a contact post of an existing wall. A candidate needs the BFS check iff ≥2 of its 3 posts are anchored; otherwise it's trivially legal (≤1 anchor ⇒ a length-2 wall can't complete a cut ⇒ always routable-around).

```rust
// post index in 0..100 for a 10x10 post grid
#[inline]
fn post_idx(px: i32, py: i32) -> u32 {
    (px * 10 + py) as u32
}

/// Bitset (u128, 100 posts) of all contact posts occupied by existing walls.
fn occupied_posts(s: &GameState) -> u128 {
    let mut bits = 0u128;
    let mut hm = s.h_mask;
    while hm != 0 {
        let i = hm.trailing_zeros() as i32;
        hm &= hm - 1;
        let (c, r) = (i % 8, i / 8);
        for px in [c, c + 1, c + 2] {
            bits |= 1u128 << post_idx(px, r + 1);
        }
    }
    let mut vm = s.v_mask;
    while vm != 0 {
        let i = vm.trailing_zeros() as i32;
        vm &= vm - 1;
        let (c, r) = (i % 8, i / 8);
        for py in [r, r + 1, r + 2] {
            bits |= 1u128 << post_idx(c + 1, py);
        }
    }
    bits
}

#[inline]
fn is_boundary_post(px: i32, py: i32) -> bool {
    px == 0 || px == 9 || py == 0 || py == 9
}

/// The three contact posts of a candidate wall (orient 0=H, 1=V).
fn wall_posts(c: i32, r: i32, orient: u8) -> [(i32, i32); 3] {
    if orient == 0 {
        [(c, r + 1), (c + 1, r + 1), (c + 2, r + 1)]
    } else {
        [(c + 1, r), (c + 1, r + 1), (c + 1, r + 2)]
    }
}

/// True if this candidate could possibly complete a cut (needs the path BFS).
/// Conservative: only returns false (skip BFS) when the wall touches the barrier
/// graph at <2 of its 3 posts, which is provably always-legal for a length-2 wall.
fn needs_path_check(occupied: u128, c: i32, r: i32, orient: u8) -> bool {
    let mut anchored = 0;
    for (px, py) in wall_posts(c, r, orient) {
        if is_boundary_post(px, py) || (occupied >> post_idx(px, py)) & 1 != 0 {
            anchored += 1;
        }
    }
    anchored >= 2
}
```

Rewrite `legal_walls` (keep `overlaps`/`with_wall` unchanged):
```rust
pub fn legal_walls(s: &GameState) -> Vec<(i32, i32, u8)> {
    if s.walls_left[s.turn as usize] == 0 {
        return Vec::new();
    }
    let occupied = occupied_posts(s);
    let mut res = Vec::new();
    for orient in [0u8, 1u8] {
        for c in 0..8 {
            for r in 0..8 {
                if overlaps(s, c, r, orient) {
                    continue;
                }
                if needs_path_check(occupied, c, r, orient) {
                    let s2 = with_wall(s, c, r, orient);
                    if path_exists(&s2, 0) && path_exists(&s2, 1) {
                        res.push((c, r, orient));
                    }
                } else {
                    res.push((c, r, orient)); // <2 anchored posts -> always legal
                }
            }
        }
    }
    res
}
```

- [ ] **Step 4: Build and run the full differential suite** (the oracle):
```bash
source .venv/bin/activate && maturin develop -m native/Cargo.toml -q && python -m pytest tests/test_native_game.py -q
```
Expected: PASS (both the open-board test and BOTH differential tests, including the new wall-dense one — proving the fast-path matches the exact Python BFS over thousands of wall-dense positions). If any mismatch: the dumped position shows where `needs_path_check` wrongly skipped — fix the post geometry (it must never skip a wall that the BFS would reject).

- [ ] **Step 5: Micro-benchmark the speedup** (informational):
```bash
source .venv/bin/activate && python -c "
import time, random, barricades_native as bn
from core.state import initial_state
from core import rules
from tests.test_native_game import to_native
rng=random.Random(1); s=initial_state()
# build a wall-dense midgame position
for _ in range(40):
    if rules.is_terminal(s): break
    from core.state import Wall
    ms=rules.legal_moves(s); ws=[m for m in ms if isinstance(m,Wall)]
    s=rules.apply_move(s, rng.choice(ws) if ws and rng.random()<0.7 else rng.choice(ms))
ns=to_native(s)
t0=time.perf_counter()
for _ in range(2000): bn.legal_moves(ns)
print(f'legal_moves on a {len(s.h_walls)+len(s.v_walls)}-wall position: {2000/(time.perf_counter()-t0):.0f}/s')
"
```
Report the rate (compare to your sense of the prior ~hundreds/s). No assertion — just record the win.

- [ ] **Step 6: Full suite + commit**
```bash
source .venv/bin/activate && python -m pytest -q
git add native/src/movegen.rs tests/test_native_game.py
git commit -m "perf(native): floating-wall legality fast-path (skip BFS when <2 contact posts anchored)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```
Expected: full suite green.

---

## Task 2 (Unit 2): Left-right symmetry data augmentation

Add `augment_lr` (Python) doubling training data via the L-R mirror, with a fixed action permutation. **Critical subtlety:** the plane mirror differs for pawn planes (cols 0–8, `c→8-c`) vs wall planes (anchors live in cols 0–7, `cc→7-cc`) vs the constant walls-left planes (unchanged). A naive `np.flip` of all columns is a silent bug. The commutation test catches it.

**Files:**
- Modify: `agents/az/train.py` (add `LR_PERM`, `mirror_planes`, `augment_lr`)
- Modify: `scripts/campaign.py` (apply `augment_lr` before `form_dense_targets`)
- Test: `tests/test_lr_augment.py` (create)

- [ ] **Step 1: Write the failing commutation test** `tests/test_lr_augment.py`:

```python
import random
import numpy as np
from core.state import GameState, Step, Wall, initial_state
from core import rules
from agents.az import encoding as enc
from agents.az.train import mirror_planes, LR_PERM, augment_lr


def lr_mirror_state(s):
    # pawn col c->8-c ; wall anchor c->7-c (both H and V); rows/walls/turn unchanged
    pawns = tuple((8 - c, r) for (c, r) in s.pawns)
    h = frozenset((7 - c, r) for (c, r) in s.h_walls)
    v = frozenset((7 - c, r) for (c, r) in s.v_walls)
    return GameState(pawns, h, v, s.walls_left, s.turn)


def lr_mirror_move(m):
    if isinstance(m, Step):
        return Step((8 - m.to_cell[0], m.to_cell[1]))
    return Wall(7 - m.c, m.r, m.orient)


def test_perm_is_involution():
    assert np.array_equal(LR_PERM[LR_PERM], np.arange(140))


def test_planes_mirror_commutes_with_encoding():
    rng = random.Random(5)
    checked = 0
    for _ in range(40):
        s = initial_state()
        for _ in range(60):
            if rules.is_terminal(s):
                break
            got = mirror_planes(enc.encode_planes(s))
            want = enc.encode_planes(lr_mirror_state(s))
            assert np.array_equal(got, want), f"plane mirror mismatch at {s}"
            for m in rules.legal_moves(s):
                a = enc.move_to_action(m, s)
                a_mir = enc.move_to_action(lr_mirror_move(m), lr_mirror_state(s))
                assert LR_PERM[a] == a_mir
            checked += 1
            s = rules.apply_move(s, rng.choice(rules.legal_moves(s)))
    assert checked > 1500


def test_augment_lr_doubles_and_preserves_z_feats():
    s = initial_state()
    planes = enc.encode_planes(s)
    pi = np.zeros(140, dtype=np.float32); pi[enc.move_to_action(Step((4, 1)), s)] = 1.0
    ex = [(planes, pi, 1.0, np.array([3.0, 5, 5, 7], dtype=np.float32))]
    out = augment_lr(ex)
    assert len(out) == 2
    # the mirror's pi mass is at the permuted action; z and feats unchanged
    orig_a = int(np.argmax(out[0][1])); mir_a = int(np.argmax(out[1][1]))
    assert LR_PERM[orig_a] == mir_a
    assert out[1][2] == 1.0
    assert np.array_equal(np.asarray(out[1][3]), np.asarray(out[0][3]))
```

- [ ] **Step 2: Run it, confirm it fails**
```bash
source .venv/bin/activate && python -m pytest tests/test_lr_augment.py -q
```
Expected: FAIL (`mirror_planes`/`LR_PERM`/`augment_lr` don't exist).

- [ ] **Step 3: Add to `agents/az/train.py`** (after the existing functions; `numpy as np` already imported):

```python
def _build_lr_perm():
    """Fixed length-140 L-R action permutation. Steps: dx->-dx. Walls: cc->7-cc."""
    perm = np.empty(140, dtype=np.int64)
    step_map = {0: 0, 1: 1, 2: 3, 3: 2, 4: 4, 5: 5, 6: 7, 7: 6, 8: 9, 9: 8, 10: 11, 11: 10}
    for i in range(12):
        perm[i] = step_map[i]
    for off in (12, 76):                 # 12..75 = H walls, 76..139 = V walls
        for cr in range(8):
            for cc in range(8):
                perm[off + cr * 8 + cc] = off + cr * 8 + (7 - cc)
    return perm


LR_PERM = _build_lr_perm()


def mirror_planes(planes):
    """Left-right mirror of the (6,9,9) [plane,row,col] encoding. Pawn planes flip
    cols 0..8 (c->8-c); wall planes flip only cols 0..7 (anchors; cc->7-cc, col 8
    stays 0); walls-left planes (constant) are unchanged."""
    m = np.array(planes, dtype=np.float32)          # copy
    m[0] = planes[0][:, ::-1]                        # me pawn
    m[1] = planes[1][:, ::-1]                        # opp pawn
    m[2] = 0.0; m[2][:, 0:8] = planes[2][:, 0:8][:, ::-1]   # H walls (cols 0..7)
    m[3] = 0.0; m[3][:, 0:8] = planes[3][:, 0:8][:, ::-1]   # V walls (cols 0..7)
    m[4] = planes[4]                                 # walls_left (constant plane)
    m[5] = planes[5]
    return m


def augment_lr(examples):
    """Return examples + their L-R mirrors (planes mirrored, pi permuted, z/feats
    unchanged). Doubles training data via the board's left-right symmetry."""
    out = list(examples)
    for planes, pi, z, feats in examples:
        out.append((mirror_planes(planes),
                    np.asarray(pi, dtype=np.float32)[LR_PERM],
                    z, feats))
    return out
```

- [ ] **Step 4: Run the test, confirm it passes**
```bash
source .venv/bin/activate && python -m pytest tests/test_lr_augment.py -q
```
Expected: PASS (`3 passed`). If `test_planes_mirror_commutes_with_encoding` fails on a wall position, the wall-plane vs pawn-plane mirror handling is the bug.

- [ ] **Step 5: Wire `augment_lr` into the campaign** — in `scripts/campaign.py`, import it and apply before forming targets. Change the import line:
```python
from agents.az.train import form_dense_targets, train_minibatched, save_checkpoint, augment_lr
```
And in the loop, change:
```python
        batch = form_dense_targets(examples, lam=lam, device="cpu")
```
to:
```python
        batch = form_dense_targets(augment_lr(examples), lam=lam, device="cpu")
```

- [ ] **Step 6: Confirm the campaign smoke still passes + commit**
```bash
source .venv/bin/activate && python -m pytest tests/test_lr_augment.py tests/test_campaign_smoke.py -q
git add agents/az/train.py scripts/campaign.py tests/test_lr_augment.py
git commit -m "feat(phase2): left-right symmetry data augmentation (free 2x training data)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```
Expected: PASS.

---

## Task 3 (Unit 3a): Endgame race solver core

A depth-bounded exact race negamax for `walls_left==(0,0)`, in a new `native/src/endgame.rs`, exposed to Python as `bn.solve_race(state)`, gated by a differential test against an independent Python negamax.

**Files:**
- Create: `native/src/endgame.rs`
- Modify: `native/src/lib.rs` (add `mod endgame;`), `native/src/pyiface.rs` (expose `solve_race`)
- Test: `tests/test_endgame_solver.py` (create)

- [ ] **Step 1: Write the failing differential test** `tests/test_endgame_solver.py`:

```python
import random
import barricades_native as bn
from core.state import GameState, Step
from core import rules
from tests.test_native_game import to_native, mv_to_tuple

PLY_BOUND = 36


def py_solve_race(s, depth, memo):
    """Independent reference: depth-bounded negamax over pawn moves only.
    Returns value for the side to move: +1 win, -1 loss, 0 draw-at-bound."""
    w = rules.winner(s)
    if w is not None:
        return 1 if w == s.turn else -1
    if depth == 0:
        return 0
    key = (s.pawns, s.turn, depth)
    if key in memo:
        return memo[key]
    best = -1
    for c in rules.legal_steps(s):
        v = -py_solve_race(rules.apply_move(s, Step(c)), depth - 1, memo)
        if v > best:
            best = v
        if best == 1:
            break
    memo[key] = best
    return best


def _random_zero_wall_position(rng):
    # play a game until both players are out of walls (walls_left==(0,0)), non-terminal
    for _ in range(200):
        s = rules.initial_state() if hasattr(rules, "initial_state") else None
        from core.state import initial_state
        s = initial_state()
        for _ in range(120):
            if rules.is_terminal(s):
                break
            if s.walls_left == (0, 0):
                return s
            moves = rules.legal_moves(s)
            from core.state import Wall
            walls = [m for m in moves if isinstance(m, Wall)]
            # spend walls aggressively to reach (0,0)
            s = rules.apply_move(s, rng.choice(walls) if walls else rng.choice(moves))
    return None


def test_solve_race_matches_reference():
    rng = random.Random(7)
    checked = 0
    for _ in range(40):
        s = _random_zero_wall_position(rng)
        if s is None or rules.is_terminal(s):
            continue
        assert s.walls_left == (0, 0)
        val, mv = bn.solve_race(to_native(s))
        ref = py_solve_race(s, PLY_BOUND, {})
        assert val == ref, f"value mismatch {val} vs {ref} at {s}"
        assert tuple(mv) in {mv_to_tuple(m) for m in rules.legal_moves(s)}
        # if it's a win, the returned move must preserve the win (ref says win after it)
        if val == 1:
            after = rules.apply_move(s, Step((mv[1], mv[2])))
            assert -py_solve_race(after, PLY_BOUND - 1, {}) == 1
        checked += 1
    assert checked >= 10
```

- [ ] **Step 2: Run it, confirm it fails**
```bash
source .venv/bin/activate && python -m pytest tests/test_endgame_solver.py -q
```
Expected: FAIL (`bn.solve_race` missing).

- [ ] **Step 3: Create `native/src/endgame.rs`**:

```rust
use std::collections::HashMap;

use crate::movegen::legal_steps;
use crate::state::{apply_move, winner, GameState, Move};

const RACE_PLY_BOUND: u32 = 36; // 4*N: ample for either pawn to reach its goal

fn pawn_moves(s: &GameState) -> Vec<Move> {
    legal_steps(s).into_iter().map(|(c, r)| Move::Step { c, r }).collect()
}

fn negamax(s: &GameState, depth: u32,
           memo: &mut HashMap<((u8, u8), (u8, u8), u8, u32), i32>) -> i32 {
    if let Some(w) = winner(s) {
        return if w == s.turn as usize { 1 } else { -1 };
    }
    if depth == 0 {
        return 0; // draw at the ply bound
    }
    let key = (s.pawns[0], s.pawns[1], s.turn, depth);
    if let Some(&v) = memo.get(&key) {
        return v;
    }
    let mut best = i32::MIN;
    for m in pawn_moves(s) {
        let v = -negamax(&apply_move(s, &m), depth - 1, memo);
        if v > best {
            best = v;
        }
        if best == 1 {
            break; // cannot beat a forced win
        }
    }
    let v = if best == i32::MIN { -1 } else { best }; // no moves => stuck => loss
    memo.insert(key, v);
    v
}

/// Exact value for the side to move in a frozen-wall race (precondition:
/// walls_left == (0,0), not terminal). +1 win / -1 loss / 0 draw-at-bound, plus
/// the optimal move. Depth-bounded + memoized => total and fast.
pub fn solve_race(s: &GameState) -> (i32, Move) {
    let mut memo = HashMap::new();
    let mut best_val = i32::MIN;
    let moves = pawn_moves(s);
    let mut best = moves[0];
    for m in &moves {
        let v = -negamax(&apply_move(s, m), RACE_PLY_BOUND - 1, &mut memo);
        if v > best_val {
            best_val = v;
            best = *m;
        }
        if best_val == 1 {
            break;
        }
    }
    (best_val, best)
}
```

- [ ] **Step 4: Add `mod endgame;` to `native/src/lib.rs`** (alphabetical: after `mod encoding;`... actually after `mod coords;` — keep the existing ordering style, place `mod endgame;` between `mod encoding;` and `mod mcts;`).

- [ ] **Step 5: Expose `solve_race` in `native/src/pyiface.rs`** — add the pyfunction (uses the existing `parse_state` + `Move: IntoPyObject`):
```rust
#[pyfunction]
fn solve_race_py(state: &Bound<'_, PyAny>) -> PyResult<(i32, Move)> {
    Ok(crate::endgame::solve_race(&parse_state(state)?))
}
```
Register it (matching the `#[pyo3(name=...)]` style used for the other functions):
```rust
    m.add_function(wrap_pyfunction!(solve_race_py, m)?)?;
    m.add("solve_race", m.getattr("solve_race_py")?)?;
```
(Or add `#[pyo3(name = "solve_race")]` to the function and just `m.add_function(...)` — match whatever the file currently does.)

- [ ] **Step 6: Build + run the differential test**
```bash
source .venv/bin/activate && maturin develop -m native/Cargo.toml -q && python -m pytest tests/test_endgame_solver.py -q
```
Expected: PASS (`1 passed`, with `checked >= 10` real `(0,0)` positions). If a value mismatch: the Rust negamax and Python reference must use the SAME ply bound (36) and the same terminal sign convention.

- [ ] **Step 7: Commit**
```bash
git add native/src/endgame.rs native/src/lib.rs native/src/pyiface.rs tests/test_endgame_solver.py
git commit -m "feat(native): exact endgame race solver for walls_left==(0,0)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 4 (Unit 3b): Integrate the solver (MCTS leaf · self-play truncation · inference)

Wire `solve_race` into the three use sites. MCTS-leaf and agent-inference are unconditional (exact = always correct); self-play truncation is behind `Config.endgame_solve` (it changes training data) and counts hit-rate.

**Files:**
- Modify: `native/src/mcts.rs` (endgame leaf eval in `prepare_leaf`)
- Modify: `native/src/selfplay.rs` (`Config.endgame_solve`, truncation in `commit_move`, `forced_outcome` in `finalize`, hit-rate counter + accessor)
- Modify: `native/src/pyiface.rs` (`SelfPlayPool` ctor gains `endgame_solve`; add `games_solved()` accessor)
- Modify: `agents/native_agent.py` (inference: play the solver move at `(0,0)`)
- Test: `tests/test_endgame_integration.py` (create)

- [ ] **Step 1: Write the failing integration test** `tests/test_endgame_integration.py`:

```python
import numpy as np
import barricades_native as bn
from core.state import GameState, Step
from core import rules
from tests.test_native_game import to_native, mv_to_tuple


def _state(p0, p1, wl, turn=0, h=(), v=()):
    return GameState((p0, p1), frozenset(h), frozenset(v), wl, turn)


def test_native_agent_plays_solver_move_at_zero_walls():
    from agents.native_agent import NativeMctsAgent
    # both out of walls, p0 to move, clearly winning race -> agent must play the
    # solver's optimal (winning) move, not a wandering MCTS move.
    s = _state((4, 6), (0, 1), (0, 0), turn=0)
    val, mv = bn.solve_race(to_native(s))
    agent = NativeMctsAgent(sims=50, seed=0)   # heuristic mode is fine
    chosen = agent.select_move(s)
    from core.state import Step as St
    assert (chosen.to_cell if isinstance(chosen, St) else None) == (mv[1], mv[2])


def test_carryover_pool_with_endgame_solve_smoke():
    # A pool with endgame_solve on still produces well-formed examples and drains.
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
    assert pool.games_solved() >= 0   # accessor exists; >=0 always
```

- [ ] **Step 2: Run it, confirm it fails**
```bash
source .venv/bin/activate && python -m pytest tests/test_endgame_integration.py -q
```
Expected: FAIL (`SelfPlayPool` has no `endgame_solve`/`games_solved`; agent doesn't use the solver).

- [ ] **Step 3: MCTS leaf endgame eval — `native/src/mcts.rs`**

In `prepare_leaf`, immediately AFTER the `is_terminal` block and BEFORE `encode_planes`/parking, add the endgame check:
```rust
        if self.nodes[node].state.walls_left == [0, 0] {
            let st = self.nodes[node].state;
            let (val_mover, _) = crate::endgame::solve_race(&st);
            let v = if st.turn as usize == self.root_player {
                val_mover as f64
            } else {
                -(val_mover as f64)
            };
            self.backup(node, v);
            return Leaf::Terminal; // exact value, no net eval needed
        }
```
(Do the same insertion in `run_heuristic`'s descent loop right after its terminal-backup block, so the net-free heuristic agent also uses exact endgame values.)

- [ ] **Step 4: Self-play truncation + hit-rate — `native/src/selfplay.rs`**

Add `pub endgame_solve: bool,` to `Config`. Add `forced_outcome: Option<Option<usize>>` to `Slot` (outer Some = solved/forced; inner = winner index or None for draw) — initialize `None` in both the constructor and `refill`. Add `solved_games: u32` to `SelfPlayPool` (init 0).

Add `use crate::endgame::solve_race;` to the imports.

In `commit_move`, replace the single terminal/cap guard with the ordered checks (terminal first, then `(0,0)` truncation, then cap):
```rust
        if is_terminal(&next) {
            return false; // natural win -> finalize uses winner(next)
        }
        if cfg.endgame_solve && next.walls_left == [0, 0] {
            let (val_mover, _) = solve_race(&next);
            let w = if val_mover > 0 {
                Some(next.turn as usize)
            } else if val_mover < 0 {
                Some(1 - next.turn as usize)
            } else {
                None // draw at bound
            };
            self.slots[i].forced_outcome = Some(w);
            self.solved_games += 1;
            return false; // truncate: race is decided
        }
        if slot.ply >= cfg.max_plies {
            return false; // cap -> draw
        }
```
NOTE: `self.slots[i].forced_outcome = ...` and `self.solved_games += 1` need `self`, but `slot` holds `&mut self.slots[i]`. Resolve the borrow exactly like the existing `seed` pattern: compute `w` into a local, then drop the `slot` borrow before touching `self` — e.g. do the `(0,0)` block using `self.slots[i]` directly (not the `slot` alias) since this branch returns immediately. Adjust as the compiler requires; no `unsafe`.

In `finalize`, use the forced outcome when present:
```rust
        let w = match self.slots[i].forced_outcome {
            Some(fw) => fw,                       // solved/truncated game
            None => winner(&self.slots[i].game),  // natural end or max_plies draw
        };
```
(The rest of `finalize` is unchanged.)

- [ ] **Step 5: pyiface — `native/src/pyiface.rs`**

In `SelfPlayPool::new`, add `endgame_solve=false` to the `#[pyo3(signature = ...)]` and the arg list, and thread it into `Config { ... }`:
```rust
    #[pyo3(signature = (n_games, total_games, sims, c_puct=1.5, seed=0,
                        dirichlet_alpha=0.5, dirichlet_eps=0.25,
                        temp_moves=10, max_plies=200, carryover=true,
                        endgame_solve=false))]
    fn new(n_games: u32, total_games: u32, sims: u32, c_puct: f64, seed: u64,
           dirichlet_alpha: f64, dirichlet_eps: f64, temp_moves: u32, max_plies: u32,
           carryover: bool, endgame_solve: bool) -> SelfPlayPool {
        let cfg = Config { sims, c_puct, dirichlet_alpha, dirichlet_eps,
                           temp_moves, max_plies, carryover, endgame_solve };
        SelfPlayPool { inner: CorePool::new(n_games, total_games, cfg, seed) }
    }
```
Add a `games_solved` accessor to the `SelfPlayPool` pyclass (and a `pub fn games_solved(&self) -> u32 { self.solved_games }` on the core pool):
```rust
    fn games_solved(&self) -> u32 {
        self.inner.games_solved()
    }
```

- [ ] **Step 6: Agent inference — `agents/native_agent.py`**

In `NativeMctsAgent.select_move`, before building the tree, short-circuit to the solver at `(0,0)`:
```python
    def select_move(self, state):
        if state.walls_left == (0, 0) and not _is_terminal(state):
            _val, mv = bn.solve_race(_to_native(state))
            return _from_tuple(mv)
        # ... existing tree-building / run_heuristic / eval_fn path ...
```
Add a small helper at module level (reuse `core.rules`):
```python
def _is_terminal(state):
    from core.rules import is_terminal
    return is_terminal(state)
```
(Read the current `agents/native_agent.py` to place this cleanly; keep the existing heuristic/net paths intact below the short-circuit.)

- [ ] **Step 7: Build + run the integration test + existing native suites**
```bash
source .venv/bin/activate && maturin develop -m native/Cargo.toml -q && python -m pytest tests/test_endgame_integration.py tests/test_native_carryover.py tests/test_native_pool.py tests/test_native_tree.py -q
```
Expected: PASS. If `test_native_agent_plays_solver_move_at_zero_walls` fails, check the `(0,0)` short-circuit fires and returns the solver's move.

- [ ] **Step 8: Measure the `(0,0)` hit-rate** (sizing the truncation benefit) — informational:
```bash
source .venv/bin/activate && python -c "
import numpy as np, barricades_native as bn
pool = bn.SelfPlayPool(n_games=64, total_games=128, sims=16, seed=1, max_plies=160, endgame_solve=True)
while pool.games_remaining()>0:
    pl=pool.step()
    if pl is not None:
        b=np.asarray(pl).shape[0]; pool.feed(np.full((b,140),1/140,np.float32), np.zeros(b,np.float32))
    pool.drain()
pool.drain()
print(f'(0,0) solved/truncated: {pool.games_solved()}/128 games = {100*pool.games_solved()/128:.0f}%')
"
```
Report the fraction (with a uniform-policy net; the real net's rate will differ, but this sizes it).

- [ ] **Step 9: Full suite + commit**
```bash
source .venv/bin/activate && python -m pytest -q
git add native/src agents/native_agent.py tests/test_endgame_integration.py
git commit -m "feat(native): integrate endgame solver (MCTS leaf + self-play truncation + inference); hit-rate counter

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 5 (Unit 4): Minimax strength ladder

A parallel eval harness reporting a checkpoint's win-rate vs greedy + minimax d1/d2/d3 + time-budgeted — the scalable strength anchor. Then run it on the existing `campaign10k` checkpoint.

**Files:**
- Create: `scripts/eval_ladder.py`

- [ ] **Step 1: Create `scripts/eval_ladder.py`** (mirrors `scripts/eval_az.py`'s parallel structure):

```python
"""Strength ladder: a net checkpoint's win-rate vs greedy + a minimax ladder.
The scalable reference ("our Stockfish") for tracking AZ strength.

Usage: python scripts/eval_ladder.py [checkpoint] [games_per_rung] [az_sims]
"""
import os
import sys
import math
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from concurrent.futures import ProcessPoolExecutor
from core.state import initial_state
from core.rules import apply_move, is_terminal, winner

CKPT = sys.argv[1] if len(sys.argv) > 1 else "models/campaign10k/campaign_final.pt"
GAMES = int(sys.argv[2]) if len(sys.argv) > 2 else 20
AZ_SIMS = int(sys.argv[3]) if len(sys.argv) > 3 else 100

RUNGS = ["greedy", "minimax-d1", "minimax-d2", "minimax-d3", "minimax-t0.25"]


def make_opponent(rung, seed):
    if rung == "greedy":
        from agents.greedy_agent import GreedyAgent
        return GreedyAgent(seed=seed)
    from agents.minimax_agent import MinimaxAgent
    if rung.startswith("minimax-d"):
        return MinimaxAgent(max_depth=int(rung.split("d")[1]), time_budget=10.0, seed=seed)
    if rung.startswith("minimax-t"):
        return MinimaxAgent(time_budget=float(rung.split("t")[1]), seed=seed)
    raise ValueError(rung)


def make_az(seed):
    from agents.native_agent import NativeMctsAgent
    net = _load_net()

    def eval_fn(planes):
        x = torch.from_numpy(np.asarray(planes))           # CPU in workers
        with torch.no_grad():
            out = net(x)
            return torch.softmax(out[0], 1).numpy(), out[1].squeeze(1).numpy()
    return NativeMctsAgent(sims=AZ_SIMS, seed=seed, eval_fn=eval_fn)


_NET = None
def _load_net():
    global _NET
    if _NET is None:
        from agents.az.model import QuoridorNet
        n = QuoridorNet(32, 3)
        n.load_state_dict(torch.load(CKPT, map_location="cpu"), strict=False)
        _NET = n.eval()
    return _NET


def play_one(args):
    rung, i = args
    az, opp = make_az(i), make_opponent(rung, 5000 + i)
    players = (az, opp) if i % 2 == 0 else (opp, az)
    s = initial_state()
    for _ in range(400):
        if is_terminal(s):
            break
        s = apply_move(s, players[s.turn].select_move(s))
    w = winner(s)
    return 1 if ((w == 0 and i % 2 == 0) or (w == 1 and i % 2 == 1)) else 0


def main():
    if not os.path.exists(CKPT):
        print(f"checkpoint missing: {CKPT}"); return
    print(f"ckpt={CKPT} games/rung={GAMES} az_sims={AZ_SIMS}")
    for rung in RUNGS:
        t0 = time.time()
        with ProcessPoolExecutor() as ex:
            wins = sum(ex.map(play_one, [(rung, i) for i in range(GAMES)]))
        se = 100 * math.sqrt(0.25 / GAMES)
        print(f"  AZ vs {rung:14s}: {wins}/{GAMES} = {100*wins/GAMES:5.1f}% (±{2*se:.0f}%)  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-run on a tiny config** (correctness, not strength):
```bash
source .venv/bin/activate && python scripts/eval_ladder.py models/campaign10k/campaign_final.pt 2 16 2>&1 | tail -8
```
Expected: prints win-rate lines for all 5 rungs, no errors, fractions in [0,100]%.

- [ ] **Step 3: Run the real ladder on the campaign10k checkpoint** (evaluation, NOT training — this is allowed under "prepare only"):
```bash
source .venv/bin/activate && python scripts/eval_ladder.py models/campaign10k/campaign_final.pt 20 100
```
Use a long timeout (minimax-d3 is slow; up to 600000 ms). Report the full table — the current net's strength vs the ladder (we expect ~0% vs d2/d3 given the earlier 0/10 spot-check, near-100% vs greedy if it learned to race; this is the baseline reading).

- [ ] **Step 4: Commit**
```bash
git add scripts/eval_ladder.py
git commit -m "feat(eval): minimax strength ladder (greedy + minimax d1/d2/d3 + time-budgeted)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage**

| Spec element | Task |
|---|---|
| Unit 1: floating-wall fast-path (predicate + skip BFS) | Task 1 |
| Unit 1: differential oracle extended to wall-dense positions | Task 1 (step 1) |
| Unit 2: L-R augmentation (mirror planes + action perm) | Task 2 |
| Unit 2: commutation test | Task 2 (step 1) |
| Unit 2: wired into the campaign | Task 2 (step 5) |
| Unit 3: exact `(0,0)` race solver, depth-bounded + cycle/draw handling | Task 3 (`RACE_PLY_BOUND`, draw-at-bound) |
| Unit 3: differential vs Python negamax | Task 3 (step 1) |
| Unit 3: MCTS leaf exact value | Task 4 (step 3) |
| Unit 3: self-play truncation + exact z (flagged) | Task 4 (step 4) |
| Unit 3: inference conversion (solver plays endgame) | Task 4 (step 6) |
| Unit 3: hit-rate instrumentation | Task 4 (steps 5, 8) |
| Unit 4: minimax ladder eval harness | Task 5 |
| Unit 4: run on existing checkpoint | Task 5 (step 3) |
| "prepare only — no training campaign" | Honored: no `run_campaign`/long self-play launched; only eval (Task 5 step 3) and short instrumentation runs |

**Placeholder scan:** none. Every code step shows complete code; the only "adjust as the compiler requires" note is the borrow-resolution in Task 4 step 4, with the exact pattern (the existing `seed` approach) given.

**Type consistency:** `solve_race(state) -> (i32, Move)` is consistent across Rust (`endgame.rs`), pyiface (`(i32, Move)` → Python `(int, move_tuple)`), the differential test, and the integrations (mcts uses `val_mover as f64`; selfplay maps `val_mover` sign → winner; agent uses the move). `Config` gains `carryover` (existing) + `endgame_solve` (new) — every `Config{...}` literal (only in pyiface) updated. `augment_lr`/`mirror_planes`/`LR_PERM` names match between `train.py`, the campaign import, and the test. `Slot.forced_outcome: Option<Option<usize>>` and `finalize`'s use of it are consistent.

---

## Out of scope (per spec)

- Broadening the endgame solver past `walls_left==(0,0)`.
- Opponent-curriculum training; small-board vs-perfect anchor; transposition table; minimax-vs-minimax value pretraining.
- **The next training campaign** (this plan is preparation only).
