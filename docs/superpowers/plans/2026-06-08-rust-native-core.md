# Rust Native Core — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the Quoridor game logic, BFS, encoding, and PUCT MCTS into a Rust PyO3 extension (`barricades_native`) and drive batched-MPS AlphaZero self-play where Rust does all CPU work (rayon, GIL released) and Python/PyTorch does only the MPS forward pass.

**Architecture:** Python-driven stepper. Rust owns N concurrent games + trees; `SelfPlayPool.step()` returns a contiguous `(M,6,9,9)` float32 batch of pending leaves; Python runs the MPS forward; `pool.feed(policy,value)` expands+backs-up+advances; `pool.drain()` yields training examples. The Rust core is validated **byte-for-byte against the existing Python `core`** via differential tests before anything long runs.

**Tech Stack:** Rust (edition 2024) · PyO3 0.28.3 (`extension-module`) · rust-numpy 0.28 · rayon 1 · rand 0.10 · rand_distr 0.6 · maturin 1.13 · Python 3.14 · PyTorch 2.12 (MPS).

**Spec:** `docs/superpowers/specs/2026-06-08-rust-core-design.md`

---

## Conventions (read first — every task depends on these)

**Build + test loop (run from repo root `/Users/Ethan_1/barricades`).** Every Rust change must be rebuilt before pytest sees it:

```bash
source .venv/bin/activate && maturin develop -m native/Cargo.toml -q && python -m pytest <testfile> -q
```

For benchmarks use the release build: `maturin develop -m native/Cargo.toml -r -q`.

**Python-facing state tuple** (`state`): `((c0,r0),(c1,r1), h_list, v_list, (n0,n1), turn)` where `h_list`/`v_list` are lists of `(c,r)` wall anchors (c,r in 0..7), `walls_left` is `(n0,n1)`, `turn` is 0/1. Functions that *return* a state return `h_list`/`v_list` **sorted ascending** so equality holds.

**Python-facing move tuple** (`move`): a step is `("step", c, r)`; a wall is `("wall", c, r, "H")` or `("wall", c, r, "V")`.

**Internal Rust types** (defined in Task 1): `GameState{ pawns:[(u8,u8);2], h_mask:u64, v_mask:u64, walls_left:[u8;2], turn:u8 }` with wall slot bit index `= r*8 + c` (c,r in 0..7). `enum Move { Step{c:i32,r:i32}, Wall{c:i32,r:i32,orient:u8} }` with `orient` 0=H, 1=V.

**Perspective conventions** (mirror the Python exactly):
- BFS `bfs_dist(state, player)` ignores the opponent; goal row is 8 for player 0, 0 for player 1.
- Encoding is current-player-relative: `flip = (turn == 1)`; cells flip `r → 8-r`, wall anchors flip `r → 7-r`. Planes are `[plane, row, col]`, flat index `plane*81 + row*9 + col`.
- MCTS stores `W` in **root-player perspective**; the net value at a leaf is from the leaf's player-to-move perspective and is negated into root perspective on backup (`value if leaf.turn==root_player else -value`), matching `agents/az/mcts_nn.py`.
- Net policy: the driver passes a **full 140-way softmax** of the logits; Rust selects the legal entries and renormalizes (which equals softmax-over-legal, as in `NetWrapper.predict`).

**pyo3 0.28 note:** this version uses the `IntoPyObject` trait (not the old `IntoPy`/`into_py`) and `Python::detach` (not `allow_threads`). The code below targets 0.28; if a trait signature differs by a hair, follow the compiler's suggestion — the data shapes are fixed.

---

## File Structure

| File | Responsibility |
|------|----------------|
| `native/Cargo.toml` | crate metadata + deps |
| `native/src/lib.rs` | `#[pymodule]` entry → delegates to `pyiface::register` |
| `native/src/coords.rs` | `N`, `on_board`, `goal_row`, `DIRS4` |
| `native/src/state.rs` | `GameState`, `Move`, `apply_move`, `winner`, `is_terminal`, `initial_state` |
| `native/src/bitboard.rs` | u128 flood-fill `bfs_dist`, `path_exists`, `can_move_masks` |
| `native/src/movegen.rs` | `is_blocked`, `legal_steps`, `legal_walls`, `legal_moves` |
| `native/src/encoding.rs` | `encode_planes`, `move_to_action`, `action_to_move` |
| `native/src/mcts.rs` | `Tree` node arena: select / expand / backup / best_move / heuristic |
| `native/src/selfplay.rs` | `SelfPlayPool`: step / feed / drain |
| `native/src/pyiface.rs` | all `#[pyfunction]`/`#[pyclass]` wrappers + state/move marshaling |
| `agents/native_agent.py` | `NativeMctsAgent` (Python `Agent` wrapping the Rust `Tree`) |
| `scripts/selfplay_native.py` | pool ↔ MPS self-play driver |
| `scripts/bench_selfplay.py` | throughput benchmark + 100k projection (the gate) |
| `tests/test_native_game.py` | differential: legal moves / BFS / apply / is_blocked |
| `tests/test_native_encoding.py` | differential: planes equal + action round-trip |
| `tests/test_native_tree.py` | MCTS sanity (legal, immediate win, concentration, beats random) |
| `tests/test_native_pool.py` | self-play pool smoke (well-formed examples, drains all games) |

---

## Task 1: Rust game core + Python differential gate

Implement the whole game core (`coords`, `state`, `bitboard`, `movegen`) and the Python primitives in `pyiface`, gated by a differential test that fuzzes thousands of random positions against the Python `core`.

**Files:**
- Modify: `native/Cargo.toml`
- Create: `native/src/coords.rs`, `native/src/state.rs`, `native/src/bitboard.rs`, `native/src/movegen.rs`, `native/src/pyiface.rs`
- Modify: `native/src/lib.rs`
- Test: `tests/test_native_game.py`

- [ ] **Step 1: Write the failing differential test**

Create `tests/test_native_game.py`:

```python
import random
import barricades_native as bn
from core.state import GameState, Step, Wall, initial_state
from core import rules


def to_native(s):
    return (tuple(s.pawns), sorted(s.h_walls), sorted(s.v_walls),
            tuple(s.walls_left), s.turn)


def mv_to_tuple(m):
    if isinstance(m, Step):
        return ("step", m.to_cell[0], m.to_cell[1])
    return ("wall", m.c, m.r, m.orient)


def test_open_board_basics():
    s = to_native(initial_state())
    assert bn.shortest_path_len(s, 0) == 8
    assert bn.shortest_path_len(s, 1) == 8
    assert bn.winner(s) is None
    assert bn.is_terminal(s) is False
    # 3 opening steps + all legal walls; compare as a set to Python core
    py = {mv_to_tuple(m) for m in rules.legal_moves(initial_state())}
    assert set(bn.legal_moves(s)) == py


def test_differential_over_random_games():
    rng = random.Random(12345)
    checked = 0
    for _ in range(80):
        s = initial_state()
        for _ in range(80):
            if rules.is_terminal(s):
                break
            ns = to_native(s)
            # legal moves agree as sets
            assert set(bn.legal_moves(ns)) == {mv_to_tuple(m) for m in rules.legal_moves(s)}
            # BFS distances agree for both players
            for p in (0, 1):
                assert bn.shortest_path_len(ns, p) == rules.shortest_path_len(s, p)
            # winner/terminal agree
            assert bn.winner(ns) == rules.winner(s)
            assert bn.is_terminal(ns) == rules.is_terminal(s)
            # is_blocked agrees for every orthogonally-adjacent cell pair
            me = s.pawns[s.turn]
            for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                b = (me[0] + dx, me[1] + dy)
                if 0 <= b[0] < 9 and 0 <= b[1] < 9:
                    assert bn.is_blocked(ns, me, b) == rules.is_blocked(s, me, b)
            # apply_move agrees for a sampled legal move
            mv = rng.choice(rules.legal_moves(s))
            assert bn.apply_move(ns, mv_to_tuple(mv)) == to_native(rules.apply_move(s, mv))
            checked += 1
            s = rules.apply_move(s, mv)
    assert checked > 2000
```

- [ ] **Step 2: Run it and confirm it fails**

```bash
source .venv/bin/activate && python -m pytest tests/test_native_game.py -q
```
Expected: FAIL — `AttributeError: module 'barricades_native' has no attribute 'shortest_path_len'`.

- [ ] **Step 3: Add Rust dependencies**

Replace `native/Cargo.toml` `[dependencies]` with:

```toml
[dependencies]
pyo3 = { version = "0.28.3", features = ["extension-module"] }
numpy = "0.28"
rayon = "1"
rand = "0.10"
rand_distr = "0.6"
```

- [ ] **Step 4: Implement `native/src/coords.rs`**

```rust
pub const N: i32 = 9;
pub const DIRS4: [(i32, i32); 4] = [(0, 1), (0, -1), (1, 0), (-1, 0)];

#[inline]
pub fn on_board(c: i32, r: i32) -> bool {
    c >= 0 && c < N && r >= 0 && r < N
}

#[inline]
pub fn goal_row(player: usize) -> i32 {
    if player == 0 { N - 1 } else { 0 }
}
```

- [ ] **Step 5: Implement `native/src/state.rs`**

```rust
use crate::coords::goal_row;

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct GameState {
    pub pawns: [(u8, u8); 2], // (col, row)
    pub h_mask: u64,          // slot bit = r*8 + c, c,r in 0..7
    pub v_mask: u64,
    pub walls_left: [u8; 2],
    pub turn: u8, // 0 or 1
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Move {
    Step { c: i32, r: i32 },
    Wall { c: i32, r: i32, orient: u8 }, // 0 = H, 1 = V
}

impl GameState {
    #[inline]
    pub fn has_h(&self, c: i32, r: i32) -> bool {
        c >= 0 && c < 8 && r >= 0 && r < 8 && (self.h_mask >> (r * 8 + c)) & 1 != 0
    }
    #[inline]
    pub fn has_v(&self, c: i32, r: i32) -> bool {
        c >= 0 && c < 8 && r >= 0 && r < 8 && (self.v_mask >> (r * 8 + c)) & 1 != 0
    }
}

pub fn initial_state() -> GameState {
    GameState {
        pawns: [(4, 0), (4, 8)],
        h_mask: 0,
        v_mask: 0,
        walls_left: [10, 10],
        turn: 0,
    }
}

pub fn apply_move(s: &GameState, m: &Move) -> GameState {
    let mut g = *s;
    match *m {
        Move::Step { c, r } => {
            g.pawns[s.turn as usize] = (c as u8, r as u8);
        }
        Move::Wall { c, r, orient } => {
            g.walls_left[s.turn as usize] -= 1;
            let bp = (r * 8 + c) as u64;
            if orient == 0 {
                g.h_mask |= 1u64 << bp;
            } else {
                g.v_mask |= 1u64 << bp;
            }
        }
    }
    g.turn = 1 - s.turn;
    g
}

pub fn winner(s: &GameState) -> Option<usize> {
    for p in 0..2 {
        if s.pawns[p].1 as i32 == goal_row(p) {
            return Some(p);
        }
    }
    None
}

pub fn is_terminal(s: &GameState) -> bool {
    winner(s).is_some()
}
```

- [ ] **Step 6: Implement `native/src/bitboard.rs`** (u128 flood-fill; mirrors `core/bitboard.py`)

```rust
use crate::state::GameState;

const FULL: u128 = (1u128 << 81) - 1;

#[inline]
fn bit(c: i32, r: i32) -> u128 {
    1u128 << (r * 9 + c)
}

fn row_mask(r: i32) -> u128 {
    let mut m = 0u128;
    for c in 0..9 {
        m |= bit(c, r);
    }
    m
}

fn col_mask(c: i32) -> u128 {
    let mut m = 0u128;
    for r in 0..9 {
        m |= bit(c, r);
    }
    m
}

fn can_move_masks(s: &GameState) -> (u128, u128, u128, u128) {
    let (mut bn, mut bs, mut be, mut bw) = (0u128, 0u128, 0u128, 0u128);
    let mut hm = s.h_mask;
    while hm != 0 {
        let i = hm.trailing_zeros() as i32;
        hm &= hm - 1;
        let (a, b) = (i % 8, i / 8);
        bn |= bit(a, b) | bit(a + 1, b); // N blocked from (a,b),(a+1,b)
        bs |= bit(a, b + 1) | bit(a + 1, b + 1); // S blocked from (a,b+1),(a+1,b+1)
    }
    let mut vm = s.v_mask;
    while vm != 0 {
        let i = vm.trailing_zeros() as i32;
        vm &= vm - 1;
        let (a, b) = (i % 8, i / 8);
        be |= bit(a, b) | bit(a, b + 1); // E blocked from (a,b),(a,b+1)
        bw |= bit(a + 1, b) | bit(a + 1, b + 1); // W blocked from (a+1,b),(a+1,b+1)
    }
    let can_n = FULL & !row_mask(8) & !bn;
    let can_s = FULL & !row_mask(0) & !bs;
    let can_e = FULL & !col_mask(8) & !be;
    let can_w = FULL & !col_mask(0) & !bw;
    (can_n, can_s, can_e, can_w)
}

#[inline]
fn expand(frontier: u128, m: (u128, u128, u128, u128)) -> u128 {
    let (cn, cs, ce, cw) = m;
    let n = (frontier & cn) << 9;
    let s = (frontier & cs) >> 9;
    let e = (frontier & ce) << 1;
    let w = (frontier & cw) >> 1;
    (n | s | e | w) & FULL
}

pub fn bfs_dist(s: &GameState, player: usize) -> Option<u32> {
    let (c, r) = (s.pawns[player].0 as i32, s.pawns[player].1 as i32);
    let goal = if player == 0 { row_mask(8) } else { row_mask(0) };
    let start = bit(c, r);
    if start & goal != 0 {
        return Some(0);
    }
    let masks = can_move_masks(s);
    let mut visited = start;
    let mut frontier = start;
    let mut dist = 0u32;
    while frontier != 0 {
        let nxt = expand(frontier, masks) & !visited;
        if nxt == 0 {
            return None;
        }
        dist += 1;
        if nxt & goal != 0 {
            return Some(dist);
        }
        visited |= nxt;
        frontier = nxt;
    }
    None
}

pub fn path_exists(s: &GameState, player: usize) -> bool {
    bfs_dist(s, player).is_some()
}
```

- [ ] **Step 7: Implement `native/src/movegen.rs`** (mirrors `core/rules.py`)

```rust
use crate::bitboard::path_exists;
use crate::coords::{on_board, DIRS4};
use crate::state::{GameState, Move};

pub fn is_blocked(s: &GameState, a: (i32, i32), b: (i32, i32)) -> bool {
    let (ax, ay) = a;
    let (bx, by) = b;
    let (dx, dy) = (bx - ax, by - ay);
    if dy == 1 {
        return s.has_h(ax, ay) || s.has_h(ax - 1, ay);
    }
    if dy == -1 {
        return s.has_h(ax, by) || s.has_h(ax - 1, by);
    }
    if dx == 1 {
        return s.has_v(ax, ay) || s.has_v(ax, ay - 1);
    }
    // dx == -1
    s.has_v(bx, ay) || s.has_v(bx, ay - 1)
}

pub fn legal_steps(s: &GameState) -> Vec<(i32, i32)> {
    let me = s.pawns[s.turn as usize];
    let me = (me.0 as i32, me.1 as i32);
    let opp = s.pawns[1 - s.turn as usize];
    let opp = (opp.0 as i32, opp.1 as i32);
    let mut dests = Vec::new();
    for (dx, dy) in DIRS4 {
        let adj = (me.0 + dx, me.1 + dy);
        if !on_board(adj.0, adj.1) || is_blocked(s, me, adj) {
            continue;
        }
        if adj != opp {
            dests.push(adj);
            continue;
        }
        let landing = (opp.0 + dx, opp.1 + dy);
        if on_board(landing.0, landing.1) && !is_blocked(s, opp, landing) {
            dests.push(landing);
        } else {
            for (px, py) in DIRS4 {
                if (px, py) == (dx, dy) || (px, py) == (-dx, -dy) {
                    continue;
                }
                let diag = (opp.0 + px, opp.1 + py);
                if on_board(diag.0, diag.1) && !is_blocked(s, opp, diag) {
                    dests.push(diag);
                }
            }
        }
    }
    dests
}

fn overlaps(s: &GameState, c: i32, r: i32, orient: u8) -> bool {
    if orient == 0 {
        s.has_h(c, r) || s.has_h(c - 1, r) || s.has_h(c + 1, r) || s.has_v(c, r)
    } else {
        s.has_v(c, r) || s.has_v(c, r - 1) || s.has_v(c, r + 1) || s.has_h(c, r)
    }
}

fn with_wall(s: &GameState, c: i32, r: i32, orient: u8) -> GameState {
    let mut g = *s;
    let bp = (r * 8 + c) as u64;
    if orient == 0 {
        g.h_mask |= 1u64 << bp;
    } else {
        g.v_mask |= 1u64 << bp;
    }
    g
}

pub fn legal_walls(s: &GameState) -> Vec<(i32, i32, u8)> {
    if s.walls_left[s.turn as usize] == 0 {
        return Vec::new();
    }
    let mut res = Vec::new();
    for orient in [0u8, 1u8] {
        for c in 0..8 {
            for r in 0..8 {
                if overlaps(s, c, r, orient) {
                    continue;
                }
                let s2 = with_wall(s, c, r, orient);
                if path_exists(&s2, 0) && path_exists(&s2, 1) {
                    res.push((c, r, orient));
                }
            }
        }
    }
    res
}

pub fn legal_moves(s: &GameState) -> Vec<Move> {
    let mut out: Vec<Move> = legal_steps(s)
        .into_iter()
        .map(|(c, r)| Move::Step { c, r })
        .collect();
    for (c, r, orient) in legal_walls(s) {
        out.push(Move::Wall { c, r, orient });
    }
    out
}
```

- [ ] **Step 8: Implement `native/src/pyiface.rs`** (state/move marshaling + game primitives)

```rust
use pyo3::prelude::*;
use pyo3::types::PyModule;

use crate::bitboard::bfs_dist;
use crate::movegen::{is_blocked, legal_moves};
use crate::state::{apply_move, is_terminal, winner, GameState, Move};

// ---- marshaling -----------------------------------------------------------

pub fn parse_state(state: &Bound<'_, PyAny>) -> PyResult<GameState> {
    let pawns: ((i32, i32), (i32, i32)) = state.get_item(0)?.extract()?;
    let h: Vec<(i32, i32)> = state.get_item(1)?.extract()?;
    let v: Vec<(i32, i32)> = state.get_item(2)?.extract()?;
    let wl: (u8, u8) = state.get_item(3)?.extract()?;
    let turn: u8 = state.get_item(4)?.extract()?;
    let mut g = GameState {
        pawns: [
            (pawns.0 .0 as u8, pawns.0 .1 as u8),
            (pawns.1 .0 as u8, pawns.1 .1 as u8),
        ],
        h_mask: 0,
        v_mask: 0,
        walls_left: [wl.0, wl.1],
        turn,
    };
    for (c, r) in h {
        g.h_mask |= 1u64 << (r * 8 + c);
    }
    for (c, r) in v {
        g.v_mask |= 1u64 << (r * 8 + c);
    }
    Ok(g)
}

pub fn parse_move(m: &Bound<'_, PyAny>) -> PyResult<Move> {
    let kind: String = m.get_item(0)?.extract()?;
    let c: i32 = m.get_item(1)?.extract()?;
    let r: i32 = m.get_item(2)?.extract()?;
    if kind == "step" {
        Ok(Move::Step { c, r })
    } else {
        let o: String = m.get_item(3)?.extract()?;
        Ok(Move::Wall { c, r, orient: if o == "H" { 0 } else { 1 } })
    }
}

impl<'py> IntoPyObject<'py> for Move {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;
    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        match self {
            Move::Step { c, r } => Ok(("step", c, r).into_pyobject(py)?.into_any()),
            Move::Wall { c, r, orient } => {
                Ok(("wall", c, r, if orient == 0 { "H" } else { "V" })
                    .into_pyobject(py)?
                    .into_any())
            }
        }
    }
}

// state -> sorted python tuple
fn state_to_py(
    py: Python<'_>,
    g: &GameState,
) -> PyResult<PyObject> {
    let mut h: Vec<(i32, i32)> = Vec::new();
    let mut v: Vec<(i32, i32)> = Vec::new();
    for i in 0..64 {
        if (g.h_mask >> i) & 1 != 0 {
            h.push((i as i32 % 8, i as i32 / 8));
        }
        if (g.v_mask >> i) & 1 != 0 {
            v.push((i as i32 % 8, i as i32 / 8));
        }
    }
    h.sort();
    v.sort();
    let pawns = (
        (g.pawns[0].0 as i32, g.pawns[0].1 as i32),
        (g.pawns[1].0 as i32, g.pawns[1].1 as i32),
    );
    let wl = (g.walls_left[0], g.walls_left[1]);
    Ok((pawns, h, v, wl, g.turn).into_pyobject(py)?.into_any().unbind())
}

// ---- pyfunctions ----------------------------------------------------------

#[pyfunction]
fn legal_moves_py(state: &Bound<'_, PyAny>) -> PyResult<Vec<Move>> {
    Ok(legal_moves(&parse_state(state)?))
}

#[pyfunction]
fn shortest_path_len_py(state: &Bound<'_, PyAny>, player: usize) -> PyResult<Option<u32>> {
    Ok(bfs_dist(&parse_state(state)?, player))
}

#[pyfunction]
fn is_blocked_py(state: &Bound<'_, PyAny>, a: (i32, i32), b: (i32, i32)) -> PyResult<bool> {
    Ok(is_blocked(&parse_state(state)?, a, b))
}

#[pyfunction]
fn apply_move_py(
    py: Python<'_>,
    state: &Bound<'_, PyAny>,
    mv: &Bound<'_, PyAny>,
) -> PyResult<PyObject> {
    let g = apply_move(&parse_state(state)?, &parse_move(mv)?);
    state_to_py(py, &g)
}

#[pyfunction]
fn winner_py(state: &Bound<'_, PyAny>) -> PyResult<Option<usize>> {
    Ok(winner(&parse_state(state)?))
}

#[pyfunction]
fn is_terminal_py(state: &Bound<'_, PyAny>) -> PyResult<bool> {
    Ok(is_terminal(&parse_state(state)?))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(legal_moves_py, m)?)?;
    m.add("legal_moves", m.getattr("legal_moves_py")?)?;
    m.add_function(wrap_pyfunction!(shortest_path_len_py, m)?)?;
    m.add("shortest_path_len", m.getattr("shortest_path_len_py")?)?;
    m.add_function(wrap_pyfunction!(is_blocked_py, m)?)?;
    m.add("is_blocked", m.getattr("is_blocked_py")?)?;
    m.add_function(wrap_pyfunction!(apply_move_py, m)?)?;
    m.add("apply_move", m.getattr("apply_move_py")?)?;
    m.add_function(wrap_pyfunction!(winner_py, m)?)?;
    m.add("winner", m.getattr("winner_py")?)?;
    m.add_function(wrap_pyfunction!(is_terminal_py, m)?)?;
    m.add("is_terminal", m.getattr("is_terminal_py")?)?;
    Ok(())
}
```

> Note: the `#[pyfunction]` Rust names carry a `_py` suffix to avoid clashing with the Rust `legal_moves`/`winner`/etc. imported from the core modules; the `m.add("name", ...)` calls expose them to Python under the clean names the tests use. If you prefer, use `#[pyo3(name = "legal_moves")]` on each function and drop the alias lines — either approach must expose exactly: `legal_moves`, `shortest_path_len`, `is_blocked`, `apply_move`, `winner`, `is_terminal`.

- [ ] **Step 9: Wire up `native/src/lib.rs`**

```rust
use pyo3::prelude::*;

mod bitboard;
mod coords;
mod movegen;
mod pyiface;
mod state;

#[pymodule]
fn barricades_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    pyiface::register(m)
}
```

- [ ] **Step 10: Build and run the differential test**

```bash
source .venv/bin/activate && maturin develop -m native/Cargo.toml -q && python -m pytest tests/test_native_game.py -q
```
Expected: PASS (`2 passed`). If `legal_moves` set comparison fails, dump the differing position and compare `is_blocked`/`legal_walls` against `core/rules.py` for that exact state.

- [ ] **Step 11: Confirm the existing suite still passes (no regressions)**

```bash
source .venv/bin/activate && python -m pytest -q
```
Expected: `214 passed` (212 baseline + 2 new).

- [ ] **Step 12: Commit**

```bash
git add native/Cargo.toml native/Cargo.lock native/src tests/test_native_game.py
git commit -m "feat(native): Rust game core (bitboard BFS, movegen) + differential gate"
```

---

## Task 2: Encoding parity (planes + 140-action map)

**Files:**
- Create: `native/src/encoding.rs`
- Modify: `native/src/lib.rs` (add `mod encoding;`), `native/src/pyiface.rs` (add functions + register)
- Test: `tests/test_native_encoding.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_native_encoding.py`:

```python
import random
import numpy as np
import barricades_native as bn
from core.state import initial_state
from core import rules
from agents.az import encoding as enc
from tests.test_native_game import to_native, mv_to_tuple


def test_planes_match_initial():
    s = initial_state()
    got = bn.encode_planes(to_native(s))
    want = enc.encode_planes(s)
    assert got.shape == (6, 9, 9)
    assert got.dtype == np.float32
    assert np.array_equal(got, want)


def test_encoding_differential_over_random_games():
    rng = random.Random(7)
    checked = 0
    for _ in range(60):
        s = initial_state()
        for _ in range(80):
            if rules.is_terminal(s):
                break
            ns = to_native(s)
            assert np.array_equal(bn.encode_planes(ns), enc.encode_planes(s))
            for m in rules.legal_moves(s):
                mt = mv_to_tuple(m)
                idx = bn.move_to_action(mt, ns)
                assert idx == enc.move_to_action(m, s)
                # action_to_move round-trips back to the same canonical action
                assert bn.move_to_action(bn.action_to_move(idx, ns), ns) == idx
            checked += 1
            s = rules.apply_move(s, rng.choice(rules.legal_moves(s)))
    assert checked > 1500
```

- [ ] **Step 2: Run it and confirm it fails**

```bash
source .venv/bin/activate && python -m pytest tests/test_native_encoding.py -q
```
Expected: FAIL — `module 'barricades_native' has no attribute 'encode_planes'`.

- [ ] **Step 3: Implement `native/src/encoding.rs`** (mirrors `agents/az/encoding.py`)

```rust
use crate::state::{GameState, Move};

pub const N_ACTIONS: usize = 140;

const DIRS12: [(i32, i32); 12] = [
    (0, 1), (0, -1), (1, 0), (-1, 0),       // 0..3 steps
    (0, 2), (0, -2), (2, 0), (-2, 0),       // 4..7 straight jumps
    (1, 1), (-1, 1), (1, -1), (-1, -1),     // 8..11 diagonal jumps
];

fn dir_index(d: (i32, i32)) -> usize {
    DIRS12.iter().position(|&x| x == d).expect("non-canonical step delta")
}

#[inline]
fn cf_cell(c: i32, r: i32, flip: bool) -> (i32, i32) {
    (c, if flip { 8 - r } else { r })
}

#[inline]
fn cf_wall(c: i32, r: i32, flip: bool) -> (i32, i32) {
    (c, if flip { 7 - r } else { r })
}

/// Write the 6x9x9 planes (row-major: plane*81 + row*9 + col) into `out` (len 486),
/// which the caller must pre-zero.
pub fn encode_planes(s: &GameState, out: &mut [f32]) {
    let flip = s.turn == 1;
    let me = s.pawns[s.turn as usize];
    let opp = s.pawns[1 - s.turn as usize];
    let mc = cf_cell(me.0 as i32, me.1 as i32, flip);
    let oc = cf_cell(opp.0 as i32, opp.1 as i32, flip);
    out[(mc.1 * 9 + mc.0) as usize] = 1.0; // plane 0
    out[81 + (oc.1 * 9 + oc.0) as usize] = 1.0; // plane 1
    let mut hm = s.h_mask;
    while hm != 0 {
        let i = hm.trailing_zeros() as i32;
        hm &= hm - 1;
        let (cc, cr) = cf_wall(i % 8, i / 8, flip);
        out[2 * 81 + (cr * 9 + cc) as usize] = 1.0;
    }
    let mut vm = s.v_mask;
    while vm != 0 {
        let i = vm.trailing_zeros() as i32;
        vm &= vm - 1;
        let (cc, cr) = cf_wall(i % 8, i / 8, flip);
        out[3 * 81 + (cr * 9 + cc) as usize] = 1.0;
    }
    let w_me = s.walls_left[s.turn as usize] as f32 / 10.0;
    let w_op = s.walls_left[1 - s.turn as usize] as f32 / 10.0;
    for k in 0..81 {
        out[4 * 81 + k] = w_me;
        out[5 * 81 + k] = w_op;
    }
}

pub fn move_to_action(m: &Move, s: &GameState) -> usize {
    let flip = s.turn == 1;
    match *m {
        Move::Step { c, r } => {
            let me = s.pawns[s.turn as usize];
            let mc = cf_cell(me.0 as i32, me.1 as i32, flip);
            let dest = cf_cell(c, r, flip);
            dir_index((dest.0 - mc.0, dest.1 - mc.1))
        }
        Move::Wall { c, r, orient } => {
            let (cc, cr) = cf_wall(c, r, flip);
            let off = if orient == 0 { 0 } else { 64 };
            (12 + off + cr * 8 + cc) as usize
        }
    }
}

pub fn action_to_move(idx: usize, s: &GameState) -> Move {
    let flip = s.turn == 1;
    if idx < 12 {
        let (dx, dy) = DIRS12[idx];
        let me = s.pawns[s.turn as usize];
        let mc = cf_cell(me.0 as i32, me.1 as i32, flip);
        let real = cf_cell(mc.0 + dx, mc.1 + dy, flip); // flip is its own inverse
        Move::Step { c: real.0, r: real.1 }
    } else {
        let a = idx - 12;
        let orient = if a < 64 { 0u8 } else { 1u8 };
        let a = a % 64;
        let (cr, cc) = ((a / 8) as i32, (a % 8) as i32);
        let real = cf_wall(cc, cr, flip);
        Move::Wall { c: real.0, r: real.1, orient }
    }
}
```

- [ ] **Step 4: Add the pyfunctions to `native/src/pyiface.rs`**

Add near the top imports:

```rust
use crate::encoding::{action_to_move, encode_planes, move_to_action, N_ACTIONS};
use numpy::{IntoPyArray, PyArray3};
```

Add these functions before `register`:

```rust
#[pyfunction]
fn encode_planes_py<'py>(
    py: Python<'py>,
    state: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    let g = parse_state(state)?;
    let mut buf = vec![0f32; 6 * 81];
    encode_planes(&g, &mut buf);
    let arr = numpy::ndarray::Array3::from_shape_vec((6, 9, 9), buf)
        .expect("shape 6x9x9");
    Ok(arr.into_pyarray(py))
}

#[pyfunction]
fn move_to_action_py(mv: &Bound<'_, PyAny>, state: &Bound<'_, PyAny>) -> PyResult<usize> {
    Ok(move_to_action(&parse_move(mv)?, &parse_state(state)?))
}

#[pyfunction]
fn action_to_move_py(idx: usize, state: &Bound<'_, PyAny>) -> PyResult<Move> {
    Ok(action_to_move(idx, &parse_state(state)?))
}
```

Add to `register` (before `Ok(())`):

```rust
    m.add_function(wrap_pyfunction!(encode_planes_py, m)?)?;
    m.add("encode_planes", m.getattr("encode_planes_py")?)?;
    m.add_function(wrap_pyfunction!(move_to_action_py, m)?)?;
    m.add("move_to_action", m.getattr("move_to_action_py")?)?;
    m.add_function(wrap_pyfunction!(action_to_move_py, m)?)?;
    m.add("action_to_move", m.getattr("action_to_move_py")?)?;
    let _ = N_ACTIONS; // documents the action-space size
```

- [ ] **Step 5: Add `mod encoding;` to `native/src/lib.rs`**

Insert `mod encoding;` in the module list (keep alphabetical: after `mod coords;`).

- [ ] **Step 6: Build and run the encoding test**

```bash
source .venv/bin/activate && maturin develop -m native/Cargo.toml -q && python -m pytest tests/test_native_encoding.py -q
```
Expected: PASS (`2 passed`). If planes differ, check the `flip` row math (`8-r` for cells, `7-r` for wall anchors) against `agents/az/encoding.py`.

- [ ] **Step 7: Commit**

```bash
git add native/src tests/test_native_encoding.py
git commit -m "feat(native): encoding parity (6x9x9 planes + 140-action map)"
```

---

## Task 3: MCTS tree + native agent

Implement the PUCT tree with two leaf-eval sources: an **external net** (`prepare_leaf`/`receive`, for the pool) and an **internal heuristic** (`run_heuristic`, for a net-free bot). Expose a `Tree` pyclass and a Python `NativeMctsAgent`.

**Files:**
- Create: `native/src/mcts.rs`, `agents/native_agent.py`
- Modify: `native/src/lib.rs` (add `mod mcts;`), `native/src/pyiface.rs` (add `Tree` pyclass + register)
- Test: `tests/test_native_tree.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_native_tree.py`:

```python
import random
import numpy as np
import barricades_native as bn
from core.state import GameState, Step, initial_state
from core import rules
from tests.test_native_game import to_native, mv_to_tuple


def _state(p0, p1, wl=(10, 10), turn=0, h=(), v=()):
    return GameState((p0, p1), frozenset(h), frozenset(v), wl, turn)


def test_tree_returns_legal_move_heuristic():
    t = bn.Tree(to_native(initial_state()), 1.5, 0)
    mv = t.run_heuristic(120)
    assert mv in {mv_to_tuple(m) for m in rules.legal_moves(initial_state())}


def test_tree_takes_immediate_win():
    # player 0 at (4,7) can step to (4,8) and win.
    s = _state((4, 7), (0, 0))
    t = bn.Tree(to_native(s), 1.5, 0)
    mv = t.run_heuristic(160)
    assert mv == ("step", 4, 8)


def test_prepare_receive_protocol_runs():
    # Drive the net path with a uniform stub eval; must end with a legal move + pi.
    s = initial_state()
    t = bn.Tree(to_native(s), 1.5, 1)
    evals, guard = 0, 0
    while evals < 64 and guard < 512:
        guard += 1
        planes = t.prepare_leaf()
        if planes is None:
            continue
        policy = np.full(140, 1.0 / 140, dtype=np.float32)
        t.receive(policy, 0.0)
        evals += 1
    mv, pi = t.best_move(0.0)
    assert mv in {mv_to_tuple(m) for m in rules.legal_moves(s)}
    pi = np.asarray(pi, dtype=np.float32)
    assert pi.shape == (140,)
    assert abs(float(pi.sum()) - 1.0) < 1e-4


def test_native_agent_beats_random():
    from agents.native_agent import NativeMctsAgent
    from agents.random_agent import RandomAgent
    wins = 0
    for g in range(20):
        a, b = NativeMctsAgent(sims=120, seed=g), RandomAgent(seed=1000 + g)
        players = (a, b) if g % 2 == 0 else (b, a)
        s = initial_state()
        for _ in range(300):
            if rules.is_terminal(s):
                break
            s = rules.apply_move(s, players[s.turn].select_move(s))
        w = rules.winner(s)
        if (w == 0 and g % 2 == 0) or (w == 1 and g % 2 == 1):
            wins += 1
    assert wins >= 16  # MCTS+heuristic should crush random
```

- [ ] **Step 2: Run it and confirm it fails**

```bash
source .venv/bin/activate && python -m pytest tests/test_native_tree.py -q
```
Expected: FAIL — `module 'barricades_native' has no attribute 'Tree'`.

- [ ] **Step 3: Implement `native/src/mcts.rs`**

```rust
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::bitboard::bfs_dist;
use crate::encoding::{move_to_action, N_ACTIONS};
use crate::movegen::legal_moves;
use crate::state::{apply_move, is_terminal, winner, GameState, Move};

const HEUR_SCALE: f64 = 5.0; // squashes path-difference into (-1, 1)

struct Node {
    state: GameState,
    parent: i32,
    mv: Option<Move>, // move from parent (None at root)
    prior: f32,
    children: Vec<u32>,
    n: u32,
    w: f64, // root-player perspective
    expanded: bool,
}

/// Outcome of one `prepare_leaf` descent.
pub enum Leaf {
    /// A non-terminal unexpanded leaf is parked; its planes are written to the
    /// caller-provided buffer. The caller must call `receive` next.
    Parked,
    /// The descent hit a terminal node and backed up its value (a "free" sim).
    Terminal,
}

pub struct Tree {
    nodes: Vec<Node>,
    root: u32,
    root_player: usize,
    c_puct: f64,
    parked: i32,
    noised: bool,
    rng: StdRng,
}

impl Tree {
    pub fn new(state: GameState, c_puct: f64, seed: u64) -> Tree {
        let root = Node {
            state,
            parent: -1,
            mv: None,
            prior: 0.0,
            children: Vec::new(),
            n: 0,
            w: 0.0,
            expanded: false,
            children_capacity_hint: (),
        };
        // NOTE: drop the `children_capacity_hint` field — shown only to flag that
        // Node has no extra fields. The real Node literal omits it.
        Tree {
            nodes: vec![root],
            root: 0,
            root_player: state.turn as usize,
            c_puct,
            parked: -1,
            noised: false,
            rng: StdRng::seed_from_u64(seed),
        }
    }

    fn select_child(&self, node: usize) -> usize {
        let sqrt_n = (self.nodes[node].n as f64).sqrt();
        let parent_turn = self.nodes[node].state.turn as usize;
        let mut best = self.nodes[node].children[0] as usize;
        let mut best_score = f64::NEG_INFINITY;
        for &ci in &self.nodes[node].children {
            let ch = &self.nodes[ci as usize];
            let mut q = if ch.n > 0 { ch.w / ch.n as f64 } else { 0.0 };
            if parent_turn != self.root_player {
                q = -q;
            }
            let u = self.c_puct * ch.prior as f64 * sqrt_n / (1.0 + ch.n as f64);
            let score = q + u;
            if score > best_score {
                best_score = score;
                best = ci as usize;
            }
        }
        best
    }

    fn backup(&mut self, mut node: usize, v: f64) {
        loop {
            self.nodes[node].n += 1;
            self.nodes[node].w += v;
            let p = self.nodes[node].parent;
            if p < 0 {
                break;
            }
            node = p as usize;
        }
    }

    /// Expand `node` with priors derived from a full 140-way policy (legal
    /// entries are renormalized), then back up `value` (leaf-player perspective).
    fn expand_with_policy(&mut self, node: usize, policy: &[f32], value: f64) {
        let st = self.nodes[node].state;
        let legal = legal_moves(&st);
        let mut sum = 0f32;
        let mut probs = Vec::with_capacity(legal.len());
        for m in &legal {
            let p = policy[move_to_action(m, &st)];
            sum += p;
            probs.push(p);
        }
        for (i, m) in legal.iter().enumerate() {
            let prior = if sum > 0.0 {
                probs[i] / sum
            } else {
                1.0 / legal.len() as f32
            };
            let child = Node {
                state: apply_move(&st, m),
                parent: node as i32,
                mv: Some(*m),
                prior,
                children: Vec::new(),
                n: 0,
                w: 0.0,
                expanded: false,
            };
            let idx = self.nodes.len() as u32;
            self.nodes.push(child);
            self.nodes[node].children.push(idx);
        }
        self.nodes[node].expanded = true;
        let v = if st.turn as usize == self.root_player { value } else { -value };
        self.backup(node, v);
    }

    /// Descend from root to a leaf. If terminal, back up ±1 and return Terminal.
    /// Otherwise park the leaf, write its planes into `planes_out`, return Parked.
    pub fn prepare_leaf(&mut self, planes_out: &mut [f32]) -> Leaf {
        let mut node = self.root as usize;
        while self.nodes[node].expanded && !is_terminal(&self.nodes[node].state) {
            node = self.select_child(node);
        }
        if is_terminal(&self.nodes[node].state) {
            let w = winner(&self.nodes[node].state).unwrap();
            let v = if w == self.root_player { 1.0 } else { -1.0 };
            self.backup(node, v);
            return Leaf::Terminal;
        }
        crate::encoding::encode_planes(&self.nodes[node].state, planes_out);
        self.parked = node as i32;
        Leaf::Parked
    }

    pub fn receive(&mut self, policy: &[f32], value: f64) {
        let node = self.parked as usize;
        self.parked = -1;
        self.expand_with_policy(node, policy, value);
    }

    /// Internal heuristic value for `s` from its player-to-move perspective.
    fn heuristic_value(s: &GameState) -> f64 {
        if let Some(w) = winner(s) {
            return if w == s.turn as usize { 1.0 } else { -1.0 };
        }
        let d_self = bfs_dist(s, s.turn as usize).unwrap_or(1000) as f64;
        let d_opp = bfs_dist(s, 1 - s.turn as usize).unwrap_or(1000) as f64;
        ((d_opp - d_self) / HEUR_SCALE).tanh()
    }

    /// Run a full search using the internal heuristic (no Python round-trip):
    /// uniform priors, BFS-based leaf value. Returns the greedy best move.
    pub fn run_heuristic(&mut self, sims: u32) -> Move {
        let uniform = vec![1.0f32 / N_ACTIONS as f32; N_ACTIONS];
        let mut evals = 0u32;
        let mut guard = 0u32;
        let cap = sims * 8 + 64;
        while evals < sims && guard < cap {
            guard += 1;
            let mut node = self.root as usize;
            while self.nodes[node].expanded && !is_terminal(&self.nodes[node].state) {
                node = self.select_child(node);
            }
            if is_terminal(&self.nodes[node].state) {
                let w = winner(&self.nodes[node].state).unwrap();
                let v = if w == self.root_player { 1.0 } else { -1.0 };
                self.backup(node, v);
                continue;
            }
            let val = Tree::heuristic_value(&self.nodes[node].state);
            self.expand_with_policy(node, &uniform, val);
            evals += 1;
        }
        self.best_move(0.0).0
    }

    /// Add Dirichlet noise to the root's children priors (call once, after the
    /// root has been expanded). `alpha` is the Dirichlet concentration.
    pub fn apply_root_noise(&mut self, alpha: f64, eps: f64) {
        if self.noised || !self.nodes[self.root as usize].expanded {
            return;
        }
        use rand_distr::{Distribution, Gamma};
        let kids: Vec<u32> = self.nodes[self.root as usize].children.clone();
        if kids.is_empty() {
            return;
        }
        let gamma = Gamma::new(alpha, 1.0).unwrap();
        let mut g: Vec<f64> = (0..kids.len()).map(|_| gamma.sample(&mut self.rng)).collect();
        let tot: f64 = g.iter().sum::<f64>().max(1e-12);
        for x in g.iter_mut() {
            *x /= tot;
        }
        for (i, &ci) in kids.iter().enumerate() {
            let p = self.nodes[ci as usize].prior as f64;
            self.nodes[ci as usize].prior = ((1.0 - eps) * p + eps * g[i]) as f32;
        }
        self.noised = true;
    }

    /// Root visit-count policy over 140 actions, plus the chosen move.
    /// `temp == 0.0` → greedy (argmax with random tie-break); else sample ∝ visits.
    pub fn best_move(&mut self, temp: f64) -> (Move, [f32; N_ACTIONS]) {
        let kids = self.nodes[self.root as usize].children.clone();
        let root_state = self.nodes[self.root as usize].state;
        let mut pi = [0f32; N_ACTIONS];
        let total: u32 = kids.iter().map(|&c| self.nodes[c as usize].n).sum();
        if total > 0 {
            for &c in &kids {
                let a = move_to_action(self.nodes[c as usize].mv.as_ref().unwrap(), &root_state);
                pi[a] = self.nodes[c as usize].n as f32 / total as f32;
            }
        }
        let chosen = if temp == 0.0 {
            let top = kids.iter().map(|&c| self.nodes[c as usize].n).max().unwrap();
            let winners: Vec<u32> = kids
                .iter()
                .cloned()
                .filter(|&c| self.nodes[c as usize].n == top)
                .collect();
            winners[self.rng.random_range(0..winners.len())]
        } else {
            let r: f32 = self.rng.random::<f32>() * total.max(1) as f32;
            let mut acc = 0f32;
            let mut pick = kids[0];
            for &c in &kids {
                acc += self.nodes[c as usize].n as f32;
                if acc >= r {
                    pick = c;
                    break;
                }
            }
            pick
        };
        (self.nodes[chosen as usize].mv.unwrap(), pi)
    }
}
```

> **Implementer note:** the `Node` literal in `Tree::new` above shows a bogus
> `children_capacity_hint: ()` field with a comment — delete that line; `Node`
> has exactly the seven fields declared in the `struct Node` block. It is called
> out so you do not accidentally add fields.

- [ ] **Step 4: Add the `Tree` pyclass to `native/src/pyiface.rs`**

Add imports:

```rust
use crate::mcts::{Leaf, Tree as CoreTree};
use numpy::{PyArrayMethods, PyReadonlyArray1};
```

Add the pyclass and its methods:

```rust
#[pyclass]
pub struct Tree {
    inner: CoreTree,
}

#[pymethods]
impl Tree {
    #[new]
    fn new(state: &Bound<'_, PyAny>, c_puct: f64, seed: u64) -> PyResult<Tree> {
        Ok(Tree { inner: CoreTree::new(parse_state(state)?, c_puct, seed) })
    }

    /// Descend to a leaf. Returns the (6,9,9) planes if a leaf was parked
    /// (caller must call `receive` next), or None if a terminal was backed up.
    fn prepare_leaf<'py>(
        &mut self,
        py: Python<'py>,
    ) -> Option<Bound<'py, PyArray3<f32>>> {
        let mut buf = vec![0f32; 6 * 81];
        match self.inner.prepare_leaf(&mut buf) {
            Leaf::Parked => {
                let arr = numpy::ndarray::Array3::from_shape_vec((6, 9, 9), buf).unwrap();
                Some(arr.into_pyarray(py))
            }
            Leaf::Terminal => None,
        }
    }

    fn receive(&mut self, policy: PyReadonlyArray1<f32>, value: f64) -> PyResult<()> {
        self.inner.receive(policy.as_slice()?, value);
        Ok(())
    }

    fn run_heuristic(&mut self, sims: u32) -> Move {
        self.inner.run_heuristic(sims)
    }

    #[pyo3(signature = (alpha, eps=0.25))]
    fn apply_root_noise(&mut self, alpha: f64, eps: f64) {
        self.inner.apply_root_noise(alpha, eps);
    }

    fn best_move(&mut self, temp: f64) -> (Move, Vec<f32>) {
        let (mv, pi) = self.inner.best_move(temp);
        (mv, pi.to_vec())
    }
}
```

Add to `register`:

```rust
    m.add_class::<Tree>()?;
```

- [ ] **Step 5: Add `mod mcts;` to `native/src/lib.rs`**

Insert `mod mcts;` in the module list.

- [ ] **Step 6: Implement `agents/native_agent.py`**

```python
"""MCTS agent backed by the Rust core (barricades_native.Tree).

Default mode uses the Rust-internal heuristic eval (no neural net) — a fast,
self-contained bot for tournaments and the analysis board. Pass an `eval_fn`
mapping a batch of planes (B,6,9,9) float32 to (policy (B,140), value (B,)) to
drive it with a network instead.
"""
import numpy as np
import barricades_native as bn
from agents.base import Agent, Analysis
from agents.az.encoding import action_to_move  # for reporting, if needed


def _to_native(state):
    return (tuple(state.pawns), sorted(state.h_walls), sorted(state.v_walls),
            tuple(state.walls_left), state.turn)


def _from_tuple(state, mv):
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
        tree = bn.Tree(_to_native(state), self.c_puct, self.seed)
        if self.eval_fn is None:
            mv = tree.run_heuristic(self.sims)
            return _from_tuple(state, mv)
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
        return _from_tuple(state, mv)
```

- [ ] **Step 7: Build and run the tree tests**

```bash
source .venv/bin/activate && maturin develop -m native/Cargo.toml -q && python -m pytest tests/test_native_tree.py -q
```
Expected: PASS (`4 passed`). If `test_tree_takes_immediate_win` fails, verify the terminal value sign in `prepare_leaf`/`run_heuristic` (`+1` when `winner == root_player`).

- [ ] **Step 8: Commit**

```bash
git add native/src agents/native_agent.py tests/test_native_tree.py
git commit -m "feat(native): PUCT tree (net + heuristic eval) + NativeMctsAgent"
```

---

## Task 4: SelfPlayPool (the batched stepper)

N concurrent games; `step()` commits ready moves + parks one fresh leaf per active slot and returns the batch; `feed()` expands+backs-up; `drain()` yields finished training examples carrying `z` + heuristic features.

**Files:**
- Create: `native/src/selfplay.rs`
- Modify: `native/src/lib.rs` (add `mod selfplay;`), `native/src/pyiface.rs` (add `SelfPlayPool` pyclass + register)
- Test: `tests/test_native_pool.py`

- [ ] **Step 1: Write the failing smoke test**

Create `tests/test_native_pool.py`:

```python
import numpy as np
import barricades_native as bn


def _uniform_eval(planes):
    b = planes.shape[0]
    policy = np.full((b, 140), 1.0 / 140, dtype=np.float32)
    value = np.zeros(b, dtype=np.float32)
    return policy, value


def test_pool_produces_wellformed_examples_and_drains_all_games():
    pool = bn.SelfPlayPool(
        n_games=8, total_games=8, sims=16, c_puct=1.5, seed=0,
        dirichlet_alpha=0.5, dirichlet_eps=0.25, temp_moves=4, max_plies=120,
    )
    examples = []
    guard = 0
    while pool.games_remaining() > 0 and guard < 200_000:
        guard += 1
        planes = pool.step()
        if planes is None:
            continue
        planes = np.asarray(planes, dtype=np.float32)
        assert planes.ndim == 4 and planes.shape[1:] == (6, 9, 9)
        assert planes.shape[0] >= 1
        policy, value = _uniform_eval(planes)
        pool.feed(policy, value)
        examples.extend(pool.drain())
    assert pool.games_remaining() == 0
    assert len(examples) > 0
    for planes, pi, z, feats in examples:
        planes = np.asarray(planes, dtype=np.float32)
        pi = np.asarray(pi, dtype=np.float32)
        feats = np.asarray(feats, dtype=np.float32)
        assert planes.shape == (6, 9, 9)
        assert pi.shape == (140,)
        assert abs(float(pi.sum()) - 1.0) < 1e-4
        assert z in (-1.0, 0.0, 1.0)
        assert feats.shape == (4,)  # path_diff, walls_left_own, walls_left_opp, plies_to_end
```

- [ ] **Step 2: Run it and confirm it fails**

```bash
source .venv/bin/activate && python -m pytest tests/test_native_pool.py -q
```
Expected: FAIL — `module 'barricades_native' has no attribute 'SelfPlayPool'`.

- [ ] **Step 3: Implement `native/src/selfplay.rs`**

```rust
use rayon::prelude::*;

use crate::bitboard::bfs_dist;
use crate::encoding::encode_planes;
use crate::mcts::{Leaf, Tree};
use crate::state::{apply_move, is_terminal, initial_state, winner, GameState, Move};

const N_FEATS: usize = 4;

pub struct Example {
    pub planes: Vec<f32>,    // len 486
    pub pi: Vec<f32>,        // len 140
    pub z: f32,              // filled at game end
    pub feats: [f32; N_FEATS], // path_diff, wl_own, wl_opp, plies_to_end (filled at end)
}

struct Pending {
    planes: Vec<f32>, // len 486
}

enum Phase {
    AwaitingEval, // parked a leaf this step; expects feed
    ReadyToMove,  // sim budget reached; next step commits the move
}

struct Slot {
    game: GameState,
    tree: Tree,
    sims_done: u32,
    ply: u32,
    phase: Phase,
    // per-move records awaiting a z + plies_to_end stamp at game end
    records: Vec<(Vec<f32>, Vec<f32>, usize, [f32; N_FEATS])>, // planes, pi, player, feats(partial)
    active: bool,
    pending: Option<Pending>,
}

#[derive(Clone, Copy)]
pub struct Config {
    pub sims: u32,
    pub c_puct: f64,
    pub dirichlet_alpha: f64,
    pub dirichlet_eps: f64,
    pub temp_moves: u32,
    pub max_plies: u32,
}

pub struct SelfPlayPool {
    slots: Vec<Slot>,
    cfg: Config,
    next_seed: u64,
    launched: u32,
    total_games: u32,
    finished: u32,
    out_examples: Vec<Example>,
    // slots (in order) that parked a leaf in the most recent step()
    last_pending: Vec<usize>,
}

fn features(g: &GameState) -> [f32; N_FEATS] {
    let mover = g.turn as usize;
    let d_self = bfs_dist(g, mover).unwrap_or(1000) as f32;
    let d_opp = bfs_dist(g, 1 - mover).unwrap_or(1000) as f32;
    [
        d_opp - d_self, // path_diff (positive = good for mover)
        g.walls_left[mover] as f32,
        g.walls_left[1 - mover] as f32,
        0.0, // plies_to_end, stamped at game end
    ]
}

impl SelfPlayPool {
    pub fn new(n_games: u32, total_games: u32, cfg: Config, seed: u64) -> SelfPlayPool {
        let mut next_seed = seed;
        let mut slots = Vec::with_capacity(n_games as usize);
        let mut launched = 0u32;
        for _ in 0..n_games.min(total_games) {
            let g = initial_state();
            slots.push(Slot {
                game: g,
                tree: Tree::new(g, cfg.c_puct, next_seed),
                sims_done: 0,
                ply: 0,
                phase: Phase::AwaitingEval,
                records: Vec::new(),
                active: true,
                pending: None,
            });
            next_seed = next_seed.wrapping_add(1);
            launched += 1;
        }
        SelfPlayPool {
            slots,
            cfg,
            next_seed,
            launched,
            total_games,
            finished: 0,
            out_examples: Vec::new(),
            last_pending: Vec::new(),
        }
    }

    fn refill(&mut self, i: usize) {
        if self.launched < self.total_games {
            let g = initial_state();
            self.slots[i].game = g;
            self.slots[i].tree = Tree::new(g, self.cfg.c_puct, self.next_seed);
            self.slots[i].sims_done = 0;
            self.slots[i].ply = 0;
            self.slots[i].phase = Phase::AwaitingEval;
            self.slots[i].records.clear();
            self.slots[i].active = true;
            self.slots[i].pending = None;
            self.next_seed = self.next_seed.wrapping_add(1);
            self.launched += 1;
        } else {
            self.slots[i].active = false;
            self.slots[i].pending = None;
        }
    }

    /// Commit a ready move for slot i (records example, applies move, resets tree
    /// or finishes the game). Returns false if the game ended (slot to refill).
    fn commit_move(&mut self, i: usize) -> bool {
        let cfg = self.cfg;
        let slot = &mut self.slots[i];
        let temp = if slot.ply < cfg.temp_moves { 1.0 } else { 0.0 };
        let (mv, pi) = slot.tree.best_move(temp);
        let pre = slot.game;
        let mut planes = vec![0f32; 6 * 81];
        encode_planes(&pre, &mut planes);
        slot.records
            .push((planes, pi.to_vec(), pre.turn as usize, features(&pre)));
        let next = apply_move(&pre, &mv);
        slot.game = next;
        slot.ply += 1;
        if is_terminal(&next) || slot.ply >= cfg.max_plies {
            return false; // game over
        }
        slot.tree = Tree::new(next, cfg.c_puct, self.next_seed_for(i));
        slot.sims_done = 0;
        slot.phase = Phase::AwaitingEval;
        true
    }

    fn next_seed_for(&mut self, _i: usize) -> u64 {
        let s = self.next_seed;
        self.next_seed = self.next_seed.wrapping_add(1);
        s
    }

    /// Stamp z + plies_to_end onto a finished slot's records and move them out.
    fn finalize(&mut self, i: usize) {
        let w = winner(&self.slots[i].game); // None if capped at max_plies
        let n = self.slots[i].records.len();
        let recs = std::mem::take(&mut self.slots[i].records);
        for (k, (planes, pi, player, mut feats)) in recs.into_iter().enumerate() {
            let z = match w {
                None => 0.0,
                Some(win) => if win == player { 1.0 } else { -1.0 },
            };
            feats[3] = (n - k) as f32; // plies_to_end from this state
            self.out_examples.push(Example { planes, pi, z, feats });
        }
        self.finished += 1;
    }

    /// Advance every active slot: commit ready moves, then park one fresh leaf.
    /// Returns the parked planes as a flat (M*486) buffer plus M (row count);
    /// the caller reshapes to (M,6,9,9). M == 0 means nothing to evaluate.
    pub fn step(&mut self) -> (Vec<f32>, usize) {
        // 1) commit ready moves serially (mutates pool-level counters/examples)
        let n = self.slots.len();
        for i in 0..n {
            if !self.slots[i].active {
                continue;
            }
            if matches!(self.slots[i].phase, Phase::ReadyToMove) {
                let alive = self.commit_move(i);
                if !alive {
                    self.finalize(i);
                    self.refill(i);
                }
            }
        }
        // 2) park one leaf per active slot, in parallel (each slot owns its tree)
        let sims = self.cfg.sims;
        self.slots.par_iter_mut().for_each(|slot| {
            slot.pending = None;
            if !slot.active {
                return;
            }
            let mut buf = vec![0f32; 6 * 81];
            loop {
                match slot.tree.prepare_leaf(&mut buf) {
                    Leaf::Parked => {
                        slot.phase = Phase::AwaitingEval;
                        slot.pending = Some(Pending { planes: buf });
                        break;
                    }
                    Leaf::Terminal => {
                        slot.sims_done += 1;
                        if slot.sims_done >= sims {
                            slot.phase = Phase::ReadyToMove;
                            break; // no leaf this step; commit next step
                        }
                    }
                }
            }
        });
        // 3) assemble the batch in slot order
        self.last_pending.clear();
        let mut out = Vec::new();
        for i in 0..n {
            if let Some(p) = self.slots[i].pending.take() {
                out.extend_from_slice(&p.planes);
                self.last_pending.push(i);
            }
        }
        let m = self.last_pending.len();
        (out, m)
    }

    /// Apply the network results to the slots parked in the last `step()`.
    pub fn feed(&mut self, policy: &[f32], value: &[f32]) {
        let pending = self.last_pending.clone();
        let sims = self.cfg.sims;
        let (alpha, eps) = (self.cfg.dirichlet_alpha, self.cfg.dirichlet_eps);
        // collect per-row slices first (immutable borrow), then mutate slots
        for (row, &i) in pending.iter().enumerate() {
            let pol = &policy[row * 140..row * 140 + 140];
            let v = value[row] as f64;
            let slot = &mut self.slots[i];
            slot.tree.receive(pol, v);
            // root just expanded on the first eval of this move → add noise once
            if slot.sims_done == 0 && alpha > 0.0 {
                slot.tree.apply_root_noise(alpha, eps);
            }
            slot.sims_done += 1;
            if slot.sims_done >= sims {
                slot.phase = Phase::ReadyToMove;
            }
        }
    }

    pub fn drain(&mut self) -> Vec<Example> {
        std::mem::take(&mut self.out_examples)
    }

    pub fn games_remaining(&self) -> u32 {
        self.total_games - self.finished
    }

    pub fn active(&self) -> usize {
        self.slots.iter().filter(|s| s.active).count()
    }
}
```

> **Implementer notes (correctness, not optional):**
> 1. `feed` runs serially because it touches pool-level `last_pending`; the
>    expensive part (`prepare_leaf` descent) is already parallelized in `step`.
>    `receive` per slot is cheap (one expansion). If profiling later shows `feed`
>    hot, parallelize the per-slot `receive` with rayon (slots are disjoint) and
>    move the counter/phase updates after the join.
> 2. The Dirichlet-noise trigger uses `slot.sims_done == 0` meaning "the eval we
>    just fed was the root expansion (the move's first eval)". `apply_root_noise`
>    is idempotent (guarded by `self.noised`), so a double-call is harmless.
> 3. `commit_move` returning `false` covers both terminal and `max_plies` cap;
>    `finalize` reads `winner` (None ⇒ draw ⇒ z=0) so the cap path yields z=0.

- [ ] **Step 4: Add the `SelfPlayPool` pyclass to `native/src/pyiface.rs`**

Add imports:

```rust
use crate::selfplay::{Config, SelfPlayPool as CorePool};
use numpy::{PyArray4, PyReadonlyArray1, PyReadonlyArray2};
```

Add the pyclass:

```rust
#[pyclass]
pub struct SelfPlayPool {
    inner: CorePool,
}

#[pymethods]
impl SelfPlayPool {
    #[new]
    #[pyo3(signature = (n_games, total_games, sims, c_puct=1.5, seed=0,
                        dirichlet_alpha=0.5, dirichlet_eps=0.25,
                        temp_moves=10, max_plies=200))]
    fn new(
        n_games: u32,
        total_games: u32,
        sims: u32,
        c_puct: f64,
        seed: u64,
        dirichlet_alpha: f64,
        dirichlet_eps: f64,
        temp_moves: u32,
        max_plies: u32,
    ) -> SelfPlayPool {
        let cfg = Config { sims, c_puct, dirichlet_alpha, dirichlet_eps, temp_moves, max_plies };
        SelfPlayPool { inner: CorePool::new(n_games, total_games, cfg, seed) }
    }

    /// Returns the parked-leaf batch as (M,6,9,9) float32, or None if M == 0.
    fn step<'py>(&mut self, py: Python<'py>) -> Option<Bound<'py, PyArray4<f32>>> {
        let (buf, m) = py.detach(|| self.inner.step());
        if m == 0 {
            return None;
        }
        let arr = numpy::ndarray::Array4::from_shape_vec((m, 6, 9, 9), buf).unwrap();
        Some(arr.into_pyarray(py))
    }

    fn feed(&mut self, policy: PyReadonlyArray2<f32>, value: PyReadonlyArray1<f32>) -> PyResult<()> {
        let pol = policy.as_slice()?;
        let val = value.as_slice()?;
        self.inner.feed(pol, val);
        Ok(())
    }

    /// Drain finished examples as (planes(6,9,9), pi(140), z, feats(4)) tuples.
    fn drain<'py>(&mut self, py: Python<'py>) -> PyResult<Vec<PyObject>> {
        let mut out = Vec::new();
        for ex in self.inner.drain() {
            let planes = numpy::ndarray::Array3::from_shape_vec((6, 9, 9), ex.planes)
                .unwrap()
                .into_pyarray(py);
            let pi = numpy::ndarray::Array1::from_vec(ex.pi).into_pyarray(py);
            let feats = numpy::ndarray::Array1::from_vec(ex.feats.to_vec()).into_pyarray(py);
            out.push((planes, pi, ex.z, feats).into_pyobject(py)?.into_any().unbind());
        }
        Ok(out)
    }

    fn games_remaining(&self) -> u32 {
        self.inner.games_remaining()
    }

    fn active(&self) -> usize {
        self.inner.active()
    }
}
```

Add to `register`:

```rust
    m.add_class::<SelfPlayPool>()?;
```

- [ ] **Step 5: Add `mod selfplay;` to `native/src/lib.rs`**

Insert `mod selfplay;` in the module list.

- [ ] **Step 6: Build and run the pool smoke test**

```bash
source .venv/bin/activate && maturin develop -m native/Cargo.toml -q && python -m pytest tests/test_native_pool.py -q
```
Expected: PASS (`1 passed`). If it hangs, the `guard` cap will trip the assertion — check the `step` terminal-loop never spins forever (a slot that reaches `ReadyToMove` must break out, not loop).

- [ ] **Step 7: Run the whole suite (no regressions)**

```bash
source .venv/bin/activate && python -m pytest -q
```
Expected: all green (218 = 212 baseline + 6 new across tasks 1–4).

- [ ] **Step 8: Commit**

```bash
git add native/src tests/test_native_pool.py
git commit -m "feat(native): SelfPlayPool batched stepper (step/feed/drain + features)"
```

---

## Task 5: Self-play driver + throughput benchmark (the gate)

**Files:**
- Create: `scripts/selfplay_native.py`, `scripts/bench_selfplay.py`

- [ ] **Step 1: Implement `scripts/selfplay_native.py`**

```python
"""Batched-MPS self-play driver: Rust SelfPlayPool ↔ PyTorch net on MPS.

Usage: python scripts/selfplay_native.py [total_games] [n_games] [sims] [device]
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

import barricades_native as bn
from agents.az.model import QuoridorNet


def run_selfplay(total_games=256, n_games=256, sims=100, device="mps",
                 channels=32, blocks=3, ckpt=None, seed=0):
    net = QuoridorNet(channels=channels, blocks=blocks)
    if ckpt and os.path.exists(ckpt):
        net.load_state_dict(torch.load(ckpt, map_location="cpu")["model"])
    net = net.to(device).eval()

    pool = bn.SelfPlayPool(n_games=n_games, total_games=total_games, sims=sims,
                           seed=seed)
    examples, batches, batch_pos = [], 0, 0
    t0 = time.perf_counter()
    while pool.games_remaining() > 0:
        planes = pool.step()
        if planes is None:
            continue
        x = torch.from_numpy(np.asarray(planes)).to(device)
        with torch.no_grad():
            logits, value = net(x)
            policy = torch.softmax(logits, dim=1).cpu().numpy()
            value = value.squeeze(1).cpu().numpy()
        pool.feed(np.ascontiguousarray(policy, dtype=np.float32),
                  np.ascontiguousarray(value, dtype=np.float32))
        examples.extend(pool.drain())
        batches += 1
        batch_pos += x.shape[0]
    dt = time.perf_counter() - t0
    return examples, dict(games=total_games, seconds=dt, batches=batches,
                          mean_batch=batch_pos / max(batches, 1),
                          games_per_sec=total_games / dt,
                          examples=len(examples))


if __name__ == "__main__":
    total = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    ngames = int(sys.argv[2]) if len(sys.argv) > 2 else 256
    sims = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    device = sys.argv[4] if len(sys.argv) > 4 else "mps"
    _, stats = run_selfplay(total, ngames, sims, device)
    print(stats)
```

- [ ] **Step 2: Smoke-run the driver on CPU (fast, no MPS dependency)**

```bash
source .venv/bin/activate && python scripts/selfplay_native.py 8 8 16 cpu
```
Expected: prints a stats dict with `games=8`, `examples>0`, `mean_batch` between 1 and 8, no errors.

- [ ] **Step 3: Implement `scripts/bench_selfplay.py`**

```python
"""Benchmark native batched-MPS self-play throughput and project the 100k run.

Usage: python scripts/bench_selfplay.py [sims]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from scripts.selfplay_native import run_selfplay


def main():
    sims = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device={dev} sims={sims}")
    # warm up MPS / caches with a tiny run
    run_selfplay(total_games=16, n_games=16, sims=sims, device=dev)
    # measured run
    _, st = run_selfplay(total_games=512, n_games=256, sims=sims, device=dev)
    gps = st["games_per_sec"]
    proj = 100_000 / gps / 3600.0
    print(f"  games/sec={gps:.1f}  mean_batch={st['mean_batch']:.0f}  "
          f"batches={st['batches']}  examples={st['examples']}")
    print(f"  ==> projected 100k games: {proj:.2f} hours")
    if st["mean_batch"] < 128:
        print("  WARNING: mean batch < 128 — MPS underfed; raise n_games.")
    if proj > 2.0:
        print("  GATE: projection > 2h. Lever options before the 100k run:")
        print("   - lower sims, raise n_games, or add subtree carryover (see spec).")
    else:
        print("  GATE PASSED: projection <= 2h. Cleared to launch the 100k campaign.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the benchmark**

```bash
source .venv/bin/activate && maturin develop -m native/Cargo.toml -r -q && python scripts/bench_selfplay.py 100
```
Expected: prints `games/sec`, `mean_batch` (should be ≥128 with n_games=256), and the projected 100k hours with a GATE line. **This is the decision gate.** Record the numbers. Do NOT launch a 100k run from this plan — that is a separate, explicitly-authorized campaign (see the spec's "later spec/plan").

- [ ] **Step 5: Commit**

```bash
git add scripts/selfplay_native.py scripts/bench_selfplay.py
git commit -m "feat(native): batched-MPS self-play driver + throughput benchmark (gate)"
```

- [ ] **Step 6: Report the benchmark result**

Summarize for the user: device, sims, games/sec, mean batch size, projected 100k wall-clock, and whether the gate passed. If `mean_batch < 128` or projection `> 2h`, recommend the specific lever (raise `n_games`, lower `sims`, or implement subtree carryover) rather than launching.

---

## Self-Review

**1. Spec coverage**

| Spec element | Task |
|---|---|
| Bitboard game core (BFS, movegen, apply) | Task 1 |
| Differential testing against Python core | Tasks 1 (game) + 2 (encoding) |
| 6×9×9 planes + 140-action encoding parity | Task 2 |
| PUCT MCTS (select/expand/backup), heuristic + net eval | Task 3 |
| `RustMctsAgent` for UI/tournaments | Task 3 (`NativeMctsAgent`) |
| Dirichlet root noise + temperature | Task 3 (`apply_root_noise`, `best_move` temp) |
| Python-driven stepper (`step`/`feed`/`drain`) | Task 4 |
| Rayon, GIL released | Task 4 (`par_iter_mut` inside `py.detach`) |
| One-leaf-per-game batching (v1) | Task 4 (`step` parks one leaf per slot) |
| Reward-signal features recorded (path_diff, walls, plies_to_end) | Task 4 (`features`, stamped in `finalize`) |
| Self-play driver ↔ MPS | Task 5 |
| Benchmark + gate before 100k | Task 5 |
| Subtree carryover | **Deferred** (out-of-scope below) — `commit_move` rebuilds the tree per move (matches Python's known-correct per-move tree); carryover is the first lever if the benchmark misses ≤2h. |
| Virtual loss | Deferred (spec out-of-scope) |

**2. Placeholder scan:** none. The only intentional "delete this" marker is the bogus `children_capacity_hint: ()` field in `Tree::new`, explicitly flagged with a note to remove it.

**3. Type consistency:** state tuple `((c,r),(c,r), h, v, (n,n), turn)` and move tuples `("step",c,r)` / `("wall",c,r,"H"/"V")` are used identically across the tests, `pyiface` marshaling, and `native_agent.py`. `GameState`/`Move` field names match between `state.rs`, `bitboard.rs`, `movegen.rs`, `encoding.rs`, `mcts.rs`, `selfplay.rs`. `prepare_leaf(&mut [f32])`/`receive(&[f32], f64)`/`best_move(f64)->(Move,[f32;140])` signatures are consistent between `mcts.rs` and both `Tree` and `SelfPlayPool` callers. Reward features are a fixed 4-vector everywhere (`N_FEATS = 4`).

---

## Out of scope (deferred to a later spec/plan)

- **Subtree carryover** between moves (the per-move tree is rebuilt; carryover is the first throughput lever if the benchmark misses ≤2h — it reduces eval count, not batch size, so v1 throughput shape is unaffected).
- **Virtual loss / multi-leaf-per-tree** batching.
- **The 100k self-play campaign** and **the reward-signal training experiments** (eval-blended / length-discounted / aux distance head) — the features are recorded here; consuming them in `train.py` is the next spec.
- **Wiring `NativeMctsAgent` into the server registry / web UI** (the agent class exists and is tournament-usable; registry wiring is a small follow-up).
- **Distributed / multi-process** self-play.
