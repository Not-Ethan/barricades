# Rust Native Core — Batched-MPS Self-Play Engine

**Date:** 2026-06-08
**Status:** Approved (design phase)
**Branch:** `rust-core`
**Crate:** `barricades_native` (PyO3 0.28.3, maturin, edition 2024)

## Overview

Port the throughput-critical parts of the Quoridor engine — game logic, BFS,
and the PUCT MCTS tree — from Python into a Rust extension module, then drive
**batched AlphaZero self-play** where Rust does all the CPU work (multithreaded,
GIL released) and Python/PyTorch does only the neural-net forward pass on MPS.

**Goal:** generate ~100k self-play games in **1–2 hours** on this Mac (Apple
GPU via MPS), and provide a fast native MCTS usable by the web UI and
tournaments — without re-implementing the rules a third time incorrectly.

### Why this shape (the throughput logic)

Profiling the existing Python self-play showed the cost split as roughly:

| Component        | Share | Where it goes |
|------------------|-------|---------------|
| BFS game logic   | ~65%  | Rust          |
| Tree + apply     | ~22%  | Rust          |
| NN forward       | ~13%  | Python / MPS  |

So **87% of the work is CPU game/tree logic.** Rust's job is to crush that
until the NN forward becomes the binding constraint, then keep MPS saturated.

MPS inference (measured, `scripts/bench_mps.py`, 32ch/3blk net):

| batch | CPU pos/s | MPS pos/s | speedup |
|-------|-----------|-----------|---------|
| 1     | 1,692     | 1,009     | 0.6×    |
| 32    | 5,616     | 10,066    | 1.8×    |
| 128   | 9,294     | 69,319    | **7.5×**|
| 512+  | ~9,000    | ~68,000   | ~7.5×   |

The cliff at batch≥128 is the whole ballgame: **MPS only wins when the batch is
large.** Batch=1 is *slower* than CPU. Therefore the architecture must always
present MPS with ≥128 positions at once.

### Throughput budget for 100k games

- 100k games × ~60 plies × 100 sims ≈ **600M leaf evals** (worst case).
- At 68k pos/s sustained ⇒ ~2.4 hr if every sim is a unique forward pass.
- Subtree carryover between moves + leaf dedup typically removes 30–50% of
  unique evals, and early-game sims can be fewer ⇒ realistically **~1.2–1.7 hr.**

The 1–2 hr target is reachable *only if* (a) Rust removes the CPU bottleneck and
(b) batches stay ≥128. Both fall out of the design below. The big run uses the
**32ch/3blk** net; 64ch drops MPS to ~17k pos/s (~10 hr), which blows the target.

## Architecture: Python-driven stepper (Approach A)

Python owns the self-play loop. Rust owns a pool of concurrent games and all
CPU compute. PyTorch never leaves Python.

```
Python driver                          Rust (barricades_native)
─────────────                          ─────────────────────────
load net on MPS                        SelfPlayPool { N games, each: tree+root }
loop:
  planes = pool.step()        ───────▶ advance every game's MCTS until each has
                                        a pending leaf; collect encoded planes
                              ◀─────── return (M,6,9,9) float32  (M ≤ N)
  logits,val = net(planes→mps)
  policy = softmax(logits)
  pool.feed(policy, value)    ───────▶ expand priors, backup value, advance trees;
                                        on completed move record (state, π, player);
                                        on game end record outcome z
  for ex in pool.drain():     ◀─────── pull finished (planes, π, z, features)
    replay.add(ex)
until pool.games_remaining()==0
```

**Why this and not the alternatives:**

- **A (chosen) — Python-driven stepper.** PyTorch stays in its native habitat;
  batch = number of active games (≥128 for free); one big zero-copy numpy array
  per boundary crossing; Rust runs the tree/BFS work across all cores with the
  GIL released (`Python::detach` / `allow_threads`).
- **B — Rust-driven with a PyO3 callback into Python for eval.** Inverts
  control, juggles the GIL on every batch, makes batching awkward. Rejected.
- **C — Full Rust, NN inference in Rust (tch-rs/candle/ort).** Highest ceiling,
  but tch-rs-on-MPS is immature, reimplements the net, and can't cleanly reuse
  the trained PyTorch checkpoints. Too risky now; revisit only if
  MPS-in-Python proves to be the wall.

### Batching policy (v1)

**One pending leaf per game per tick.** With N=256 active games, every `step()`
naturally yields a batch of up to 256 leaves — comfortably past the 128 cliff,
with zero virtual-loss bias. Virtual loss (multiple in-flight leaves per tree
per tick, to push the batch even larger) is a **noted future lever**, not v1.

As games finish, the active count shrinks and the batch with it; when it would
drop below an efficient size, the pool refills from the remaining game budget so
the batch stays large until the very end of the run.

## Rust crate layout

```
native/
  Cargo.toml            pyo3, rayon, numpy (rust-numpy), (smallvec)
  src/
    lib.rs              #[pymodule] — exports + registration
    coords.rs           cell <-> index, goal rows, adjacency
    bitboard.rs         u128/[u64;2] board masks; flood-fill BFS
    state.rs            GameState (Copy), apply_move, terminal/winner
    movegen.rs          legal_steps (jumps), legal_walls (path guard)
    mcts.rs             PUCT node arena, select/expand/backup, carryover
    encoding.rs         encode_planes (6×9×9), move<->action (140), parity
    selfplay.rs         SelfPlayPool: step/feed/drain, rayon, GIL release
    agent.rs            RustMctsAgent: single-position search for UI/tournaments
    pyiface.rs          #[pyclass]/#[pyfunction] wrappers, ndarray marshaling
```

Mirrors the Python `core`/`agents/az` modules name-for-name so the
differential tests map one-to-one.

### Game core (`bitboard.rs`, `state.rs`, `movegen.rs`, `coords.rs`)

- `GameState` is a small **`Copy`** struct (no heap): two pawn cell indices
  (`u8`), wall masks, two `walls_left` counts (`u8`), `turn` (`u8`). Cheap to
  clone in search — replaces Python's frozen-dataclass copy-on-move.
- Walls stored as bitmasks over the 8×8 slot grid (64 h-slots, 64 v-slots), so
  overlap/cross checks and the BFS "is this edge blocked" test are bit ops.
- **BFS** (`bfs_dist`) is the hot function: bitwise flood-fill from a pawn to its
  goal row, mirroring the existing Python `core/bitboard.py` algorithm
  (`_can_move_masks` / `_expand`). Returns distance or "no path".
- `legal_steps` handles adjacency, wall-blocking, straight jump over the
  opponent, and diagonal jump when a wall/edge sits behind the opponent —
  identical semantics to `core/rules.py::legal_steps`.
- `legal_walls` enforces in-bounds, no overlap/cross, walls-left > 0, **and the
  path-existence guard** (both pawns must still reach their goal) via two BFS
  calls. Matches `core/rules.py::legal_walls`.

These are the four functions that must be **byte-identical** to Python; they get
dedicated differential tests.

### Encoding (`encoding.rs`)

Exactly reproduces `agents/az/encoding.py`:

- **6 planes × 9×9** input encoding (`encode_planes`), with the same
  current-player-relative orientation/parity convention.
- **140-action canonical move encoding** (`move_to_action` / `action_to_move`):
  steps + jumps + horizontal walls + vertical walls, same index layout.

A round-trip differential test (`Python.encode == Rust.encode`,
`move_to_action` agree) guarantees the policy head's action space is shared, so
a net trained against Python encoding works unchanged with the Rust pool.

### MCTS (`mcts.rs`)

- Node **arena** (`Vec<Node>`, indices not pointers): per node `N, W, Q`,
  child `(action, prior, child_idx)` edges, expanded flag.
- **PUCT selection** with the standard `Q + c_puct · P · √ΣN / (1+N)`.
- **Expansion** consumes a net `(policy, value)` for the leaf: legal-move mask
  applied to the policy, priors normalized over legal actions, value backed up.
- **Backup** along the selection path, alternating sign by player.
- **Subtree carryover**: after a move is chosen, the chosen child becomes the new
  root and its subtree is retained (re-rooting the arena), so visits aren't
  thrown away between plies — a major cut to unique evals.
- **Dirichlet root noise** + temperature on the root visit distribution for
  self-play exploration (config: `dirichlet_alpha`, `dirichlet_eps`,
  `temp`, `temp_moves`), matching `agents/az/mcts_nn.py`.

### SelfPlayPool (`selfplay.rs`) — the stepper

State per slot: a game (`GameState` plus the per-move example records awaiting a
`z` stamp), its MCTS tree, sims done this move, and a phase
(`ready-to-move` → `awaiting-eval`).

Clean split of responsibilities: **`feed` only absorbs eval results; `step` only
advances game state and produces leaves.** A move is committed exactly once, at
the top of `step`, never inside `feed`.

- **`step() -> ndarray (M,6,9,9) f32`**: under `py.detach(...)` (GIL released),
  rayon-parallel over slots. For each slot:
  1. If the slot is `ready-to-move` (its sim budget for the current move is
     satisfied), **commit the move**: pick it from the temperature-weighted root
     visits, record the training example (planes, visit-count π over 140
     actions, the player to move, and the heuristic features below), apply the
     move, carry the subtree over. If that move ends the game, stamp the outcome
     `z` onto all of that game's pending example records and mark the slot for
     drain + refill, then skip to the next slot.
  2. Run PUCT selection down to a leaf, park at that leaf, and write its encoded
     planes into a shared output buffer.

  `M` = number of slots that parked at a fresh leaf this tick. Returns the buffer
  to Python as a zero-copy numpy array.
- **`feed(policy: ndarray (M,140) f32, value: ndarray (M,) f32)`**: maps each row
  back to its parked slot, applies the legal-move mask to the policy, expands the
  leaf with normalized priors, backs the value up the path, and increments that
  slot's sim counter. When the counter reaches `sims`, it flips the slot to
  `ready-to-move` (the actual move is committed next `step`).
- **`drain() -> list[Example]`**: hands finished examples to Python:
  `(planes f32[6,9,9], pi f32[140], z f32, features f32[K])`.
- **`games_remaining() / active() / stats()`** for the driver loop and progress.

Concurrency: rayon parallelizes the per-slot CPU work inside `step`/`feed` with
the GIL released. The only Python-touching work is allocating/filling the numpy
buffers, done once per tick on contiguous memory.

### RustMctsAgent (`agent.rs`)

A single-position search for the **web UI and tournaments**: `search(state,
sims) -> (best_move, visit_policy, value, stats)`. Internally one tree + the same
PUCT code; the net eval is supplied by Python (same stepper trick, batch=1, or a
provided callable) — or by the built-in heuristic for a net-free fast bot. This
also gives the differential tests a second consumer of the Rust core and a
faster MCTS for the existing arena.

## Pluggable reward signals (cheap to experiment)

The user explicitly wants alternative reward signals to speed AZ convergence.
Design principle: **self-play records raw signals; Python forms the value target
at training time.** So every reward-signal variant is a training-config change,
**never** a self-play re-run.

Each example carries, besides `z` (game outcome in {−1,0,+1} from the recorded
player's POV) and `pi`, a small `features` vector that Rust computes essentially
for free during BFS:

- `path_diff` — own shortest-path minus opponent shortest-path (the core eval
  signal),
- `walls_left_own`, `walls_left_opp`,
- `plies_to_end` — distance (in plies) from this state to the game's terminal.

Python's training step can then build the value target as any of:

- **baseline** — pure `z`;
- **eval-blended** — `λ·z + (1−λ)·tanh(path_diff / s)` (anneal `λ→1`), giving the
  net a dense early signal before outcomes are informative;
- **length-discounted** — `z · γ^plies_to_end`, valuing faster wins;
- **auxiliary distance head** — add a small head predicting `path_diff`, trained
  alongside value (multi-task), to regularize the trunk.

The features are recorded once; the experiments are config flags in `train.py`.

## Testing strategy — differential against the Python core

The Rust core is **not trusted until it provably matches Python.** Two layers:

1. **Ported unit tests** (`tests/` Python, calling the Rust module): the
   existing `core`/encoding test cases — every jump variant, wall
   overlap/cross, path-blocking scenario, terminal/winner, the 140-action
   round-trip — re-run against `barricades_native` and must pass identically.

2. **Fuzz / property differential** (`tests/test_native_diff.py`): play
   thousands of random games; at **every** position assert Rust and Python agree
   exactly on:
   - `legal_moves` (as a set),
   - `shortest_path_len` for both players,
   - `is_blocked` for sampled edges,
   - `apply_move` result state,
   - `encode_planes` (array-equal) and `move_to_action` for all legal moves.

   Any mismatch fails with the exact position for debugging.

3. **MCTS sanity** (not differential — MCTS is stochastic): with a fixed net and
   seed, the Rust search returns a legal move, takes an immediate win when one
   exists, and its visit distribution concentrates on sensible moves on crafted
   positions.

4. **End-to-end smoke**: a tiny self-play run (e.g. 4 games, 16 sims) produces
   well-formed examples (planes shape, π sums to 1, z ∈ {−1,0,+1}) and the
   driver loop completes.

Only after layers 1–2 are green do we benchmark or run anything long.

## Build order (each step independently testable)

1. **Game core** — `coords`, `bitboard` (BFS), `state`, `movegen`; differential
   tests green (legal moves, BFS, apply, is_blocked).
2. **Encoding** — `encode_planes` + 140-action map; round-trip differential green.
3. **MCTS** — `mcts.rs` + `RustMctsAgent`; sanity tests green; sanity-check it in
   the existing arena vs Python MCTS (similar strength at equal sims).
4. **SelfPlayPool** — `selfplay.rs` stepper; end-to-end smoke green.
5. **Python self-play driver** (`scripts/selfplay_native.py`) wiring pool↔MPS.
6. **Benchmark** (`scripts/bench_selfplay.py`) — measure games/sec and mean MPS
   batch size; confirm batch≥128 and project the 100k wall-clock. **Gate: only
   launch the 100k run if the projection lands ≤2 hr.**
7. **(Later, separate spec/plan)** reward-signal experiments + the 100k campaign
   + benchmark vs the existing bot pool.

## Out of scope (YAGNI for now)

- NN inference in Rust (Approach C) — Python/MPS owns the forward pass.
- Virtual loss / multi-leaf-per-tree batching — v1 uses one leaf per game.
- Distributed / multi-process self-play — single process, rayon threads.
- Re-training pipeline changes beyond the pluggable value target hook.
- Replacing the Python `core` — it stays as the reference oracle for the
  differential tests; the web server keeps using it unless/until we wire the
  Rust agent in behind the existing `Agent` interface.
