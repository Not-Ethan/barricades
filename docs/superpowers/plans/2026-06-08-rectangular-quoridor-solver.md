# Rectangular Quoridor Solver — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax. Each task is TDD: the provided test is the **contract** — write it first, watch it fail, implement until it passes, then run the full crate test suite.

**Goal:** A fresh, standalone Rust crate that *exactly solves* rectangular Quoridor, validated against the reference writeup's published results, then used to measure the cost of solving 6×5 / 7×5 / 7×7 and produce a firm RunPod estimate.

**Architecture:** New crate `solver/` (`quoridor_solver`) — a `lib` + a `bin`. `u64` bitboards, **runtime** board dimensions (`Board { w, h, walls }` precomputes masks). Rules mirror the already-validated `smallboard` Python engine (itself differential-tested vs the production `core`). Solver = iterative-deepening negamax + alpha-beta + transposition table + move ordering, with a retrograde fallback for draws. Correctness-first; performance levers (df-pn, const-generic monomorphization, GPU tablebase) are Phase-1 experiments.

**Tech Stack:** Rust 1.92 (edition 2024), `cargo`. No PyO3 (standalone solver). Spec: `docs/superpowers/specs/2026-06-08-rectangular-quoridor-solver-design.md`.

---

## Conventions

- **Coordinates:** cell `(c, r)`, `c ∈ 0..w`, `r ∈ 0..h`. Bit index `idx(c,r) = r*w + c`; cell bitboard `1u64 << idx`. Player 0 starts `(w/2, 0)`, goal row `h-1`; player 1 starts `(w/2, h-1)`, goal row `0`.
- **Wall anchors:** `(wc, wr)`, `wc ∈ 0..w-1`, `wr ∈ 0..h-1` (so `(w-1)*(h-1)` anchors per orientation). Anchor bit index `wr*(w-1)+wc`. A horizontal wall at `(wc,wr)` sits between rows `wr` and `wr+1` spanning columns `wc,wc+1`. A vertical wall at `(wc,wr)` sits between columns `wc,wc+1` spanning rows `wr,wr+1`.
- **Blocking** (mirror `smallboard/engine.py::is_blocked`): N move `(ax,ay)→(ax,ay+1)` blocked iff h-wall at `(ax,ay)` or `(ax-1,ay)`. S move blocked iff h-wall at `(ax,ay-1)` or `(ax-1,ay-1)`. E move `(ax,ay)→(ax+1,ay)` blocked iff v-wall at `(ax,ay)` or `(ax,ay-1)`. W move blocked iff v-wall at `(ax-1,ay)` or `(ax-1,ay-1)`.
- **Wall overlap/cross** (mirror `smallboard/engine.py::_overlaps`): H-wall `(c,r)` illegal if any of `(c,r),(c-1,r),(c+1,r)` already in h-walls, or `(c,r)` in v-walls. V-wall `(c,r)` illegal if any of `(c,r),(c,r-1),(c,r+1)` in v-walls, or `(c,r)` in h-walls.
- **Run/test** (from repo root): `cd solver && cargo test` (debug) and `cargo test --release` for the slow validation tests. Build the CLI: `cargo build --release --bin solve`.
- **The oracle:** `smallboard/engine.py` for rules (square boards) and the writeup's published values for solver outcomes. A subagent must NOT weaken any validation target to make a test pass.

---

## Task 1: Crate scaffold + `Board` + `State`

**Files:** Create `solver/Cargo.toml`, `solver/src/lib.rs`, `solver/src/board.rs`, `solver/src/state.rs`.

- [ ] **Step 1: `solver/Cargo.toml`**
```toml
[package]
name = "quoridor_solver"
version = "0.1.0"
edition = "2024"

[lib]
name = "quoridor_solver"
path = "src/lib.rs"

[[bin]]
name = "solve"
path = "src/bin/solve.rs"

[dependencies]
rustc-hash = "2"            # fast FxHashMap for the transposition table

[profile.release]
opt-level = 3
lto = true
codegen-units = 1
```

- [ ] **Step 2: Write the failing test** in `solver/src/state.rs` (`#[cfg(test)]`):
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Board;

    #[test]
    fn initial_state_5x5() {
        let b = Board::new(5, 5, 3);
        let s = b.initial();
        assert_eq!(s.pawn[0], b.idx(2, 0));
        assert_eq!(s.pawn[1], b.idx(2, 4));
        assert_eq!(s.walls_left, [3, 3]);
        assert_eq!(s.turn, 0);
        assert!(!b.is_terminal(&s));
        assert!(b.winner(&s).is_none());
    }

    #[test]
    fn winner_on_goal_row() {
        let b = Board::new(3, 3, 1);
        let mut s = b.initial();
        s.pawn[0] = b.idx(1, 2);           // player 0 on its goal row (h-1=2)
        assert_eq!(b.winner(&s), Some(0));
        assert!(b.is_terminal(&s));
    }
}
```

- [ ] **Step 3: Implement `board.rs` + `state.rs`.** `State` is `Copy`:
```rust
// state.rs
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct State {
    pub pawn: [u8; 2],       // cell indices
    pub h_walls: u64,        // anchor bitset
    pub v_walls: u64,
    pub walls_left: [u8; 2],
    pub turn: u8,            // 0 or 1
}
```
`Board` holds `w, h, walls` and precomputed masks (cell `FULL` mask, per-player goal-row masks). Provide: `new(w,h,walls)`, `idx(c,r)->u8`, `cr(idx)->(u8,u8)`, `goal_row(player)->u8`, `initial()->State`, `winner(&State)->Option<u8>` (player p wins iff `cr(pawn[p]).1 == goal_row(p)`), `is_terminal`. Wall-anchor index helpers `hbit(wc,wr)`, `vbit(wc,wr)` and accessors `has_h/has_v`.

- [ ] **Step 4: `lib.rs`** declares `pub mod board; pub mod state;` (add later modules as created).

- [ ] **Step 5: Run** `cd solver && cargo test` → 2 passed. **Step 6: Commit** (`feat(solver): crate scaffold + Board/State`).

---

## Task 2: Bitboard BFS — reachability + distance

**Files:** Create `solver/src/bitboard.rs`. The floodfill that powers wall-legality and move ordering.

- [ ] **Step 1: Failing test** (`bitboard.rs` tests):
```rust
#[cfg(test)]
mod tests {
    use crate::board::Board;
    use crate::state::State;

    #[test]
    fn open_board_distance_5x5() {
        let b = Board::new(5, 5, 3);
        let s = b.initial();
        // player 0 at row 0, goal row 4 -> distance 4 on an empty board
        assert_eq!(b.dist_to_goal(&s, 0), Some(4));
        assert_eq!(b.dist_to_goal(&s, 1), Some(4));
    }

    #[test]
    fn wall_lengthens_path() {
        let b = Board::new(3, 3, 1);
        let mut s = b.initial();             // p0 at (1,0), goal row 2, dist 2
        assert_eq!(b.dist_to_goal(&s, 0), Some(2));
        // box player 0 in partially; path still exists but is longer or equal
        s.h_walls |= 1 << b.hbit(0, 0);      // wall between rows 0,1 at cols 0,1
        assert!(b.has_path(&s, 0));          // must still have a path (going right)
    }
}
```

- [ ] **Step 2: Implement** `dist_to_goal(&State, player) -> Option<u32>` (BFS distance from pawn to goal row, `None` if unreachable) and `has_path(&State, player) -> bool`. Use a bitboard frontier expand: neighbors via `<<w` (N), `>>w` (S), `<<1`/`>>1` (E/W with column-edge masks to stop wrap), each masked by the not-blocked condition derived from `h_walls`/`v_walls`. Compute blocked-edge masks per direction from the wall bitsets (vectorized: e.g. an N-move from row r is blocked by h-walls; build the "north-blocked cells" mask once). A correct, simple version: BFS over individual cells using the `is_blocked` predicate from the Conventions; optimize to bitboard-frontier only after the test passes. **Step 3: Run** → pass. **Step 4: Commit** (`feat(solver): bitboard BFS reachability + distance`).

---

## Task 3: Move generation (steps + jumps + walls + floating-wall fast-path)

**Files:** Create `solver/src/movegen.rs`. Add `Move` enum to `state.rs`.

```rust
// state.rs
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Move {
    Step(u8),                 // destination cell index
    Wall { wc: u8, wr: u8, horiz: bool },
}
```

- [ ] **Step 1: Failing tests** — (a) opening step count on 3×3 matches `smallboard`; (b) **fast-path soundness**: `legal_walls` with the floating-wall fast-path equals the brute-force version (always BFS-check) over random play; (c) a generated **differential fixture** vs `smallboard` (see Step 4).
```rust
#[cfg(test)]
mod tests {
    use crate::board::Board;
    #[test]
    fn fast_path_matches_bruteforce_5x5() {
        let b = Board::new(5, 5, 3);
        // random playout; at each node, legal_walls (fast-path) must equal
        // legal_walls_bruteforce (always path-check). Seeded LCG, 40 games x 30 plies.
        // assert set equality at every node, >1000 nodes checked.
        assert!(crate::movegen::selfcheck_fast_path(&b, 123) > 1000);
    }
}
```

- [ ] **Step 2: Implement** in `movegen.rs`:
  - `legal_steps(&Board,&State) -> Vec<u8>` — orthogonal moves with jump rules, mirroring `smallboard/engine.py::legal_steps` exactly (straight jump over opponent; if landing blocked/off-board, the two diagonal jumps).
  - `apply(&Board,&State,Move) -> State` — step moves pawn + flips turn; wall sets the anchor bit, decrements `walls_left`, flips turn. Mirror `smallboard::apply_move`.
  - `legal_walls(&Board,&State) -> Vec<Move>` — for each orientation/anchor with `walls_left[turn]>0`, skip if `_overlaps`; then if `needs_path_check(wall)` run `has_path` for both players, else accept. `needs_path_check` = the wall touches the board boundary OR shares an endpoint post with an existing wall (the writeup's "edge or ≥2 contact points" condition; specify precisely and conservatively — when in doubt, check).
  - `legal_walls_bruteforce` — same but always path-checks (no fast-path).
  - `legal_moves` = steps + walls. `selfcheck_fast_path(&Board, seed) -> usize` — the random-playout differential used by the test.

- [ ] **Step 3: Run** → pass.

- [ ] **Step 4: Cross-language rules differential (belt-and-suspenders).** Add `solver/tests/diff_vs_smallboard.rs`: load `solver/tests/fixtures/smallboard_5x5.json` (a list of `{state, legal_moves}` records) and assert the Rust `legal_moves` set equals the recorded set per position. Generate the fixture with a committed helper script `scripts/gen_solver_fixture.py` (walks random `smallboard` games at 3×3 and 5×5, dumps canonical state + sorted move keys to JSON). Commit the script AND the fixture. **Step 5: Run** `cargo test` → all pass. **Step 6: Commit** (`feat(solver): movegen (jumps + walls + floating-wall fast-path) + smallboard differential`).

---

## Task 4: Exact solver — ID negamax + alpha-beta + TT + ordering + retrograde fallback

**Files:** Create `solver/src/solver.rs`, `solver/src/endgame.rs`.

- [ ] **Step 1: Failing tests** — solver vs an independent in-crate brute-force negamax on 3×3 (value + that the start is a P2 win), and the no-walls race base case.
```rust
#[cfg(test)]
mod tests {
    use crate::board::Board;
    use crate::solver::{Solver, Value};
    #[test]
    fn solver_matches_bruteforce_3x3() {
        let b = Board::new(3, 3, 1);
        let s = b.initial();
        let mut sol = Solver::new(&b);
        assert_eq!(sol.solve(&s), crate::solver::brute_value(&b, &s, 14));
    }
    #[test]
    fn three_by_three_is_second_player_win() {
        let b = Board::new(3, 3, 1);
        let mut sol = Solver::new(&b);
        assert_eq!(sol.solve(&b.initial()), Value::Loss); // side-to-move (p0) loses
    }
}
```

- [ ] **Step 2: Implement.** `Value { Loss, Draw, Win }` (side-to-move relative), ordered `Loss < Draw < Win`, with `negate()`. `Solver` holds a `FxHashMap<(State,u32), (Value, Flag)>` TT (bound-flagged EXACT/LOWER/UPPER, like `smallboard/solver.py`). Iterative deepening: raise `max_depth` until the root value stabilizes across two successive bounds (or a configured ceiling), returning the stable value. Interior: depth-bounded negamax (`depth==0 -> Draw`), alpha-beta with move ordering by `dist_to_goal` advantage. On `walls_left==[0,0]`, call `endgame::race_value` (Task 4b) instead of recursing — exact race result (turn/jumps/parity included), the first endgame tablebase slice. `brute_value` = a plain depth-bounded negamax (no AB/TT) for the cross-check. **Retrograde fallback:** if iterative deepening does not stabilize by the ceiling (the draw signature), fall back to a retrograde fixpoint over the reachable race/low-wall states to resolve the value (the 8×3 case). Document the GHI reasoning inline.

- [ ] **Step 3: `endgame.rs`** — `race_value(&Board,&State) -> Value`: exact negamax over pawn-only moves with frozen walls, depth-bounded by `2*(w+h)` with TT; the retrograde tablebase generalization is a Phase-1 experiment. **Step 4: Run** `cargo test` → pass. **Step 5: Commit** (`feat(solver): ID negamax + alpha-beta + TT + race endgame`).

---

## Task 5: Validation vs the writeup (the correctness gate)

**Files:** `solver/tests/writeup_values.rs`. No new library code — this proves the solver.

- [ ] **Step 1: Write the validation tests** (run with `--release`; mark slow ones `#[ignore]` if needed and run explicitly):
```rust
use quoridor_solver::board::Board;
use quoridor_solver::solver::{Solver, Value};

fn start_value(w: u8, h: u8, walls: u8) -> Value {
    let b = Board::new(w, h, walls);
    Solver::new(&b).solve(&b.initial())
}

#[test]
fn v_3x3_p2_win() { assert_eq!(start_value(3,3,1), Value::Loss); }

#[test]
fn v_5x5_transition() {
    // writeup: 5x5 is P2 (side-to-move loses) at <=4 walls, P1 (win) at >=5.
    assert_eq!(start_value(5,5,4), Value::Loss);
    assert_eq!(start_value(5,5,5), Value::Win);
}

#[test]
fn v_even_height_is_p1_win() {
    // PLAN: pin a concrete even-height board + published value from the writeup's
    // results table (H even => P1 win at all wall counts). e.g. a width-2 even-height
    // board that solves quickly. Implementer: confirm the exact (w,h,walls) and value
    // from the writeup before finalizing; assert == Value::Win.
}

#[ignore] // slow; run explicitly in CI/locally
#[test]
fn v_8x3_three_walls_is_draw() {
    // The GHI canary: 8x3 at 3 walls per player is a DRAW from the start.
    assert_eq!(start_value(8,3,3), Value::Draw);
}
```

- [ ] **Step 2: Run** `cargo test --release` (and the ignored 8×3 explicitly: `cargo test --release -- --ignored v_8x3`). All pass. If 8×3 reveals a GHI bug, fix the retrograde fallback in Task 4 (do NOT weaken the assertion). **Step 3: Commit** (`test(solver): reproduce writeup values (3x3, 5x5 transition, even-height P1, 8x3 draw)`).

---

## Task 6: Wall-configuration counter

**Files:** Create `solver/src/configcount.rs`; tests inline.

- [ ] **Step 1: Failing test** — reproduce the writeup's exact counts:
```rust
#[cfg(test)]
mod tests {
    use crate::board::Board;
    use crate::configcount::count_wall_configs;
    #[test]
    fn matches_writeup_small() {
        assert_eq!(count_wall_configs(&Board::new(5,5,99)), 2_532_560);
        assert_eq!(count_wall_configs(&Board::new(4,5,99)), 70_944);
        assert_eq!(count_wall_configs(&Board::new(3,5,99)), 1_880);
        assert_eq!(count_wall_configs(&Board::new(2,5,99)), 60);
    }
}
```

- [ ] **Step 2: Implement** `count_wall_configs(&Board) -> u64` — count all geometrically-legal wall arrangements (any number of non-overlapping/non-crossing walls; ignore path constraint and walls_left, matching the writeup's `CONFIG_TOTALS`). Start with exhaustive backtracking enumeration (correct, fine ≤5×5). If it's too slow for 6×5+, add a column-profile transfer-matrix DP (gate it by the same small-board equalities). **Step 3: Run** → pass. **Step 4: Commit** (`feat(solver): exact wall-configuration counter (validated vs writeup)`).

---

## Task 7 (Phase 1): Profiling CLI + instrumentation

**Files:** Create `solver/src/bin/solve.rs`; add counters to `solver.rs`.

- [ ] **Step 1:** Instrument `Solver` with `nodes`, `tt_len`, and peak-RAM sampling (resident set via a small platform helper or `tt.len()*entry_size` estimate). The CLI `solve W H WALLS` prints: value, nodes, TT entries, est. peak RAM, wall-clock. Add a `--bruteforce` flag for the unoptimized baseline. **Step 2:** A smoke test asserting the CLI solves 3×3 and prints a `Value`. **Step 3: Commit** (`feat(solver): profiling CLI + node/TT/RAM instrumentation`).

- [ ] **Step 4: Measurement run** — solve 5×5 (all wall counts up to 5) and 4×7, recording nodes/TT/RAM/time. Capture into `docs/superpowers/solver-phase1-measurements.md`. This is the per-area growth curve.

---

## Task 8 (Phase 1): df-pn and endgame-tablebase experiments

**Files:** `solver/src/dfpn.rs` (experimental), extend `endgame.rs`.

- [ ] **Step 1:** Implement df-pn (depth-first proof-number search) as an alternative driver, GHI-handled, gated by the **same** writeup-value tests as Task 5 (it must produce identical values). **Step 2:** Extend the endgame tablebase to a tunable `k`-wall retrograde precompute. **Step 3:** Compare on 5×5/4×7: nodes, RAM, wall-clock — (a) ID-alpha-beta vs df-pn, (b) tablebase depth 0 vs 1 vs 2. Append results to `solver-phase1-measurements.md`. **Step 4: Commit** (`feat(solver): df-pn + k-wall tablebase experiments (Phase 1)`).

---

## Task 9 (Phase 1): Config counts for targets + the RunPod estimate

**Files:** `docs/superpowers/solver-phase1-estimate.md`.

- [ ] **Step 1:** Run `count_wall_configs` for 6×5, 7×5, 7×7 (use the DP if enumeration is too slow; if 7×7 is infeasible to count exactly, extrapolate from the W×5 trend and label it an estimate). **Step 2:** Combine the config counts + the Task 7/8 growth curve + the best (search, tablebase) config to project, for **6×5** and **7×5**: peak RAM, single-thread and N-core wall-clock, the matching RunPod instance class, and a **$ figure** (with explicit assumptions + uncertainty). Give a go/no-go on 7×7. **Step 3: Commit** (`docs(solver): Phase 1 measurements + firm 6x5/7x5 RunPod estimate`).

**This estimate doc is the deliverable the user reviews before any RunPod spend.** Stop here and report; Phases 2–3 get their own spec.

---

## Self-Review

**Spec coverage:** fresh crate ✅ (T1); rectangular rules mirroring smallboard ✅ (T1–T3); floating-wall fast-path ✅ (T3); ID negamax+AB+TT+ordering ✅ (T4); race endgame + retrograde fallback ✅ (T4); writeup validation incl. 8×3 draw ✅ (T5); config counts validated ✅ (T6); profiling ✅ (T7); df-pn + tablebase experiments ✅ (T8); config counts for targets + firm estimate ✅ (T9). Symmetry/killer-history are noted as Phase-1 perf levers (T8 territory) — acceptable; correctness doesn't depend on them.

**Placeholder scan:** one deliberate deferral — the concrete even-height validation board in T5 (implementer pins it from the writeup table). Flagged explicitly, not silent.

**Type consistency:** `Board`/`State`/`Move`/`Value`/`Solver` signatures are consistent across tasks. `Value` is side-to-move-relative everywhere (`Loss/Draw/Win` + `negate`). `count_wall_configs(&Board)->u64` and `Solver::solve(&State)->Value` used consistently in tests and CLI.

---

## Out of scope (per spec)

RunPod solves (Phases 2–3); frontier-enumeration + GPU tablebase build; anything neural/AZ; parameterizing the 9×9 `native/` crate.
