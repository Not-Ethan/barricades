# Solver — Phase 1 Baseline Measurements

**Date:** 2026-06-09
**Solver:** Phase-0 build (`solver/`, commit `4a9791d`) — single-pass depth-bounded
negamax + alpha-beta + transposition table + move ordering + a per-leaf race solve.
**No** Phase-1 optimizations yet (no endgame tablebase reuse, no symmetry, no
per-config legality tables). Machine: Apple M1, 16 GB, single-threaded.

## Baseline sweep (`solve W H WALLS`, 90 s timeout)

| Board (W×H) | walls | value | nodes | TT entries | time |
|---|---|---|---|---|---|
| 3×3 | 0/1/2/3 | Loss | 52 / 1.3K / 620 / 636 | 0/91/393/399 | ~0 |
| 4×4 | 0/1/2/3 | Win | 18 / 7.6K / 111K / 25.8K | 0/132/1656/6385 | ≤0.02s |
| 5×5 | 0 | Loss | 2.2K | 0 | 0.000s |
| 5×5 | 1 | Loss | **1.85M** | 1,189 | 0.086s |
| 5×5 | 2 | Loss | **112.3M** | 48,873 | 5.03s |
| 5×5 | 3 | — | — | — | **TIMEOUT >90s** |
| 3×5 | 1/2/3 | Loss/Loss/Win | 0.43M / 1.5M / 0.14M | 370/10.9K/64.7K | ≤0.2s |
| 4×5 | 1/2 | Loss | 0.95M / **19.8M** | 801/35,968 | 0.06 / 1.16s |
| 2×5 | 2/3 | Loss | 345 / 345 | 200 | ~0 |
| **6×5** | 0 | Loss | 2.8K | 0 | 0.000s |
| **6×5** | 1 | Loss | **2.78M** | 1,554 | 0.144s |
| **6×5** | 2 | Loss | **140.3M** | 58,329 | 6.69s |
| 4×7 | 0/1 | Loss | 14K / **65.4M** | 0/3,175 | 0.001 / 2.63s |
| 7×5 | 0/1 | Loss | 3.6K / **3.67M** | 0/1,949 | 0.000 / 0.19s |
| 7×7 | 0/1 | Loss | 47K / **249.5M** | 0/8,470 | 0.002 / 9.02s |

## Findings

1. **Wall-count scaling is the wall (≈50–100× per added wall).** 5×5: W1→W2 = 61×,
   W2→W3 ≥ 18× (timeout). 6×5: W1→W2 = 50×. Solving a board at the wall counts that
   matter (the parity transition needs up to ~W5–7, like 5×5 flipping at W5) is
   astronomically out of reach for this solver: 6×5 W3 ≈ minutes, W4 ≈ hours, W5 ≈ infeasible.

2. **The dominant cost is re-solving the race endgame, not the "real" search.**
   Across the board, **nodes ≫ TT entries** (6×5 W2: 140M nodes / 58K TT entries ≈ 2,400
   nodes per main-TT entry). Reason: `race_value` is invoked at every walls-exhausted leaf
   and **re-solves the race from scratch each time**, with no caching across leaves. Proof:
   5×5 W0 (a single race from the start) = 2.2K nodes; 5×5 W1 = 1.85M ≈ **800 re-solves of
   that same ~2.2K-node race**. The main transposition table stays tiny because the main
   (with-walls) tree at low wall counts is small — the explosion is entirely in repeated
   race search.

3. **The board's pure-race cost (W0) is cheap and scales mildly** with size (3×3: 52 →
   6×5: 2.8K → 7×7: 47K). So the race itself is not expensive — *re-deriving it millions
   of times* is.

## Implication for the estimate (Task 9)

The Phase-0 solver **cannot** be profiled to a meaningful 6×5 estimate, because its cost
is dominated by an artifact (race re-search) that the writeup's central optimization
removes. The single highest-impact fix is exactly the writeup's headline technique:
**precompute the race / endgame values once per wall-configuration (a retrograde
tablebase) and reuse them**, instead of re-solving per leaf. Expected impact: orders of
magnitude on every low-wall board. Secondary levers (each multiplicative): a **persistent
race memo** (cheap interim version of the tablebase), **horizontal symmetry** (~2×), the
**rotate+swap symmetry** (~2×), and **killer/history move ordering**.

**Therefore Task 8 (the endgame-tablebase + symmetry build) is a prerequisite for a
trustworthy Task 9 estimate** — not an optional experiment. The plan's "measure the
un-optimized solver and extrapolate" assumption does not hold; we must build the core
optimization first, then measure the optimized solver to project 6×5 / 7×5.

## CRITICAL bugs found by adversarial audit + fixed (2026-06-09)

An adversarial audit found **two CRITICAL exactness bugs**. Both are now fixed and
gated by exact-repro regression tests (`solver/tests/wall_legality.rs`,
`solver/tests/race_exactness.rs`). **All measurements in the table above were taken
with these bugs live and must be treated as pre-fix / untrustworthy** wherever the bug
could have been exercised.

- **Bug 1 (wall legality, even-width boards):** the floating-wall fast-path in
  `legal_walls` (`needs_path_check`) skipped the connectivity BFS for walls that were
  interior and ≥ Chebyshev-2 from every existing wall. Two collinear same-orientation
  walls whose anchors are Chebyshev-2 apart *along the axis* share a lattice endpoint and
  form a connected barrier; a "keystone" wall in the gap completes a goal-spanning barrier
  yet looks "floating" to the predicate, so an **illegal pawn-stranding wall was admitted**.
  This inverted solver values on even-width boards (e.g. a 6×4 keystone position that is a
  true first-player **Win** was returned as **Loss**). **Fix:** the fast-path is deleted;
  `legal_walls` now always runs the two-player `has_path` BFS on every non-overlapping
  candidate (identical to the brute-force reference).

- **Bug 2 (race endgame, long-path mazes):** `endgame.rs::race_value` used a fixed depth
  bound `2*(w+h)`. A frozen-wall maze can force a proof line longer than that bound, so the
  depth-0 floor fired and `race_value` returned a bogus **Draw** — but a wall-less race is
  provably never a true draw. The wrong Draw escaped into `ab()` and flipped ancestor
  values, and was order-dependent (reused vs fresh `Solver` disagreed). **Fix:**
  **iterative-deepening** negamax (no fixed floor) — on the `Loss<Draw<Win` lattice the
  floor can only ever taint a result up to `Draw` and never flips a true Win/Loss, so the
  first definitive Win/Loss the deepening yields is exact; clean (full-proof) Win/Loss are
  memoised and reused across deepening iterations and leaves. A **mandatory panic guard**
  (`race_value` panics if a wall-less race ever fails to resolve to Win/Loss within the
  `2*(w*h)^2` hard ceiling) makes a silent wrong Draw impossible.

### Re-validated values after both fixes

| Board | old value (pre-fix) | new value (post-fix) | changed? |
|---|---|---|---|
| 6×5 W0 | Loss | **Loss** | no |
| 6×5 W1 | Loss | **Loss** | no |
| 6×5 W2 | Loss | **Loss** | no |
| 5×5 W3 | Loss (75.5 s, **Bug 2 live**) | **timeout** (> 400 s) — not re-confirmable in budget | n/a |

The 6×5 W0/W1/W2 values are unchanged (the keystone bug existed but did not flip these
specific opening values), and are now trustworthy. 5×5 W3 was computed with Bug 2 live, so
its earlier "Loss" is **untrustworthy**; post-fix the exact solve no longer completes in the
time budget (the mandatory BFS-on-every-wall and iterative-deepening race search slow the
un-optimised solver further), so its trustworthy value awaits the Task-8 optimisations. The
old 6×5 numbers in the table above predate the fix and should be read with that caveat.
