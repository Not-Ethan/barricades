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
