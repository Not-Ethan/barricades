# 6×5 Quoridor, Weakly Solved for 0-10 Walls

**Date:** 2026-06-10 · **Branch:** `solver-and-az` (renamed from `az-bootstrap`) · **Raw logs:** `docs/superpowers/raw/ladder_logs/`

## The result

6×5 Quoridor (6 columns × 5 rows; player 0 starts bottom-center, races to the top
row; standard jump and wall rules; W walls per player) is **exactly solved for every
wall count W = 0…10**:

| W (walls/player) | 0 | 1 | 2 | 3 | **4** | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **winner (perfect play)** | P2 | P2 | P2 | P2 | **P1** | P1 | P1 | P1 | P1 | P1 | P1 |

This is, to our knowledge, **the first board of area > 28 ever solved** (the prior
frontier: grantslatton.com/solving-quoridor, area ≤ 28 on 128 GB), and the first
mapped parity transition beyond it. 6×5 behaves like 5×5 (single clean flip — the
2nd player's jump-parity advantage rules low wall counts until the 1st player's tempo
advantage takes over, here at **W4**) and shows none of 4×7's oscillation.

## Per-rung cost (16-thread lazy-SMP except as noted)

| W | value | nodes | wall-clock | where |
|---|---|---|---|---|
| 0 | P2 | 1.4 K | 0 s | M1 |
| 1 | P2 | 0.35 M | 0.1 s | M1 |
| 2 | P2 | 6.5 M | 2.4 s | M1 |
| 3 | P2 | 268 M | ~144 s CPU | M1 |
| 4 | **P1** | 12.9 B | 3.7 h (≈40 min uncontended) | M1, 8 threads |
| 5 | P1 | 38.2 B | 38 min | pod |
| 6 | P1 | 40.0 B | 70 min | pod |
| 7 | P1 | 28.8 B | 86 min | pod, 64 GiB TT |
| 8 | P1 | 32.4 B | 101 min | pod |
| 9 | P1 | 31.5 B | 98 min | pod |
| 10 | P1 | 35.8 B | 111 min | pod |

**The cost-curve finding:** solving cost peaks just *past* the value transition
(W5–W6) and then plateaus/declines — decisive positions prove cheaply (alpha-beta
cutoffs), the board's useful wall placements saturate, and factorial transposition
merging absorbs the deeper budgets. Practical heuristic for solving such games:
*budget compute around the transition rung, not the deepest rung.* A second
structural observation: the race endgame **vanishes** with budget depth — W5 touched
1.9 M race configs; W10 touched **zero**. Deep Quoridor is pure wall labyrinth.

## Why the values can be trusted

- **Engine lineage:** rules differentially tested against two independent
  implementations (the production `core` and `smallboard` engines, 2,000+ positions);
  solver cross-checked against an unpruned brute-force oracle on complete reachable
  graphs (hundreds of thousands of positions, repeatedly, after every change);
  external validation against all published writeup values (3×3, 4×4, 5×5 incl. the
  W5 transition, the 8×3 draw machinery).
- **Decisive-value soundness:** every ladder value is a Win/Loss from full-window
  alpha-beta — exact regardless of depth ceiling, TT eviction, thread count, or cache
  sizes (only Draws are ceiling-limited; none occurred).
- **Independent confirmations:** W3 computed identically on three engine generations;
  W4–W10 on x86/Linux after the full 78-test gate suite passed there; W5 partially
  recomputed on ARM/macOS (40.4 B nodes, killed by host OOM — values en route agreed);
  a sound independent df-pn engine exists as a cross-check oracle.
- **Adversarial discipline:** two value-inverting bugs (the keystone illegal-wall
  fast-path; the race depth-floor false-Draw) were found by adversarial audit *before*
  any novel rung was trusted, fixed, and pinned by regression tests. Two further
  pruning ideas (df-pn, footprint/Theorem-4) were implemented, *verified exact*, and
  measured as net losses — kept as documented negative results rather than shipped.

## The engine that did it

Alpha-beta negamax, depth-bounded with exact retrograde race endgames; depth-folded
packed-key fixed-capacity sharded TT; bounded config-granular LRU race cache (RwLock
shards + atomic LRU ticks — sharding it was a 4.2× wall-clock win found by live
profiling); killer/history ordering; horizontal-mirror canonicalization; lazy-SMP
parallelism (superlinear — 84× on 5×5-W3 — via shared-TT portfolio effects); DSU-on-
posts wall-legality filter (sound by planar duality; 1.5–2× fewer flood fills than
the writeup's contact heuristic, measured under controls — see
`solver-legality-filter-comparison.md`); live heartbeat observability.

## Resources

Apple M1 16 GB (development, validation, W0–W4) + one RunPod cpu5m pod
(16 vCPU/128 GB-class, $1.04/hr) for W5–W10: ≈ 8 pod-hours ≈ **$10.0** of a $15
allocation, including a corrected-predicate benchmark re-run. Total wall-clock from
"let's solve 6×5" to the full ladder: ~2 days, most of it verification by choice.
