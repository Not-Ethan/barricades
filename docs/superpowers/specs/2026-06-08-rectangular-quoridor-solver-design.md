# Rectangular Quoridor Solver — Design (Phase 0 + Phase 1)

**Date:** 2026-06-08
**Status:** Approved (design phase)
**Branch:** `az-bootstrap`

## Overview

Build an **exact solver** for rectangular Quoridor — parameterized by width `W`, height
`H`, and walls-per-player `k` — that pushes past the area-28 frontier of the reference
writeup (grantslatton.com/solving-quoridor) to **solve 6×5** (area 30), then **7×5**
(area 35), with **7×7** (area 49) as a stretch. "Solve" means *prove* the game-theoretic
value (1st-player win / 2nd-player win / draw) from the start position under perfect play
— a proof, not a strong-playing bot.

**Notation (fixed):** `W×H = width × height`. Columns `0..W-1`, rows `0..H-1`. Player 0
starts row 0 (goal row `H-1`), player 1 starts row `H-1` (goal row 0). "Even-height" = `H`
even. The user's target "6×5" = **width 6, height 5** (odd height). The writeup uses the
same width×height convention.

**Why these boards.** Odd-height boards are the *open* cases: player 2 has a jump-parity
advantage at low wall counts, player 1 a tempo advantage that dominates at high wall
counts — and whether the transition is clean or has anomalies (as 7×3/8×3 hint) is
unknown past area 28. 6×5 would be the **first area-30 board ever solved**; 7×5 is the
writeup author's named "next frontier."

**Compute & cost discipline.** Heavy solves run on **RunPod** (CPU + RAM — the search is
*not* GPU-friendly; GPU only helps table precompute). The user wants a **firm cost
estimate before committing spend**, so the project is staged: **Phases 0–1 run free on the
local 16 GB M1 and produce that estimate**; only then do we rent anything.

## Scope

- **This spec covers Phase 0 + Phase 1.** Phases 2–3 are sketched as follow-ons and get
  their own spec once Phase 1's estimate lands.
- **Validation-first** is non-negotiable: we never trust a value from an unvalidated
  solver, because a single unsound shortcut yields a *wrong proof*.

## Approach

A **fresh, self-contained Rust crate** (`solver/`, crate `quoridor_solver`),
parameterized by board dimensions + wall budget. We do **not** reuse the existing
9×9-hardwired `native/` crate (it's baked to 9×9 across ~25 constants and tuned for AZ
self-play); we reimplement the bitboard/BFS *techniques* cleanly for arbitrary `W×H`.
`u64` bitboards suffice for every target (7×7 = 49 cells, 36 wall-slots/orientation, all
< 64).

**Core method (Phase 0 — the writeup's validated approach, chosen for correctness-first):**
iterative-deepening **negamax + alpha-beta**, depth-bounded (unresolved lines score as
draw; raise the bound until the value stabilizes), with `(0,∞)`-style bounds for extra
cutoffs, plus a **retrograde-analysis fallback** for genuine draws. Values are
win/draw/loss for the side to move.

**Soundness-preserving optimizations** (every one is value-preserving — an exact solve
forbids any approximate pruning):

- **Transposition table** (the shared DP/memo of solved positions) with bound flags +
  memory-bounded replacement.
- **Symmetry:** left-right mirror canonicalization (2×); the additional
  (180°-rotate + swap-players)-with-negation fold evaluated in Phase 1 (potential extra 2×).
- **Move ordering:** distance-to-goal first; killer/history heuristics added in Phase 1.
- **Floating-wall fast-path:** skip the path-existence BFS for walls that provably can't
  block (only check walls touching an edge or another wall at ≥2 points).
- **Endgame tablebase:** retrograde-precompute the 0-wall race slice (extensible to a
  tunable `k`-wall depth); the main search stops at the boundary and reads an exact value
  instead of searching into the endgame. CPU build first; the GPU/SIMD bulk build is a
  later accelerator.
- **df-pn** (depth-first proof-number search) evaluated against ID-alpha-beta in Phase 1;
  adopted only if it wins on nodes/RAM *and* preserves correctness.

**The correctness landmine — graph-history interaction (repetition draws).** Quoridor
allows endless pawn shuffling, so a position's value can depend on path (draw by
repetition). Phase 0's depth-bounded ID approach caps cycles (unresolved-within-bound =
draw) and the retrograde fallback resolves true draws. The **8×3-at-3-walls draw is the
canary test** — the exact case that forced the writeup author into a retrograde fallback.

## Components (modules of `solver/`)

- `board`/`state` — dims, state struct, `apply_move`, `winner`, `is_terminal`.
- `bitboard` — floodfill BFS reachability + distance-to-goal, parameterized by `W,H`.
- `movegen` — legal pawn steps (incl. jumps) + wall placements, with the floating-wall
  fast-path.
- `solver` — ID negamax + alpha-beta + TT + move ordering + symmetry; retrograde fallback.
- `endgame` — retrograde race tablebase (0-wall; extensible to `k`-wall).
- `configcount` — exact count of legal wall configurations (validated vs the writeup).
- `bin`/CLI — run a solve for `(W,H,k)`, print value + stats (nodes, TT size, peak RAM,
  wall-clock).
- `tests` — rules differential vs the `smallboard` Python engine (square sizes); solver vs
  the writeup's published values; config-count vs the writeup; the 8×3 draw.

## Phase 0 — build + validate (free, local M1)

**Deliverable:** a solver that reproduces the writeup's published results *exactly*:

- 3×3 = **player-2 win**.
- 5×5 = **player-2 at ≤4 walls, player-1 at ≥5** (the parity→tempo transition).
- 4×5 — value per the writeup table.
- An **even-height** board (`H` even) confirmed **player-1 win** (plan pins a concrete one
  with a published value).
- **8×3 at 3 walls = draw** (the GHI canary).
- **Config counts:** 5×5 = **2,532,560** and 4×5 = **70,944** reproduced exactly.

Plus a rules differential test: the new Rust movegen/winner must match the `smallboard`
Python engine on square boards (3×3, 5×5) over random play.

## Phase 1 — measure → estimate (free, local M1)

- **Exact wall-config counts** for 6×5, 7×5, 7×7 (validated against the 5×5/4×5 numbers;
  if 7×7 enumeration is itself too slow, a transfer-matrix DP or a documented extrapolation
  from the W×5 growth trend).
- **Profile** node count, TT size, peak RAM, wall-clock on already-solved boards (5×5,
  4×7) → extract the per-area cost-and-RAM growth curve.
- **Experiment A:** df-pn vs ID-alpha-beta (nodes, RAM) on solved boards.
- **Experiment B:** endgame-tablebase pruning impact (nodes/time with vs without).
- **Output:** a firm RunPod instance + hours + **$ estimate for 6×5**, a defensible range
  for 7×5, and a go/no-go on 7×7 — the artifact the user needs before any spend.

## Sketched follow-ons (separate specs after Phase 1)

- **Phase 2 — solve 6×5 on RunPod.** High-mem CPU pod; current ballpark ~$5–30.
- **Phase 3 — attempt 7×5.** Gated on a **frontier-enumeration** table (to fit RAM) +
  multicore parallelism; ballpark ~$100–500 (the author's "few hundred dollars").
- **7×7 — research stretch**, only if 7×5 lands and the machinery generalizes.

## Success criteria

- **Phase 0:** solver reproduces every named writeup result above, exactly; rules match
  `smallboard` at square sizes; the 8×3 draw is found.
- **Phase 1:** produces the config counts, the cost/RAM growth curve, the df-pn and
  tablebase experiment numbers, and a firm 6×5 RunPod estimate.

## Out of scope

- The actual RunPod solves (Phases 2–3) — separate spec once the estimate lands.
- Frontier-enumeration and the GPU/SIMD tablebase build — sketched; specced later.
- Anything neural/AZ (a different project).
- Parameterizing the existing 9×9 `native/` crate.

## Risks / open questions

- **df-pn GHI-correctness** — keep ID-alpha-beta as the trusted baseline; adopt df-pn only
  if Phase 1 proves it correct *and* faster.
- **7×5 RAM (~300 GB full table)** — needs frontier-enumeration; the Phase-3 gating risk.
- **7×7** — likely infeasible with known methods; treat as research, not a deliverable.
- **Counting 7×7 configs (~10¹²)** — enumeration may be infeasible; fall back to DP or a
  documented trend extrapolation, clearly labeled as an estimate.
