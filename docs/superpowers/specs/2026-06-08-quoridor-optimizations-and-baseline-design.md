# Quoridor-Specific Optimizations + Strength Baseline

**Date:** 2026-06-08
**Status:** Approved (design phase)
**Branch:** `az-bootstrap` (current HEAD; the native core + AZ campaign are merged here)
**Depends on:** the native core (`barricades_native`), the AZ campaign (`scripts/campaign.py`), and the dense-reward training (`agents/az/train.py`).

## Overview

Four pieces, motivated by the article *Solving Quoridor* (grantslatton.com) and a 10k-game campaign that showed the dense-reward design works (game length 76→28 as λ ramped) but the resulting net loses 0/10 to depth-2 minimax — i.e., the pipeline is correct but undertrained, and we need (a) more training throughput per wall-clock and (b) a real way to measure strength.

1. **Wall-legality fast-path** — skip the path-existence BFS for "floating" walls, attacking the dominant self-play cost.
2. **Left-right symmetry augmentation** — free 2× training data → better sample efficiency.
3. **Endgame race solver (`walls_left==(0,0)` only)** — exact values for the frozen-wall race, used in MCTS, self-play truncation, and inference.
4. **Minimax strength ladder** — a scalable reference baseline (our "Stockfish") to measure the net's strength.

**This is preparation only — no training campaign is launched as part of this work.** The strength ladder (Unit 4) may be *run on existing checkpoints* (that's evaluation, not training).

**Stack:** Rust + PyO3 (`native/src/`), PyTorch on MPS, Python 3.14. Differential testing against the Python `core` remains the correctness oracle for Rust changes.

---

## Unit 1 — Floating-wall legality fast-path

**Where:** `native/src/movegen.rs::legal_walls`.

**Problem:** today, for every non-overlapping candidate wall, `legal_walls` runs **two** `path_exists` BFS (one per player) — up to ~128 candidates × 2 BFS per node expansion. This is the dominant self-play CPU cost.

**Optimization:** a 2-segment wall can disconnect a pawn from its goal *only* if it extends the existing barrier structure (other walls ∪ board edges) into a closed cut, which requires it to contact that structure at **≥2 of its endpoints**. A wall contacting at <2 points can always be routed around → it is **trivially legal**, no BFS needed. In the opening/midgame almost every candidate is "floating," so the BFS runs only for the few walls that could actually complete a fence.

**Design:**
- A cheap predicate `needs_path_check(state, c, r, orient) -> bool`: returns `true` iff the candidate wall touches an existing wall endpoint or the board boundary at ≥2 of its contact points (computed with bit ops on `h_mask`/`v_mask` + edge checks). The exact contact geometry: a wall has three lattice contact points along its length (two ends + middle); count how many coincide with an existing perpendicular/collinear wall end or the board border.
- `legal_walls`: for each non-overlapping candidate, if `!needs_path_check(...)` → push as legal (skip BFS); else run the existing 2× BFS.
- **Conservative-by-construction:** the predicate must only return `false` (skip) when the wall is *provably* safe; if uncertain, return `true` (fall back to BFS). A predicate that is too eager to skip is a bug; one that is too eager to BFS is merely slow.

**Correctness (the safety net):** the Python `core.legal_walls` — which BFS-checks every wall — is the **oracle**. Extend the existing differential test (`tests/test_native_game.py`) so the random-playout fuzz includes **wall-dense midgame/endgame positions** (where blocking is actually possible — e.g., bias the fuzz toward wall moves, and run longer games), asserting `set(bn.legal_moves) == set(core.legal_moves)` over tens of thousands of positions. Any geometry error in the predicate surfaces as a differential mismatch. Optionally, a debug-only Rust assertion that runs both the fast-path and full-BFS and asserts agreement.

**Testing:** the extended differential fuzz (must include many positions with ≥6 walls placed, and positions where some candidate walls are genuinely illegal). A micro-benchmark (`legal_walls` calls/sec before vs after on a wall-dense position) to confirm the speedup.

**Out of scope:** incremental/union-find barrier tracking (Approach C) — bigger, deferred.

---

## Unit 2 — Left-right symmetry data augmentation

**Where:** Python training (`agents/az/train.py` — a new `augment_lr`; applied in the campaign before `form_dense_targets`).

**Symmetry:** Quoridor is invariant under left-right reflection. In our coordinates: pawn `(c,r) → (8-c, r)`; wall anchor `(c,r) → (7-c, r)` for **both** H and V walls; `walls_left`, `turn` unchanged. The canonical encoding already exploits the *player-swap* (row-flip) symmetry; L-R is the orthogonal, additional free 2×.

**Design:** `augment_lr(examples) -> examples'` returns each example plus its mirror:
- **planes:** flip the 6×9×9 array along the column axis (`planes[:, :, c] -> planes[:, :, 8-c]`).
- **policy π (140):** apply a fixed L-R action permutation:
  - steps (indices 0–11): `dx → -dx` — i.e. E↔W (2↔3), 2E↔2W (6↔7), NE↔NW (8↔9), SE↔SW (10↔11); N/S/2N/2S unchanged.
  - H-walls: `12 + cr*8 + cc  →  12 + cr*8 + (7-cc)`.
  - V-walls: `76 + cr*8 + cc  →  76 + cr*8 + (7-cc)` (the +64 block).
- **`z`, `feats`:** unchanged (a mirror changes neither who wins nor the path difference).

The permutation is precomputed once (a fixed length-140 index array). Mirroring twice is the identity.

**Why:** doubles training data → better sample efficiency → the net learns wall tactics with fewer games → games shorten and strength rises faster per wall-clock.

**Testing (commutation):** over random states `s` and their legal moves `m`,
- `mirror_planes(encode_planes(s)) == encode_planes(lr_mirror_state(s))`, and
- `lr_action_perm[move_to_action(m, s)] == move_to_action(lr_mirror_move(m), lr_mirror_state(s))`.
Plus: applying the permutation twice is the identity; `augment_lr` doubles the example count and preserves `z`/`feats`. A wrong plane-flip or action index fails the commutation check.

---

## Unit 3 — Endgame race solver (`walls_left == (0,0)`)

**Where:** new `native/src/endgame.rs` (+ integration in `mcts.rs`, `selfplay.rs`, `agent.rs`/`pyiface.rs`).

**Insight:** once **both** players are out of walls, the wall structure is frozen and the game is a pure pawn **race** with a tiny branching factor (≤ ~5 pawn moves/side, no walls). It is exactly solvable by a small **depth-bounded** memoized minimax over pawn moves. This is the Quoridor analogue of an endgame tablebase, trivially computable.

**Cycle/draw handling (important):** pawns can move backward, so the race graph has cycles (a stalling player shuffles back and forth) — a naïve negamax would recurse forever. Bound the search by a ply limit ample for either pawn to reach its goal under perfect play (e.g. `4*N = 36` plies; either pawn's frozen-wall shortest path is < that). Within the bound the result is normally **decisive** — the leader can force progress; the trailer cannot block (only jump-over) — but the bound + a `draw (0)` fallback at the limit makes stalling/cycles safe and the function total. So `solve_race` returns win(+1)/loss(−1)/draw(0).

**Design:**
- `solve_race(state) -> (i32 value_for_mover, Move best)`: depth-bounded exact negamax over pawn moves only (walls frozen), memoized on `(pawns, turn, plies_remaining)` (or `(pawns, turn)` with path-cycle guarding) within the call. `value_for_mover` ∈ {+1 win, 0 draw-at-bound, −1 loss} with perfect play. Precondition: `walls_left == (0,0)` and not terminal.
- **Three uses (one component, three call sites):**
  1. **MCTS leaf eval** (`mcts.rs`): if a non-terminal leaf has `walls_left==(0,0)`, back up the exact `solve_race` value (converted to root-player perspective) instead of calling the net. Sharper values, no net call in solved subtrees.
  2. **Self-play truncation** (`selfplay.rs`): when a slot's game reaches `walls_left==(0,0)`, compute `solve_race`, stamp the exact outcome `z` on that game's recorded examples, and finalize the game (skip the mechanical racing tail). Saves plies and gives perfect value labels.
  3. **Inference / agent play** (`NativeMctsAgent.select_move`): if the live position has `walls_left==(0,0)`, return `solve_race`'s `best` move directly — perfect conversion. **This is why truncating self-play training is safe: the solver, not the net, plays the endgame at inference, so no "conversion" skill needs to be learned.**
- **Scope: the exact `(0,0)` case only.** Broadening to "walls remain but cannot change the outcome" is harder (requires reasoning over future wall placements) and is explicitly deferred.

**Hit-rate measurement:** because games may end (~30–44 plies in the 10k run) before both players spend all 20 walls, the `(0,0)` case may be infrequent. **As part of this unit, instrument and report the fraction of self-play games that reach `walls_left==(0,0)`** (e.g., a counter exposed from `SelfPlayPool` or measured in a short self-play run). This sizes the training-throughput benefit honestly; the inference-conversion and MCTS-leaf benefits apply whenever it fires regardless.

**Testing:**
- **Differential vs full search:** on random `(0,0)` positions, `solve_race` agrees with an independent exact solve (a small Python negamax over pawn moves using `core`) on both the value and that the chosen move is winning when a win exists.
- **Conversion:** from a winning `(0,0)` race position, the solver's move sequence actually reaches the goal (no dawdling, no draw).
- **MCTS/agent integration:** a `(0,0)` won position → the agent plays the solving move; an MCTS leaf at `(0,0)` gets the exact value.
- **No regression:** the existing native + campaign suites stay green (the solver only changes behavior at `(0,0)`).

---

## Unit 4 — Minimax strength ladder (the real baseline)

**Where:** new `scripts/eval_ladder.py` (reusing the parallel match harness pattern from `scripts/eval_az.py` and the staged `winrate_vs` opponents).

**Rationale:** random and heuristic-MCTS are too weak to read strength (the 10k net beat both but lost 0/10 to depth-2 minimax). Our scalable reference — our "Stockfish" — is **minimax at increasing depth/time**. Anchoring win-rate against fixed minimax rungs gives an interpretable, monotone strength scale.

**Design:** `eval_ladder.py [checkpoint]` plays a net-driven `NativeMctsAgent` (configurable sims) vs a ladder of opponents, N games each (alternating colors), reporting win-rate (± stderr) per rung:
- `greedy`
- `minimax` depth 1, depth 2, depth 3
- `minimax` time-budgeted (e.g. 0.25s/move) — the "as strong as we cheaply can" rung
The harness is parallel (ProcessPoolExecutor) like `eval_az.py`. Output is a single table: the net's win-rate vs each rung — the strength curve we track across future training runs.

**Note on the in-loop campaign eval:** the staged change to `scripts/campaign.py` already generalized the per-iteration eval to `winrate_vs(..., opponent=...)` defaulting to **greedy**; `eval_ladder.py` is the heavier, periodic strength check (vs the full minimax ladder), run on saved checkpoints rather than every iteration.

**Testing:** a smoke run on a tiny config (2 games/rung, depth-1 only, cpu) completes and returns valid fractions; a sanity check that a strong agent (e.g. minimax-d3 itself) scores ~100% vs greedy and ~50% vs its own rung.

---

## Build order (each independently testable; no campaign launched)

1. **Unit 1** (wall-legality) — extend the differential fuzz first, then optimize; gate on the oracle.
2. **Unit 2** (L-R aug) — commutation tests, then wire into the campaign.
3. **Unit 3** (endgame solver) — differential vs Python negamax, then the three integrations + hit-rate instrumentation.
4. **Unit 4** (minimax ladder) — the eval harness; run it on `models/campaign10k/campaign_final.pt` to record the current net's strength vs the ladder (evaluation, not training).

## Out of scope / deferred

- Broadening the endgame solver beyond `walls_left==(0,0)`.
- **Opponent-curriculum training** (net-MCTS vs minimax/heuristic-MCTS as a sparring partner, annealing to self-play) — discussed, promising, separate spec.
- Small-board weak-solve + "vs perfect play" absolute anchor.
- Transposition table / symmetry in MCTS search.
- Minimax-vs-minimax supervised value pretraining.
- **The actual next training campaign** (this work is preparation only).
