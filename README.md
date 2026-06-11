# Barricades — Quoridor Engines & the 6×5 Solve

Engines, solvers, and experiments for **Quoridor** (the wall-and-pawn race game),
built around two questions: *can small Quoridor boards be solved exactly beyond the
known frontier?* and *can an AlphaZero-style agent learn strong 9×9 play?*

## Headline result

**6×5 Quoridor is weakly solved for every wall count** — the first board of area > 28 with a known game value (the prior frontier:
[Solving Quoridor](https://grantslatton.com/solving-quoridor), area ≤ 28):

| walls per player | 0 | 1 | 2 | 3 | **4** | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **winner (perfect play)** | P2 | P2 | P2 | P2 | **P1** | P1 | P1 | P1 | P1 | P1 | P1 |

A clean parity→tempo transition at 4 walls, no oscillation. Full result, methodology,
and trust argument: **[`docs/superpowers/6x5-solved-results.md`](docs/superpowers/6x5-solved-results.md)** ·
raw solve logs: [`docs/superpowers/raw/ladder_logs/`](docs/superpowers/raw/ladder_logs/).

## Repository map

| dir | what |
|---|---|
| [`solver/`](solver/) | **The exact solver** (Rust, standalone). Parallel lazy-SMP alpha-beta, depth-folded sharded TT, exact retrograde race endgames, DSU wall-legality filter, live heartbeat observability. Differential-tested against two independent rules engines and an unpruned brute-force oracle; reproduces the published values for 3×3 / 4×4 / 5×5 / the 8×3 draw. |
| [`core/`](core/) | Reference Python rules engine (the correctness oracle). |
| [`native/`](native/) | Rust/PyO3 engine for AlphaZero self-play: bitboard movegen, PUCT MCTS with subtree carryover, batched parallel self-play pool. |
| [`agents/`](agents/) | Baselines (random/greedy/minimax) + the AZ stack (3-head net, dense-reward training). |
| [`smallboard/`](smallboard/) | Self-contained N×N AZ validation lab — proved the training pipeline converges to exact-solver-optimal play on 3×3–5×5. |
| [`docs/superpowers/`](docs/superpowers/) | Results & research notes (see below). |

## Key documents

- [`6x5-solved-results.md`](docs/superpowers/6x5-solved-results.md) — the consolidated result.
- [`solver-legality-filter-comparison.md`](docs/superpowers/solver-legality-filter-comparison.md) —
  controlled benchmark: exact connectivity (union-find over wall posts, sound by planar
  duality) vs the contact heuristic for wall-legality filtering.
- [`solver-pruning-theorems.md`](docs/superpowers/solver-pruning-theorems.md) — novel,
  falsification-tested pruning theorems for Quoridor (wall-insertion invariance /
  "mustplay" footprints; one-sided frozen-race bounds) — sound, validated, and honestly
  reported as *subsumed by a good transposition table* at this board size.
- [`solver-phase1-measurements.md`](docs/superpowers/solver-phase1-measurements.md) —
  the chronological lab notebook: optimization ladder, profiling, negative results
  (df-pn measured 10–19× *worse* than alpha-beta here; why), bug post-mortems.
- [`smallboard-validation-results.md`](docs/superpowers/smallboard-validation-results.md) —
  AZ pipeline validation against exact solvers (90–100 % optimal-move agreement).

## Quick start (the solver)

```bash
cd solver
cargo test --release          # exactness gate suite
cargo build --release --bin solve
./target/release/solve 6 5 4  # solve 6x5 at 4 walls/player -> value=Win (P1)
```

Useful env knobs: `QS_THREADS` (default: all cores), `QS_TT_MB` / `QS_RACE_MB`
(cache budgets — keep total under free RAM), `QS_PROGRESS_SECS` (heartbeat interval).
Small boards solve in seconds; 6×5 at high wall counts wants ~16 threads, tens of GB
of cache, and 1–2 hours per wall count.

## Status

- **Exact solving:** 6×5 complete. 7×5 (area 35) is next.
- **AlphaZero 9×9:** training campaign in progress (see
  [`docs/az-cloud-training-handoff.md`](docs/az-cloud-training-handoff.md)).

Development was AI-assisted (Claude-based coding agents) under human direction, with
the verification discipline documented throughout the research notes: every
optimization gated on differential equality against unpruned oracles, adversarial
falsification for novel theorems, and negative results reported alongside positive ones.
