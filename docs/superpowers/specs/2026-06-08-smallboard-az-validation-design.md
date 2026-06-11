# Small-Board AZ Convergence Validation

**Date:** 2026-06-08
**Status:** Approved (design phase)
**Branch:** `az-bootstrap`

## Overview

The 9×9 AZ net (10k-game campaign) loses 0% vs greedy and the whole minimax ladder — it beats only random. That leaves an open question: is the net weak because it's **undertrained**, or because the **pipeline has a flaw**? This project answers it.

On boards small enough to **solve perfectly** (3×3, 5×5 — the writeup *Solving Quoridor* solves area ≤28), we can establish ground-truth optimal play and check whether our AZ pipeline **converges to it**. If AZ reaches near-optimal on small boards, the pipeline is validated and 9×9 is purely a scale/compute problem. If it does not, we've found a real pipeline bug — the most valuable thing to know.

**Approach:** a self-contained `smallboard/` Python package, parameterized by board size `N` and walls-per-player `W`, **independent of the 9×9 Rust/Python production core**. Tiny boards run fine in pure Python, and a separate stack can't destabilize the heavily-tested production code.

**Stack:** pure Python + PyTorch (CPU is fine at these sizes). Mirrors the AZ patterns already in `agents/az/` but parameterized by `N`.

## Architecture

```
smallboard/
  engine.py     N-parameterized Quoridor (state, moves, jumps, wall legality, apply, winner, BFS)
  solver.py     exact full-game perfect solver (negamax + alpha-beta + transposition table)
  encoding.py   6xNxN planes + the (12 + 2*(N-1)^2) action space, parameterized by N
  model.py      small CNN (policy + value heads) over NxN
  mcts.py       PUCT search (net-guided)
  selfplay.py   AZ self-play (dense path-diff reward shaping, like the 9x9 work)
  train.py      self-play -> targets -> train loop
  validate.py   the experiment: solver cross-check + AZ-vs-solver convergence metrics
tests/smallboard/  unit + differential tests per module
```

Each module has one responsibility and is small. `engine.py` is the single source of truth for rules; everything else sits on top.

## Components

### `engine.py` — N-parameterized rules
- `Board(N, W)`: config (board size, walls per player). Goal rows: player 0 → `N-1`, player 1 → `0`. Starts: player 0 at `(N//2, 0)`, player 1 at `(N//2, N-1)`.
- `State`: pawns `((c,r),(c,r))`, `h_walls`/`v_walls` (frozensets of anchors, `c,r ∈ 0..N-2`), `walls_left`, `turn`.
- `legal_steps` (adjacency, blocked-by-wall, straight + diagonal jumps), `legal_walls` (in-bounds, no overlap/cross, walls-left>0, path exists for both via BFS — with the floating-wall fast-path), `legal_moves`, `apply_move`, `winner`, `is_terminal`, `shortest_path_len` (BFS, for the dense reward).
- This is a clean reimplementation of `core/rules.py` parameterized by `N` (not a copy of the 9×9 code). Validated differentially against `core` at `N=9` to prove the parameterization is faithful.

### `solver.py` — exact full-game perfect solver
- `solve(state) -> (value, best_moves)`: game-theoretic value for the side to move (+1 win / 0 draw / −1 loss) and the set of optimal moves, via **negamax + alpha-beta + a transposition table** keyed on the full state `(pawns, h_walls, v_walls, walls_left, turn)`.
- **Draw handling (the key correctness risk):** walls only accumulate (each player places ≤`W`), so wall structure is monotone (a DAG); draws arise from **pawn cycling** (a losing side stalling). Use a **depth bound** generous enough that any forced win is found within it, with unresolved lines scored as draws (value 0); plus path-cycle detection in the TT where cheap. Once both players are out of walls it reduces to the bounded race (the same logic as the 9×9 endgame solver).
- **Performance:** TT + alpha-beta + move ordering (by shortest-path-difference) + the floating-wall fast-path. 3×3 is trivial; 5×5 with modest `W` is tractable (the writeup solves 5×5 in minutes). **If pure-Python 5×5 solving is too slow, port only `solve`'s hot loop to Rust** — flagged as a contingency, not v1.

### AZ stack (`encoding.py`, `model.py`, `mcts.py`, `selfplay.py`, `train.py`)
- `encoding.py`: `encode_planes(state)` → `(6, N, N)` (me, opp, h-walls, v-walls, walls-left_me, walls-left_opp), current-player-relative (the canonical row-flip). `N_ACTIONS = 12 + 2*(N-1)**2`; `move_to_action`/`action_to_move` parameterized by `N`.
- `model.py`: a small CNN (a few conv layers over `N×N`) with policy (`N_ACTIONS`) and value (`tanh`) heads. (No distance head needed for the validation; keep it minimal — policy + value.)
- `mcts.py`: PUCT guided by the net (the same algorithm as `agents/az/mcts_nn.py`, parameterized encoding/action space).
- `selfplay.py`/`train.py`: AZ self-play producing `(planes, π, z)` examples with the **path-diff dense-reward shaping** (`v_target = λz + (1−λ)tanh(path_diff/scale)`, annealed) that worked at 9×9; minibatched training.

### `validate.py` — the experiment
The deliverable. Two stages:
1. **Validate the solver** (prove the ground truth): reproduce the writeup's known results — 5×5 is a 2nd-player win at ≤4 walls (1st-player at >4); even-height boards are 1st-player wins. On 3×3, additionally cross-check `solve` against an **exhaustive brute force** (no alpha-beta/TT) over a sample of positions — they must agree on value and optimal-move set.
2. **Validate AZ convergence** (solver as ground truth), three metrics:
   - **(a) AZ vs the perfect solver:** play AZ as the player who *should* win/draw at the start; does it achieve the game-theoretic result (win won starts, hold drawn ones, never lose a won one) against optimal solver play? Both seatings.
   - **(b) Optimal-move agreement:** over a sample of reachable positions, the fraction where AZ's chosen move (greedy, post-search) is in the solver's optimal-move set.
   - **(c) Value accuracy:** correlation / sign-agreement of AZ's value head vs the solver's exact value over sampled positions.

## Success criterion

AZ, trained on a small board, **(a)** achieves the game-theoretic result vs the perfect solver and **(b)** reaches high optimal-move agreement (target ≥~90% on 3×3). Met ⇒ the AZ pipeline is **proven sound** (9×9 is a scale problem). Not met ⇒ a real pipeline bug, surfaced where it's debuggable.

## Scope & sequence

1. **3×3 first** — trivially solvable; proves the entire engine→solver→AZ→validate loop end-to-end quickly and cheaply.
2. **5×5 next** — the meaningful validation (real wall tactics); Python solver, with the Rust-port contingency if needed.

Default configs to start (expandable): **3×3 with `W=1`**, **5×5 with `W=2`** (a known 2nd-player win). The solver computes the actual ground truth for whatever `(N,W)` we choose; matching the writeup is the solver cross-check, not a dependency.

## Testing

- **engine:** unit tests (jumps, wall overlap/cross, path-must-exist); **differential vs `core` at N=9** (the parameterization must reproduce the production rules exactly).
- **solver:** vs exhaustive brute force on 3×3; vs the writeup's known win/loss/draw results on 5×5 (and the even-height = P1-win property); determinism.
- **AZ stack:** encoding round-trip + a commutation/sanity check; a self-play smoke (well-formed `(planes, π, z)`); a training smoke (loss decreases).
- **validate:** the metrics run end-to-end on 3×3 and produce the convergence numbers.

## Out of scope

- Parameterizing the 9×9 Rust/Python production core (this is a separate lean stack by design).
- GPU/MPS (CPU is fine at N≤5).
- Boards beyond 5×5 in this effort (the writeup goes to area ≤28; 7×3 etc. are future).
- Using the small-board solver as a 9×9 opponent (different board).
- The stronger-9×9-heuristics work (the *second* approved direction — its own spec after this).
