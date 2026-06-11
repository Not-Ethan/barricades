# AZ Campaign Phase 2 — Fast Self-Play + Dense Rewards

**Date:** 2026-06-08
**Status:** Approved (design phase)
**Branch:** `rust-core` (continues the native-core work)
**Depends on:** the native core (`barricades_native`: `SelfPlayPool`, `Tree`, encoding, game core) and `docs/superpowers/specs/2026-06-08-rust-core-design.md`.

## Overview

The native core made self-play CPU work negligible (~4%), but a benchmark gate
revealed two blockers to running a real AlphaZero campaign on this Mac:

1. **Throughput is pipeline-bound.** Batched MPS self-play achieves only
   ~8–12k pos/s versus the ~68k forward-only ceiling — a ~6× gap — because the
   driver is *synchronous*: host→device transfer → GPU forward → `.cpu()` sync →
   Rust `step()`, all serialized so the GPU idles during the CPU phases. Bigger
   batches do **not** close it (128→2048 moved pos/s only 7k→9.8k). Projected
   100k self-play: ~17–46h depending on net/sims.
2. **The reward signal is too sparse.** With a weak/uncalibrated policy, games
   meander ~140 plies (a sane Quoridor game is ~25–40), and every position of a
   game gets the *same* terminal win/loss label (or 0 for a capped draw). The net
   can't assign credit, so it never learns to stop wandering. Measured: the
   bootstrap net produced games as long as the random net (142 vs 140 plies) —
   game length is a reward-signal problem, not a net-capacity problem, and it
   won't fix itself until the net actually learns.

**Goal of Phase 2:** an iterated self-play↔train campaign that (a) generates
self-play at **≤2h-per-100k** throughput and (b) escapes the wandering regime via
**dense rewards**, validated by **game length dropping** and **win-rate vs the
existing bot pool rising** across iterations.

**Stack:** the existing Python AZ stack (`agents/az/model.py`, `train.py`) +
the Rust `barricades_native` self-play pool + PyTorch on MPS.

## Architecture

Three workstreams, each independently testable, built and **gated in sequence**:

```
A. Throughput        async double-buffer driver  +  Rust subtree carryover  -> benchmark gate (<=2h/100k)
B. Dense rewards     aux distance head  +  blended/annealed value target     -> unit tests
C. Campaign          iterate {self-play(A) -> dense targets(B) -> train -> eval} -> smoke (len down, winrate up)
```

Workstreams A and B are independent (A is self-play speed; B is training
quality). They converge in C, the campaign loop that runs A's self-play, forms
B's targets, trains, and evaluates each iteration.

## Workstream A — Throughput (self-play speed)

### A1. Async double-buffer driver

**Insight:** MPS executes `net(x)` asynchronously and only blocks at `.cpu()`.
So while batch A's forward runs on the GPU, the CPU is free to run a *different*
pool's `step()` (pure Rust, GIL released). Two pools ping-ponging overlap GPU
and CPU with no threads and **no Rust change**:

```
P0, P1 = two SelfPlayPool (each ~half the desired concurrency)
prime: submit forward(P0.step())            # async, no .cpu()
loop while either pool has games_remaining:
    planes_other = P_other.step()           # Rust CPU, overlaps the in-flight GPU forward
    pol, val = inflight.result.cpu()         # sync the prior forward (GPU already done)
    inflight.pool.feed(pol, val); collect inflight.pool.drain()
    submit forward(planes_other)             # async
    swap pools
drain both pools fully at the end (the example-loss fix from Phase 1 applies)
```

Replaces `scripts/selfplay_native.py`'s `run_selfplay` with a pipelined version.
Reusable signature exposing `n_games, total_games, sims, max_plies, temp_moves,
ckpt, device, channels, blocks`. A `ckpt` is loaded as a raw `state_dict`
(`models/az_bootstrap.pt` is a bare `OrderedDict`, not `{"model": ...}`).

**Target:** recover most of the ~6× gap → ~40–50k pos/s.

### A2. Subtree carryover (Rust `Tree` + `SelfPlayPool`)

Today each move rebuilds the tree from scratch (`Tree::new`), discarding the
previous search. Carryover keeps the chosen child's subtree:

- `Tree::advance(&mut self, mv: Move)` — find the root child matching `mv`, make
  it the new root (`parent = -1`), **negate `w` for every retained node** (the
  root player flips each ply, and `w` is stored in root-player perspective, so a
  sign flip re-bases it), keep `n`/priors/`expanded`, drop the rest of the arena
  (compact to the retained subtree), reset `parked`/`noised`, and set
  `root_player = new_root.state.turn`.
- `SelfPlayPool::commit_move` calls `tree.advance(mv)` instead of
  `Tree::new(next, ...)` when carryover is enabled; `sims_done` resets to
  `min(retained_root_visits, sims)` (so the move runs the *remaining* evals, and
  commits immediately if the retained subtree already exceeds the budget).
- **Dirichlet-noise timing under carryover:** with a fresh tree, the pool applies
  root noise on the first eval (the root-expansion `receive`). A carried-over
  root is *already expanded*, so that trigger never fires — instead `advance`
  applies root noise to the new (already-expanded) root immediately (resetting
  `noised`), so every move still gets root exploration noise.
- **Behind a config flag** (`carryover: bool`, default on for the pool;
  exposable). A/B-benchmarked for **strength** before trusting it, per the
  project rule that engine-behavior changes are validated empirically, not
  assumed.

**Target:** ~1.5–2× fewer evals per game.

### A3. Benchmark + config sweep

Extend `scripts/bench_selfplay.py` to report pos/s, mean batch, and the 100k
projection across `sims ∈ {50, 100}`, `max_plies ∈ {60, 80, 200}`, and
carryover on/off. **Gate: the campaign's full run is only launched once a config
projects ≤2h-per-100k.** A move cap (`max_plies ~80`) is a config lever here:
140-ply meandering games are mostly noise for a weak net; capping ~halves eval
count and is standard early-iteration AZ practice — made *safe* by the dense
reward (B), since a capped draw is no longer 0 signal.

## Workstream B — Dense rewards (training quality)

The `SelfPlayPool` already records, per example, the raw material:
`feats = [path_diff, walls_left_own, walls_left_opp, plies_to_end]` (from the
recorded mover's perspective). B turns these into a dense learning signal.

### B1. Aux distance head (`agents/az/model.py`)

`QuoridorNet.forward` returns `(policy, value, dist)`, where `dist` is a scalar
prediction of `path_diff` (normalized). Implemented as a third small head off the
shared trunk (mirroring the value head's structure). **Backward-compatible:**
old 2-head checkpoints load via `load_state_dict(..., strict=False)`; the new
head trains from fresh init. `NetWrapper.predict` ignores `dist` (MCTS only needs
policy+value), so the self-play pool path is unaffected.

### B2. Dense value target + annealed schedule (`agents/az/train.py`)

For each example, with `z` = outcome from the mover's POV, `k = plies_to_end`,
`pd = path_diff`:

```
shaped_outcome = z * gamma ** k                         # faster-win discount
potential      = tanh(pd / SCALE)                       # dense per-position signal
v_target       = lam * shaped_outcome + (1 - lam) * potential
dist_target    = pd / DIST_NORM
```

Loss: `policy_loss + value_loss(value, v_target) + BETA * mse(dist, dist_target)`.

`lam` (λ) is **annealed 0→1 across campaign iterations** (e.g. linear over the
first N iters): early training leans on the dense potential to escape wandering,
then fades so the final value head predicts the true outcome and play is not
capped at the heuristic's strength. `gamma`, `SCALE`, `DIST_NORM`, `BETA`, and
the λ schedule are config.

Perspective consistency: `z`, `pd`, and the encoding are all current-player /
mover-relative, so the blend is perspective-correct without extra sign handling.

### B3. Batch/step plumbing

Extend `examples_to_batch` to consume the pool's `(planes, π, z, feats)`
4-tuples and produce `(planes, π, v_target, dist_target)` given the current λ;
extend `train_step` for the 3-head loss. Keep the old 3-tuple path working (or
remove it if no longer used — check callers first).

## Workstream C — Campaign (orchestration + validation)

### C1. Campaign loop (`scripts/campaign.py`)

Iterate, carrying the net forward:

```
for it in range(iterations):
    lam = anneal(it)
    examples = run_selfplay(net=current, games=G, sims=S, max_plies=cap, device="mps")  # A1
    batches  = form_dense_targets(examples, lam, gamma, ...)                              # B2
    for epoch: train_step(net, opt, batch)                                                # B3
    save_checkpoint(net, f"models/campaign_it{it}.pt")
    metrics = eval_vs_pool(net, opp=[random, greedy, minimax, mcts], games=...)            # reuse eval_az.py
    log(it, mean_game_len, lam, loss, metrics)                                             # the trend record
```

Reuses `scripts/eval_az.py`'s parallel match harness for evaluation.

### C2. Short smoke (the proof)

A tiny campaign (2–3 iterations, small G/S) asserting the **trends**, not
absolute strength: mean game length **decreases** across iterations and win-rate
vs random (and ideally greedy) **increases**. This is the empirical evidence that
the dense signal escapes wandering and that the loop is wired correctly.

### C3. Full campaign + bot-pool eval

The payoff: a longer campaign reaching real strength, evaluated against the bot
pool. **Separately authorized** — not launched from the implementation plan;
gated on A3 (≤2h projection) and C2 (trends confirmed).

## Testing strategy

- **A1 (pipeline):** benchmark shows higher pos/s than the synchronous baseline
  at equal mean batch; correctness — total examples drained equals the
  synchronous driver's for the same seed/config (no loss, no duplication).
- **A2 (carryover):** (i) a re-root invariant test — after `advance(mv)`, the new
  root's retained children carry their prior visit counts and sign-flipped W
  correctly, and `best_move` returns a legal move; (ii) a strength A/B —
  carryover-on vs carryover-off over a fixed budget must not regress win-rate vs
  random/the heuristic (measure, don't assume).
- **B (dense target):** unit tests for `form_dense_targets` — the λ blend,
  γ-discount, and perspective signs match hand-computed values on crafted
  examples; aux-head output shape; a `train_step` on a fixed batch reduces total
  loss over a few steps.
- **C (campaign):** the C2 smoke trend assertions (length down, win-rate up).
- **No regression:** the existing 225-test suite stays green.

## Out of scope (YAGNI / later)

- **MCTS-value bootstrap** (KataGo-style value mixing) — considered, deferred;
  path-diff shaping + aux head + discount capture most of the benefit.
- **Virtual-loss in-pool pipelining** and **threaded producer/consumer** drivers
  — the 2-pool MPS-async double-buffer is simpler and sufficient.
- **Richer potentials** (second-shortest / vertex-disjoint paths, wall geometry)
  — `path_diff` is the high-value signal; revisit only if needed.
- **The full campaign run itself** (C3) — a separately authorized compute job.
- **Distributed / multi-process** self-play.
