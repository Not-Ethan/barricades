# Design: AlphaZero 9×9 Race-Plateau Ablation + Definitive Campaign

**Date:** 2026-06-10 · **Branch:** `az-cloud-training` (off `az-bootstrap`) · **Target:** RunPod CUDA

## 1. Problem

The 25k-game AZ campaign collapsed into a **pure-race local optimum**: mean game length
fell 79 → 17 plies by iteration ~13 and pinned there; the net never learned wall tactics
and loses to the minimax ladder. The pipeline itself is proven sound (small-board
validation: 100% optimal-move agreement on 3×3, 98.3% on 5×5, never loses a won
position vs an exact solver), so this is a training/dynamics failure, not a pipeline bug.

### Mechanism (the diagnosis driving this design)
A wall is +1 **only when it lengthens the opponent's path without lengthening your own** —
i.e. when the pawns occupy different regions so the wall is *asymmetric*. Head-on racing
is exactly the configuration where that's impossible: both pawns share a corridor, so any
wall in the contested zone obstructs both. Wall skill is really the skill of *maneuvering
into an asymmetric position first, then walling* — a multi-ply, delayed-payoff sequence.

Two compounding causes:
1. **The dense reward rewards racing.** `v_target = λ·(z·γ^plies) + (1−λ)·tanh(path_diff/scale)`
   scores your *instantaneous* shortest-path lead. Racing maximizes that lead for free;
   the wall setup *costs* lead now for an asymmetric wall later. The dense term is symmetric
   and myopic — it optimizes the exact quantity that misleads here, and λ anneals 0→1 so the
   dense signal *dominates early*, matching the iter-~13 collapse.
2. **Self-play co-adaptation.** Once both players race, walls are genuinely low-value
   *against a racer* — the policy sits at a real local Nash of its own sparring partner.
   PUCT exploration is anchored to the prior, so once walls get ~0 prior probability, MCTS
   stops expanding them and never generates the counterfactual data to correct the prior.
   The basin self-seals at both the policy and search level.

## 2. Goal & hypotheses

Run a **causal ablation** to attribute the plateau to its cause(s), then run the definitive
campaign on the winning design. Two hypotheses, tested as a 2×2 factorial:

- **H-reward:** the dense path-diff term biases toward racing.
- **H-coadapt:** self-play against a pure racer removes any gradient toward walls.

## 3. Scope — three phases (this spec covers Phase 0–2)

### Phase 0 — gates + cost (pod)
Handoff smoke gates: native test suite green on Linux; tiny `campaign.py 2 64 60 cuda`
produces sane checkpoints/log; CPU↔CUDA policy/value agreement to ~1e-4; throughput tune
over `n_games ∈ {256,512,1024}` at `sims=100`. **Deliverable: real games/s + projected
total cost, confirmed with the user before any sustained spend** (handoff guardrail ~$25;
expected well under, but confirmed with measured numbers, not a guess).

### Phase 1 — ablation @ ~25k games/arm
The plateau historically appears by iter ~13, so 25k is sufficient to observe escape-or-not.
Four arms (the 2×2):

| | Self-play (control opp.) | Opponent pool |
|---|---|---|
| **Dense reward (control)** | **A** — reproduce plateau at scale (required baseline) | **C** — isolates H-coadapt |
| **Drop dense (λ≡1)** | **B** — isolates H-reward | **D** — combined best-bet |

All arms run with the agreed cheap hedges: slower λ anneal (`warmup_frac=0.8`), ladder eval
every iteration, per-iteration diagnostics (§6), all checkpoints kept. Run sequentially on
one pod (the stepper is CPU-bound; parallel arms would contend for vCPUs). Start FRESH — no
`init_ckpt` (warm-starting the 25k net would bake in the race optimum).

### Phase 2 — definitive campaign @ 100k games
The winning design from Phase 1, run to the full project goal, with ladder eval per
checkpoint and the cost/throughput report. **Reward fix is earned, not assumed:** if Phase 1
shows reward is causal, Phase 2 uses **PBS** (§4) rather than merely dropping dense; if the
*pool* is the decisive lever, Phase 2 keeps dense+pool.

### Phase 3 — full-scale convergence (FUTURE, not designed here)
After Phase 1–2, design a full-scale run (~1M games, or whatever the Phase-1 read indicates
is needed to converge) on the validated design. Explicitly deferred — designed once we know
which levers matter and what the throughput/cost curve looks like.

## 4. Reward treatment

- **Ablation level = drop dense (`λ≡1`, pure outcome `z·γ^plies`).** Zero-risk, unambiguous
  diagnostic that directly answers H-reward. Implemented by forcing λ=1 (skip the blend /
  anneal), no new math.
- **PBS held in reserve as the Phase-2 fix.** Potential-based shaping (Ng–Harada–Russell)
  `F = γ·Φ(s′) − Φ(s)`, `Φ = path_diff`, is *policy-invariant* — it accelerates learning
  toward the true optimum (which values walls) without biasing toward racing. It is **not**
  used in the ablation because it is a fix, not a diagnostic, and is subtle to implement
  correctly (the value head trains to `V−Φ`, requiring a compensating shift at MCTS leaf
  eval — exactly the silent-error class the handoff warns against). Applied only if/when
  Phase 1 shows reward is causal, and then carefully with its own verification.

## 5. Opponent pool (Arms C, D) — pure-Python, no native changes

The frozen native crate exposes a complete single-game MCTS driver via the `Tree` pyclass
(`prepare_leaf()` → (6,9,9) planes, `receive(policy[140], value)`, `apply_root_noise`,
`best_move(temp)` → (move, π[140]), `advance(mv)`). This lets us drive a game where **each
side's leaves route to a different net** entirely in Python.

- **New script `scripts/selfplay_pool.py`:** batches N games' `prepare_leaf()` outputs into
  one forward per tick, routing each game's leaf to the **learner net** or a **frozen
  checkpoint net** by whose turn it is. Mirrors `SelfPlayPool` knobs (`c_puct=1.5`,
  `dirichlet α=0.5 / ε=0.25`, `temp_moves=10`, `sims`, `max_plies`).
- **Pool = last K checkpoints.** Each pool-game samples one opponent from the pool; the
  learner takes a random side. A configurable fraction (default 50%) of each iteration's
  games are pool-games; the rest are vanilla `SelfPlayPool` self-play (keeps the learner
  sharp and the GPU fed via the fast native path).
- **Only the learner's positions become training examples** (standard league practice); the
  frozen opponent's moves are not trained on.
- The fast native `SelfPlayPool` remains the path for Arms A/B and the self-play fraction of
  C/D. The Python `Tree` driver is slower (per-tick Python overhead, two forwards) but
  acceptable at the 25k ablation budget and bounded by the pool-game fraction.

## 6. Diagnostics (in scope; derivable from drained examples)

Per iteration, log alongside loss / mean game-length / ladder winrate:
- **Wall-placement rate** — fraction of selected moves with action index ≥ 12 (indices 0–11
  are pawn moves per `encoding.py`; ≥12 are walls).
- **Root-policy entropy** — entropy of the visit-count policy π.

These turn the verdict from "length hit 17 again" into a mechanism read: does the wall prior
collapse, and does the collapse track the dense-dominated early phase?

## 7. Success criterion

An arm **escapes** the plateau if, by end of run: mean game length stays well above ~17,
**and** wall-placement rate stays non-trivial, **and** ladder winrate vs minimax climbs
above ~0. Ladder eval (`scripts/eval_ladder.py`) every iteration — greedy-only winrate is a
known-weak signal.

## 8. Boundaries

- **No edits** to `solver/`, `smallboard/`, `core/`, or `native/` (Rust). All work is new
  Python (`scripts/selfplay_pool.py`; additions to `campaign.py` / `train.py`) plus the
  report.
- Commit `az:`-prefixed on `az-cloud-training`. Never commit the RunPod API key or
  checkpoints >100 MB (rsync those off-pod; pods die, checkpoints are the product).
- Budget guardrail: confirm projected cost with the user if it exceeds ~$25; in any case
  report the Phase-0 measured estimate before launching Phase 1.

## 9. Components & interfaces (new/changed)

- `scripts/selfplay_pool.py` (NEW) — `run_selfplay_pool(total_games, n_games, sims, device,
  learner_net, pool_nets, pool_frac, side_random, seed, max_plies, temp_moves)` →
  `(examples, stats)`. Same return shape as `run_selfplay` so `campaign.py` can consume it
  interchangeably.
- `scripts/campaign.py` (CHANGED) — add: `warmup_frac` (λ anneal knob, default 0.8 here),
  `reward_mode ∈ {dense, outcome}` (outcome ⇒ λ≡1), `opponent ∈ {self, pool}` + pool knobs,
  per-iteration diagnostics logging, ladder eval hook every iteration. Backward-compatible
  defaults so existing behavior is unchanged.
- `scripts/eval_ladder.py` (REUSE) — called per iteration for Arms A–D.
- A small throwaway CPU↔CUDA agreement check (Phase 0 gate 3), not committed.

## 10. Testing

- Phase 0 gates are the integration tests for the port (native suite, tiny campaign, CPU↔CUDA
  agreement).
- `scripts/selfplay_pool.py`: a unit test that a 2-game pool run with `pool_frac=1.0` and a
  pool of one known checkpoint produces well-formed examples (shapes (6,9,9)/(140)/scalar/(4))
  and that learner-only examples are collected. Differential sanity: with `pool_frac=0` it
  must reproduce `run_selfplay` example statistics within noise.
- Diagnostics: unit-test wall-rate and entropy computations against a hand-constructed π batch.
