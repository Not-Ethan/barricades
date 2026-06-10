# Results: AZ 9×9 Reward Ablation (A vs B), Cloud Run 2026-06-10

**Setup:** RunPod RTX 3090 (community, $0.22/hr), branch `az-cloud-training` off
`az-bootstrap`. Each arm = 25 iterations × 1000 games = 25k games, `n_games=512`,
`sims=100`, `warmup_frac=0.8`, minimax eval every 5 iters, all checkpoints kept.

This run executed the **reward** factor of the planned 2×2 ablation. The **opponent-pool**
factor (arms C/D) was **descoped** this round — see "Pool arms" below.

## The two arms

- **A — dense (control):** `v_target = λ·(z·γ^plies) + (1−λ)·tanh(path_diff/scale)`,
  λ annealed 0→1 over the first 80% of iterations.
- **B — drop-dense:** `λ≡1` (pure outcome `z·γ^plies`), no path-diff shaping.

Everything else identical (same engine, sims, schedule, seeds-by-arm).

## Headline result

**Both arms collapsed into the race basin. Dropping the dense reward did not prevent it.**

| | game length (start → end) | wall_argmax (end) | winrate vs minimax |
|---|---|---|---|
| **A — dense**      | 78 → **18.5** (pinned from it13) | ~0.13 | **0.0** at every eval |
| **B — drop-dense** | 77 → **22.5** (settled it17+)    | ~0.13 | **0.0** at every eval |

Arm A's collapse (game length 39→24→19.5 at it12–13, then pinned ~18) is an exact
reproduction of the original 25k failure mode (length → ~17). Arm B collapsed too, to
~22 plies. **Neither arm ever beat minimax.**

## Interpretation

1. **The dense path-diff shaping is NOT the root cause of the plateau.** Pure-outcome
   self-play (B) races just as surely. Reward tweaks alone will not fix this.
2. **But dense is a real accelerant.** A collapsed ~18% shorter than B (18.5 vs 22.5
   plies) with a sharper mid-training wall-usage dip (`wall_mass` 0.53→0.18 by it8 in A,
   vs B holding ~0.40). So the shaping makes racing worse — an accelerant, not the cause.
3. **This implicates self-play co-adaptation** — the lever the opponent-pool arms (C/D)
   were designed to test. The reward ablation has, in effect, earned that experiment:
   the evidence now points to opponent diversity / exploration, not reward design.

**Caveat:** both nets end with non-trivial `wall_mass` (~0.42–0.46) but `wall_argmax`
~0.13 and 18–22-ply games — "race with an occasional tactical wall," never positional
wall play. Short length + 0 winrate vs minimax are the decisive race-collapse signatures.

## Strength

Both arms' **inline minimax eval read 0.0 winrate at every checkpoint** (iters
0/5/10/15/20/24), i.e. neither net ever beat the time-budgeted minimax — the decisive
strength signal, consistent with the race-collapse curves.

The full `scripts/eval_ladder.py` ladder (greedy + minimax d1/d2/d3) was **not** completed:
on this 64-core host its default `ProcessPoolExecutor` (one worker per core) × torch's
per-worker intra-op threads oversubscribes to ~thousands of threads and thrashes (load avg
hit 88). Re-run with `OMP_NUM_THREADS=1` and a capped worker count if a detailed ladder is
wanted; the verdict here does not depend on it (inline winrate=0 already establishes
"loses to minimax").

## Pool arms (C/D) — descoped, and why

The opponent-pool design needs a different net on each side of a game. The native
`SelfPlayPool` drives all games with one net, so C/D used a pure-Python two-net driver
(`scripts/selfplay_pool.py`) over the exposed `Tree` API. It is **correct** (validated
engine-equivalent to the native pool) but **~100× too slow**: profiling shows `prepare_leaf`
(MCTS step; movegen runs a wall-blocking BFS) at ~4 ms/call dominating 95% of time. Native
runs that **rayon-parallel with the GIL released**; the Python driver runs it **serially**
(~0.12 g/s vs ~12 g/s). Multiprocessing buys only ~8×. So C/D would take ~7 h/arm.

**To run the pool arms (and to reach the 1M-game goal), the fast path is a minimal native
addition**: teach `SelfPlayPool` to carry a per-game net assignment and tag each pending
leaf with the net that should evaluate it, so Python routes learner-vs-frozen evals while
the rayon stepping stays in Rust. This touches the pool *orchestration*, not the
oracle-validated movegen, and should be re-checked with the differential tests.

## Recommended next step

The reward question is answered (shaping is an accelerant, not the cure). The priority is
now the **co-adaptation lever**: implement the native two-net pool addition, then run arms
C/D (and likely a past-checkpoint league / curriculum) at scale. Reward-only redesign is
de-prioritized by this result.
