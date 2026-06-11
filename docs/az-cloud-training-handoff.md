# Handoff: AlphaZero 9×9 Training — Cloud Campaign (RunPod)

**Date:** 2026-06-10 · **Branch:** `solver-and-az` (renamed from `az-bootstrap`) · **Audience:** the agent taking over AZ
training. The solver effort (`solver/`, `smallboard/`) is a SEPARATE workstream owned by
another session — read those dirs for reference only, do not modify them.

---

## 1. Mission

Port the validated 9×9 AlphaZero self-play stack from local Apple-MPS to a RunPod CUDA
pod and run the full-scale campaign (target: **100k self-play games**, the original
project goal), with the evaluation ladder wired in so we learn definitively whether
scale breaks the known training plateau (§4). Deliverables:

1. Linux/CUDA port verified by the smoke gates (§6) — no silent numerical drift.
2. The 100k campaign run to completion with per-iteration checkpoints + logs.
3. Eval-ladder results per iteration (vs greedy + minimax ladder), reported with the
   final checkpoint synced back to the repo machine.
4. A measured cost/throughput report (games/s on the pod vs the M1 baseline ~5–22/s).

## 2. What exists and works (all on `solver-and-az` (renamed from `az-bootstrap`), all tests green)

### The game
Quoridor ("barricades") 9×9: move 1 step orthogonally (with straight/diagonal jump
rules) or place one of 10 walls (2-segment, may not fully block either pawn's path).
First to the opposite row wins. Perfect information, deterministic.

### Python engine + oracle layers
- `core/` — the reference rules engine (the correctness oracle for everything).
- `agents/random_agent.py`, `agents/greedy_agent.py`, `agents/minimax_agent.py` —
  baselines. Greedy races the shortest path; minimax has a time-budget knob.
- `scripts/eval_ladder.py` — the strength ladder (greedy, minimax d1/d2/d3,
  time-budgeted minimax). USE THIS for evaluation, not just winrate-vs-greedy.

### Rust native engine — `native/` (crate `barricades_native`)
PyO3 0.28.3 + maturin, edition 2024. u128-bitboard movegen/BFS, canonical
(6,9,9)-plane + 140-action encoding, PUCT MCTS `Tree` **with subtree carryover**
(`advance`), `SelfPlayPool` — a rayon-parallel batched stepper that releases the GIL
(`py.detach`). Differential-tested against `core` (tests in `tests/test_native_*.py`,
`tests/test_endgame_solver.py` etc.). Race-endgame solver integrated (flag-gated:
`Config.endgame_solve` for self-play truncation; MCTS leaf eval defaults on for
inference).
- Build: `cd native && maturin develop --release` (or `maturin build --release` + pip
  install the wheel). Pure CPU crate — no platform-specific code; builds on Linux.

### The AZ stack
- `agents/az/model.py` — `QuoridorNet(channels=32, blocks=3)`: **3 heads** — policy
  (140), value (tanh), auxiliary distance head.
- `agents/az/train.py` — the training core:
  - `form_dense_targets(examples, lam, gamma=0.99, scale=5.0, dist_norm=10.0)`:
    **the dense-reward shaping** — `v_target = lam·(z·gamma^plies_to_end) +
    (1−lam)·tanh(path_diff/scale)`; `dist_target = path_diff/dist_norm`.
  - `train_minibatched(net, opt, batch, epochs=4, batch_size=2048, beta=1.0, device)` —
    keep the full batch on CPU, move minibatches to device (avoids GPU OOM).
  - `augment_lr(examples)` — L-R mirror augmentation (`LR_PERM`, `mirror_planes`).
    Doubles data; already wired into the campaign.
  - `save_checkpoint` / `load_checkpoint`.
- `scripts/selfplay_native.py` — `run_selfplay(total_games, n_games, sims, device, net,
  seed, max_plies)`: the **async 2-pool double-buffer** — overlaps Rust `step()` with
  the GPU forward. Device-parameterized; nothing MPS-specific.
- `scripts/campaign.py` — the orchestrator. `run_campaign(iterations, games_per_iter,
  n_games, sims=100, max_plies=80, epochs=4, lr=1e-3, device, seed, channels=32,
  blocks=3, init_ckpt=None, out_dir="models", eval_games, eval_opponent)`:
  per iteration: self-play → `augment_lr` → `form_dense_targets(lam=anneal)` →
  `train_minibatched` → winrate eval → checkpoint `models/campaign_it{i}.pt`.
  - `anneal_lambda`: λ ramps 0→1 over the first 60% of iterations (dense shaping →
    pure outcome). 
  - CLI: `python scripts/campaign.py <iters> <games_per_iter> <sims> <device>`.
  - **Keep `n_games < games_per_iter`** (default `min(256, gpi)`): the pool refill
    keeps GPU batches full — without it the batch decays in the tail of each iteration.
- `scripts/bench_selfplay.py`, `scripts/bench_mps.py` — throughput benchmarks (the
  latter is MPS-only; ignore on CUDA, or adapt the forward/train benches).

### Device plumbing — port status
`"mps"` appears ONLY as default argument values and in `bench_mps.py`. The pipeline is
`.to(device)` throughout — **pass `device="cuda"` and it should just run.** Verified by
grep on 2026-06-10; if you find any hidden MPS sync call, it's in a bench script, not
the pipeline.

## 3. Validated results you inherit (do not re-derive)

- **The pipeline is proven sound.** A separate small-board validation
  (`smallboard/`, `docs/superpowers/smallboard-validation-results.md`) trained the same
  AZ recipe on exactly-solved boards: **100% optimal-move agreement on 3×3, 98.3% on
  5×5(W1), 89.7% on 4×4(W2), never loses a won position** vs an exact solver. Weak 9×9
  nets are a scale/training-budget issue, NOT a pipeline bug.
- **The 25k-game campaign result** (M1, ~50 min): the net is stronger than the 10k run
  (draws ~50% vs greedy; beats random easily) but collapsed into a **pure-race local
  optimum** — mean game length 79 → 17 plies by iteration ~13 and pinned there; it
  never learned wall tactics (those cost a tempo now for payoff later). Loses to the
  minimax ladder. Full log shape: λ anneal 0→1 over ~15 iters; loss ~1.9 plateau.
- Throughput on M1 MPS: 5–22 games/s (rises as games shorten); 100k games projected
  ~3 h locally. A 4090/A100 should beat this substantially **if the CPU side keeps up**
  (see §5 pod sizing).

## 4. The science flag (read before spending)

Scale alone may reproduce the 25k plateau. The known failure mode: outcome-driven play
converges to racing because wall play costs tempo with delayed payoff. The dense
path-diff shaping + annealing was added to counter exactly this; 100k games with a
slower anneal may break through — or may not. **Your job is to run the definitive
experiment, not to redesign training.** If the plateau recurs (game length collapses
to ~17 and winrate-vs-minimax stays ~0), STOP after the run completes, report, and
leave curriculum/exploration redesign as a decision for the humans. Suggested cheap
hedges within scope: (a) slower λ anneal (`warmup_frac` 0.8), (b) eval vs the ladder
every iteration so the plateau is visible early, (c) keep all checkpoints.

Start FRESH (no `init_ckpt`): warm-starting from the 25k net would bake the race
optimum in. (`models/campaign_final.pt` is that plateaued net — useful only as a
comparison baseline in evals.)

## 5. RunPod runbook

**Pod sizing (important):** 1× RTX 4090 (or A100 if spot is cheap) **with ≥16 vCPUs
and ≥32 GB RAM**. The self-play stepper is rayon-parallel Rust on CPU — with too few
vCPUs the GPU starves and you'll see low games/s no matter the GPU.

```bash
# --- security: keep the API key in env only; never commit/echo it ---
export RUNPOD_API_KEY=...        # provided separately by the user

# --- on the pod (ubuntu/pytorch image with CUDA matching torch) ---
apt-get update && apt-get install -y build-essential curl git
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
. "$HOME/.cargo/env"
pip install maturin numpy torch --index-url https://download.pytorch.org/whl/cu124  # match pod CUDA

# repo: clone/rsync the solver-and-az branch (rsync from the dev machine or a git remote)
cd barricades/native && maturin develop --release && cd ..
python -m pytest tests/ -q -k "native or endgame"     # gate 1: Rust crate on Linux
```

**Smoke gates (run ALL before the big run):**
1. Native test suite green on Linux (above).
2. Tiny campaign end-to-end on CUDA:
   `python scripts/campaign.py 2 64 60 cuda` — must produce checkpoints + sane log.
3. Numerical sanity: same net, same batch of planes → CPU vs CUDA policy/value outputs
   agree to ~1e-4 (write a 10-line throwaway; catches dtype/non-determinism issues).
4. Throughput tune: try `n_games` ∈ {256, 512, 1024} at `sims=100` and record games/s —
   CUDA tolerates much larger batches than MPS; pick the knee. Report games/s and the
   projected cost for 100k games BEFORE launching it.

**The run:**
```bash
# inside tmux; log everything; checkpoints land in models/
python scripts/campaign.py 100 1000 100 cuda 2>&1 | tee campaign_100k.log
#   = 100 iterations × 1000 games  (adjust split as desired; keep n_games tuned)
```
- Wire the ladder eval: either set `eval_opponent="minimax"` (uses
  `MinimaxAgent(time_budget=0.1)`) or run `scripts/eval_ladder.py` against each
  checkpoint after the fact. Greedy-only winrate is a weak signal (see history).
- Monitor: game-length trend (collapse to ~17 = the plateau), loss, winrate trend,
  GPU/CPU utilization (CPU pegged + GPU idle ⇒ raise n_games / lower sims; reverse ⇒
  more vCPUs needed).
- Sync `models/*.pt` + the log back to the dev machine periodically (rsync). Pods die;
  checkpoints are the product.

## 6. Acceptance criteria

- [ ] All smoke gates pass (incl. CPU/CUDA agreement) — paste outputs in your report.
- [ ] 100k games completed; per-iteration checkpoints + full log preserved off-pod.
- [ ] Ladder eval per checkpoint (or at least every 5th) — table in the report.
- [ ] Cost + games/s actuals vs the M1 baseline.
- [ ] Verdict: did scale break the race plateau? (game-length + ladder evidence)

## 7. Boundaries

- **Do not modify** `solver/`, `smallboard/`, `docs/superpowers/solver-*` — active
  parallel workstream. `core/` and `native/` movegen semantics are frozen (oracle-
  validated); if a Linux build issue requires touching `native/`, keep diffs minimal
  and run the differential tests.
- Commit your work (logs/doc/report, any port fixes) on `solver-and-az` (renamed from `az-bootstrap`) with clear
  `az:`-prefixed messages; never commit the API key or checkpoints >100 MB (rsync
  those, or use a release artifact).
- Budget guardrail: confirm projected cost with the user if it exceeds ~$25.
