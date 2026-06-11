# Solver → RunPod Runbook (6×5 ladder W5–W10)

**Why cloud:** the 16 GB M1 is memory-starved — three SIGABRTs root-caused to
system-wide malloc exhaustion (crash reports: identical allocator frames, victims
`race_value`/`ab`); the W5+ rungs want multi-GB TT + race caches plus headroom, and the
search scales near-linearly with cores (lazy-SMP, ~5.4/8 utilization measured, mutex
bottleneck eliminated at e568006).

**Pod spec (CPU only — GPUs are useless for exact tree search):**
- 32–64 vCPU (EPYC-class), **128–256 GB RAM**, ~50 GB disk.
- RunPod CPU pods run ~$0.5–1.5/hr at this size. Estimated rungs at 32 vCPU
  (extrapolating from the M1's 6–8.7 M nodes/s at ~5.4 effective cores → expect
  ~25–40 M nodes/s): **W5 ≈ 20–90 min**, W6 ≈ hours, W7+ = the real frontier
  (unknown; the decelerating per-wall factor decides). Budget guidance: W5+W6 ≲ $10;
  full-ladder attempt $50–300 — confirm with the user past ~$50.

**Steps:**
1. Provision via the RunPod API/console (API key stays in env, never committed).
2. `rsync -az --exclude target --exclude .git/objects ~/barricades root@POD:~/` (or
   clone `solver-and-az` (renamed from `az-bootstrap`)).
3. `bash ~/barricades/scripts/runpod_solver_setup.sh` — toolchain, build, **full
   exactness gate suite** (any failure = stop), pins the binary.
4. `tmux new -s ladder` then `bash ~/barricades/scripts/runpod_solver_ladder.sh 5 10`
   — auto-sizes caches to the pod (≈53/27 % of RAM for TT/race), heartbeats every
   5 min into per-rung logs, stops on first failure.
5. Periodically `rsync` `~/ladder_logs/` back; on completion append results +
   shadow-benchmark bucket tables to `docs/superpowers/solver-phase1-measurements.md`.

**Knobs cheat sheet:** `QS_THREADS` (=nproc), `QS_TT_MB`, `QS_RACE_MB`,
`QS_RACE_SHARDS` (set 4× threads on big pods; env-tunable as of this commit),
`QS_PROGRESS_SECS` (heartbeat), `QS_SHADOW=0` (disable benchmark counters),
`QS_DSU_WALLS=0` / `QS_T4=1` / `QS_FOOTPRINT=1` (A/B knobs — leave defaults).

**Operational lessons encoded here (learned locally, the hard way):**
- Run from a **pinned binary copy**, never the cargo build path.
- Caches + system residents must fit RAM with ~20 % headroom or malloc aborts.
- Don't run the legacy `zz_sign_audit`/`zzz_audit_*` suites (multi-CPU-hour,
  coverage subsumed).
- Watch the heartbeat: nodes/s collapsing or `dmesg` OOM-killer lines = resize, not
  retry.

**What the ladder answers:** 6×5 values at W5–W10 (transition shape: clean flip at W4
like 5×5-at-W5, or oscillation like the writeup's anomalous 4×7), completing the first
area-30 board ever solved — plus the density-bucketed legality-filter benchmark
(writeup-vs-DSU) from the shadow counters, free in the same runs.
