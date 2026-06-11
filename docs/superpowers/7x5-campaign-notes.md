# 7×5 campaign notes (in progress)

Live log of the 7×5 (area 35) solving campaign. Started 2026-06-11. Pod:
32 vCPU / 128 GB-cgroup RunPod (cpu5g, $1.47/hr), pinned binary, same gated
build as the 6×5 ladder.

## Values so far

| W | value | nodes | wall-clock | status |
|---|---|---|---|---|
| 3 | **P2** (Loss for mover) | 3.53 B | 32 s (32 thr) | solved |
| 4 | **P2** (Loss for mover) | 225.8 B | 67 min (32 thr) | solved (attempt 5) |
| 5 | ? | — | — | in progress |

W0–W2 back-fill running (assumed cheaper than W3).

**The transition does NOT track 6×5.** On 6×5 the flip to P1 happens at W4;
on 7×5, W4 is still a P2 win — the transition is at W5 or later. Pattern so
far: 5×5 flips at W5, 6×5 at W4, 7×5 at ≥W5 — *not* monotone in area. A
width-parity hypothesis fits: on odd-width boards the pawns face off in a
shared center column and the second player's jump-parity advantage survives
more wall budget; even-width boards break the face-off and the first
player's tempo converts sooner. 7×7 (odd, area 49) would be the test, but
is priced out of this campaign.

**Cost scaling at fixed W (area 30 → 35):** W3 13×, W4 17.5× — call it
~15× per rung. Extrapolation for W5: ~570 B nodes ≈ 2.3 h at 70 M nps.

## The W4 saga — three attempts, two findings

**Attempt 1** (TT 128 GiB + race 32 GiB): SIGKILL at 42 s. Root cause: the
pod's **cgroup memory limit is 128 GB even though `free` reports the host's
251 GB** — the ask exceeded the container limit outright. Operational rule:
size caches against `/sys/fs/cgroup/memory/memory.limit_in_bytes`, never
against `free`.

**Attempt 2** (TT 64 GiB + race 24 GiB ≈ 88 GiB nominal): SIGKILL at 41 min,
129 B nodes in. `memory.max_usage_in_bytes` == the 128 GB limit exactly — a
true OOM despite ~31 GiB of nominal headroom. Two mechanisms account for the
gap:
1. **Hash scatter makes the TT fully resident at ~1% fill.** Entries land
   uniformly across the array, so with 128 entries/4K-page, ~all pages are
   touched after capacity/128 stores. A fixed-capacity TT's RSS cost is its
   *capacity*, not its fill — capacity beyond need is pure waste.
2. **Allocator arena bloat under race-cache LRU churn.** glibc grows up to
   8×cores arenas; 32 threads churning billions of small alloc/frees can
   hold tens of GB above live bytes.

**Attempt 3** (TT 16 GiB + race 16 GiB + `MALLOC_ARENA_MAX=2`, RSS logged
every 60 s): **stable at 29 GB RSS for 3.5 h** — flat curve, OOM fixed,
which fingers arena bloat as attempt 2's main unaccounted consumer (the
arena cap is the dominant delta). But the halved race cache cost ~3× in
throughput: 65 M nps early → 18 M nps steady-state with the race cache
pegged at its cap (1.07 B entries) and recompute churn replacing cached
endgame values. Timed out at 221 B nodes (4 h cap) without finishing.

**Attempt 4** (TT 16 GiB + race 64 GiB + `MALLOC_ARENA_MAX=2`): killed after
36 min at **15 M nps** — *slower* than attempt 3 despite a 4× race cache.
This refuted the "race-cache size = throughput" theory and exposed the real
culprit: **`MALLOC_ARENA_MAX=2` serializes 32 allocation-heavy threads onto
2 malloc arenas.** Both arena-capped runs (v3, v4) crawled at 15–18 M nps;
the uncapped attempt 2 ran 54–65 M nps. The cap fixed the OOM by paying ~4×
in throughput.

**Attempt 5** (TT 16 GiB + race 24 GiB + `MALLOC_ARENA_MAX=16`): the
compromise — enough arenas for parallel allocation, bounded bloat. First
heartbeat: **70.4 M nps** (fastest yet), RSS 44.7 GB and stable. Throughput
restored AND memory safe. Running.

### Findings worth keeping regardless of outcome

- **7×5 W4 ≥ 450 B nodes** (sum of distinct partial searches; single-run
  lower bound 221 B without completing). Compare: 6×5 W4 = 12.9 B *total*.
  The transition rung's cost grew **≥ 35×** for a 5-cell area increase —
  far steeper than the ~10× observed at W1 (6×5 → 7×5). The transition is
  where area bites; the plateau may be gentler (see the 6×5 W15 result:
  saturation flattens cost completely at high W).
- **The race endgame dominates 7×5 low-W solves.** The race cache pegs at
  any size tried (1.6 B entries at 24 GiB; 1.07 B at 16 GiB) and its size
  sets the node rate almost linearly. This is the opposite regime from
  deep-wall 6×5 (race vanishes by W10). If 7×5 W4 remains out of reach,
  the highest-value engine change is a *bigger/cheaper race representation*,
  not better wall-phase search.

## Memory & throughput rules (validated by the saga)

For a 128 GB-cgroup, 32-thread pod: `TT_real ≈ TT_capacity` (hash scatter —
size the TT to *need*, not to available RAM), `race_real ≈ 1.0–1.5× nominal`,
arena bloat ≈ tens of GB uncapped / negligible at `MALLOC_ARENA_MAX=16`.
**Do NOT use `MALLOC_ARENA_MAX=2`** — it serializes allocation and costs ~4×
throughput on 32 threads. `MALLOC_ARENA_MAX=16` keeps full speed (70 M nps,
the fastest configuration tested) with bounded bloat. Always run an RSS
logger (`memory.usage_in_bytes`, 60 s cadence).

## Raw artifacts

- `docs/superpowers/raw/7x5_w3.txt` — W3 solve.
- `docs/superpowers/raw/7x5_w4_v3_timeout.txt` — attempt 3's heartbeat trail
  (the 221 B-node incomplete run; evidence for the lower bound).
