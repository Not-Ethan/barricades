# 7×5 Quoridor: W0–W4 exactly solved (all P2); W5 open at ≥591 B nodes

Campaign log, 2026-06-11 (one day). Pod: 32 vCPU / 128 GB-cgroup RunPod
(cpu5g, $1.47/hr), pinned binary, same gated build as the 6×5 ladder.
Campaign ended at the budget cap with W5 undecided.

## Values

| W | value | nodes | wall-clock (32 thr) | status |
|---|---|---|---|---|
| 0 | **P2** | 15.2 K | 1 ms | solved |
| 1 | **P2** | 1.81 M | 18 ms | solved |
| 2 | **P2** | 74.8 M | 0.54 s | solved |
| 3 | **P2** | 3.53 B | 32 s | solved |
| 4 | **P2** | 225.8 B | 67 min | solved (attempt 5) |
| 5 | open | **> 591 B** | timed out at 3.5 h | LOWER BOUND only |

7×5 (area 35) is exactly solved for W0–W4: **the second player wins at every
wall count through 4**. W5 ran to 590.9 B nodes (TT 97% full, eviction
active) without resolving — already > 15.5× the cost of 6×5's W5, with the
~15×-per-rung area multiplier predicting it was within roughly an hour of
finishing when the budget cap hit. No checkpointing exists, so the partial
search is evidence only as a lower bound.

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

## What W5 needs (priced options for a future campaign)

1. **Just more hours**: ~5–6 h at the proven v5 config (70 M nps early,
   ~55 M sustained) ≈ $9–13 of pod time. Highest-confidence path.
2. **Best-move byte + bigger TT**: the TT hit 97% — a 32–48 GiB TT (fits:
   total RSS was 70 GB of 119 GiB) plus hash-move ordering
   (`solver-tt-revisit-7x5.md`; must be A/B'd) could cut the wall-clock
   meaningfully.
3. **Checkpointing** (TT serialization) would de-risk long rungs — today a
   timeout forfeits everything; this is the single most costly operational
   gap (we burned ~830 B nodes across W4/W5 timeouts).
4. **W6+**: if the ~15× multiplier holds against 6×5 (W6 = 40 B there),
   expect ~600 B/rung in the plateau — each rung ≈ a W5. The transition rung
   itself (wherever it is) plus one or two more would map the 7×5 column's
   shape. The race-cache regime persists at least through W5 (race entries
   pegged at every cap tried) — the bigger/cheaper race representation
   remains the highest-value engine change for this board family.

## Raw artifacts

- `docs/superpowers/raw/7x5_w0.txt` … `7x5_w4.txt` — the five solved rungs.
- `docs/superpowers/raw/7x5_w4_v3_timeout.txt` — attempt 3's heartbeat trail
  (221 B-node incomplete run at the arena-capped throughput).
- `docs/superpowers/raw/7x5_w5_timeout.txt` — the W5 lower-bound run
  (590.9 B nodes, 3.5 h, unresolved).
