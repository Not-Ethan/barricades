# Wall-Legality Filters: an Exact Connectivity Filter vs the Contact Heuristic

**Date:** 2026-06-10 · **Code:** `solver/src/bin/legality_bench.rs` (controlled harness),
`movegen.rs` (`legal_walls_bruteforce` / `legal_walls_writeup_bench` / `legal_walls`-DSU).

## Question

Quoridor-class solvers must test every candidate wall for legality (both pawns must
keep a goal path). The naive check is two flood fills per candidate. Two sound filters
can skip provably-harmless candidates:

- **Contact heuristic** (the *Solving Quoridor* writeup, read faithfully): check only
  candidates touching {board edge ∪ existing walls} at **≥ 2 contact points** combined.
  Sound: < 2 attachments cannot close a curve. (An earlier mis-parse of this rule —
  firing on any single border contact — was caught in review and corrected; see
  measurements doc.)
- **DSU connectivity filter** (ours): union-find over wall-endpoint posts, border
  pre-merged; check only candidates whose posts land in an **already-connected
  component** (the only way a curve can close, by planar duality). Strictly more
  precise: ignores harmless *different-component* contacts (merges, extensions,
  border-peninsula attachments).

## Method (controls)

Seeded wall-biased random playouts generate the position set **once**; all three modes
replay the identical set in identical order. **Correctness gate first**: all three
modes must return identical move sets at every position (any divergence aborts the
experiment). Positions grouped by wall-density bucket; each (mode, bucket) batch timed
separately; warmup pass per mode; median of 5 reps; single-threaded (Apple M1).

## Results

### 6×5, W=5 budget — 1,500,000 positions

| | weighted total | vs always-BFS |
|---|---|---|
| always-BFS (control) | 22.38 s | 1.00× |
| writeup (faithful) | 7.73 s | 2.90× |
| **DSU** | **5.20 s** | **4.31×** |

Per-position cost by density (ns; bfs→dsu ratio):

| bucket | always-BFS | writeup | DSU | writeup/DSU |
|---|---|---|---|---|
| 0 | 17,743 | 352 | 396 | **0.89** |
| 1 | 18,238 | 1,418 | 664 | 2.14 |
| 2 | 18,466 | 2,946 | 1,155 | **2.55** |
| 3 | 18,013 | 4,539 | 1,885 | 2.41 |
| 5 | 15,762 | 7,008 | 3,937 | 1.78 |
| 7 | 12,173 | 7,671 | 5,888 | 1.30 |
| 9 | 8,058 | 6,499 | 6,221 | 1.04 |

### 8×5, W=5 — 750,000 positions (geometry contrast)

always-BFS 18.11 s · writeup 4.75 s (3.81×) · **DSU 2.43 s (7.46×)** ·
**DSU/writeup = 1.96×**.

## Findings

1. **Both filters are large wins over naive checking** (2.9–3.8× and 4.3–7.5×
   end-to-end on the legality step).
2. **The DSU beats the contact heuristic 1.5×–2.0× overall**, with the per-density
   advantage peaking in the sparse-to-mid game (2.1–2.6× at buckets 1–3) and
   converging to parity at saturation (bucket 9: 1.04×) — once most contacts are
   same-component, exactness buys nothing extra.
3. **Honest detail:** on an *empty* board the writeup heuristic is slightly *faster*
   (0.89×) — its few bitops beat the DSU's per-call build (border pre-merge + finds).
   The DSU only pulls ahead once walls exist.
4. **The gap grows with board width** (1.49× on 6×5 → 1.96× on 8×5): wider boards
   mean costlier flood fills per fall-through and proportionally less border, i.e.
   more harmless different-component contacts for the exact filter to skip. The
   advantage should be larger still on 9×9.
5. Caveats: this measures the *legality step* in isolation, single-threaded; in-search
   benefit is diluted by all other per-node work (TT, ordering, race solves). Position
   distribution follows wall-biased playouts, not the search's true visit measure (the
   in-search shadow counters cover that; see measurements doc).

## Conclusion

The union-find connectivity filter is a strict, measurable improvement over the
(correctly implemented) contact heuristic for wall-legality filtering — modest on
small saturated boards, growing with board size, and never unsound by construction
(curve-closing candidates always fall through to the exact check). Combined finding
with the in-search shadow data: the writeup's heuristic was a good idea correctly
aimed; connectivity tracking is its natural completion.

## Addendum: the three-stage cascade (contact → DSU → BFS)

Layering the cheap contact count BEFORE the DSU (skip <2-contact candidates with
pure bitops; consult the DSU only for ≥2-contact ones; BFS only for genuine
curve-closers — each stage sound, so the cascade is) plus a per-board border
TEMPLATE for the DSU (struct copy + 2 unions/wall instead of re-running the
border pre-merge): measured at 600K positions, 4 modes, set-equality verified
throughout. Result: cascade == DSU overall (4.43× vs 4.42× vs naive), winning
+8–9% at densities 0–1 (363ns — faster than BOTH pure modes; fixes the DSU's
empty-board inversion) and fading to −1% at saturation (stage-1 evaluation is
overhead once most candidates reach stage 2 anyway). Architecturally correct,
practically neutral here because the DSU's per-call constant was already
~400ns; kept as a bench mode (`legal_walls_cascade_bench`), not wired into the
engine. Incremental DSU maintenance across the search tree was evaluated and
rejected: it requires rollback machinery + a make/unmake search architecture to
save ~20 array ops per node vs the template copy.

## 6×5 W7 = FIRST-PLAYER WIN — node counts now DECLINING (plateau confirmed)

`solve 6 5 7`: value=Win, 28,768,125,565 nodes, 5,174s (86 min, 64GiB TT @95%).
**W4 12.9B → W5 38.2B → W6 40.0B → W7 28.8B**: solving cost peaked just past
the value transition and is falling, as predicted (decisiveness + board
saturation + factorial transposition merging). Ladder: P2 at W0–W3, P1 at
W4–W7.
