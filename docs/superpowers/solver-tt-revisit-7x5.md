# TT revisit for 7×5 — best-move byte & DTM mode (analysis, not implemented)

Status: **analysis only** (2026-06-11). Decision: the 7×5 ladder runs on the
unmodified production engine ("no new code optimizations unless the ladder
stalls"); DTM mode is recorded as a curiosity, not project-critical. Findings
below were produced by parallel code-analysis passes and adversarially
re-verified against the source; line references are to that day's `master`.

## Option 1 — best-move byte in the TT entry

**Verdict: free on memory, small to build, soundness-neutral; its *search*
benefit is unproven here and must be A/B'd. Build it only if the 7×5 ladder
stalls or when a full-ladder PV pass is wanted.**

- **Zero memory cost.** `Entry { key: u128, value, flag, depth: u16 }` pads to
  32 bytes (16-byte alignment from the u128 key): 20 bytes used, **12 bytes of
  tail padding** (offsets 20..31). A compiled layout check confirms adding a
  `best_move: u8` keeps `size_of::<Entry>() == 32`.
- **No concurrency hazard.** The main TT is `Vec<Mutex<DenseTt>>` — every
  probe/store takes the shard mutex, so torn reads are structurally
  impossible, and the full-u128 key compare on probe means a stored move is
  position-correct by construction. (Caveat for anyone generalizing: the
  *race* memo `RaceTt` is RwLock-sharded with lock-free per-entry writes —
  a different design; don't copy this argument across.)
- **Move encoding.** 7×5 has 83 distinct moves (35 step destinations + 24
  anchors × 2 orientations). The existing dense `move_index` (max 191 for
  boards ≤ 7×7) truncates to u8 with `0xFF` = no-move sentinel.
- **The load-bearing gotcha is mirror canonicalization.** TT keys are the
  mirror-canonical state; a stored move must be saved canonical and mirrored
  back on probe when `canonical(s) != s` (Step dest `c → w-1-c`; wall
  `wc → w-2-wc`). A `mirror_move` helper is required — same discipline the
  footprint masks already implement.
- **Plumbing** (≈8 functions): record at the `v > best` update and the
  beta-cutoff break in `Worker::ab`, store with the existing `tt.store`;
  surface on probe even when `sdepth < qdepth` (a depth-unqualified hit's
  move is still valid ordering advice); consume in `ordered_moves` ahead of
  the heuristic order, and in `pv.rs::extract_pv` to skip the candidate loop.
- **Soundness story unchanged.** The byte is *ordering advice only*. The PV
  walk keeps its per-ply value-invariant assert and terminal winner check; a
  stale/evicted/wrong move fails the invariant and falls back to today's
  candidate requery, which is already proven correctness-neutral.
- **Eviction is depth-stratified, not uniform.** Slot 0 is depth-preferred,
  and PV ply k's entry carries depth `ceiling-k`: near-root entries are
  effectively eviction-proof; tail-of-line entries are leaf-like and
  unprotected. At 12% fill (6×5 W4) chain loss is negligible; at 98% fill
  (W10) expect the root-side prefix to survive and the last plies gone —
  fallback covers it.
- **Honest expectation for hash-move ordering: modest, possibly ~zero.**
  Literature says 5–25% node reduction *when killers/history already exist*,
  but three local facts argue for the low end: (1) no iterative deepening —
  a single fixed-ceiling pass, so there's no cheap previous iteration to seed
  moves from; (2) the distance heuristic is the dominant primary key, and a
  prior experiment showed displacing it cost ~9× nodes; (3) two prior
  theoretically-sound optimizations (Theorem-4, footprint) measured as net
  slowdowns here and ship default-OFF. Must be A/B'd on a mid-ladder rung
  before being trusted.

### Corrected understanding of the PV requery tail

Earlier discussion undersold it: the depth-fold reuse guard
(`sdepth >= qdepth`) rejects **even the on-line child's own entry** — at PV
ply k the child was stored at depth `ceiling-(k+1)`, but the requery asks at
the full ceiling. So every per-ply probe re-searches its *top* node; the warm
TT accelerates the interior but never short-circuits the probe itself.
Per-ply costs: loser-side plies = exactly one solve (every child of a Loss
parent is value-preserving); winner-side plies probe candidates in PV order
until one preserves the value — worst case all legal moves (~53 at 7×5, not
the ~130 of 9×9), each miss costing an exact solve of a subtree the original
search beta-cut past. Race-phase plies are exempt (the walls-exhausted
short-circuit returns the memoized race value with no alpha-beta).

## Option 2 — DTM solve mode (distance-carrying values)

**Verdict: shelved as a curiosity (user call, 2026-06-11). A weak solve is
the project's goal. Recorded here so the price is known if it's ever wanted.**

- **Memory is NOT the problem**: an i16 mate score also fits in the entry
  padding. The cost is structural:
  1. **Loss of the first-win cutoff.** Today `Win` is lattice-top, so
     alpha≥beta fires on the first winning child — boolean proof trees expand
     ~1 winning child per winner node. DTM must distance-discriminate **all**
     winning siblings (mitigated by the strong admissible BFS distance bound,
     i.e. mate-distance pruning).
  2. **Weakened TT reuse.** Exact DTM entries stay position-local facts
     (folding survives for them), but Lower/Upper *distance bounds* keep the
     depth guard and resolve strictly fewer queries. The deep-serves-shallow
     fraction of today's TT hits is unmeasured — a one-line counter on
     `sdepth > qdepth` vs `==` at the probe would settle it; structurally it
     is plausibly the majority of hits.
  - Honest blowup range for a forward DTM search: **3–30×** nodes.
    Anchors: checkers was solved WDL-only (DTM judged infeasible at scale);
    Awari got exact scores only via full-space retrograde; chess 6-man
    Nalimov DTM (~1.2 TB) vs syzygy WDL/DTZ (~150 GB) is ~an order of
    magnitude.
- **The "binary-search the root horizon" trick is unsound** with this TT:
  the depth-fold deliberately makes `ab(s, d)` *not* a win-within-d
  predicate (a depth-40 Win entry answers a depth-12 query). Fixing that
  needs per-depth keys — the exact design this codebase abandoned for size.
- **The cheap hybrid (~1.1–2× a boolean solve), half-built already:** keep
  the boolean folded solve for values, then run a DTM-only branch-and-bound
  over the proven-value cone — winner nodes minimize over value-preserving
  moves, loser nodes maximize over all moves — using the warm boolean TT as
  a free value oracle, `race_dtm_map` at walls-exhausted leaves, and BFS
  distance as the admissible bound. Requires a separate non-folded DTM memo
  with cycle-safe handling (forward DFS memoization has the same GHI
  pathology the df-pn work hit — wave or iterative-deepening discipline).
- **Race phase is already done**: `race_dtm_map` computes exact (Value, dtm)
  by wave retrograde; PV race suffixes are exactly DTM-optimal today.
  But a full PV line's length is **not** a sound DTM bound in either
  direction (winner doesn't minimize; loser's resistance is heuristic).
- **What full DTM would buy** (beyond exact perfect-play length per wall
  count): independent re-derivation of all ladder values (replication);
  a quantitative "walls buy delay" law (DTM growth vs W); a sound dtm rank
  that could revive the footprint-certificate extraction; uniqueness/tension
  metrics (count of DTM-optimal moves per position); max-DTM record
  positions; and a canonical fastest-win policy that removes the PV's
  cosmetic survival heuristic.

## Decision log

| Date | Decision |
|------|----------|
| 2026-06-10 | Pay-per-replay PV chosen for 6×5; best-move byte deferred ("worth revisiting at 7×5"). |
| 2026-06-11 | Analysis above. 7×5 ladder runs unmodified; best-move byte stays in pocket (build + A/B only if the ladder stalls); DTM mode shelved as curiosity — weak solve is the goal. |
