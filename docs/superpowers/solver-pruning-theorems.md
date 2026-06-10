# Solver Pruning Theorems — Wall-Relevance Footprints + Inferior-Wall Analysis

**Status: falsification-tested research, ready for staged implementation.**

Synthesis of two theory tracks (Track A: wall-relevance footprint theorem; Track B:
dead/dominated/equivalent walls + race bounds) and their independent falsification phase.
All rule semantics are the EXACT semantics of `quoridor_solver` at HEAD ~`e8cf0b8`
(`solver/src/movegen.rs`, `src/bitboard.rs`, `src/board.rs`, `src/state.rs`,
`src/solver.rs`, `src/endgame.rs`). Every theorem below is stated in its **final,
post-falsification form** — where falsification narrowed or refuted a claim, the
narrowed form is what appears here, with the original noted.

Raw track notes are archived under `docs/superpowers/raw/` (commit `d5f1a46`).

---

## 0. Shared preliminaries

### 0.1 Value semantics (what every claim is stated in)

The engine has **no repetition rule**. Define the depth-indexed family `V_d`
(exactly `brute_value`): `V_0 = Draw` at non-terminals; terminals are Win/Loss for
the side to move; `V_d(s) = max_m negate(V_{d-1}(apply(s,m)))` on the lattice
`Loss < Draw < Win`. `V_d` is *refining* in `d` (a decided Win/Loss never flips;
Draw can become decided). The **true value** `V_∞` is the limit; the state graph is
finite (walls only accumulate), so `V_∞` is the unique negamax fixpoint.
`endgame::race_value` computes `V_∞` exactly on wall-exhausted (`[0,0]`) states,
draws included. `Solver::solve`'s contract: returned Win/Loss are true forced
values; **Draw means "undecided within the depth ceiling"** (load-bearing caveat,
see Gap G1). A position with no legal moves is a **Loss for the mover** (matches
`ab`'s empty move loop and the retrograde seeding).

**Soundness criterion for pruning a move B at state s** (uniform in depth, because
the depth-folded TT queries the same node at many depths): there must exist a kept
move A with `V_d(apply(s,A)) ≤ V_d(apply(s,B))` **for all d ≥ 0** (child values are
opponent-to-move). Every proven theorem below discharges this per-depth, so
GHI/path-dependence has no loophole.

### 0.2 The four traps every theorem must survive

1. **Genuine draws exist** (frozen-wall blockade races; confirmed 7x5 example).
2. **Walls CREATE moves** via the diagonal-jump rule: a wall blocking a straight
   jump legalizes up to two diagonal jumps — a "far away" wall can hand the
   opponent NEW moves. (Empirically weaponized: T3 family below, six verified
   positions where a wall Chebyshev ≥ 2 from both pawns and the whole shortest
   path flips Win → Loss.)
3. **Wall-overlap rules**: a wall can block ANOTHER wall's placement
   (overlap-denial). This killed Conjecture 2A (§B.6).
4. **Repetition/path-dependence (GHI)**: handled by stating everything in `V_d`;
   re-enters only through strategy *extraction* (Gap G4 — a verified rank is
   mandatory, confirmed by in-the-wild cycles).

### 0.3 Notation

State `s = (pawn[2], h_walls, v_walls, walls_left[2], turn)`; `idx(c,r) = r*w+c`;
anchor bit `wr*(w−1)+wc`. H-anchor `(wc,wr)` blocks the north edges of `(wc,wr)`
and `(wc+1,wr)`; V-anchor blocks the east edges of `(wc,wr)` and `(wc,wr+1)`
(exactly `step_blocked`). Conflict sets from `overlaps`:
`Conflict(H@(c,r)) = {H(c−1,r), H(c,r), H(c+1,r), V(c,r)}`,
`Conflict(V@(c,r)) = {V(c,r−1), V(c,r), V(c,r+1), H(c,r)}`, clipped to range.
`flip(s)` toggles `turn` only. `ins(u,w,Z)` sets `w`'s bit and decrements
`walls_left[Z]`, same side to move. `Y` = certificate owner, `Z = 1−Y` = inserter.

### 0.4 Falsification methodology (evidence base for everything below)

Two independent throwaway harnesses (both deleted after use, nothing committed):

- **Track A harness**: independent forward-reachability + retrograde attractor
  labeling over COMPLETE game graphs closed under turn-flip (so `V(flip(s))` is
  always exact, genuine draws included), cross-checked against the verified
  `Solver` at 400 random states per board (2,000 checks, 0 mismatches). Boards:
  3x3-w1 (2,860 states), 4x3-w2 (54,708), 3x5-w2 (392,696, 32 genuine draws),
  4x4-w2 (685,492), 6x4-w1 (369,156, 12 draws).
- **Track B harness**: complete reachable graphs extended over all initial budget
  pairs and both root turns; exact `V_∞` by retrograde labeling and `V_d` for
  `d = 0..52` by value iteration; validated graph `V_d == brute_value` (d ≤ 5) at
  ~150 strided states/board, the refinement invariant at every state for all
  d ≤ 52, and retrograde `V_∞ == race_value` at all 831,586 non-terminal `[0,0]`
  states. Total 2,121,148 states across five boards. Zero harness discrepancies.

---

# PART A — Wall-Relevance Footprint Theorems

## A.1 Certificates and footprints (definitions)

**Definition 1 (certificate).** A certificate `P = (T, σ, leaves)` for `Y`
guaranteeing `V₀ ∈ {Draw, Win}` from `u₀` is a set `T` of positions with
`u₀ ∈ T`, closed as follows:

- at every Y-node `u ∈ T` (non-terminal): a designated legal move `σ(u)` with
  `apply(u, σ(u)) ∈ T`;
- at every Z-node `u ∈ T` (non-terminal): **every** `m ∈ legal_moves(u)` has
  `apply(u, m) ∈ T` (a Z-stuck node is a Y-win leaf);
- every terminal in `T` has `winner = Y`;
- for `V₀ = Win` additionally a rank `ρ: T → ℕ` strictly decreasing along every
  closure edge. For `V₀ = Draw` no rank — cyclic play is allowed.

**Definition 2 (footprint `R(P)`).** Protected edges `R_E` + forbidden anchor
slots `R_W`:

- **(A) Y-move edges.** For each Y-node with a step/jump `σ(u)`: the edges whose
  UNBLOCKED status certifies legality (plain step: 1 edge; straight jump:
  mover→opp and opp→landing; diagonal jump: mover→opp and opp→diag-dest). The
  blocked straight edge enabling a diagonal is deliberately NOT included —
  blocking is monotone under insertion.
- **(B) Z anti-growth (the jump trap).** For each Z-node where Z is adjacent to Y
  in direction `d` with the straight-jump path currently open (edge Z→Y unblocked,
  `Y+d` on-board, edge `Y→(Y+d)` unblocked): protect edge `Y→(Y+d)`. Blocking it
  is the ONLY mechanism by which insertion can ADD a pawn move (straight→diagonal
  conversion, including the off-board case — verified by code-level audit of
  `legal_steps` to be the unique additive transition).
- **(C) Anti-pre-block.** For each Y-node where `σ(u)` is a wall `x`:
  `Conflict(x) ⊆ R_W`.
- **(D) Legality witnesses.** For each such `(u,x)`: goal paths `π_Y, π_Z` for
  BOTH players in `apply(u,x)` (they exist since `x` was legal); all their edges
  ∈ `R_E`. (This excludes the joint-stranding case: two individually legal walls
  jointly stranding a pawn.)

**Out-of-footprint wall:** blocks no `R_E` edge (≤ 2 blocking anchors per edge)
and `∉ R_W`. The whole test compiles to two u64 masks (forbidden H / V anchors);
membership is one bit test.

## A.2 THEOREM 1 (Wall-Insertion Invariance) — **PROVEN; zero violations at scale**

> **Statement.** Let `P` be a certificate for `Y` guaranteeing
> `V₀ ∈ {Draw, Win}` from `u₀`, and `w` any wall with `u₀.walls_left[Z] ≥ 1`
> outside `R(P)`. Then `P` itself, executed verbatim, certifies `V₀` from
> `ins(u₀, w, Z)`. Hence `V_Y(ins(u₀,w,Z)) ≥ V₀`; for `V₀ = Win` exactly `= Win`,
> with win length ≤ the original (rank transfers unchanged).

**Proof status: PROVEN.** Six obligations each discharged against code semantics
(terminal agreement; read-set determinism of `legal_steps`; Z-step antitonicity —
the only additive transition is straight→diagonal, exactly component (B); Z-wall
antitonicity; Y-wall legality via (C)+(D); budget bookkeeping). The falsification
phase independently re-derived all six at code level against `movegen.rs` —
no hole found.

**Empirical validation (Win direction): ZERO violations everywhere.**
Five complete game graphs, faithful Definition-2/§A.5 extraction
(components A/B/C/D, all-Z-replies closure, exact attractor-level rank):
49,230 certificate roots, **537,771 wall checks, ~403K pruned walls, 0
mismatches**. Plus targeted families:

- **T1** (5x5 corridor win): extracted footprint EXACTLY `{H(1,3), H(2,3)}`,
  30/32 walls pruned; both in-footprint walls flip to Loss — the footprint
  isolates exactly the value-critical walls.
- **T2** (jump win): exactly the 4 claimed anchors, 28/32 pruned.
- **T3** (six adversarial far-wall value-flippers — walls Chebyshev ≥ 2 from both
  pawns and from every shortest-path cell that flip Win→Loss): all six land
  INSIDE the extracted footprints; all out-of-footprint walls preserve Win. Any
  geometric/shortest-path-based footprint is empirically unsound; Definition 2
  catches all known flippers.
- **T5** (wall-forced wins, unique winning move is a wall): footprint contains
  `Conflict(H(3,1))` as required.
- Randomized 5x5 sweep (2 random pre-walls, wl `[2,2]`): 60 trials, 1,387 wall
  checks, 765 pruned, 0 violations.

**Ablations prove every component load-bearing** (violation counts when a
component is deleted): A-only / A+B / A+B+C = 1,256 / 1,192 / 542 (4x3-w2),
625 / 616 / 391 (3x5-w2), — / — / 303 (4x4-w2), 16 / >0 / 0 (6x4-w1);
(D)-deletion produces 2 real violations at T5.0 (walls `H(3,2)`, `H(3,3)` flip
Win→non-Win). **Do not ship a partial footprint.**

**Empirical validation (Draw direction):** exhaustive over every genuine-draw
root with Z holding a wall on the complete graphs: 32 roots, 368 wall checks,
32 pruned, 0 violations. All 32 pruned walls came back **Loss-for-Z** — strictly
worse than Draw, confirming Corollary 2's cap (not equality) is the right form.

### Corollary 1 (mustplay pruning at a Z-to-move node) — PROVEN

Let `s` have Z to move, `walls_left[Z] ≥ 1`, and `P` certify `V₀` for Y from
`flip(s)`. For every `w ∈ legal_walls(s)` outside `R(P)`:

- `V₀ = Win`: `V(apply(s,w)) = Win` for Y — `w` is refuted with zero search.
  When proving `V(s) = Loss` for Z, the wall mustplay set is exactly
  `{walls hitting R(P)}`; pawn moves are always searched.
- `V₀ = Draw`: `V(apply(s,w)) ≤ Draw` for Z — `w` can never be a winning move
  for Z.

### Corollary 2 (budget+tempo cap) — PROVEN, deliberately a cap not an equality

`V_Z(apply(s,w)) ≤ negate(V₀)`. For `V₀ = Win` this is exact (`= Loss`). For
`V₀ = Draw` only the INEQUALITY holds — a wasted wall can convert Z's draw into
a loss by tempo (empirically: all 32 draw-direction pruned walls were Loss-for-Z).
The equality form `value(s·w) == value(s)` is **false in general**; the cap is
what pruning needs.

### Alpha-beta / df-pn integration rule (exactness guard) — PROVEN

- **Win-direction prunes are EXACT child values** (Loss for Z): skipping them
  never changes the max. Safe unconditionally.
- **Draw-direction prunes are UPPER bounds** (≤ Draw for Z): skip them only once
  `alpha ≥ Draw` at the node, or when the query is one-sided ("can Z win?" —
  df-pn disproof). If all searched moves return Loss for Z and pruned walls
  exist, the exact value is NOT determined (Loss vs Draw) — the engine MUST then
  expand pruned walls. **This guard is mandatory in any implementation.**
- df-pn symmetry: proof sets ("Y wins") and disproof sets ("Z cannot win" = a
  Y ≥ Draw certificate) are both Definition-2 instances; both directions covered.

## A.3 THEOREM 2 (restricted sibling transfer) — **PROVEN with stated assumptions; verified**

> **Statement.** Let `P₁` certify `V₀` for Y from `s·w₁`. If
> **(T1)** `w₂` is outside `R(P₁)`;
> **(T2)** `w₁` blocks NO edge consulted by ANY `legal_steps` evaluation at any
> node of `P₁` — *including blocked-consults* (removing `w₁` flips no consult);
> **(T3)** at every Z-node of `P₁`, Z has `walls_left = 0` (`w₁` was Z's last
> wall);
> then `P₁` certifies `V₀` from `s·w₂`.

**Empirical validation:** 1,342,075 sibling pairs across four complete boards
(with T1+T2+T3 checked, including the consulted-edge tracking for T2): **0
violations.** Practically relevant exactly where walls dominate cost: last-wall
layers feeding the race endgame.

### Conjecture C1 (general sibling transfer) — **REFUTED, do not implement**

Naive sibling reuse (drop T2/T3) fails at scale, exactly as predicted:
**51,651 counterexamples in 2,103,045 pairs**, including last-wall
(T3-satisfying) layers where only T2 fails. Example: 3x3-w1
`s = {pawn:[4,7], h:0x0, v:0x0, wl:[1,1], turn:0}`: `w₁ = H(0,1)` refuted, but
`w₂ = H(0,0)` outside `R(P₁)` is **Loss** for Y. T2 (consulted-edge cleanliness)
is independently load-bearing, not just T3. Any future generalization needs
explicit "anti-unlock" sets covering every wall whose legality is
`w₁`-dependent at every node — potentially huge; treated as dead for now.

## A.4 Gaps and non-claims (mandatory reading before implementing)

- **G1 — horizon draws are not certificates.** `Solver::solve` returning Draw
  may mean "undecided within ceiling". The Draw direction of Theorem 1 requires
  a GENUINE safety certificate (race-retrograde draw label or completed fixpoint
  proof). Using a horizon-Draw "footprint" is UNSOUND — there is no strategy
  object. Win/Loss solver outputs are true values and safe.
- **G2 — removal direction** only under Theorem 2's T1+T2+T3; C1 refuted.
- **G3 — witness-path freedom.** (D) protects ONE chosen witness pair per
  placement: sound but conservative. Minimal-witness/cut-set selection is an
  optimization question, not correctness.
- **G4 — extraction rank is NOT optional.** A "strategy" picking any Win-valued
  child can cycle through Win-valued positions forever and is NOT a certificate.
  **Confirmed in the wild**: greedy win-preserving σ produced non-well-founded
  closures at 3 of 8 5x5 targets (T3.0 needed 4 repair iterations / 75 boosted
  Y-nodes; also T3.1, T5.1). See the §A.5 amendment.
- **G5 — mirror symmetry.** Footprints must live in the REAL orientation of `s`.
  Extract by re-solving children of real-orientation states, never by reading
  strategy moves out of canonical TT entries; mirror cached masks when the
  canonical representative is the mirrored state.
- **G6 — TT bounds.** Lower/Upper TT entries are bounds, not strategies;
  extraction must use Exact results or fresh solves.
- **G7 — claim scope.** Theorem 1 says nothing when the precondition
  `V(flip(s)) ≥ V₀` fails. Negative control T4 (7x5 blockade + one Z wall):
  precondition fails and nearly every Z wall wins — the theorem correctly stays
  silent; any implementation that prunes there is buggy.
  *Erratum (falsification ground truth):* T4 has **25** legal Z walls of which
  **24** win for Z (not the originally noted 27/26); the qualitative claim holds.

## A.5 Footprint extraction algorithm — **AMENDED (original §5 rank computation was unsound)**

**Falsification finding (soundness-relevant):** the original prescription
`dtw(u) := min d with ab(u,d) decisive, TT-hot` is broken by the engine's
depth-fold TT reuse — deep decisive entries are returned at shallow query
depths, so the computed `dtw` is a nonuniform under-estimate NOT guaranteed
strictly decreasing (empirically deviates from the true attractor level at
4,477/13,776 4x3-w2 roots and 62/348 3x3-w1 roots). It violates G4's own
requirement. **Use one of the two validated replacements:**

- **(a) Exact attractor levels** from retrograde labeling — race slices already
  provide this; full layers via the walls-only-accumulate layer DAG; or
- **(b) Build-then-verify** (validated on every 5x5 instance): build the closure
  with any win-preserving σ, then verify well-foundedness post-hoc by retrograde
  labeling **on the closure graph itself** (all nodes decided ⟺ a valid
  Definition-1 rank exists; the decided order IS the rank). On failure, locally
  re-pick σ at cycle-participating Y-nodes ("K-boosting") and re-verify —
  converged in ≤ 4 iterations on every tested instance. **Abort = sound
  no-prune fallback.**

```
extract(u0, V0) -> Option<(h_mask, v_mask)>:     # u0 = flip(s), Y to move
  if V0 == Win: require solve(u0) == Win          # true value (G6: Exact only)
  else:         require GENUINE >=Draw certificate  # race-retrograde / fixpoint (G1)
  visited := {}; RE := {}; RW := {}; budget := N_nodes  # ~10^4
  dfs(u):
    if u in visited: return                       # DAG-collapse
    visited += u; if --budget < 0: ABORT          # sound fallback: no pruning
    if winner(u) != None: return
    if u.turn == Y:
      m := sigma(u)        # Win: rank-decreasing child per (a)/(b) above;
                           # Draw: the safety-strategy move
                           # prefer steps over walls, shortest wins => small R
      add component (A) edges of m
      if m is Wall x:
        RW += Conflict(x)                                          # (C)
        RE += edges(BFS_path(apply(u,x), Y) + BFS_path(apply(u,x), Z))  # (D)
      dfs(apply(u, m))
    else:                                         # Z to move
      if pawns adjacent && straight-jump path open:
        RE += straight-jump landing edge          # (B)
      for m in legal_moves(u): dfs(apply(u, m))   # ALL replies incl. Z walls
  dfs(u0)
  if V0 == Win: VERIFY rank well-founded on closure (scheme (b)); on failure ABORT
  compile RE -> anchors (<=2 per edge); return (RW | compiled) as h/v u64 masks
```

Race-endgame leaves (`walls_left == [0,0]`): extract from `endgame.rs` retrograde
labels — Y-nodes pick a retrograde-rank-decreasing child (Win) or any non-losing
child inside the draw fixpoint (Draw); Z-nodes take all step children; components
A/B only. Sound over-approximation fallback: all unblocked edges of the pawns'
connected region (bigger footprint = less pruning, never wrong).

**Where to apply.** At a Z-to-move node about to iterate walls: probe/solve
`flip(s)` (TT-shared; one tempo better for Y, hence typically much shallower),
extract, prune walls outside the masks subject to the §A.2 integration guard.
Cache `(masks, V₀)` per canonical position, mirroring masks per G5. Heuristic
gate: attempt only when `dist_to_goal(Y) − dist_to_goal(Z)` suggests Y clearly
ahead; cap extraction at `N_nodes ≈ 10⁴`.

**Expected impact.** Wall-anchor universe `2(w−1)(h−1)` = 32 (5x5) / 40 (6x5) /
48 (7x5); walls are 85–90% of branching when in hand. Measured: T1 footprint
2/32 anchors (~94% of wall replies pruned at that node), T2 4/32 (~88%);
randomized sweep pruned 765/1,387 (~55%). Deep proofs accrete (A) edges across
all Z replies plus (D) witnesses → realistic mid-game footprints 30–60% of
anchors → per-node wall-branching cut 1.5–3x; near-endgame race-funnel proofs
stay corridor-local → 3–10x per node. **Honest overall estimate: 2–6x node
reduction** on 6x5/7x5-class solves, concentrated in the refutation-heavy
layers that dominate cost. Hex's 25x is NOT promised (Quoridor footprints must
cover Z's reply fan; Hex carriers don't). Instrument refuted-Z-node wall counts
before/after; report geometric mean.

---

# PART B — Inferior-Wall Analysis (dead / dominated / equivalent / fill-in)

## B.1 Lemma L1 (fresh-edge lemma) — **NARROWED by falsification**

> **Final statement.** Every legal wall candidate blocks exactly two
> currently-unblocked grid edges, **disjoint from all PLACED walls' edge sets**.
> Two simultaneously legal same-orientation candidates MAY share exactly one
> edge. Equal edge sets between distinct walls remain impossible.

**Status: proven (narrowed).** The original second clause ("any two
simultaneously legal walls block disjoint edge sets") is **FALSE** for
candidate-vs-candidate pairs: e.g. 3x3-w1
`{pawn:[1,7], h:0x0, v:0x0, wl:[0,1], turn:1}` — candidates `H(0,0)` and
`H(1,0)` are both legal yet both block the north edge of cell (1,0). The
original proof only covered placed-vs-candidate. Non-load-bearing downstream:
consequence (a) (the "duplicate-edges dead wall" subfamily is vacuous — the
overlap rule excludes it) and consequence (b) (**no set-theoretic wall
subsumption**) both survive, (b) via equal-sets-impossible rather than
disjointness.

**Evidence:** clause 1 verified on 1,115,696 sampled legal candidates
(full-board edge-diff == 2 everywhere); counterexamples to the old clause 2 on
every tested board.

## B.2 Lemma L2 (confinement, jump-inclusive) — **PROVEN, verified**

In any continuation of `s`, pawn `p` only ever occupies cells of
`comp_s(pawn_p)` (its component over unblocked edges). Every legal destination —
step, straight jump, diagonal jump — is linked to the mover's cell by ≤ 2
unblocked edges at move time, and components only shrink (Lemma L3:
`R(t) := comp_t(p0) ∪ comp_t(p1) ⊆ R(s)`).
**Evidence:** 4,213,012 step moves (jumps included), zero escapes.

## B.3 Lemma M (wall-budget monotonicity) — **PROVEN, verified at massive scale**

> If `t, t′` are identical except `t′.walls_left[p] ≥ t.walls_left[p]`, then for
> ALL `d`, `t′`'s value for `p` is ≥ `t`'s value for `p`.

Proof by induction on `d`: only the mover's budget is ever read, only as
`== 0`, so the mover's move set is a superset and the opponent's is identical;
dodges trap 2 by construction (varies budgets, never placed walls).
**Evidence:** 1,200,048 budget-adjacent pairs checked at `V_∞` AND all 53 depths
(~63.6M inequality checks) — zero violations. This is the load-bearing lemma
for Theorem B.4.

## B.4 THEOREM 4 (one-sided frozen-race bounds) — **PROVEN, verified; main Track-B deliverable**

> **Statement.** Non-terminal `s`, mover `p`, budgets `[a,b]` (mover, opp),
> `r = race_value(s with walls_left := [0,0])` (true value, draws included,
> same side to move):
> - `b = 0, a > 0`: `V_∞(s) ≥ r`; in particular `r = Win ⇒ s is a true forced
>   Win` (return immediately); `r = Draw ⇒ V_d(s) ≥ Draw` for all `d` (Lower
>   bound; a Loss return is impossible).
> - `a = 0, b > 0`: dually `V_∞(s) ≤ r`; `r = Loss ⇒ true forced Loss`;
>   `r = Draw ⇒ V_d(s) ≤ Draw` (Upper bound).

Proof: Lemma M zeroes the non-exhausted side's budget + exactness of
`race_value` on `[0,0]` states + the refinement property for the per-depth Draw
bounds.

**Evidence (strongest in the program):** 750,836 one-sided-exhaustion nodes on
complete graphs: `V_∞` vs race bound pointwise + per-depth Draw bounds (no
`V_d` ever on the wrong side of Draw, all d ≤ 52) — zero violations. The
375,402 decisive cases were then implemented as the ACTUAL pruning (node
replaced by leaf = race value, full graph relabeled): reproduced `V_∞` exactly
at **all 2,121,148 states on all five boards** — zero mismatches. P5a/b hold;
P5c (sharp draw-boundary case on the documented 7x5 blockade) holds —
`[2,0] → Loss ≤ Draw`, `[0,2] → Win ≥ Draw`; the trap-2 diagonal-jump attack
fails exactly as the theorem predicts.

**Implementation spec (drop-in for `Worker::ab`):** at a node with one-sided
exhaustion, synthesize a TT-style hit
`(val = r, flag = Lower if opp exhausted / Upper if mover exhausted, depth = ∞)`
through the existing probe logic; decisive cases (Win-Lower / Loss-Upper)
return immediately. The synthesized entries are uniformly-in-depth valid
(decided cases are true `V_∞`; the Draw-bound invariant is closed under negamax
and the depth-fold reuse rule — verified at every depth). Composes with /
strictly extends the existing `[0,0]` short-circuit. Cost: O(1) budget test +
a `race_value` call already memoized per wall config (`RaceTt`) — expected
amortized-free since the `[0,0]` path would eventually need the same configs.

**Expected impact:** 3.47% / 3.80% of 6x5 random-playout positions (walls=3/4)
have one-sided exhaustion; in-search incidence should be substantially higher
(search concentrates on wall-burning lines), and **each fire cuts an entire
subtree**. Instrument in-search fire rate behind a counter before quoting a
node-reduction factor.

## B.5 Dead walls: Theorem B-1 (geometric nullity) + 1M (monotonicity) — **PROVEN; detection only, NOT pruning**

> **Definition.** Slot X is *dead in s* iff every endpoint of both edges it
> blocks lies outside `R(s)` (per-slot test: `H(wc,wr)` dead ⟺ cells `(wc,wr)`
> and `(wc+1,wr)` ∉ R; `V(wc,wr)` dead ⟺ `(wc,wr)` and `(wc,wr+1)` ∉ R).
> **Theorem B-1.** Placing a dead wall changes nothing game-relevant except:
> mover budget −1, turn flip, and overlap-exclusion of `excl(W)`. In every
> continuation, erasing its anchor changes no `legal_steps` (jumps included —
> trap 2 discharged: every jump-consulted edge is incident to a pawn cell, which
> stays in `R(s)`), no `dist_to_goal`, no `has_path`, and no candidate-wall
> legality verdict. **Theorem B-1M.** Once dead, always dead (R only shrinks);
> classification cacheable, recheck only after wall placements.

**Evidence:** 60,024 dead placed-anchor instances along continuations — anchor
erasure changed no step set (both turns), no distance (both players), no
candidate `has_path` verdict; deadness monotone along every dead-placement
edge; the 2-cell test ≡ the 4-endpoint definition on every legal candidate.
(Earlier Track-B run: 7,081 paired positions on P1, identical.)

**NON-THEOREM (refuted, calibrated): dead walls are NOT removable from the move
list.** A dead wall is exactly *pass + burn a wall + forbid `excl(W)`*. Naive
dead-wall pruning broke `V_∞` at 324 states (4x3-w2) + 224 (3x5-w2) — e.g.
`{pawn:[3,11], h:0, v:0, wl:[2,2], turn:0}` — confirming the pass/tempo
refutation AND proving the harness detects unsound pruning. Notably, NO value
spread among same-node dead children was observed anywhere (46,596 dead moves,
2.1M states): the demonstrated unsoundness is **purely the pass-tempo mechanism**,
not dead-twin inequivalence — so a future "dead-cluster as pass-counting game"
theory (sum-of-games on the Cram-like anchor cluster + race parity) is the
right next target.

**What ships:** detection only — one O(w·h) flood fill from both pawns + 2 bit
tests per slot, gated on `R ≠ full` — used for move ordering (search dead walls
last) and as the precondition for Theorem 3 folding. Measured incidence on 6x5
random playouts: positions with ≥1 dead legal wall 0.33% / 0.89% (walls=3/4);
dead share of legal wall moves 0.16% / 0.48% (biased sample; in-search rates
need instrumentation).

## B.6 THEOREM 3 (dead-twin equivalence) — **PROVEN but provably VACUOUS; documentation only**

> Dead walls `W ≠ W′` with `excl(W) \ {W′} = excl(W′) \ {W}` have
> `V_d(apply(s,W)) = V_d(apply(s,W′))` for all `d` (bit-swap isomorphism); one
> may be pruned.

Falsification upgrade: zero qualifying pairs in all 2.1M states, AND an
exhaustive geometric scan of all boards 2x2..8x8 shows the excl-hypothesis is
satisfiable **only on the degenerate 2x2 board** (1 pair total). The theorem is
sound but **fully vacuous on any real board** (stronger than the original
"near-vacuous"). Keep only as documentation of the exact obstruction (excl-set
mismatch) that makes naive dead-wall folding unsound. **Do not implement.**

## B.7 Refuted dominance families (kept as negative results / regression targets)

- **Edge-subset subsumption** between walls: impossible (L1 — equal edge sets
  impossible).
- **Race-tempo dominance** ("mover strictly ahead in BFS distance ⇒ some pawn
  step attains the value"): **FALSE**. P4 = `Board::new(5,4,2)`,
  `{pawn:[7,18], h:0x400 (H(2,2)), v:0x80 (V(3,1)), wl:[1,1], turn:0}` — p0
  ahead 3 vs 4, `V = Win`, yet EVERY pawn step loses; only a wall wins.
  Re-verified standing in falsification. (This is the exact trap from the AZ
  campaign history, now with a certified counterexample.)
- **Deferral** ("place W later instead of now"): refuted in concept — the reply
  can occupy an overlapping slot, delegalize W via connectivity, or exploit the
  line W would have blocked.
- **Conjecture 2A (mover-only walls dominated)** — **REFUTED**, via the
  suspected overlap-denial mechanism (trap 3). Minimal confirmed counterexample,
  `Board::new(5,4,4)`:
  `{pawn:[6,19], h:0x1 (H(0,0)), v:0x504 (V(2,0),V(0,2),V(2,2)), wl:[1,1], turn:0}`
  — components split; exact value Win for p0 (brute_value d=26 AND Solver
  agree); the UNIQUE value-attaining move is `V(1,2)`, which is mover-only;
  every kept move is Loss. Mechanism: `V(1,2)` pre-occupies the overlap region,
  denying the opponent's delayed killer slots (`H(1,2)`/`V(1,1)`) inside the
  mover's own corridor. 446 such positions on 5x4/6x4 fence+maze families, one
  independently confirmed with raw `brute_value(d=26)` only. **Struck from the
  program**; any future mover-only-dominance claim must at minimum except walls
  whose excl-set intersects opponent-playable half-live slots in the mover's
  component.
- **Mirror-pair move folding**: proven (rests on the crate's test-gated mirror
  automorphism), but the 6x5 campaign root is not self-symmetric and child-level
  TT canonicalization already captures most of the benefit. Skip.

---

# C. Implementation program

## C.1 SAFE TO IMPLEMENT NOW (proven + falsification-clean)

| # | Item | Theorem | Cost/node | Expected impact |
|---|------|---------|-----------|-----------------|
| 1 | One-sided race bounds as synthesized TT hits | B Thm 4 | O(1) + memoized race solve | whole-subtree cutoffs at 3.5–3.8%+ of nodes (in-search rate likely higher) |
| 2 | Win-direction footprint mustplay pruning at Z-nodes | A Thm 1 + Cor 1 (Win) | one `flip(s)` solve + extraction (capped 10⁴ nodes, cached masks) | per-node wall-branching cut 1.5–3x midgame, 3–10x near endgame; ~2–6x overall node reduction |
| 3 | Draw-direction footprint pruning **with the alpha-guard and G1 genuine-certificate gate** | A Thm 1 + Cor 1/2 (Draw) | as above | smaller; matters for draw-heavy blockade layers |
| 4 | Sibling transfer in last-wall layers (T1+T2+T3 checked) | A Thm 2 | consulted-edge bookkeeping during refutation | reuse of refutations exactly where walls dominate cost |
| 5 | Dead-slot detection for move ordering (never pruning) | B Thm B-1/B-1M | one flood fill, gated | small; enables future cluster theory |

Non-negotiable implementation constraints baked into the proofs:

- **Footprint extraction MUST use a verified rank** (§A.5 amendment): exact
  attractor levels, or build-then-verify with K-boost repair; abort ⇒ no-prune.
  The original TT-hot `dtw` is unsound with the depth-fold TT — do not use it.
- **Full footprint only** (A+B+C+D) — every ablation produced violations.
- **Draw direction**: genuine certificates only (G1); prune only under
  `alpha ≥ Draw` or one-sided queries; expand pruned walls if the node would
  otherwise resolve to all-Loss (§A.2 guard).
- **Mirror discipline** (G5): extract in real orientation; mirror cached masks.
- **Theorem 2**: all three side conditions checked, including blocked-consult
  tracking for T2 — C1's 51,651 counterexamples include T3-satisfying layers.

## C.2 Recommended implementation order

1. **Theorem 4 race bounds** (Track B). Smallest diff (`Worker::ab` probe-path
   synthesis), zero new data structures, proven uniformly in depth, verified by
   exact whole-graph relabeling at 2.1M states. Instrument fire-rate counter.
2. **Win-direction footprint pruning** (Track A Thm 1/Cor 1) with the §A.5
   build-then-verify extractor, mask cache, and refuted-Z-node instrumentation.
   This is the big lever (walls are 85–90% of branching) and the bulk of the
   2–6x estimate.
3. **Draw-direction pruning** behind the alpha-guard, fed by race-retrograde
   draw certificates (the only genuine ≥Draw certificates currently available).
4. **Theorem 2 sibling transfer** in `walls_left[Z] = 1` layers, reusing the
   extractor's consulted-edge logs.
5. **Dead-slot ordering** + counters; revisit the dead-cluster pass game
   (research gap) only if counters show clusters matter on 6x5+.

Gate each stage on: `cargo test --release` green from
`/Users/Ethan_1/barricades/solver`, plus an A/B exact-value regression (same
solved values, node counts compared) on the §A/§B test-position suites
(T1–T5, P1–P5c, P4) and a randomized sweep.

## C.3 NOT safe / needs more proof work

- **Conjecture C1** (general sibling transfer): refuted; would need anti-unlock
  sets — currently dead.
- **Conjecture 2A** (mover-only dominance): refuted outright; struck.
- **Dead-wall removal / dead-twin folding**: removal refuted (pass-tempo);
  folding vacuous. Open theory target: dead-cluster pass-counting game
  (sum-of-games + race parity) — the no-spread empirical finding says only the
  pass mechanism matters, which makes this tractable.
- **Witness-path minimization** (G3): correctness-irrelevant optimization; do
  after measuring footprint sizes in real solves.
- **All frequency numbers** are random-playout or small-board estimates;
  in-search rates require the stage-1/2 counters before quoting speedups.

## C.4 Soundness ledger

| Claim | Final status |
|---|---|
| A Thm 1, Win direction (full footprint) | **Proven**; 537,771 complete-graph checks + targeted + sweep, 0 violations |
| A Thm 1, Draw direction | **Proven given genuine certificate (G1)**; exhaustive over all 32 draw roots, 0 violations |
| A Cor 1/2 + integration guards | **Proven**; cap (not equality) empirically confirmed |
| A Thm 2 (T1+T2+T3) | **Proven**; 1,342,075 pairs, 0 violations |
| A Conj C1 (general sibling) | **REFUTED** (51,651 counterexamples) |
| A §5 TT-hot dtw rank | **UNSOUND as originally written**; replaced by verified-rank schemes (§A.5) |
| B Lemma L1 | **Proven as narrowed** (candidate-pair disjointness refuted; no-subsumption survives) |
| B Lemma L2/L3, Lemma M | **Proven**; 4.2M / 63.6M checks, 0 violations |
| B Thm 4 race bounds + TT spec | **Proven**; 750,836 nodes + full-graph relabel at 2.1M states, 0 mismatches |
| B Thm B-1/B-1M dead detection | **Proven** (detection only; removal refuted by tempo) |
| B Thm 3 dead-twin folding | Proven but **provably vacuous** for w,h ≥ 3 — documentation only |
| B Conj 2A | **REFUTED** (446 counterexamples, overlap-denial mechanism) |
| B race-tempo dominance / deferral / edge subsumption | **REFUTED** (P4 et al.) |

Both falsification harnesses were throwaway (deleted, nothing committed); the
parallel df-pn task's working-tree files were never touched by either track.
