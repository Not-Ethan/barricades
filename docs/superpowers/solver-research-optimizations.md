# Search-Complexity Research — Verified Findings & Optimization Plan

**Date:** 2026-06-09
**Method:** deep-research workflow — 6 search angles, 29 sources fetched, 144 claims
extracted, top 25 adversarially verified (3 independent votes each): 24 confirmed, 1 killed.
**Question:** provably-sound techniques to cut the search complexity of our exact
rectangular-Quoridor solver, whose mid-game wall-placement branching blows up ~20–100×
per added wall (6×5: W2 = 6.5 M nodes, W3 = 268 M, W4 > 10–20 B).

## Headline (all claims verified against primary sources)

1. **Proof-number search (df-pn family), not alpha-beta, is the dominant technique for
   exact game solving** — it underlies the Checkers draw proof, Fanorona, Go-Moku, and
   all 8×8–10×10 Hex results. Direct benchmark (LOA, 314 commonly-solved positions):
   alpha-beta searched **~19× more nodes** than PN-search. *(Winands & van den Herik PN
   chapter; ICGA "First Twenty Years" survey.)*

2. **GHI (graph-history interaction) is a practical soundness hazard, not theoretical:**
   a GHI-ignoring df-pn returned **wrong answers on 18/200** checkers problems (87,181
   corrupted TT entries found). The sound fix — **Kishimoto–Müller (AAAI-04)**: 64-bit
   path signatures, base/twin TT entries, simulation re-verification — costs **~0–2.5 %
   nodes**. Mandatory for Quoridor (repetition draws). Binary VALUE stays correct under
   arbitrary TT replacement; only the recovered proof tree needs the retention condition
   (Kishimoto-05 root reconstruction fixes that).

3. **The 1+ε trick** (Pawlewicz & Lew CG-06) keeps df-pn effective when the search space
   exceeds TT capacity (our exact regime): re-expansions drop from linear to O(log pt);
   **36× speedup** memory-starved (Atari-Go 6×6: 95 s vs 3403 s at a 2¹⁴-entry TT);
   combine with least-work TT replacement. SPDFPN sustained progress with a search space
   **100,000× larger than the TT**.

4. **PNS has a known pathology in exactly our regime** (many legal moves, slowly changing
   between parent/child — our ~40 wall slots): it degenerates to breadth-first search.
   Fixes that preserve exactness: **Dynamic Widening** (Yoshizoe-08, >30× on 19×19 Go
   capture) and **FDFPN** (Henderson: child limit = base + ⌈fraction × live⌉, every child
   eventually considered; disabling it cost 141–214 % time in Hex ablations).

5. **The largest measured node reducer anywhere is Hex's mustplay/relevance pruning
   (~25× nodes)** plus inferior-cell/fill-in analysis (~10×). These are proof-set-derived
   *theorems*, not heuristics — but they are Hex-specific (draw-free, monotone, acyclic)
   and **must be re-proven for Quoridor** (which is loopy, has genuine draws, and where
   walls *create* moves via the diagonal-jump rule). No Quoridor version exists — novel
   research territory.

6. **SPDFPN parallel df-pn:** ~11.8× on 16 threads (tuned; 0.59 efficiency default).
   **Published version has no GHI handling** (Hex is acyclic) — ours must keep
   Kishimoto–Müller machinery under parallelism.

7. **Refuted/open:** the claim "df-pn is provably incomplete on cyclic graphs even with
   GHI handling" failed verification (1-2) — termination of df-pn on loopy games is
   genuinely unsettled; we mitigate with a cycle guard + alpha-beta fallback.
   **Nothing Quoridor-specific survived verification** — no formalization of the
   illegality-frontier idea, no Quoridor endgame DBs, no exact-solving prior art. Also
   unanswered: tablebase compression specifics, rotate+swap symmetry soundness, ETC/
   aspiration measurements.

## Ranked plan for our bottleneck (expected node reduction)

| # | Technique | Verified magnitude (source domain) | Status |
|---|---|---|---|
| 1 | Mustplay-analog **wall-relevance pruning** | ~25× nodes (Hex) | needs novel theorem — theory track A |
| 2 | **df-pn + K-M GHI + 1+ε** + least-work TT | ~19× vs AB; +36× memory-starved | build pipeline stage 1 |
| 3 | **FDFPN dynamic widening** | 1.4–2.1× (Hex), targets our pathology | build pipeline stage 2 |
| 4 | **Inferior-wall / dominance analysis** | ~10× (Hex fill-in) | needs novel theorem — theory track B |
| 5 | **SPDFPN parallelization** | ~11.8× / 16 threads | build pipeline stage 3 |

(2)+(3)+(5) compose into one architecture change; (1) and (4) are orthogonal multipliers.
Hex precedent: the combined knowledge-heavy solver beat the prior best by ~660× nodes
(7×7 ablation) — knowledge cost is polynomial, pruning gains exponential.

## Execution

- **Theory workflow** (`solver-theory-tracks`): tracks 1 & 4 — formal theorem development
  with adversarial falsification: candidate pruning rules are *empirically tested against
  the exact solver over complete reachable graphs* before any proof is trusted.
- **Build workflow** (`solver-dfpn-pipeline`): tracks 2, 3, 5 as three exactness-gated
  stages; the alpha-beta solver is kept intact as the differential oracle, with fallback
  on df-pn cycle stalls.

## Key sources (all verified primary)

- Kishimoto & Müller, *A General Solution to the GHI Problem*, AAAI-04
- Winands & van den Herik, PN-search chapter; ICGA survey *The First Twenty Years*
- Pawlewicz & Lew, *The 1+ε trick*, CG-06
- Pawlewicz & Hayward, *SPDFPN / Scalable Parallel DFPN*, CG-13
- Henderson, PhD thesis (Hex: mustplay, inferior cells, FDFPN); *Solving 8×8 Hex*, IJCAI-09
