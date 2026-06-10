//! Df-pn(+) exact solving engine — the SECOND driver beside the alpha-beta
//! `Solver` (which stays intact as the differential oracle).
//!
//! This module implements depth-first proof-number search (Nagai 2002) with
//! three published enhancements, each followed precisely:
//!
//!   1. **Kishimoto–Müller GHI handling** (AAAI-04, "A General Solution to the
//!      Graph History Interaction Problem"): 64-bit path signatures per TT
//!      entry, base/twin entry splitting (path-dependent (dis)proofs are tagged
//!      with the signature of the path they were established on), and
//!      Kawano-style *simulation* re-verification when a stored (dis)proof is
//!      reached via a new path. Root thresholds are set to `INF-1` (not `INF`)
//!      per their df-pn modification, and a base entry's pn/dn are re-set to
//!      1/1 whenever a (dis)proof is saved into a twin entry.
//!   2. **1+ε trick** (Pawlewicz & Lew, CG-06): the selected child's threshold
//!      on the binding number is `min(parent_threshold, ceil(second_best ·
//!      (1+ε)))` instead of `second_best + 1`, cutting re-expansions from
//!      linear to logarithmic in the parent threshold. ε defaults to 1/4 (the
//!      paper's empirical optimum for df-pn) and is tunable via `QS_EPS`.
//!   3. **Least-work-replacement TT**: a fixed-capacity table (`QS_DFPN_MB`,
//!      default 1024 MiB) probed in small clusters
//!      of `CLUSTER` cells; on overflow the entry with the least accumulated
//!      subtree work is evicted. Eviction is exactness-neutral for the VALUE
//!      (a missing entry only forces re-search; see the soundness notes).
//!
//! # Game-value semantics (MUST match the alpha-beta solver)
//!
//! The value of a position is the HISTORY-FREE fixpoint value on the
//! `Loss < Draw < Win` lattice — the unique labeling the retrograde race
//! solver (`endgame.rs`) computes on its subgame, and the limit the alpha-beta
//! solver's depth-bounded negamax (draw at the horizon) converges to: `Win` if
//! the side to move can force reaching its goal in finitely many plies, `Loss`
//! if the opponent can, `Draw` otherwise (perpetual shuffling / blockade).
//! Quoridor is loopy; genuine draws exist.
//!
//! **Repetition rule used by the search**: a move that closes a cycle (the
//! successor's *canonical* position already occurs on the current root-to-node
//! path) is adjudicated `Draw` — the shuffling side makes no progress. This is
//! a SEARCH device, not a change of game: the truncated game tree (repetition
//! = draw terminal) has the same value at the ROOT as the fixpoint game,
//! because a forced win admits a strategy that strictly decreases the
//! plies-to-win measure, so no position (hence no canonical class, the mirror
//! being a value-preserving automorphism) can repeat along any line of it;
//! draws/losses are likewise preserved since truncation only ever introduces
//! Draw terminals, never wins. The alpha-beta solver realizes the same
//! semantics differently (a shuffle burns depth and unresolved-at-horizon
//! scores Draw), so both engines compute the same game value.
//!
//! At INTERIOR nodes, however, the truncated value is relative to the path —
//! exactly the Graph History Interaction problem — which is why the
//! Kishimoto–Müller machinery below is mandatory for soundness, not optional.
//!
//! # Three-valued result via two binary runs (documented choice)
//!
//! Df-pn proves binary questions. We follow the standard two-run scheme (the
//! same one Kishimoto–Müller use for checkers, and our research brief calls
//! standard practice):
//!
//!   * **Run 1**: prove/disprove "the side to move at the root (`A`) forces a
//!     Win". OR nodes are `A`-to-move nodes, AND nodes the opponent's.
//!   * **Run 2** (only if run 1 is disproven): prove/disprove "the opponent
//!     (`B`) forces a Win" — the root is then an AND node.
//!   * Result: run 1 proven → `Win`; run 2 proven → `Loss`; both disproven →
//!     `Draw`.
//!
//! In BOTH runs a repetition counts AGAINST the question's attacker (a cycle
//! is a draw, never a win), so both runs are Kishimoto–Müller's
//! *first-player-loss* scenario: only DISPROOFS can be path-dependent. Proofs
//! are grounded exclusively in real terminals (goal reached / mover stuck /
//! exact race values) and in other proofs, never in repetition adjudications,
//! so every proof is path-independent. The implementation still carries the
//! general machinery (twin entries hold either polarity) defensively.
//!
//! # Exact leaf folding
//!
//! * `winner(s)` — real terminal, path-independent.
//! * `walls_left == [0,0]` — the retrograde race solver returns the EXACT
//!   W/D/L fixpoint value; it is folded in directly. Folding a *fresh* exact
//!   value at an interior node is sound in both directions: a fresh Win for
//!   the attacker yields a real winning continuation regardless of history
//!   (fixpoint semantics ignore repetition), and a fresh non-Win implies no
//!   truncated win exists on ANY path (a truncated win would be a real win),
//!   i.e. a path-INDEPENDENT disproof.
//! * No legal moves — the mover loses (same convention as the AB solver and
//!   the race solver). This needs no special case: an OR node with zero
//!   children gets `pn = min ∅ = INF`, `dn = Σ ∅ = 0` (attacker stuck =
//!   disproof) and an AND node dually (defender stuck = proof).
//!
//! # Cycle / infinite-loop guard
//!
//! Df-pn on directed cyclic graphs can fail to terminate (the completeness
//! question is open): a node's pn/dn can become self-supporting through the
//! TT. Direct self-support through the current path is already impossible here
//! (a repetition is adjudicated BEFORE any TT probe), but we keep two guards,
//! both exactness-preserving because they resolve the offending subtree with
//! the (exact) alpha-beta solver and fold the result in as a path-independent
//! (dis)proof — fresh exact values are valid on every path (see above):
//!
//!   * **Self-support detector**: if the same child is selected `STALL_LIMIT`
//!     times in a row with identical (pn, dn) and identical thresholds — i.e.
//!     consecutive recursive calls made no progress, which an honest store
//!     cannot do (a returned child must raise pn or dn past its strictly-higher
//!     threshold) — that child is resolved by the AB fallback. A small run
//!     length is tolerated because a TT eviction can erase one store benignly.
//!   * **Expansion watchdog**: if a single non-root `mid` call loops more than
//!     `QS_DFPN_LOOP_CAP` (default 8 192) times, the NODE is resolved by the
//!     AB fallback. (The root is exempt — it loops once per top-level descent
//!     by design and is covered by the other guards.)
//!   * **Twin-escalation**: a position that demands a SECOND search-derived
//!     path-dependent (dis)proof is resolved exactly once by the AB fallback
//!     and stored path-independently. Path-dependent results never transpose,
//!     so without this guard shuffle-swamp subgraphs (pervasive in loopy
//!     Quoridor middlegames) degenerate into per-path re-search — measured on
//!     5x5-w2: the guard set turns a >600 s timeout into an ordinary solve,
//!     with the K-M machinery still handling every isolated repetition.
//!
//! Both log to stderr (first few occurrences) and are counted in the stats.
//! The root has its own ultimate fallback: if a run ends `unknown` (a
//! threshold-saturated return with neither pn nor dn at 0 — possible only via
//! the `INF-1` caps), the whole position is handed to the AB solver.
//!
//! # Soundness under TT replacement
//!
//! Per the 2012 ICGA df-pn survey (and our brief): the BINARY VALUE df-pn
//! returns is correct under arbitrary TT replacement; only recovering the
//! proof TREE needs proven nodes retained. We only consume values. Concretely:
//! evicting an entry merely forces a re-search; simulation that cannot find
//! the entries it wants simply fails to verify and the node is re-searched
//! under the current path, which is always sound.
//!
//! # Deviations from the papers (documented)
//!
//! * Path signatures hash the CANONICAL position reached at each ply
//!   (splitmix-style mix of `(canonical pack, ply)`, XOR-folded along the
//!   path) instead of Kishimoto–Müller's `R[move][depth]` random table. The
//!   position-sequence determines the truncation context exactly (it is what
//!   repetition is defined on), the ply-salting keeps the encoding
//!   order-sensitive like theirs, and no `MaxMove × MaxDepth` table is needed.
//!   Like the published scheme, signature equality is trusted at 64 bits.
//! * Nagai's df-pn stores the entry thresholds at node ENTRY so that cyclic
//!   descendants terminate early; our explicit on-path repetition check
//!   subsumes that role, so we store only at exit. The `INF-1` root-threshold
//!   discipline is kept regardless (no saturated unproven value can ever be
//!   confused with a true `INF` (dis)proof).
//! * Df-pn+ leaf initialization (the "(+)"): unproven, never-seen leaves are
//!   initialized from the goal-distance heuristic instead of (1, 1)
//!   (`QS_DFPN_H=0` restores (1, 1)). Initialization steers selection only —
//!   it can never change a proven/disproven outcome.

use crate::board::Board;
use crate::endgame::RaceTt;
use crate::movegen;
use crate::solver::{build_mirror_perm, canonical, pack_u128, Solver, Value};
use crate::state::State;
use rustc_hash::FxHashSet;
use std::sync::Arc;

/// Proof/disproof-number infinity: a TRUE (dis)proof and nothing else. All
/// unproven arithmetic saturates at `INF - 1` (`SAT`), so `INF` is unambiguous.
const INF: u64 = u64::MAX;
/// Saturation ceiling for UNPROVEN proof/disproof numbers and thresholds
/// (Kishimoto–Müller's `∞ - 1` root-threshold discipline, applied uniformly).
const SAT: u64 = u64::MAX - 1;

/// Probe-cluster width of the least-work TT (the "probe k=4 cells, evict the
/// least-accumulated-work entry" scheme from the research brief).
const CLUSTER: usize = 4;

/// Default df-pn TT budget in MiB when `QS_DFPN_MB` is unset.
const DEFAULT_DFPN_MB: usize = 1024;

/// Default ε numerator over 1024 for the 1+ε trick: 256/1024 = 1/4, the value
/// Pawlewicz & Lew found empirically best for df-pn.
const DEFAULT_EPS_NUM: u64 = 256;

/// Default per-`mid`-call expansion-loop watchdog (`QS_DFPN_LOOP_CAP`).
/// Non-root only — the root call legitimately loops once per top-level
/// descent. A node that churns this many select/recurse iterations without
/// resolving sits in a loopy shuffle swamp where df-pn makes no transposition
/// progress; punting it to the exact AB fallback EARLY is a large net win
/// (measured on 5x5-w2: >600 s timeout at 100k vs ~2 min at 4k, value equal).
const DEFAULT_LOOP_CAP: u64 = 8_192;

/// Default node budget for one Kawano simulation walk (`QS_DFPN_SIM_BUDGET`).
/// Simulation is a verification overlay: exhausting the budget merely fails
/// the verification and the node is re-searched (always sound). It MUST be
/// small: simulation is meant to borrow stored (dis)proof structure in a few
/// O(1) TT/terminal checks (K-M measure ~0-2.5% total overhead); a generous
/// budget lets failing attempts degenerate into exponential walks that
/// dominate the whole search (observed directly on 5x5-w2 with a 10k budget).
const DEFAULT_SIM_BUDGET: u64 = 128;

/// Consecutive no-progress re-expansions of the SAME child with identical
/// values and thresholds before the self-support guard fires. One repeat can
/// be a benign TT-eviction artifact; a genuine TT-mediated cycle repeats
/// forever, so a small run length separates the two cheaply.
const STALL_LIMIT: u32 = 8;

/// Cap on the failed-simulation memo before it is cleared (bounded memory;
/// clearing only allows re-attempts, never affects values).
const SIM_FAIL_CACHE_MAX: usize = 1 << 22;

/// Default MiB for the lazily created embedded AB fallback solver's TT
/// (`QS_DFPN_FALLBACK_MB`). The fallback fires rarely; it needs only a modest
/// cache.
const DEFAULT_FALLBACK_MB: usize = 256;

/// Entry kind: `Base` (one per position×question: unproven pn/dn or a
/// path-INdependent (dis)proof) or `Twin` (a path-DEPENDENT (dis)proof valid
/// for the path whose signature it carries — Kishimoto–Müller's twin table
/// entries; several may coexist for one position, one per path).
const KIND_BASE: u8 = 0;
const KIND_TWIN: u8 = 1;

/// One df-pn TT entry.
///
/// `key` is the FULL injective u128 pack of the canonical state with the
/// question's attacker folded into bit 121 and `OCCUPIED` at bit 127 (so a
/// live key is never 0 and the two binary runs never alias). The full key is
/// verified on probe: a hash collision is a miss, never a foreign hit.
///
/// `pn`/`dn` follow the standard convention: `(0, INF)` proven, `(INF, 0)`
/// disproven, otherwise both in `1..=SAT` (unproven). `work` accumulates the
/// number of search nodes spent below this entry — the least-work replacement
/// victim metric. `sig` is the 64-bit path signature (twins only; 0 in base
/// entries).
#[derive(Clone, Copy)]
struct Entry {
    key: u128,
    pn: u64,
    dn: u64,
    work: u64,
    sig: u64,
    kind: u8,
}

impl Entry {
    const EMPTY: Entry = Entry {
        key: 0,
        pn: 1,
        dn: 1,
        work: 0,
        sig: 0,
        kind: KIND_BASE,
    };
    #[inline]
    fn is_empty(&self) -> bool {
        self.key == 0
    }
    #[inline]
    fn is_proof(&self) -> bool {
        self.pn == 0
    }
    #[inline]
    fn is_solved(&self) -> bool {
        self.pn == 0 || self.dn == 0
    }
}

/// Occupied marker bit (key is never 0 for a live entry; bit 127 is far above
/// the ~100 bits `pack_u128` uses).
const OCCUPIED_BIT: u128 = 1u128 << 127;
/// Attacker bit (which binary question this entry belongs to): bit 121.
const ATTACKER_BIT: u128 = 1u128 << 121;

/// Fixed-capacity, cluster-probed, least-work-replacement transposition table.
///
/// `2^k` cells in clusters of `CLUSTER`; a key hashes to one cluster and may
/// occupy any cell in it. Several entries may share the same `key` (one base
/// plus twins for different paths) — `probe_*` therefore scan the whole
/// cluster. When a store finds no empty cell and no in-place match, the
/// least-`work` cell in the cluster is evicted (the published least-work
/// replacement scheme paired with the 1+ε trick). Eviction is value-neutral.
struct DfpnTt {
    cells: Vec<Entry>,
    /// `(cells.len() / CLUSTER) - 1`; cluster count is a power of two.
    cluster_mask: usize,
    fill: usize,
}

impl DfpnTt {
    fn with_capacity_mb(mb: usize) -> DfpnTt {
        let bytes = mb.max(1).saturating_mul(1024 * 1024);
        let want_cells = (bytes / size_of::<Entry>()).max(2 * CLUSTER);
        // Round the cluster count DOWN to a power of two (mask indexing).
        let want_clusters = (want_cells / CLUSTER).max(1);
        let nclusters = if want_clusters.is_power_of_two() {
            want_clusters
        } else {
            want_clusters.next_power_of_two() / 2
        };
        DfpnTt {
            cells: vec![Entry::EMPTY; nclusters * CLUSTER],
            cluster_mask: nclusters - 1,
            fill: 0,
        }
    }

    /// First cell index of `key`'s cluster (fast splitmix-style mix; the full
    /// key is verified on probe, so the hash only spreads, never decides).
    #[inline]
    fn cluster_start(&self, key: u128) -> usize {
        let lo = key as u64;
        let hi = (key >> 64) as u64;
        let mut x = lo ^ hi.rotate_left(32);
        x ^= x >> 33;
        x = x.wrapping_mul(0xff51afd7ed558ccd);
        x ^= x >> 33;
        ((x as usize) & self.cluster_mask) * CLUSTER
    }

    /// The base entry for `key`, if resident.
    #[inline]
    fn probe_base(&self, key: u128) -> Option<Entry> {
        let s = self.cluster_start(key);
        self.cells[s..s + CLUSTER]
            .iter()
            .find(|e| e.key == key && e.kind == KIND_BASE)
            .copied()
    }

    /// All twin entries for `key` currently resident (at most `CLUSTER - 1`).
    #[inline]
    fn probe_twins(&self, key: u128, out: &mut Vec<Entry>) {
        out.clear();
        let s = self.cluster_start(key);
        for e in &self.cells[s..s + CLUSTER] {
            if e.key == key && e.kind == KIND_TWIN {
                out.push(*e);
            }
        }
    }

    /// Store/update the base entry for `key`. In-place if resident; otherwise
    /// fills an empty cell or evicts the least-work cell of the cluster.
    /// `add_work` accumulates into the entry's work counter.
    fn store_base(&mut self, key: u128, pn: u64, dn: u64, add_work: u64) {
        let s = self.cluster_start(key);
        let cluster = &mut self.cells[s..s + CLUSTER];
        if let Some(e) = cluster
            .iter_mut()
            .find(|e| e.key == key && e.kind == KIND_BASE)
        {
            // Never let an unproven update clobber a solved base entry: a
            // solved base is a path-independent fact; "downgrading" it would
            // only cost re-search, but it can simply be kept.
            if e.is_solved() && !(pn == 0 || dn == 0) {
                return;
            }
            e.pn = pn;
            e.dn = dn;
            e.work = e.work.saturating_add(add_work);
            return;
        }
        let new = Entry {
            key,
            pn,
            dn,
            work: add_work,
            sig: 0,
            kind: KIND_BASE,
        };
        Self::place(cluster, new, &mut self.fill);
    }

    /// Re-initialize a resident base entry's pn/dn to 1/1 (the Kishimoto–
    /// Müller df-pn modification applied whenever a (dis)proof is saved into a
    /// twin entry: df-pn's inflated pre-(dis)proof numbers otherwise wedge the
    /// search on other paths). No-op when absent or already solved.
    fn reset_base(&mut self, key: u128) {
        let s = self.cluster_start(key);
        if let Some(e) = self.cells[s..s + CLUSTER]
            .iter_mut()
            .find(|e| e.key == key && e.kind == KIND_BASE)
            && !e.is_solved()
        {
            e.pn = 1;
            e.dn = 1;
        }
    }

    /// Store a path-dependent (dis)proof twin for `(key, sig)`.
    fn store_twin(&mut self, key: u128, sig: u64, proof: bool, add_work: u64) {
        let s = self.cluster_start(key);
        let cluster = &mut self.cells[s..s + CLUSTER];
        let (pn, dn) = if proof { (0, INF) } else { (INF, 0) };
        if let Some(e) = cluster
            .iter_mut()
            .find(|e| e.key == key && e.kind == KIND_TWIN && e.sig == sig)
        {
            e.pn = pn;
            e.dn = dn;
            e.work = e.work.saturating_add(add_work);
            return;
        }
        let new = Entry {
            key,
            pn,
            dn,
            work: add_work,
            sig,
            kind: KIND_TWIN,
        };
        Self::place(cluster, new, &mut self.fill);
    }

    /// Place `new` in the cluster: empty cell first, else evict the
    /// least-accumulated-work cell (the least-work replacement policy).
    #[inline]
    fn place(cluster: &mut [Entry], new: Entry, fill: &mut usize) {
        if let Some(e) = cluster.iter_mut().find(|e| e.is_empty()) {
            *e = new;
            *fill += 1;
            return;
        }
        let victim = cluster
            .iter_mut()
            .min_by_key(|e| e.work)
            .expect("cluster is non-empty");
        *victim = new;
    }

    #[inline]
    fn len(&self) -> usize {
        self.fill
    }
    #[inline]
    fn capacity(&self) -> usize {
        self.cells.len()
    }
}

/// Mix a canonical position pack and its ply into one 64-bit path-step code
/// (the `R[position][ply]` analogue of Kishimoto–Müller's `R[move][depth]`
/// random table, realized as a hash instead of a table). XOR-folding these
/// step codes along the path yields the path signature; the ply salt keeps the
/// encoding order-sensitive exactly as in the paper.
#[inline]
fn step_code(cpack: u128, ply: u32) -> u64 {
    let lo = cpack as u64;
    let hi = (cpack >> 64) as u64;
    let mut x = lo ^ hi.rotate_left(27) ^ ((ply as u64) << 1).wrapping_mul(0x9e3779b97f4a7c15);
    x = x.wrapping_mul(0xbf58476d1ce4e5b9);
    x ^= x >> 31;
    x = x.wrapping_mul(0x94d049bb133111eb);
    x ^= x >> 29;
    // Never 0, so a twin's sig can't collide with the base sentinel 0.
    x | 1
}

/// Search statistics for one `DfpnSolver` (cumulative across `solve` calls).
#[derive(Clone, Copy, Default, Debug)]
pub struct DfpnStats {
    /// `mid` node expansions (the df-pn analogue of AB's internal nodes).
    pub nodes: u64,
    /// Race-endgame nodes folded in at walls-exhausted leaves.
    pub race_nodes: u64,
    /// Repetition adjudications (cycle-closing successors seen).
    pub rep_hits: u64,
    /// Twin (path-dependent (dis)proof) entries stored.
    pub twin_stores: u64,
    /// Kawano simulations invoked / nodes walked / verifications succeeded.
    pub sim_calls: u64,
    pub sim_nodes: u64,
    pub sim_verified: u64,
    /// AB-fallback resolutions: self-support child, watchdog node, repeated
    /// path-dependent (dis)proof (twin-escalation), root run.
    pub fallback_child: u64,
    pub fallback_node: u64,
    pub fallback_twin: u64,
    pub fallback_root: u64,
}

impl DfpnStats {
    /// Total search effort, comparable to the AB solver's `nodes` counter
    /// (which likewise sums its main-search and race nodes).
    pub fn total_nodes(&self) -> u64 {
        self.nodes + self.race_nodes + self.sim_nodes
    }
    pub fn fallbacks(&self) -> u64 {
        self.fallback_child + self.fallback_node + self.fallback_twin + self.fallback_root
    }
}

/// Per-question search context: the current root-to-node path as canonical
/// packs (vector for ordered scans plus the signature stack). The path is what
/// repetition and path signatures are defined on.
struct Ctx {
    /// Which player the question "X forces a Win" is about.
    attacker: u8,
    /// Canonical packs of the positions on the current path, root first.
    path: Vec<u128>,
    /// `sigs[i]` = XOR-fold of `step_code` over `path[0..=i]`.
    sigs: Vec<u64>,
}

impl Ctx {
    #[inline]
    fn on_path(&self, cpack: u128) -> bool {
        self.path.iter().rev().any(|&p| p == cpack)
    }
    #[inline]
    fn tip_sig(&self) -> u64 {
        *self.sigs.last().expect("path is never empty")
    }
    #[inline]
    fn push(&mut self, cpack: u128) {
        let ply = self.path.len() as u32;
        let sig = self.tip_sig() ^ step_code(cpack, ply);
        self.path.push(cpack);
        self.sigs.push(sig);
    }
    #[inline]
    fn pop(&mut self) {
        self.path.pop();
        self.sigs.pop();
    }
    /// Signature a CHILD position would have if pushed now.
    #[inline]
    fn child_sig(&self, child_cpack: u128) -> u64 {
        self.tip_sig() ^ step_code(child_cpack, self.path.len() as u32)
    }
}

/// One evaluated child: successor state, canonical pack, TT key, the df-pn+
/// leaf initialization (precomputed once from the ordering distances), the
/// current pn/dn, and whether the current value is path-dependent (repetition
/// or twin derived).
struct Child {
    state: State,
    cpack: u128,
    key: u128,
    /// Df-pn+ `(pn, dn)` initialization for a never-stored leaf.
    h: (u64, u64),
    /// `(pn, dn, dep)` FIXED for the whole parent `mid` call: repetition
    /// adjudication (the path is invariant across the call's loop iterations),
    /// real terminal, or exact race fold. Computed ONCE in `gen_children`;
    /// re-deriving these per loop iteration (especially `race_value`) was
    /// measured to dominate the entire search.
    fixed: Option<(u64, u64, bool)>,
    pn: u64,
    dn: u64,
    dep: bool,
}

/// The df-pn exact solving engine. Borrows the `Board`; owns its least-work
/// TT, a bounded exact race memo (shared type with the AB solver), and a
/// lazily created embedded AB `Solver` used ONLY as the cycle-guard /
/// root-unknown fallback.
pub struct DfpnSolver<'a> {
    b: &'a Board,
    tt: DfpnTt,
    race_tt: Arc<RaceTt>,
    mirror_perm: Option<Vec<u8>>,
    /// ε numerator over 1024 for the 1+ε trick (`QS_EPS`, default 0.25).
    eps_num: u64,
    /// Per-`mid` expansion-loop watchdog (`QS_DFPN_LOOP_CAP`).
    loop_cap: u64,
    /// Node budget per simulation walk (`QS_DFPN_SIM_BUDGET`).
    sim_budget: u64,
    /// Df-pn+ leaf initialization from the distance heuristic (`QS_DFPN_H`).
    use_h: bool,
    /// GHI machinery master switch. `true` in ALL real use; `false` exists
    /// ONLY so the test suite can demonstrate that a naive (GHI-ignoring)
    /// df-pn actually mis-evaluates positions our machinery gets right.
    ghi: bool,
    /// Embedded fallback AB solver (lazily built; `QS_DFPN_FALLBACK_MB`).
    ab: Option<Box<Solver<'a>>>,
    ab_mb: usize,
    /// Cumulative statistics.
    pub stats: DfpnStats,
    /// Scratch for twin probes (avoids per-node allocation).
    twin_scratch: Vec<Entry>,
    /// Memo of FAILED simulation attempts keyed by `mix(key, path-sig)`:
    /// a verification that failed once for a (position, path) pair is not
    /// retried on every re-evaluation (Kishimoto–Müller's "reducing simulation
    /// calls" concern). Purely a performance cache — a hash collision or a
    /// clear merely re-runs or skips a verification attempt, and a skipped
    /// verification just means the node is searched normally (always sound).
    sim_fail: FxHashSet<u64>,
    /// Positions (TT keys) that have already had one SEARCH-DERIVED
    /// path-dependent (dis)proof stored. A second search-derived twin for the
    /// same position is the GHI path-enumeration symptom (path-dependent
    /// results never transpose, so shuffle-swamp subgraphs degenerate into
    /// per-path re-search); the position is then escalated to the exact AB
    /// fallback ONCE and stored as a path-independent base fact. Simulation-
    /// derived twins (cheap, bounded reuse) deliberately do NOT count.
    twin_seen: FxHashSet<u128>,
}

impl<'a> DfpnSolver<'a> {
    /// Build with budgets/knobs from the environment: `QS_DFPN_MB` (TT MiB,
    /// default 1024), `QS_RACE_MB` (race memo), `QS_EPS` (the 1+ε factor,
    /// default 0.25), `QS_DFPN_LOOP_CAP`, `QS_DFPN_SIM_BUDGET`, `QS_DFPN_H`
    /// (default on), `QS_DFPN_FALLBACK_MB`.
    pub fn new(b: &'a Board) -> DfpnSolver<'a> {
        let mb = std::env::var("QS_DFPN_MB")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .filter(|&m| m > 0)
            .unwrap_or(DEFAULT_DFPN_MB);
        DfpnSolver::with_tt_mb(b, mb)
    }

    /// Build with an explicit TT budget in MiB (test hook; `new` funnels the
    /// env budget here).
    pub fn with_tt_mb(b: &'a Board, mb: usize) -> DfpnSolver<'a> {
        let env_u64 = |k: &str| {
            std::env::var(k)
                .ok()
                .and_then(|v| v.trim().parse::<u64>().ok())
                .filter(|&m| m > 0)
        };
        let eps_num = std::env::var("QS_EPS")
            .or_else(|_| std::env::var("QS_DFPN_EPS"))
            .ok()
            .and_then(|v| v.trim().parse::<f64>().ok())
            .filter(|&e| (0.0..=8.0).contains(&e))
            .map(|e| (e * 1024.0).round() as u64)
            .unwrap_or(DEFAULT_EPS_NUM);
        let use_h = std::env::var("QS_DFPN_H").map(|v| v != "0").unwrap_or(true);
        DfpnSolver {
            b,
            tt: DfpnTt::with_capacity_mb(mb),
            race_tt: Arc::new(RaceTt::default()),
            mirror_perm: build_mirror_perm(b),
            eps_num,
            loop_cap: env_u64("QS_DFPN_LOOP_CAP").unwrap_or(DEFAULT_LOOP_CAP),
            sim_budget: env_u64("QS_DFPN_SIM_BUDGET").unwrap_or(DEFAULT_SIM_BUDGET),
            use_h,
            ghi: true,
            ab: None,
            ab_mb: env_u64("QS_DFPN_FALLBACK_MB")
                .map(|m| m as usize)
                .unwrap_or(DEFAULT_FALLBACK_MB),
            stats: DfpnStats::default(),
            twin_scratch: Vec::with_capacity(CLUSTER),
            sim_fail: FxHashSet::default(),
            twin_seen: FxHashSet::default(),
        }
    }

    /// TEST-ONLY: disable the GHI machinery (repetition (dis)proofs are then
    /// stored as path-independent base entries — the unsound, naive df-pn).
    /// Exists solely so `tests/dfpn_exact.rs` can demonstrate that naive df-pn
    /// mis-evaluates repetition-heavy positions which the full machinery
    /// handles. NEVER disable this outside that demonstration.
    pub fn set_ghi_for_unsound_demo(&mut self, on: bool) {
        self.ghi = on;
    }

    /// Occupied TT cells (reporting).
    pub fn tt_len(&self) -> usize {
        self.tt.len()
    }
    /// Fixed TT cell capacity (reporting).
    pub fn tt_capacity(&self) -> usize {
        self.tt.capacity()
    }
    /// Exact heap footprint of the TT array in bytes.
    pub fn tt_bytes(&self) -> usize {
        self.tt.capacity() * size_of::<Entry>()
    }
    /// Race-memo entry count (reporting).
    pub fn race_tt_len(&self) -> usize {
        self.race_tt.len()
    }

    /// Canonical pack of `s` under the mirror fold (shared with the AB TT).
    #[inline]
    fn cpack(&self, s: &State) -> u128 {
        pack_u128(&canonical(self.b, self.mirror_perm.as_deref(), s))
    }

    /// TT key for `(canonical pack, question attacker)`.
    #[inline]
    fn key(&self, cpack: u128, attacker: u8) -> u128 {
        cpack
            | OCCUPIED_BIT
            | if attacker == 1 { ATTACKER_BIT } else { 0 }
    }

    /// Exact fixpoint value of `s` (side to move) from the embedded AB
    /// fallback solver. Single-threaded for determinism; exactness identical.
    fn ab_value(&mut self, s: &State) -> Value {
        if self.ab.is_none() {
            let mut ab = Box::new(Solver::with_tt_mb(self.b, self.ab_mb));
            ab.set_threads(1);
            self.ab = Some(ab);
        }
        let ab = self.ab.as_mut().expect("just initialized");
        ab.solve(s)
    }

    /// Does `value` (for the side to move of the evaluated state) prove the
    /// question "`attacker` forces a Win"?
    #[inline]
    fn proves(turn: u8, attacker: u8, value: Value) -> bool {
        if turn == attacker {
            value == Value::Win
        } else {
            value == Value::Loss
        }
    }

    /// Solve `s` to its exact game value for the side to move (the same
    /// contract as `Solver::solve`).
    pub fn solve(&mut self, s: &State) -> Value {
        // Real terminals and walls-exhausted races need no search at all.
        if let Some(p) = self.b.winner(s) {
            return if p == s.turn { Value::Win } else { Value::Loss };
        }
        if s.walls_left == [0, 0] {
            let (v, n) = crate::endgame::race_value(self.b, s, &self.race_tt);
            self.stats.race_nodes += n;
            return v;
        }
        let me = s.turn;
        // Run 1: does the side to move force a win?
        match self.solve_question(s, me) {
            Some(true) => Value::Win,
            Some(false) => {
                // Run 2: does the opponent force a win?
                match self.solve_question(s, 1 - me) {
                    Some(true) => Value::Loss,
                    Some(false) => Value::Draw,
                    None => self.root_fallback(s),
                }
            }
            None => self.root_fallback(s),
        }
    }

    /// Ultimate root fallback: hand the position to the exact AB solver
    /// (logged + counted). Fires only when a df-pn run ends `unknown`.
    fn root_fallback(&mut self, s: &State) -> Value {
        self.stats.fallback_root += 1;
        eprintln!(
            "[dfpn] root run unknown (threshold-saturated); falling back to AB \
             for pawns={:?} h={:#x} v={:#x} wl={:?} turn={}",
            s.pawn, s.h_walls, s.v_walls, s.walls_left, s.turn
        );
        self.ab_value(s)
    }

    /// One binary run: prove/disprove "`attacker` forces a Win from `s`".
    /// `Some(true)` proven, `Some(false)` disproven, `None` unknown (both
    /// `INF-1` root thresholds saturated without a (dis)proof — the open
    /// non-termination guard's last resort).
    fn solve_question(&mut self, s: &State, attacker: u8) -> Option<bool> {
        let cpack = self.cpack(s);
        let mut ctx = Ctx {
            attacker,
            path: vec![cpack],
            sigs: vec![step_code(cpack, 0)],
        };
        // Root thresholds INF-1 per Kishimoto–Müller: a saturated unproven
        // number can never be mistaken for a true (dis)proof.
        let (pn, dn, _dep) = self.mid(s, SAT, SAT, &mut ctx);
        if pn == 0 {
            Some(true)
        } else if dn == 0 {
            Some(false)
        } else {
            None
        }
    }

    /// Evaluate a prospective child WITHOUT expanding it: fixed adjudication
    /// (repetition / real terminal / race fold, precomputed per `mid` call in
    /// `gen_children`), TT probe (base / sig-matched twin / simulation-
    /// verified twin), or the precomputed df-pn+ leaf init `h_init`. Returns
    /// `(pn, dn, dep)` for the question, `dep` = the value is path-dependent.
    fn child_value(
        &mut self,
        child: &State,
        cpack: u128,
        fixed: Option<(u64, u64, bool)>,
        h_init: (u64, u64),
        ctx: &mut Ctx,
    ) -> (u64, u64, bool) {
        let attacker = ctx.attacker;
        // 1-3. Repetition / terminal / race: fixed for the whole parent call.
        if let Some(f) = fixed {
            return f;
        }
        // 4. TT probe.
        let key = self.key(cpack, attacker);
        let base = self.tt.probe_base(key);
        if let Some(e) = base
            && e.is_solved()
        {
            // Path-independent (dis)proof: valid on every path.
            return (e.pn, e.dn, false);
        }
        if self.ghi {
            let child_sig = ctx.child_sig(cpack);
            let mut twins = std::mem::take(&mut self.twin_scratch);
            self.tt.probe_twins(key, &mut twins);
            // 4a. A twin recorded for THIS path: its (dis)proof applies as-is.
            if let Some(t) = twins.iter().find(|t| t.sig == child_sig) {
                let r = (t.pn, t.dn, true);
                self.twin_scratch = twins;
                return r;
            }
            // 4b. Twins from other paths: Kawano-simulate to re-verify the
            //     (dis)proof under the CURRENT path; on success store a new
            //     twin for this path and use it. Failed attempts are memoized
            //     per (position, path, polarity) so re-evaluations of the same
            //     child do not re-run a verification that already failed
            //     (K-M's "reducing simulation calls"; skipping is fail-safe —
            //     the node is simply searched normally).
            for t in &twins {
                let want_proof = t.is_proof();
                let fkey = Self::sim_fail_key(key, child_sig, want_proof);
                if self.sim_fail.contains(&fkey) {
                    continue;
                }
                self.stats.sim_calls += 1;
                let mut budget = self.sim_budget;
                ctx.push(cpack);
                let ok = self.simulate(child, want_proof, ctx, &mut budget);
                ctx.pop();
                if ok {
                    self.stats.sim_verified += 1;
                    self.stats.twin_stores += 1;
                    self.tt.store_twin(key, child_sig, want_proof, 0);
                    self.tt.reset_base(key);
                    let r = if want_proof { (0, INF, true) } else { (INF, 0, true) };
                    self.twin_scratch = twins;
                    return r;
                }
                if self.sim_fail.len() >= SIM_FAIL_CACHE_MAX {
                    self.sim_fail.clear();
                }
                self.sim_fail.insert(fkey);
            }
            self.twin_scratch = twins;
        }
        // 4c. Unproven base entry: heuristic pn/dn, path-independent by
        //     convention (K-M: unproven numbers may be path-influenced; only
        //     (dis)proofs carry correctness weight).
        if let Some(e) = base {
            return (e.pn, e.dn, false);
        }
        // 5. Never-seen leaf: df-pn+ initialization (precomputed in
        //    `gen_children` from the goal distances; selection bias only).
        (h_init.0, h_init.1, false)
    }

    /// The df-pn MID procedure (Nagai 2002) with the 1+ε child thresholds and
    /// the cycle guards. `s` is the node (side to move `s.turn`), `pt`/`dt`
    /// its proof/disproof thresholds, `ctx` the current path (with `s` already
    /// pushed by the caller — the root pushes itself in `solve_question`).
    ///
    /// Returns `(pn, dn, dep)`; `dep` is meaningful only for (dis)proofs and
    /// says the result is path-dependent (derived through a repetition or a
    /// twin entry).
    fn mid(&mut self, s: &State, pt: u64, dt: u64, ctx: &mut Ctx) -> (u64, u64, bool) {
        self.stats.nodes += 1;
        let work0 = self.stats.total_nodes();
        let attacker = ctx.attacker;
        let or_node = s.turn == attacker;
        let cpack = *ctx.path.last().expect("mid called with node on path");
        let key = self.key(cpack, attacker);

        // Children: legal moves ordered by the proven distance heuristic
        // (descending advantage), deduplicated by canonical pack (mirror-equal
        // siblings share a value; keeping one is value-safe and avoids df-pn's
        // DAG double-counting for that pair). Order affects selection
        // tie-breaks only, never values.
        let mut children = self.gen_children(s, ctx);

        // Per-child path-dependence memo (refreshed on every evaluation).
        let mut iter: u64 = 0;
        // Self-support detector state: (child index, its pn, its dn, ptc, dtc)
        // of the previous expansion. An identical tuple on consecutive
        // expansions means the recursive call made no progress — impossible
        // for an honest store (the child is called with thresholds strictly
        // above its current pn/dn, so a surviving store must raise one), hence
        // either a benign TT eviction or a TT-mediated cycle. `STALL_LIMIT`
        // consecutive repeats rule out eviction luck and fire the guard.
        let mut last_step: Option<(usize, u64, u64, u64, u64)> = None;
        let mut stall: u32 = 0;

        loop {
            // ---- Recompute child values and the node's pn/dn. ----
            for c in children.iter_mut() {
                let state = c.state;
                let (pn, dn, dep) = self.child_value(&state, c.cpack, c.fixed, c.h, ctx);
                c.pn = pn;
                c.dn = dn;
                c.dep = dep;
            }
            let (pn, dn) = Self::combine(or_node, &children);

            // ---- Solved? Store and return (the GHI-sensitive moment). ----
            if pn == 0 || dn == 0 {
                let proof = pn == 0;
                // Normalize: a (dis)proof is exactly (0, INF) / (INF, 0); the
                // non-binding number may have saturated during combination.
                let (pn, dn) = if proof { (0, INF) } else { (INF, 0) };
                let dep = self.ghi && Self::result_dep(or_node, proof, &children);
                let add_work = self.stats.total_nodes() - work0;
                if dep {
                    // Twin-escalation guard: a SECOND search-derived
                    // path-dependent (dis)proof for the same position means
                    // the search is re-deriving this position per path (the
                    // GHI path-enumeration degeneration — path-dependent
                    // results never transpose). Resolve the position EXACTLY
                    // once via the AB fallback; the fixpoint value is a
                    // path-independent fact (see module docs), so every later
                    // path reuses the solved base entry.
                    if self.twin_seen.contains(&key) {
                        self.stats.fallback_twin += 1;
                        if self.stats.fallback_twin <= 4 {
                            eprintln!(
                                "[dfpn] repeated path-dependent (dis)proof — \
                                 resolving position via AB fallback (exact)"
                            );
                        }
                        let v = self.ab_value(s);
                        let proof = Self::proves(s.turn, attacker, v);
                        let (pn, dn) = if proof { (0, INF) } else { (INF, 0) };
                        self.tt.store_base(key, pn, dn, add_work);
                        return (pn, dn, false);
                    }
                    if self.twin_seen.len() >= SIM_FAIL_CACHE_MAX {
                        self.twin_seen.clear();
                    }
                    self.twin_seen.insert(key);
                    // Path-dependent (dis)proof: twin entry tagged with this
                    // path's signature + K-M base reset to 1/1.
                    self.stats.twin_stores += 1;
                    self.tt.store_twin(key, ctx.tip_sig(), proof, add_work);
                    self.tt.reset_base(key);
                } else {
                    self.tt.store_base(key, pn, dn, add_work);
                }
                return (pn, dn, dep);
            }

            // ---- Threshold exceeded? Store unproven numbers and return. ----
            if pn >= pt || dn >= dt {
                let add_work = self.stats.total_nodes() - work0;
                self.tt.store_base(key, pn, dn, add_work);
                return (pn, dn, false);
            }

            // ---- Watchdog: a single mid call must not loop unboundedly. ----
            // Root calls (path length 1) are exempt: the root loops once per
            // top-level descent by design, and its stalls are covered by the
            // self-support and twin-escalation guards (each firing resolves a
            // child exactly, so the root's solved-children set grows
            // monotonically and the loop terminates).
            iter += 1;
            if iter > self.loop_cap && ctx.path.len() > 1 {
                self.stats.fallback_node += 1;
                if self.stats.fallback_node <= 4 {
                    eprintln!(
                        "[dfpn] expansion watchdog ({} iters) — resolving node \
                         via AB fallback (exact)",
                        self.loop_cap
                    );
                }
                let v = self.ab_value(s);
                let proof = Self::proves(s.turn, attacker, v);
                let (pn, dn) = if proof { (0, INF) } else { (INF, 0) };
                let add_work = self.stats.total_nodes() - work0;
                self.tt.store_base(key, pn, dn, add_work);
                return (pn, dn, false);
            }

            // ---- Select the best child + the 1+ε thresholds. ----
            let (bi, second) = Self::select(or_node, &children);
            let c = &children[bi];
            let (ptc, dtc) = if or_node {
                // OR: pn binds. ptc = min(pt, ceil(p2·(1+ε))) (≥ p2+1);
                // dtc = dt - dn + dn_c (the exact PN-search bound).
                (
                    pt.min(self.eps_ceil(second)),
                    Self::sub_add(dt, dn, c.dn),
                )
            } else {
                // AND: dn binds; dual formulas.
                (
                    Self::sub_add(pt, pn, c.pn),
                    dt.min(self.eps_ceil(second)),
                )
            };

            // ---- Self-support detector (TT-mediated cycle guard). ----
            let step = (bi, c.pn, c.dn, ptc, dtc);
            if last_step == Some(step) {
                stall += 1;
                if stall >= STALL_LIMIT {
                    self.stats.fallback_child += 1;
                    if self.stats.fallback_child <= 4 {
                        eprintln!(
                            "[dfpn] self-support detected ({stall} no-progress \
                             re-expansions) — resolving child via AB fallback (exact)"
                        );
                    }
                    let child_state = children[bi].state;
                    let child_key = children[bi].key;
                    let v = self.ab_value(&child_state);
                    let proof = Self::proves(child_state.turn, attacker, v);
                    let (cpn, cdn) = if proof { (0, INF) } else { (INF, 0) };
                    self.tt.store_base(child_key, cpn, cdn, 0);
                    last_step = None;
                    stall = 0;
                    continue;
                }
            } else {
                last_step = Some(step);
                stall = 0;
            }

            // ---- Recurse. ----
            let child_state = children[bi].state;
            let child_cpack = children[bi].cpack;
            ctx.push(child_cpack);
            let _ = self.mid(&child_state, ptc, dtc, ctx);
            ctx.pop();
        }
    }

    /// Generate, order (distance-advantage descending), and canonical-dedupe
    /// the children of `s`, precomputing each child's FIXED adjudication
    /// (repetition / terminal / race — constant across the parent `mid` call,
    /// because the path is invariant over the call's loop iterations).
    fn gen_children(&mut self, s: &State, ctx: &Ctx) -> Vec<Child> {
        let mover = s.turn;
        let opp = 1 - mover;
        let attacker = ctx.attacker;
        let big = 4 * (self.b.w as i64 + self.b.h as i64);
        let mut scored: Vec<(i64, Child)> = Vec::new();
        for m in movegen::legal_moves(self.b, s) {
            let child = movegen::apply(self.b, s, m);
            let cpack = self.cpack(&child);
            // Fixed adjudication, in the same precedence order the per-
            // iteration evaluation used to apply:
            //   1. Cycle-closing successor: adjudicated Draw — a disproof of
            //      "attacker forces a Win", valid ONLY on this path
            //      (path-dependent). Decided BEFORE any TT probe, which also
            //      forecloses direct self-support through an on-path entry.
            //      (Naive mode — test demo only — still adjudicates the
            //      repetition; what it omits is the path-dependence TRACKING.)
            //   2. Real terminal: the player who just moved reached goal.
            //   3. Walls exhausted: exact race value, folded in directly
            //      (path-independent in both directions; see module docs).
            let fixed = if ctx.on_path(cpack) {
                self.stats.rep_hits += 1;
                Some((INF, 0, self.ghi))
            } else if let Some(p) = self.b.winner(&child) {
                let v = if p == child.turn { Value::Win } else { Value::Loss };
                Some(if Self::proves(child.turn, attacker, v) {
                    (0, INF, false)
                } else {
                    (INF, 0, false)
                })
            } else if child.walls_left == [0, 0] {
                let (v, n) = crate::endgame::race_value(self.b, &child, &self.race_tt);
                self.stats.race_nodes += n;
                Some(if Self::proves(child.turn, attacker, v) {
                    (0, INF, false)
                } else {
                    (INF, 0, false)
                })
            } else {
                None
            };
            let d_self = self.b.dist_to_goal(&child, mover).map_or(big, |d| d as i64);
            let d_opp = self.b.dist_to_goal(&child, opp).map_or(big, |d| d as i64);
            // Df-pn+ leaf init: proving "attacker wins" is harder the farther
            // the ATTACKER is from goal; disproving is harder the farther the
            // DEFENDER is. Selection bias only — never changes a value.
            let h = if self.use_h {
                let (da, dd) = if mover == attacker {
                    (d_self, d_opp)
                } else {
                    (d_opp, d_self)
                };
                (1 + (da.max(0) as u64) / 3, 1 + (dd.max(0) as u64) / 3)
            } else {
                (1, 1)
            };
            scored.push((
                d_opp - d_self,
                Child {
                    state: child,
                    cpack,
                    key: self.key(cpack, attacker),
                    h,
                    fixed,
                    pn: 1,
                    dn: 1,
                    dep: false,
                },
            ));
        }
        scored.sort_by(|a, b| b.0.cmp(&a.0));
        let mut out: Vec<Child> = Vec::with_capacity(scored.len());
        for (_, c) in scored {
            if !out.iter().any(|o| o.cpack == c.cpack) {
                out.push(c);
            }
        }
        out
    }

    /// Node pn/dn from child values (saturating at `SAT` for unproven sums so
    /// `INF` stays reserved for true (dis)proofs). Empty child lists give the
    /// correct stuck-mover adjudication for free (`min ∅ = INF`, `Σ ∅ = 0`).
    fn combine(or_node: bool, children: &[Child]) -> (u64, u64) {
        if or_node {
            let pn = children.iter().map(|c| c.pn).min().unwrap_or(INF);
            let dn = Self::sat_sum(children.iter().map(|c| c.dn));
            // All children disproven => dn == 0 (each dn 0); else if any child
            // is unproven the sum is >= 1.
            (pn, dn)
        } else {
            let pn = Self::sat_sum(children.iter().map(|c| c.pn));
            let dn = children.iter().map(|c| c.dn).min().unwrap_or(INF);
            (pn, dn)
        }
    }

    /// Saturating sum capped at `SAT` — unless every addend is 0 (a true
    /// all-(dis)proven sum) or some addend is `INF`-irrelevant: a single `INF`
    /// addend (a (dis)proven-the-wrong-way child) pins the sum at `SAT`, never
    /// `INF`, so an unproven node can never masquerade as solved.
    fn sat_sum(it: impl Iterator<Item = u64>) -> u64 {
        let mut acc: u64 = 0;
        for v in it {
            acc = acc.saturating_add(v);
            if acc >= SAT {
                return SAT;
            }
        }
        acc
    }

    /// `ceil(x · (1+ε))`, at least `x + 1`, capped at `SAT` (the Pawlewicz–Lew
    /// child threshold on the binding number; `x` is the second-best value and
    /// may be `INF` when no competitor exists, in which case the parent's own
    /// threshold is the only cap).
    fn eps_ceil(&self, x: u64) -> u64 {
        if x >= SAT {
            return SAT;
        }
        let scaled = (x as u128 * (1024 + self.eps_num as u128)).div_ceil(1024);
        let scaled = scaled.min(SAT as u128) as u64;
        scaled.max(x.saturating_add(1)).min(SAT)
    }

    /// `bound - total + child`, the exact PN-search threshold for the
    /// non-binding number (`d0 ≥ dt ⟺ d0_child ≥ dt - d + d_child`), with
    /// saturation guards. Caller guarantees `total < bound`.
    fn sub_add(bound: u64, total: u64, child: u64) -> u64 {
        debug_assert!(total < bound);
        (bound - total).saturating_add(child).min(SAT)
    }

    /// Path-dependence of a just-derived (dis)proof, per the K-M propagation
    /// rules for the first-player-loss scenario:
    ///   * proof at OR: dep of the (a) proven child;
    ///   * proof at AND: any child's dep (all are proven; OR-fold);
    ///   * disproof at OR: any child's dep (all are disproven; OR-fold);
    ///   * disproof at AND: dep of the disproven (dn == 0) child.
    ///
    /// In our scenario proofs are always independent — see module docs — but
    /// the general rule is kept; over-tagging as dependent would only cost
    /// performance, never correctness.
    fn result_dep(or_node: bool, proof: bool, children: &[Child]) -> bool {
        if children.is_empty() {
            return false; // stuck-mover adjudication: a real, fresh terminal.
        }
        match (or_node, proof) {
            (true, true) => children.iter().find(|c| c.pn == 0).is_some_and(|c| c.dep),
            (false, false) => children.iter().find(|c| c.dn == 0).is_some_and(|c| c.dep),
            (true, false) | (false, true) => children.iter().any(|c| c.dep),
        }
    }

    /// Best-child index and the second-best binding number (OR: pn, AND: dn).
    /// Ties resolve to the earlier (better-ordered) child.
    fn select(or_node: bool, children: &[Child]) -> (usize, u64) {
        let mut bi = 0usize;
        let mut best = INF;
        let mut second = INF;
        for (i, c) in children.iter().enumerate() {
            let v = if or_node { c.pn } else { c.dn };
            if v < best {
                second = best;
                best = v;
                bi = i;
            } else if v < second {
                second = v;
            }
        }
        (bi, second)
    }

    /// Memo key of one failed simulation attempt: a mix of the TT key, the
    /// path signature it was attempted under, and the polarity attempted. A
    /// 64-bit collision merely skips one verification attempt (fail-safe).
    #[inline]
    fn sim_fail_key(key: u128, sig: u64, want_proof: bool) -> u64 {
        let mut x = (key as u64)
            ^ ((key >> 64) as u64).rotate_left(21)
            ^ sig.rotate_left(43)
            ^ (want_proof as u64);
        x = x.wrapping_mul(0x2545f4914f6cdd1d);
        x ^ (x >> 32)
    }

    /// Kawano simulation: cheaply re-verify that `s` is (dis)proven for the
    /// question UNDER THE CURRENT PATH (`ctx`, with `s` already pushed),
    /// borrowing structure from stored (dis)proofs instead of searching.
    /// Returns `true` only if the (dis)proof verifies; any missing/unproven
    /// information or budget exhaustion returns `false` (fail-safe: the node
    /// is then searched normally). Never expands the search tree: it walks
    /// terminals, repetitions, and TT entries only.
    fn simulate(&mut self, s: &State, want_proof: bool, ctx: &mut Ctx, budget: &mut u64) -> bool {
        if *budget == 0 {
            return false;
        }
        *budget -= 1;
        self.stats.sim_nodes += 1;
        let attacker = ctx.attacker;

        // Real terminal?
        if let Some(p) = self.b.winner(s) {
            let v = if p == s.turn { Value::Win } else { Value::Loss };
            return Self::proves(s.turn, attacker, v) == want_proof;
        }
        if s.walls_left == [0, 0] {
            let (v, n) = crate::endgame::race_value(self.b, s, &self.race_tt);
            self.stats.race_nodes += n;
            return Self::proves(s.turn, attacker, v) == want_proof;
        }

        let or_node = s.turn == attacker;
        let moves = movegen::legal_moves(self.b, s);
        if moves.is_empty() {
            // Stuck mover loses.
            let proves = !or_node; // defender stuck => attacker wins.
            return proves == want_proof;
        }

        // Walk the children, borrowing the stored (dis)proof structure:
        //   want_proof at OR  => SOME child verifies proven;
        //   want_proof at AND => ALL children verify proven;
        //   want disproof     => dual.
        let need_all = want_proof != or_node;
        let mut child_states: Vec<(State, u128)> = Vec::with_capacity(moves.len());
        for m in moves {
            let c = movegen::apply(self.b, s, m);
            let cp = self.cpack(&c);
            if !child_states.iter().any(|(_, p)| *p == cp) {
                child_states.push((c, cp));
            }
        }
        for (c, cp) in child_states {
            let ok = self.simulate_child(&c, cp, want_proof, ctx, budget);
            if need_all && !ok {
                return false;
            }
            if !need_all && ok {
                return true;
            }
        }
        need_all
    }

    /// Verify one child of a simulation node: repetition adjudication, then
    /// TT-claimed (dis)proofs (base directly; sig-matching twin directly;
    /// other twins by recursive descent), then terminals via the recursive
    /// call. An unproven/missing child fails the verification (fail-safe).
    fn simulate_child(
        &mut self,
        c: &State,
        cpack: u128,
        want_proof: bool,
        ctx: &mut Ctx,
        budget: &mut u64,
    ) -> bool {
        // Repetition against the (simulated) path: a Draw adjudication — it
        // verifies a DISPROOF and refutes a proof.
        if ctx.on_path(cpack) {
            return !want_proof;
        }
        let key = self.key(cpack, ctx.attacker);
        if let Some(e) = self.tt.probe_base(key)
            && e.is_solved()
        {
            // Path-independent (dis)proof: decisive either way.
            return e.is_proof() == want_proof;
        }
        let child_sig = ctx.child_sig(cpack);
        let mut twins = std::mem::take(&mut self.twin_scratch);
        self.tt.probe_twins(key, &mut twins);
        let sig_match = twins
            .iter()
            .find(|t| t.sig == child_sig)
            .map(|t| t.is_proof());
        let any_claim = twins.iter().any(|t| t.is_proof() == want_proof);
        self.twin_scratch = twins;
        if let Some(p) = sig_match {
            // A twin recorded for exactly this path: decisive.
            return p == want_proof;
        }
        if any_claim {
            // A twin from another path claims the wanted polarity: descend.
            ctx.push(cpack);
            let ok = self.simulate(c, want_proof, ctx, budget);
            ctx.pop();
            return ok;
        }
        // No claim in the TT: the child might still be a real terminal /
        // race / stuck node — let the recursive walk decide; it fails fast on
        // genuinely unproven interiors (no TT claims below either).
        if self.b.winner(c).is_some() || c.walls_left == [0, 0] {
            ctx.push(cpack);
            let ok = self.simulate(c, want_proof, ctx, budget);
            ctx.pop();
            return ok;
        }
        false
    }
}
