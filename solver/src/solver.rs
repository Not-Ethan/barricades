//! Exact full-game solver: depth-bounded negamax with alpha-beta, a
//! bound-flagged transposition table, move ordering, and a wall-less race
//! short-circuit. Returns the game-theoretic value for the side to move.
//!
//! Mirrors `smallboard/solver.py` (the reference Python solver) but specialized
//! to the three-valued `Value` lattice and the Rust engine.

use crate::board::Board;
use crate::endgame::RaceTt;
use crate::footprint::{FootprintCache, FpEntry};
use crate::state::{Move, State};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// Maximum search ply we keep killer slots for. The `solve()` depth ceiling for
/// the validation boards is well under this, and any deeper ply simply skips the
/// killer heuristic (it only affects ordering, never values).
const MAX_PLY: usize = 256;

/// Footprint pruning (Theorem 1, Win direction): minimum remaining depth at
/// which a FRESH extraction attempt is worth its cost (cached masks are used
/// at every depth — a compiled certificate is a true-value object). Cost
/// heuristic only; never affects values. Env override: `QS_FP_MIND`.
const FP_MIN_DEPTH: u32 = 6;

/// Footprint attempt gate: require the certificate owner Y to be STRICTLY
/// ahead in the raw race (`dist(Y) + margin <= dist(Z)`). Measured on 5x5-w2:
/// margin 0 admits deep tempo-race flip-proofs that cost ~2.3x total nodes;
/// margin 1 collapses the overhead to ~OFF parity while PRUNING MORE (the
/// strictly-ahead certificates are the shallow, corridor-local ones the doc's
/// near-endgame estimate is built on). Cost heuristic only; never affects
/// values. Env override: `QS_FP_MARGIN`.
const FP_DIST_MARGIN: i64 = 1;

/// Footprint attempt gate: cap on Y's goal distance (extraction concentrates
/// on near-goal funnels; `u32::MAX` disables). Cost heuristic only. Env
/// override: `QS_FP_MAXDY`.
const FP_MAX_DY: u32 = u32::MAX;

/// Footprint attempt gate: number of wall replies that must have been
/// SEARCHED at the node before a fresh extraction is attempted (cached masks
/// apply from the first wall regardless). A node that cuts off early never
/// pays an extraction; a genuine refutation fan pays after `K` walls and
/// prunes the rest. Cost heuristic only. Env override: `QS_FP_DEFER`.
const FP_DEFER_WALLS: u32 = 2;

/// Footprint pruning: a failed attempt cached at remaining depth `d` is only
/// retried when a later query is at least this much deeper (failures can be
/// depth-relative: e.g. the `flip(s)` win was not provable within `d`). Cost
/// heuristic only.
const FP_RETRY_MARGIN: u32 = 4;

/// Game-theoretic value for the side to move.
///
/// `Loss < Draw < Win` by declaration order, which the derived `Ord`/`PartialOrd`
/// rely on — keep the variants in this order.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Value {
    Loss,
    Draw,
    Win,
}

impl Value {
    /// Negamax sign flip: the value from the opponent's perspective.
    #[inline]
    pub fn negate(self) -> Value {
        match self {
            Value::Loss => Value::Win,
            Value::Draw => Value::Draw,
            Value::Win => Value::Loss,
        }
    }
}

/// Transposition-table bound flag for a stored `(value, flag)`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Flag {
    /// `value` is the exact negamax value.
    Exact,
    /// `value` is a lower bound (fail-high / beta cutoff).
    Lower,
    /// `value` is an upper bound (fail-low).
    Upper,
}

/// One dense transposition-table entry: a single CANONICAL position (one entry
/// per position — depth is NOT part of the key, it is stored in the entry and
/// used by the depth-fold reuse guard) with its proven `value`, bound `flag`,
/// and the remaining search `depth` at which that bound was established.
///
/// `key` is the FULL injective u128 pack of the canonical state (see
/// `pack_u128`). It is stored and compared in full on every probe, so a
/// hash-bucket collision can never return a foreign entry's value: a mismatched
/// key is a hard MISS (recompute), never a wrong hit.
///
/// `key == 0` is the empty-slot sentinel. To make that airtight regardless of
/// the packed layout, every stored key has a constant high "occupied" bit ORed
/// in (`OCCUPIED_BIT`, bit 127), so a live key is never 0 even for the all-zero
/// packed state (e.g. pawns at cell 0, no walls, turn 0).
#[derive(Clone, Copy)]
struct Entry {
    /// Full injective packed canonical key, with `OCCUPIED_BIT` set. 0 = empty.
    key: u128,
    value: Value,
    flag: Flag,
    /// Remaining search depth at which `value`/`flag` was proven. The depth-fold
    /// reuse rule only trusts this entry for a query depth `d <= depth`.
    depth: u16,
}

impl Entry {
    /// The empty-slot sentinel (`key == 0`).
    const EMPTY: Entry = Entry {
        key: 0,
        value: Value::Draw,
        flag: Flag::Exact,
        depth: 0,
    };
    #[inline]
    fn is_empty(&self) -> bool {
        self.key == 0
    }
}

/// A high "this slot is occupied" bit ORed into every stored key, guaranteeing a
/// live entry's `key` is never 0 (so `key == 0` is an unambiguous empty
/// sentinel) while preserving injectivity (it is a constant, so it never
/// conflates two distinct packed states). Bit 127 is far above any field
/// `pack_u128` writes (the layout uses well under 100 bits for <=7x7).
const OCCUPIED_BIT: u128 = 1u128 << 127;

/// Per-entry byte size of the dense table's element. Reported by the CLI and
/// used to derive slot capacity from the `QS_TT_MB` budget. The u128 `key`
/// forces 16-byte alignment, so `Entry` (key 16B + value/flag/depth ~4B) pads to
/// 32 bytes — the figure the `#slots = QS_TT_MB * 1MiB / TT_ENTRY_SIZE` sizing
/// divides by, making the table's heap use approximately `QS_TT_MB` MiB.
const TT_ENTRY_SIZE: usize = size_of::<Entry>();

/// Default main-TT budget in MiB when `QS_TT_MB` is unset/invalid.
const DEFAULT_TT_MB: usize = 2048;

/// Largest power of two `<= n` (with `n >= 1`); keeps the dense table's slot
/// count a power of two so the index can be a mask rather than a `%`.
#[inline]
fn prev_pow2(n: usize) -> usize {
    debug_assert!(n >= 1);
    if n.is_power_of_two() {
        n
    } else {
        1usize << (usize::BITS - 1 - n.leading_zeros()) as usize
    }
}

/// A bucket: two slots sharing one index, the standard 2-way chess-engine TT
/// layout. Slot 0 is DEPTH-PREFERRED (keep the deepest result), slot 1 is
/// ALWAYS-REPLACE (keep the most recent). Two slots per index sharply cut the
/// eviction-thrash a single slot suffers from hash collisions: a deep, valuable
/// entry in slot 0 survives a colliding shallow probe (which lands in slot 1)
/// instead of being clobbered, so the hit rate (and node count) is markedly
/// better than a 1-slot table at the same capacity.
type Bucket = [Entry; 2];

/// Dense, fixed-capacity, open-addressed transposition table.
///
/// A flat `Vec<Bucket>` of `nbuckets` 2-slot buckets (no per-entry `Option`, no
/// chaining, no `HashMap` overhead): `nbuckets` is a power of two and the bucket
/// index is `hash(key) & (nbuckets - 1)`. Capacity is FIXED at construction from
/// the `QS_TT_MB` budget and never grows.
///
/// Replacement within a bucket (after an in-place update if the key is already
/// resident in either slot):
///   * slot 0 (depth-preferred): the new entry takes it when slot 0 is empty or
///     `new.depth >= slot0.depth`; the displaced occupant falls through to slot 1;
///   * slot 1 (always-replace): unconditionally takes whatever did not win slot 0.
///
/// A DIFFERENT position may thus evict a resident one (the table is a cache; an
/// eviction only forces a recompute, never a wrong value).
///
/// EXACTNESS: this is a pure alpha-beta cache. Three independent guarantees keep
/// it sound:
///   1. The FULL u128 key is stored and verified on probe — a hash collision
///      (foreign key in the same bucket) is a MISS, never a foreign hit.
///   2. A hit is only USED when `slot.depth >= query_depth` (the depth-fold
///      reuse rule; see `Solver::ab`), which is the standard sound chess-engine
///      rule on the `Loss < Draw < Win` lattice.
///   3. Eviction / capacity are correctness-neutral: a missing or overwritten
///      entry only forces re-search; alpha-beta returns the exact value under
///      ANY subset of cached entries.
struct DenseTt {
    buckets: Vec<Bucket>,
    /// `buckets.len() - 1`. The bucket count is always a power of two, so the
    /// index is `hash & mask` — a single AND, not a 64-bit `%` (which, at two
    /// table ops per node over tens of millions of nodes, was a measurable cost).
    mask: usize,
    /// Live (occupied) slot count, for reporting only.
    fill: usize,
}

impl DenseTt {
    /// Build a fixed table whose TOTAL slot count is the largest power of two
    /// `<= min(mb-budget, max_slots)` (at least two, i.e. one bucket); the bucket
    /// count is half that. `max_slots` is a board-aware ceiling so a tiny board
    /// does not eagerly allocate (and zero) a multi-gigabyte array; pass a large
    /// value to let the MiB budget govern. Rounding DOWN to a power of two keeps
    /// the heap use within the budget while letting the index be a mask. Both
    /// ceilings are exactness-neutral (capacity only trades RAM for re-search).
    fn with_capacity(mb: usize, max_slots: usize) -> DenseTt {
        let bytes = mb.saturating_mul(1024 * 1024);
        let want_slots = (bytes / TT_ENTRY_SIZE).max(2).min(max_slots.max(2));
        // Power-of-two TOTAL slots, then halve to buckets (also a power of two).
        let nslots = prev_pow2(want_slots).max(2);
        let nbuckets = nslots / 2;
        DenseTt {
            buckets: vec![[Entry::EMPTY; 2]; nbuckets],
            mask: nbuckets - 1,
            fill: 0,
        }
    }

    /// Map a full packed key to a bucket index. Uses a fast integer mix (splitmix-
    /// style on the two halves) so the index is well-spread, then masks to the
    /// power-of-two bucket count. Correctness never depends on the hash (the probe
    /// verifies the full key), only the bucket it lands in, so any deterministic
    /// mix is fine.
    #[inline]
    fn bucket_index(&self, key: u128) -> usize {
        let lo = key as u64;
        let hi = (key >> 64) as u64;
        let mut x = lo ^ hi.rotate_left(32);
        x ^= x >> 33;
        x = x.wrapping_mul(0xff51afd7ed558ccd);
        x ^= x >> 33;
        (x as usize) & self.mask
    }

    /// Probe for `key`. Returns the stored `(value, flag, depth)` from whichever
    /// of the bucket's two slots carries the matching FULL key — a different (or
    /// empty) key in both slots is a MISS. The depth-fold reuse decision is made
    /// by the caller against `depth`.
    #[inline]
    fn probe(&self, key: u128) -> Option<(Value, Flag, u16)> {
        let b = &self.buckets[self.bucket_index(key)];
        for e in b {
            if e.key == key {
                return Some((e.value, e.flag, e.depth));
            }
        }
        None
    }

    /// Store `(value, flag, depth)` for `key` under the 2-way depth-preferred /
    /// always-replace policy (see the struct doc). `key` must already carry
    /// `OCCUPIED_BIT` (so it is never 0).
    #[inline]
    fn store(&mut self, key: u128, value: Value, flag: Flag, depth: u16) {
        let new = Entry {
            key,
            value,
            flag,
            depth,
        };
        let idx = self.bucket_index(key);
        let b = &mut self.buckets[idx];
        // 1. In-place update if the key is already resident in either slot:
        //    keep the deeper/equal-depth bound for the same position.
        for e in b.iter_mut() {
            if e.key == key {
                if depth >= e.depth {
                    *e = new;
                }
                return;
            }
        }
        // 2. Empty slot 0 -> fill it.
        if b[0].is_empty() {
            b[0] = new;
            self.fill += 1;
            return;
        }
        // 3. Empty slot 1 -> fill it.
        if b[1].is_empty() {
            b[1] = new;
            self.fill += 1;
            return;
        }
        // 4. Both slots occupied by other keys: depth-preferred eviction. If the
        //    newcomer is at least as deep as slot 0, it takes slot 0 and slot 0's
        //    occupant is demoted to the always-replace slot 1; otherwise the
        //    newcomer goes straight to slot 1. fill is unchanged (overwrite).
        if depth >= b[0].depth {
            b[1] = b[0];
            b[0] = new;
        } else {
            b[1] = new;
        }
    }

    /// Live occupied-slot count (reporting only).
    #[inline]
    fn len(&self) -> usize {
        self.fill
    }

    /// Total fixed slot capacity (`nbuckets * 2`).
    #[inline]
    fn capacity(&self) -> usize {
        self.buckets.len() * 2
    }
}

/// Default number of shards when a single thread is requested. A handful of
/// shards costs almost nothing and keeps `Mutex` granularity uniform across
/// thread counts (so the table's collision/eviction behaviour does not depend on
/// `QS_THREADS`); more threads scale it up to cut lock contention.
const MIN_SHARDS: usize = 8;

/// Thread-safe, fixed-capacity main transposition table SHARED across the
/// parallel search threads.
///
/// The dense table is split into `nshards` independent `DenseTt` segments, each
/// behind its own `Mutex`, so two threads touching different shards never
/// contend. A key routes to a shard by the HIGH bits of its hash mix and to a
/// bucket WITHIN that shard by the shard's own low-bit index — disjoint bit
/// ranges, so the two stages are independent and the whole key is still verified
/// on probe (a cross-shard or cross-bucket foreign key is a MISS, never a hit).
///
/// EXACTNESS UNDER SHARING (the load-bearing property for the parallel solver):
///   * Each stored entry is a VALID bound for its CANONICAL position regardless
///     of which thread wrote it (the bound flags and depth-fold rule are
///     position-local; see `Worker::ab`). So a probe hit written by another
///     thread is as sound as one written by this thread.
///   * Sharding/locking only changes WHICH bucket a key lands in and WHEN a
///     thread sees a peer's entry — never the value alpha-beta computes. With
///     the full `(Loss, Win)` root window, every thread's root search returns
///     the EXACT game value irrespective of the (sound) bound set it happens to
///     observe. Hence the parallel value equals the single-thread value exactly.
struct ShardedTt {
    shards: Vec<Mutex<DenseTt>>,
    /// `shards.len() - 1`; the shard count is a power of two so shard selection
    /// is a mask on the high hash bits.
    shard_mask: usize,
}

impl ShardedTt {
    /// Build a sharded table whose TOTAL capacity is the `QS_TT_MB`-equivalent
    /// `mb` budget (split evenly across `nshards` shards), each shard bounded by
    /// its share of `max_slots`. `nshards` is rounded up to a power of two and is
    /// at least `MIN_SHARDS`. Capacity is exactness-neutral (only trades RAM for
    /// re-search), so the per-shard split never changes a value.
    fn new(mb: usize, max_slots: usize, nshards: usize) -> ShardedTt {
        let nshards = nshards.max(MIN_SHARDS).next_power_of_two();
        // Split the byte budget and the slot ceiling evenly across shards (each
        // shard rounds its own slot count down to a power of two internally).
        let per_mb = (mb / nshards).max(1);
        let per_slots = (max_slots / nshards).max(2);
        let shards = (0..nshards)
            .map(|_| Mutex::new(DenseTt::with_capacity(per_mb, per_slots)))
            .collect();
        ShardedTt {
            shards,
            shard_mask: nshards - 1,
        }
    }

    /// Select a shard from the high bits of a splitmix mix of the key. Disjoint
    /// from the per-shard bucket index (which uses a different mix on the same
    /// key inside `DenseTt`), so the two stages spread independently.
    #[inline]
    fn shard_index(&self, key: u128) -> usize {
        let lo = key as u64;
        let hi = (key >> 64) as u64;
        let mut x = lo.wrapping_add(hi).wrapping_mul(0x9e3779b97f4a7c15);
        x ^= x >> 29;
        x = x.wrapping_mul(0xbf58476d1ce4e5b9);
        x ^= x >> 32;
        (x as usize) & self.shard_mask
    }

    /// Probe for `key` in its shard (briefly locking that shard only).
    #[inline]
    fn probe(&self, key: u128) -> Option<(Value, Flag, u16)> {
        let shard = &self.shards[self.shard_index(key)];
        shard.lock().expect("TT shard mutex poisoned").probe(key)
    }

    /// Store `(value, flag, depth)` for `key` in its shard (locking that shard).
    #[inline]
    fn store(&self, key: u128, value: Value, flag: Flag, depth: u16) {
        let shard = &self.shards[self.shard_index(key)];
        shard
            .lock()
            .expect("TT shard mutex poisoned")
            .store(key, value, flag, depth);
    }

    /// Live occupied-slot count summed across shards (reporting only).
    fn len(&self) -> usize {
        self.shards
            .iter()
            .map(|s| s.lock().expect("TT shard mutex poisoned").len())
            .sum()
    }

    /// Total fixed slot capacity summed across shards.
    fn capacity(&self) -> usize {
        self.shards
            .iter()
            .map(|s| s.lock().expect("TT shard mutex poisoned").capacity())
            .sum()
    }
}

/// Independent brute-force negamax — the correctness oracle. No alpha-beta, no
/// TT, no ordering; just a plain depth-bounded minimax over `Value`. Used by the
/// differential tests to pin the optimized `Solver`.
pub fn brute_value(b: &Board, s: &State, depth: u32) -> Value {
    if let Some(p) = b.winner(s) {
        return if p == s.turn { Value::Win } else { Value::Loss };
    }
    if depth == 0 {
        return Value::Draw;
    }
    let mut best = Value::Loss;
    for m in crate::movegen::legal_moves(b, s) {
        let v = brute_value(b, &crate::movegen::apply(b, s, m), depth - 1).negate();
        if v > best {
            best = v;
        }
        if best == Value::Win {
            break;
        }
    }
    best
}

/// Default thread count when `QS_THREADS` is unset/invalid: the machine's
/// available parallelism (falls back to 1 if the platform cannot report it).
fn default_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .max(1)
}

/// The optimized exact solver. Borrows a `Board` and owns a thread-safe SHARED
/// transposition table (sharded dense array) and a bounded, exact race memo,
/// both shared across the parallel search threads.
///
/// PARALLELISM (lazy SMP): `solve` runs `nthreads` worker threads, each
/// executing the SAME root alpha-beta search over the SAME shared TT and race
/// memo. With the full `(Loss, Win)` root window, every thread's search returns
/// the EXACT game-theoretic value irrespective of which (sound) bounds it
/// observes in the shared TT — so the returned parallel value is provably
/// IDENTICAL to the single-thread value on every input. The threads share
/// discovered bounds through the TT (the speedup) but never share a value:
/// alpha-beta's result is a pure function of the game tree, not of the cache
/// contents. `QS_THREADS=1` runs exactly one worker and reproduces the
/// single-thread behaviour and value.
pub struct Solver<'a> {
    b: &'a Board,
    /// Thread-safe, fixed-capacity, SHARED main transposition table (sharded
    /// dense array; see `ShardedTt`). ONE entry per canonical position
    /// (u128-packed key), remaining search depth stored IN the entry and reused
    /// under the depth-fold rule (`stored.depth >= query`). Pure alpha-beta
    /// cache: eviction/capacity only force re-search, never a wrong value, the
    /// full key is verified on probe, and every entry is a valid bound for its
    /// position regardless of which thread wrote it. Capacity from `QS_TT_MB`
    /// (default `DEFAULT_TT_MB`), shared (not per-thread) so memory stays bounded
    /// and bounds are shared.
    tt: Arc<ShardedTt>,
    /// PERSISTENT, exact, BOUNDED (config-granular LRU; `QS_RACE_MB`) race
    /// endgame memo keyed on the frozen wall config. Every entry is the
    /// position's exact game-theoretic value, so it is sound to reuse across
    /// every walls-exhausted leaf within a `solve()` call: each distinct
    /// frozen-wall race config is solved exactly once instead of per leaf. See
    /// `endgame.rs`. Thread-safe (interior `Mutex`) and SHARED across worker
    /// threads. Survives across `solve()` calls on the same `Solver` (extra
    /// reuse; values stay valid because the race value is a pure function of
    /// `State`, independent of the surrounding board's wall count).
    race_tt: Arc<RaceTt>,
    /// Precomputed horizontal-mirror permutations of the wall-anchor bit layout
    /// (`hbit`/`vbit` = `wr*(w-1)+wc`), mapping each set bit to its column-
    /// reflected anchor `wc -> w-2-wc`. `Some` only for `w >= 2`; `None` for
    /// degenerate single-column boards (no wall anchors, mirror is identity).
    mirror_perm: Option<Vec<u8>>,
    /// When false, `ordered_moves` ignores the killer/history heuristics (pure
    /// distance ordering, == baseline). Toggle for staged measurement ONLY;
    /// neither setting changes any value.
    use_ordering: bool,
    /// When false, the main TT keys on the raw state (== baseline) instead of
    /// the horizontal-mirror canonical representative. Toggle for staged
    /// measurement ONLY; both settings return identical values.
    use_symmetry: bool,
    /// THEOREM 4 (one-sided frozen-race bounds; `docs/superpowers/
    /// solver-pruning-theorems.md` §B.4): when enabled (`QS_T4=1`; default OFF — see Solver::with_tt_mb rationale
    /// disables), nodes where EXACTLY ONE side has exhausted its walls get a
    /// provably-valid Lower/Upper bound from the frozen-race value (both
    /// budgets zeroed), synthesized as a depth-infinity TT-style hit. Decisive
    /// bounds replace the whole subtree. Falsification-validated: exact-value
    /// replay at all 2,121,148 graph states across five boards, zero
    /// mismatches. Toggling is value-neutral (A/B gated by tests/theorem4.rs);
    /// OFF only forfeits the cutoffs.
    use_t4: bool,
    /// THEOREM 1 / Corollary 1, Win direction (wall-relevance footprint
    /// mustplay pruning; `docs/superpowers/solver-pruning-theorems.md` §A):
    /// when enabled (`QS_FOOTPRINT=1`; default OFF), a Z-to-move node whose
    /// turn-flipped twin `flip(s)` is a PROVEN true Win for the opponent Y
    /// gets a compiled certificate footprint (two wall-anchor masks); every Z
    /// wall outside the masks has EXACT child value Loss-for-Z (Theorem 1
    /// executes the certificate verbatim through the insertion), so the
    /// refutation loop skips it. Win-direction prunes are exact child values
    /// — skipping never changes the max (§A.2 integration rule), so toggling
    /// is value-neutral (A/B gated by tests/footprint.rs). Extraction uses
    /// the §A.5 AMENDED rank scheme: build-then-verify (Kahn on the closure
    /// graph) with K-boost repair and abort = sound no-prune fallback; race
    /// regions resolve through the engine's exact race short-circuit. The
    /// unsound TT-hot dtw rank is NOT used.
    use_footprint: bool,
    /// Shared per-position footprint cache (canonical keys, mirrored masks
    /// per the doc's G5 discipline). Pure cache of true-certificate masks.
    fp_cache: Arc<FootprintCache>,
    /// Footprint attempt-gate parameters (cost heuristics only; defaults from
    /// the FP_* constants, env-overridable for ladder experiments).
    fp_min_depth: u32,
    fp_margin: i64,
    fp_max_dy: u32,
    fp_defer: u32,
    /// Number of worker threads for `solve` (from `QS_THREADS`, default
    /// `default_threads()`). 1 reproduces single-thread behaviour/value.
    nthreads: usize,
    /// Profiling counter: total internal nodes visited, SUMMED across all worker
    /// threads of the last `solve`. Counts every `ab(...)` entry (main
    /// alpha-beta search) plus every wall-less race node entered via
    /// `race_value`. Instrumentation only — does not affect search results.
    /// (Parallel node counts vary run-to-run; only the VALUE is deterministic.)
    pub nodes: u64,
    /// Profiling counter: Theorem-4 evaluations — one-sided-exhaustion nodes
    /// where the frozen-race bound was computed — summed across the worker
    /// threads of the last `solve`. Instrumentation only.
    pub t4_fires: u64,
    /// Profiling counter: Theorem-4 cutoffs — fires whose bound resolved the
    /// node with NO move loop (decisive Win-Lower / Loss-Upper, or a Draw bound
    /// closing the window) — summed across workers. Each one replaced an entire
    /// subtree with a leaf. Instrumentation only.
    pub t4_cutoffs: u64,
    /// Profiling counter: footprint extraction ATTEMPTS (gates passed, fresh
    /// certificate build started) — summed across workers. Instrumentation only.
    pub fp_attempts: u64,
    /// Profiling counter: successful footprint extractions (a verified
    /// certificate compiled to masks). Instrumentation only.
    pub fp_extractions: u64,
    /// Profiling counter: wall moves SKIPPED by footprint mustplay pruning
    /// (each skip is an exact Loss-for-Z child the search never expanded).
    /// Instrumentation only.
    pub fp_prunes: u64,
    /// Profiling counter: sum of footprint mask popcounts over successful
    /// extractions (average footprint size = `fp_mask_bits / fp_extractions`).
    /// Instrumentation only.
    pub fp_mask_bits: u64,
}

/// One worker thread's search context for the lazy-SMP solve. Holds shared
/// references to the SHARED TT and race memo, plus PER-THREAD ordering state
/// (killers/history) and a node counter. Per-thread ordering is standard for
/// lazy SMP: it is exactness-neutral (ordering never changes a value) and lets
/// threads diversify their traversal so they explore the tree differently and
/// usefully fill the shared TT. The shared abort flag lets the first thread to
/// finish the root search signal the rest to stop early (their partial results
/// are discarded — only a thread that COMPLETES the root search contributes a
/// value, and a completed full-window search is always the exact value).
struct Worker<'a> {
    b: &'a Board,
    tt: &'a ShardedTt,
    race_tt: &'a RaceTt,
    /// Killer moves: up to two moves per search ply that previously caused a
    /// beta cutoff. ORDERING ONLY — never changes the legal-move set or value.
    killers: Vec<[Option<Move>; 2]>,
    /// History heuristic: per-move cumulative beta-cutoff count, indexed by the
    /// dense `move_index`. Biases ordering globally. ORDERING ONLY.
    history: Vec<u32>,
    mirror_perm: Option<&'a [u8]>,
    use_ordering: bool,
    use_symmetry: bool,
    /// Theorem-4 one-sided frozen-race bounds enabled (see `Solver::use_t4`).
    use_t4: bool,
    /// Footprint mustplay pruning enabled (see `Solver::use_footprint`).
    use_footprint: bool,
    /// Shared footprint mask cache (see `Solver::fp_cache`).
    fp_cache: &'a FootprintCache,
    /// True while THIS worker is inside a footprint extraction's certificate
    /// solves. Nested extraction ATTEMPTS are suppressed (pure cost control —
    /// cached masks are still USED; skipping a prune is always sound).
    fp_in_extraction: bool,
    /// Attempt-gate parameters (see the FP_* constants; resolved once at
    /// construction from the env overrides). Cost heuristics only.
    fp_min_depth: u32,
    fp_margin: i64,
    fp_max_dy: u32,
    fp_defer: u32,
    /// This worker's diversity index (0..nthreads). Perturbs ordering tie-breaks
    /// so different threads explore the tree in different orders (lazy-SMP
    /// diversification). ORDERING ONLY — never changes a value.
    tid: usize,
    /// Shared early-abort flag (see struct doc). Checked at the top of `ab`; a
    /// worker that sees it set returns an arbitrary placeholder that the caller
    /// DISCARDS (only completed searches contribute a value).
    stop: &'a AtomicBool,
    /// This worker's node count (summed into `Solver::nodes` after the join).
    nodes: u64,
    /// Shared live node counter for the progress heartbeat (observability
    /// only). Workers publish their local `nodes` in coarse batches (every
    /// ~1M nodes) so the monitor can report real-time throughput. Never read
    /// by the search itself — exactness-neutral by construction.
    live_nodes: &'a AtomicU64,
    /// How much of `nodes` has already been published to `live_nodes`.
    flushed_nodes: u64,
    /// Theorem-4 fire count for this worker (summed into `Solver::t4_fires`).
    t4_fires: u64,
    /// Theorem-4 cutoff count for this worker (summed into `Solver::t4_cutoffs`).
    t4_cutoffs: u64,
    /// Footprint counters for this worker (summed into the `Solver` fields).
    fp_attempts: u64,
    fp_extractions: u64,
    fp_prunes: u64,
    fp_mask_bits: u64,
    /// Set true if this worker hit the abort flag and returned early, so the
    /// caller knows its result is NOT a completed value and must be ignored.
    aborted: bool,
}

/// Dense move index for the killer/history tables. Steps map to their
/// destination cell index `0..64`; walls map to `64 + (horiz?0:64) + anchorbit`,
/// where `anchorbit = wr*(w-1)+wc < 64`. Range `< 192`; ordering-only, so a
/// collision (impossible here) could at worst affect speed, never values.
#[inline]
fn move_index(b: &Board, m: Move) -> usize {
    match m {
        Move::Step(d) => d as usize,
        Move::Wall { wc, wr, horiz } => {
            let bit = (wr * (b.w - 1) + wc) as usize;
            64 + if horiz { 0 } else { 64 } + bit
        }
    }
}
const MOVE_INDEX_SPAN: usize = 192;

/// Public horizontal mirror of a state for tests/tooling: pawn columns
/// `c -> w-1-c`, wall anchors column-reflected `wc -> w-2-wc`, rows/orientation/
/// walls_left/turn unchanged. A value-preserving board automorphism with the
/// SAME side to move; `mirror(mirror(s)) == s`. Builds the permutation on the
/// fly (cheap), so callers need no `Solver`.
pub fn mirror(b: &Board, s: &State) -> State {
    let perm = build_mirror_perm(b);
    mirror_state(b, perm.as_deref(), s)
}

/// Build the horizontal-mirror permutation of the wall-anchor bit layout
/// (shared with the df-pn engine, which canonicalizes its TT keys the same way).
///
/// The anchor grid is `(w-1)` columns by `(h-1)` rows; anchor `(wc, wr)` lives
/// at bit `wr*(w-1)+wc` in BOTH `h_walls` and `v_walls`. A horizontal board
/// reflection maps column `wc -> w-2-wc` (the mirror of an index in `0..w-1`),
/// leaving the row unchanged. The returned `perm[src_bit] = dst_bit` is applied
/// identically to the horizontal and vertical anchor bitsets (a horizontal wall
/// reflects to a horizontal wall, a vertical to a vertical — orientation is
/// preserved under a left-right flip). Returns `None` when `w < 2` (no anchor
/// columns exist; the mirror is the identity and there is nothing to permute).
pub(crate) fn build_mirror_perm(b: &Board) -> Option<Vec<u8>> {
    if b.w < 2 {
        return None;
    }
    let aw = (b.w - 1) as usize; // anchor columns
    let ah = (b.h - 1) as usize; // anchor rows
    let nbits = aw * ah;
    let mut perm = vec![0u8; nbits.max(1)];
    for wr in 0..ah {
        for wc in 0..aw {
            let src = wr * aw + wc;
            let mwc = aw - 1 - wc; // (w-1)-1-wc = w-2-wc
            let dst = wr * aw + mwc;
            perm[src] = dst as u8;
        }
    }
    Some(perm)
}

/// Apply a bit permutation: for each set bit `i` of `bits`, set bit `perm[i]`.
#[inline]
fn permute_bits(bits: u64, perm: &[u8]) -> u64 {
    let mut rem = bits;
    let mut out = 0u64;
    while rem != 0 {
        let i = rem.trailing_zeros() as usize;
        rem &= rem - 1;
        out |= 1u64 << perm[i];
    }
    out
}

/// Horizontal mirror of a state: pawn columns `c -> w-1-c` (idx recomputed via
/// `cr`/`idx`), wall anchors column-reflected via `perm`, rows/orientation/
/// walls_left/turn unchanged. This is a graph automorphism of Quoridor — column
/// reflection commutes with `legal_steps` (incl. jumps), `legal_walls`,
/// `winner`, and `apply` — so `value(s) == value(mirror(s))` with the SAME side
/// to move. Used only to pick a canonical TT key; never flips the value.
fn mirror_state(b: &Board, perm: Option<&[u8]>, s: &State) -> State {
    let mut t = *s;
    for p in 0..2 {
        let (c, r) = b.cr(s.pawn[p]);
        t.pawn[p] = b.idx(b.w - 1 - c, r);
    }
    if let Some(perm) = perm {
        t.h_walls = permute_bits(s.h_walls, perm);
        t.v_walls = permute_bits(s.v_walls, perm);
    }
    t
}

/// Pack a `State` into a `(u64, u64, u64)` total-order key that is cheap to
/// compare. The ordering is arbitrary but TOTAL and DETERMINISTIC, which is all
/// the canonicalization needs (it only must pick the SAME representative for `s`
/// and `mirror(s)`). The two wall bitsets are packed in full and the small
/// fields (both pawns, both walls-left counts, turn — each a `u8`) are folded
/// losslessly into the third word, so the key is injective on `State`.
#[inline]
fn pack_key(s: &State) -> (u64, u64, u64) {
    // Two 64-bit wall words plus a folded small-field word. The triple is
    // compared lexicographically by the derived tuple `Ord`.
    let small = (s.pawn[0] as u64)
        | ((s.pawn[1] as u64) << 8)
        | ((s.walls_left[0] as u64) << 16)
        | ((s.walls_left[1] as u64) << 24)
        | ((s.turn as u64) << 32);
    (s.h_walls, s.v_walls, small)
}

/// Losslessly pack a `State` into a single `u128` transposition-table key,
/// INJECTIVELY for every board of size `<= 7x7` (the solver's domain; `idx < 64`
/// per `Board`, and anchors `(w-1)*(h-1) <= 36`). The layout, low-to-high:
///
///   bits  0..6    pawn[0]  cell index (0..48 for 7x7; 6 bits hold 0..63)
///   bits  6..12   pawn[1]  cell index (6 bits)
///   bits 12..16   walls_left[0]  (4 bits; validation boards use <= 10 walls)
///   bits 16..20   walls_left[1]  (4 bits)
///   bit  20       turn  (1 bit)
///   bits 24..60   h_walls  anchor bitset (36 bits: covers 7x7's 36 anchors)
///   bits 64..100  v_walls  anchor bitset (36 bits)
///
/// Every field occupies a disjoint, fixed bit-range wide enough for the whole
/// `<= 7x7` domain, so two distinct states never collide: the pack is a bijection
/// onto its image. (The wall bitsets only ever set their low `(w-1)*(h-1) <= 36`
/// bits, so 36-bit fields are exact and lossless.) The result uses well under
/// 100 bits, leaving the top bits — including `OCCUPIED_BIT` (bit 127) — free.
///
/// INJECTIVITY is the load-bearing property: the dense TT stores this full key
/// and rejects any slot whose key differs, so injectivity guarantees a probe hit
/// is the SAME canonical position, never an aliased one. `walls_left` is bounded
/// by the 4-bit fields here; the solver's boards stay well within that (<= 10),
/// and a `debug_assert` guards it.
#[inline]
pub(crate) fn pack_u128(s: &State) -> u128 {
    debug_assert!(s.walls_left[0] < 16 && s.walls_left[1] < 16, "walls_left field is 4 bits");
    debug_assert!(s.turn < 2);
    debug_assert!(s.h_walls < (1u64 << 36) && s.v_walls < (1u64 << 36), "anchor bitset exceeds 36 bits");
    (s.pawn[0] as u128)
        | ((s.pawn[1] as u128) << 6)
        | ((s.walls_left[0] as u128) << 12)
        | ((s.walls_left[1] as u128) << 16)
        | ((s.turn as u128) << 20)
        | ((s.h_walls as u128) << 24)
        | ((s.v_walls as u128) << 64)
}

/// The canonical (orientation-folding) representative of `s`: the
/// lexicographically-smaller of `s` and its horizontal mirror, by `pack_key`.
/// Because mirror is value-preserving with the same side to move, `s` and its
/// mirror share a game value, so keying the TT on the representative is exact.
#[inline]
pub(crate) fn canonical(b: &Board, perm: Option<&[u8]>, s: &State) -> State {
    let m = mirror_state(b, perm, s);
    if pack_key(&m) < pack_key(s) {
        m
    } else {
        *s
    }
}

/// Cheap, saturating UPPER bound on the number of distinct CANONICAL positions
/// this board admits — the most slots a fixed TT could ever usefully hold (depth
/// is no longer in the key, so this is far smaller than the old per-depth count).
/// Used to cap allocation so a tiny board does not eagerly allocate (and zero) a
/// multi-gigabyte array. Saturates to `usize::MAX` on any overflow (large
/// boards), so big boards keep the full `QS_TT_MB` budget. EXACTNESS-NEUTRAL: a
/// larger table only avoids re-search; an over-tight one only forces it.
///
/// Bound = pawn_pairs * walls_left_combos * wall_layouts * 2 turns. The board
/// can hold at most `placed = 2*walls` walls total, each in one of `slots =
/// 2*anchors` orientation-positions, so `wall_layouts <= sum_{i=0..=placed}
/// C(slots, i)` (choose which positions are filled, ignoring the H/V-share-an-
/// anchor legality — a loose but valid over-count). This stays TIGHT for the
/// low-wall boards the tests hammer (so they do not eagerly allocate gigabytes)
/// while saturating to `usize::MAX` for high-wall / large boards, where the
/// `QS_TT_MB` budget then governs. EXACTNESS-NEUTRAL either way.
fn board_key_ceiling(b: &Board) -> usize {
    let cells = (b.w as usize) * (b.h as usize);
    let pawn_pairs = cells.saturating_mul(cells);
    let wl = (b.walls as usize) + 1;
    let wl_combos = wl.saturating_mul(wl);
    let anchors = (b.w as usize).saturating_sub(1) * (b.h as usize).saturating_sub(1);
    let slots = anchors.saturating_mul(2);
    let placed = (b.walls as usize).saturating_mul(2).min(slots);
    // sum_{i=0..=placed} C(slots, i), saturating.
    let mut wall_layouts: usize = 0;
    let mut binom: usize = 1; // C(slots, 0)
    for i in 0..=placed {
        wall_layouts = wall_layouts.saturating_add(binom);
        // C(slots, i+1) = C(slots, i) * (slots - i) / (i + 1).
        binom = binom
            .saturating_mul(slots.saturating_sub(i))
            .saturating_div(i + 1);
    }
    pawn_pairs
        .saturating_mul(wl_combos)
        .saturating_mul(wall_layouts)
        .saturating_mul(2)
}

impl<'a> Solver<'a> {
    /// Build a solver whose main-TT capacity is read from `QS_TT_MB` (megabytes;
    /// default `DEFAULT_TT_MB` when unset/unparseable/zero), race-memo cap from
    /// `QS_RACE_MB`, and thread count from `QS_THREADS` (default
    /// `default_threads()`, ~num_cpus). The race memo is bounded (config-granular
    /// LRU) and exact; capping never changes a value.
    pub fn new(b: &'a Board) -> Solver<'a> {
        let mb = std::env::var("QS_TT_MB")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .filter(|&m| m > 0)
            .unwrap_or(DEFAULT_TT_MB);
        Solver::with_tt_mb(b, mb)
    }

    /// Read `QS_THREADS` (>=1) or fall back to `default_threads()`.
    fn resolve_threads() -> usize {
        std::env::var("QS_THREADS")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .filter(|&n| n > 0)
            .unwrap_or_else(default_threads)
    }

    /// Build a solver with an explicit main-TT cap in MiB. Test hook for the
    /// tiny-cap eviction-stress gate, and the single place `Solver::new` funnels
    /// through after resolving `QS_TT_MB`. Sized to `min(tt_mb, board ceiling)`
    /// so small boards never allocate a multi-GiB array (see `board_key_ceiling`);
    /// both ceilings are exactness-neutral. Threads from `QS_THREADS`; the shard
    /// count scales with the thread count to cut lock contention.
    pub fn with_tt_mb(b: &'a Board, tt_mb: usize) -> Solver<'a> {
        let nthreads = Self::resolve_threads();
        // Scale shards above the thread count (4x) so threads rarely collide on
        // the same shard mutex. Exactness-neutral (only changes which bucket a
        // key lands in; the full key is verified on probe).
        let nshards = (nthreads.saturating_mul(4)).max(MIN_SHARDS);
        let tt = ShardedTt::new(tt_mb.max(1), board_key_ceiling(b), nshards);
        // Theorem-4 toggle: default OFF; `QS_T4=1` enables. Value-neutral A/B
        // knob (both settings return identical values — verified). Default OFF
        // because the measured ladder benchmark (8 threads, 6x5-w2/5x5-w3)
        // showed ON costs 1.1-2.4x MORE nodes: per-fire frozen-race solves and
        // race-cache pressure outweigh the cutoffs on these rungs. Re-evaluate
        // at higher rungs / different memory regimes.
        let use_t4 = std::env::var("QS_T4").map(|v| v.trim() == "1").unwrap_or(false);
        // Footprint-pruning toggle: default OFF; `QS_FOOTPRINT=1` enables.
        // Value-neutral A/B knob (verified). Default OFF: extraction overhead
        // plus the depth-folded TT already collapsing most maskable subtrees
        // made ON a net slowdown on every like-for-like rung measured.
        let use_footprint = std::env::var("QS_FOOTPRINT")
            .map(|v| v.trim() == "1")
            .unwrap_or(false);
        let env_u32 = |k: &str, d: u32| -> u32 {
            std::env::var(k).ok().and_then(|v| v.trim().parse().ok()).unwrap_or(d)
        };
        let fp_min_depth = env_u32("QS_FP_MIND", FP_MIN_DEPTH);
        let fp_max_dy = env_u32("QS_FP_MAXDY", FP_MAX_DY);
        let fp_defer = env_u32("QS_FP_DEFER", FP_DEFER_WALLS);
        let fp_margin = std::env::var("QS_FP_MARGIN")
            .ok()
            .and_then(|v| v.trim().parse::<i64>().ok())
            .unwrap_or(FP_DIST_MARGIN);
        Solver {
            b,
            tt: Arc::new(tt),
            race_tt: Arc::new(RaceTt::default()),
            mirror_perm: build_mirror_perm(b),
            use_ordering: true,
            use_symmetry: true,
            use_t4,
            use_footprint,
            fp_cache: Arc::new(FootprintCache::default()),
            fp_min_depth,
            fp_margin,
            fp_max_dy,
            fp_defer,
            nthreads,
            nodes: 0,
            t4_fires: 0,
            t4_cutoffs: 0,
            fp_attempts: 0,
            fp_extractions: 0,
            fp_prunes: 0,
            fp_mask_bits: 0,
        }
    }

    /// The dense table's per-entry byte size (for CLI reporting).
    pub fn tt_entry_size() -> usize {
        TT_ENTRY_SIZE
    }

    /// Number of worker threads `solve` will use (from `QS_THREADS`).
    pub fn threads(&self) -> usize {
        self.nthreads
    }

    /// Override the worker-thread count (test hook; CLI uses `QS_THREADS`).
    /// `n.max(1)`; 1 reproduces single-thread behaviour and value.
    pub fn set_threads(&mut self, n: usize) {
        self.nthreads = n.max(1);
    }

    /// Replace the race memo with a freshly built one capped at `mb` MiB (test
    /// hook for the race-cap-neutrality gate; the CLI/`new` path uses
    /// `QS_RACE_MB`). EXACTNESS-NEUTRAL: the race memo is a pure cache, so any
    /// cap yields identical values — only the re-solve work differs. Discards any
    /// previously accumulated race entries.
    pub fn set_race_cap_mb(&mut self, mb: usize) {
        self.race_tt = Arc::new(RaceTt::with_cap_mb(mb));
    }

    /// Enable/disable the killer+history move ordering (default on). For staged
    /// measurement only — does not affect values.
    pub fn set_use_ordering(&mut self, on: bool) {
        self.use_ordering = on;
    }

    /// Enable/disable horizontal-mirror TT canonicalization (default on). For
    /// staged measurement only — does not affect values.
    pub fn set_use_symmetry(&mut self, on: bool) {
        self.use_symmetry = on;
    }

    /// Enable/disable the Theorem-4 one-sided frozen-race bounds (default on;
    /// also settable via `QS_T4=0`). VALUE-NEUTRAL A/B toggle: the bounds are
    /// proven uniformly in depth (doc §B.4), so ON and OFF return identical
    /// values on every input — OFF only forfeits the whole-subtree cutoffs.
    pub fn set_use_t4(&mut self, on: bool) {
        self.use_t4 = on;
    }

    /// Enable/disable the Theorem-1 Win-direction footprint mustplay pruning
    /// (default on; also settable via `QS_FOOTPRINT=0`). VALUE-NEUTRAL A/B
    /// toggle: pruned walls have EXACT child value Loss-for-the-mover by
    /// Theorem 1 + Corollary 1, so skipping them never changes the negamax
    /// max — ON and OFF return identical values on every input (gated by
    /// tests/footprint.rs); OFF only forfeits the wall-branching cut.
    pub fn set_use_footprint(&mut self, on: bool) {
        self.use_footprint = on;
    }

    /// Number of positions in the shared footprint cache (reporting only).
    pub fn footprint_cache_len(&self) -> usize {
        self.fp_cache.len()
    }

    /// Override the footprint attempt-gate parameters (test/tooling hook; the
    /// CLI path uses the `QS_FP_*` env overrides). COST HEURISTICS ONLY —
    /// gates decide where extraction is attempted, never what may be pruned,
    /// so no setting can change a value.
    pub fn set_footprint_gates(&mut self, min_depth: u32, margin: i64, max_dy: u32, defer: u32) {
        self.fp_min_depth = min_depth;
        self.fp_margin = margin;
        self.fp_max_dy = max_dy;
        self.fp_defer = defer;
    }

    /// Number of entries currently in the (bounded) race endgame memo.
    pub fn race_tt_len(&self) -> usize {
        self.race_tt.len()
    }

    /// Number of resident frozen-wall configs in the race memo (reporting).
    pub fn race_config_count(&self) -> usize {
        self.race_tt.config_count()
    }

    /// Number of occupied slots currently in the transposition table.
    pub fn tt_len(&self) -> usize {
        self.tt.len()
    }

    /// Fixed total slot capacity of the dense transposition table.
    pub fn tt_capacity(&self) -> usize {
        self.tt.capacity()
    }

    /// Heap footprint of the dense transposition table in bytes. Unlike the old
    /// estimate this is EXACT for the dominant allocation: the table is a flat
    /// `Vec<Entry>` array of fixed `capacity` slots, fully resident regardless of
    /// fill.
    pub fn tt_bytes(&self) -> usize {
        self.tt.capacity() * TT_ENTRY_SIZE
    }

    /// Solve `s` to its game-theoretic value for the side to move.
    ///
    /// LAZY SMP: spawns `nthreads` scoped worker threads, each running the SAME
    /// root alpha-beta pass at a generous fixed depth bound `ceiling = 4*(w+h) +
    /// 2*walls + 8`, over the SAME shared TT and race memo. A forced Win/Loss
    /// proven within the bound is final (full `(Loss, Win)` window never
    /// mis-proves a forced result); only `Draw` is depth-limited.
    ///
    /// EXACTNESS OF THE PARALLEL VALUE: each thread searches the root with the
    /// full window, so each returns the EXACT game value regardless of which
    /// (always-sound) bounds it observed in the shared TT — the TT only prunes
    /// nodes, never alters the value alpha-beta computes. The first thread to
    /// COMPLETE the root search sets the shared `stop` flag; the others abort
    /// early and their partial results are DISCARDED. The reported value is the
    /// completed thread's, identical to a single-thread solve on every input.
    /// `nthreads == 1` runs exactly one worker (no abort ever fires) and
    /// reproduces the single-thread behaviour and value bit-for-bit.
    pub fn solve(&mut self, s: &State) -> Value {
        let w = self.b.w as u32;
        let h = self.b.h as u32;
        let walls = self.b.walls as u32;
        let ceiling = 4 * (w + h) + 2 * walls + 8;

        let nthreads = self.nthreads.max(1);
        let stop = AtomicBool::new(false);
        // Each completing worker publishes (value, total_nodes); aborted workers
        // publish nothing. Guarded by a mutex (written at most `nthreads` times).
        let results: Mutex<Vec<(Value, u64)>> = Mutex::new(Vec::new());
        let total_nodes = AtomicU64::new(0);
        // Observability: live node counter (batch-published by workers) + the
        // heartbeat interval. QS_PROGRESS_SECS=0 disables; default 30s. The
        // monitor prints to stderr so redirected run logs show live progress.
        let live_nodes = AtomicU64::new(0);
        let progress_secs: u64 = std::env::var("QS_PROGRESS_SECS")
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(30);
        let total_t4_fires = AtomicU64::new(0);
        let total_t4_cutoffs = AtomicU64::new(0);
        let total_fp_attempts = AtomicU64::new(0);
        let total_fp_extractions = AtomicU64::new(0);
        let total_fp_prunes = AtomicU64::new(0);
        let total_fp_mask_bits = AtomicU64::new(0);

        let tt = self.tt.as_ref();
        let race_tt = self.race_tt.as_ref();
        let fp_cache = self.fp_cache.as_ref();
        let mirror_perm = self.mirror_perm.as_deref();
        let use_ordering = self.use_ordering;
        let use_symmetry = self.use_symmetry;
        let use_t4 = self.use_t4;
        let use_footprint = self.use_footprint;
        let fp_min_depth = self.fp_min_depth;
        let fp_margin = self.fp_margin;
        let fp_max_dy = self.fp_max_dy;
        let fp_defer = self.fp_defer;
        let b = self.b;

        // Heartbeat monitor handle slot — declared OUTSIDE the scope so worker
        // borrows of it satisfy the scoped-thread lifetime.
        let mut mon_slot: Option<std::thread::Thread> = None;
        std::thread::scope(|scope| {
            // Heartbeat monitor (observability only). Exits when `stop` is set;
            // completing workers `unpark()` it so micro-solves (tests run
            // thousands) pay ~zero latency instead of a sleep quantum.
            mon_slot = if progress_secs > 0 {
                let stop = &stop;
                let live_nodes = &live_nodes;
                let tt_hb = tt;
                let race_hb = race_tt;
                let h = scope.spawn(move || {
                    let t0 = std::time::Instant::now();
                    let mut last_nodes = 0u64;
                    let mut last_t = 0u64;
                    while !stop.load(Ordering::Relaxed) {
                        std::thread::park_timeout(std::time::Duration::from_millis(250));
                        if stop.load(Ordering::Relaxed) {
                            break;
                        }
                        let el = t0.elapsed().as_secs();
                        if el < last_t + progress_secs {
                            continue;
                        }
                        let n = live_nodes.load(Ordering::Relaxed);
                        let rate = (n - last_nodes) / (el - last_t).max(1);
                        let (ttl, ttc) = (tt_hb.len(), tt_hb.capacity());
                        eprintln!(
                            "[hb] t={el}s nodes={n} ({rate}/s) tt={ttl}/{ttc} ({:.1}%) race_entries={}",
                            100.0 * ttl as f64 / ttc.max(1) as f64,
                            race_hb.len()
                        );
                        last_nodes = n;
                        last_t = el;
                    }
                });
                Some(h.thread().clone())
            } else {
                None
            };
            let mon_thread = &mon_slot;
            for tid in 0..nthreads {
                let stop = &stop;
                let results = &results;
                let total_nodes = &total_nodes;
                let live_nodes = &live_nodes;
                let total_t4_fires = &total_t4_fires;
                let total_t4_cutoffs = &total_t4_cutoffs;
                let total_fp_attempts = &total_fp_attempts;
                let total_fp_extractions = &total_fp_extractions;
                let total_fp_prunes = &total_fp_prunes;
                let total_fp_mask_bits = &total_fp_mask_bits;
                scope.spawn(move || {
                    let mut worker = Worker {
                        b,
                        tt,
                        race_tt,
                        killers: vec![[None, None]; MAX_PLY],
                        history: vec![0u32; MOVE_INDEX_SPAN],
                        mirror_perm,
                        use_ordering,
                        use_symmetry,
                        use_t4,
                        use_footprint,
                        fp_cache,
                        fp_in_extraction: false,
                        fp_min_depth,
                        fp_margin,
                        fp_max_dy,
                        fp_defer,
                        tid,
                        stop,
                        nodes: 0,
                        live_nodes,
                        flushed_nodes: 0,
                        t4_fires: 0,
                        t4_cutoffs: 0,
                        fp_attempts: 0,
                        fp_extractions: 0,
                        fp_prunes: 0,
                        fp_mask_bits: 0,
                        aborted: false,
                    };
                    let v = worker.ab(s, ceiling, 0, Value::Loss, Value::Win);
                    // Publish the unflushed tail so the heartbeat's final
                    // reading matches the true total (observability only).
                    live_nodes.fetch_add(worker.nodes - worker.flushed_nodes, Ordering::Relaxed);
                    total_nodes.fetch_add(worker.nodes, Ordering::Relaxed);
                    total_t4_fires.fetch_add(worker.t4_fires, Ordering::Relaxed);
                    total_t4_cutoffs.fetch_add(worker.t4_cutoffs, Ordering::Relaxed);
                    total_fp_attempts.fetch_add(worker.fp_attempts, Ordering::Relaxed);
                    total_fp_extractions.fetch_add(worker.fp_extractions, Ordering::Relaxed);
                    total_fp_prunes.fetch_add(worker.fp_prunes, Ordering::Relaxed);
                    total_fp_mask_bits.fetch_add(worker.fp_mask_bits, Ordering::Relaxed);
                    if !worker.aborted {
                        // A completed full-window root search: its value is the
                        // exact game value. Signal peers to stop and record it.
                        stop.store(true, Ordering::Relaxed);
                        // Wake the heartbeat monitor so it exits immediately
                        // (micro-solves must not pay its park quantum).
                        if let Some(t) = mon_thread {
                            t.unpark();
                        }
                        results.lock().expect("results mutex poisoned").push((v, worker.nodes));
                    }
                });
            }
        });

        self.nodes = total_nodes.load(Ordering::Relaxed);
        self.t4_fires = total_t4_fires.load(Ordering::Relaxed);
        self.t4_cutoffs = total_t4_cutoffs.load(Ordering::Relaxed);
        self.fp_attempts = total_fp_attempts.load(Ordering::Relaxed);
        self.fp_extractions = total_fp_extractions.load(Ordering::Relaxed);
        self.fp_prunes = total_fp_prunes.load(Ordering::Relaxed);
        self.fp_mask_bits = total_fp_mask_bits.load(Ordering::Relaxed);

        let results = results.into_inner().expect("results mutex poisoned");
        // At least one worker (with 1 thread, exactly one) completes: the FIRST
        // thread to finish the root search never saw `stop` set (nobody set it
        // earlier), so it does not abort and records its value. All completed
        // values are EXACT full-window minimax values and therefore IDENTICAL —
        // any one is the answer. (Debug-assert they all agree as a tripwire.)
        debug_assert!(
            results.iter().all(|&(v, _)| v == results[0].0),
            "parallel workers disagreed on the exact value: {:?}",
            results.iter().map(|&(v, _)| v).collect::<Vec<_>>()
        );
        let (value, _) = results.first().copied().expect(
            "at least one worker must complete the root search (the first thread to \
             finish never observes `stop` set, so it never aborts)",
        );
        value
    }

    /// TEST/TOOLING HOOK: run the Theorem-1 Win-direction footprint extraction
    /// directly on `u0` (the certificate owner to move), bypassing the
    /// in-search gates and the mask cache. Returns the forbidden (H, V)
    /// wall-anchor masks in `u0`'s REAL orientation — a wall anchored outside
    /// both masks cannot change `u0`'s Win — or `None` when no verified
    /// certificate could be built at the solve ceiling (the sound no-prune
    /// outcome). Certificate child solves run on a fresh single worker over
    /// this solver's shared TT/race memo (footprint pruning disabled inside,
    /// so the hook's behaviour is plain alpha-beta).
    pub fn extract_footprint(&mut self, u0: &State) -> Option<(u64, u64)> {
        let w = self.b.w as u32;
        let h = self.b.h as u32;
        let walls = self.b.walls as u32;
        let ceiling = 4 * (w + h) + 2 * walls + 8;
        let stop = AtomicBool::new(false);
        let live_nodes = AtomicU64::new(0); // hook runs unmonitored; sink only
        let mut worker = Worker {
            b: self.b,
            tt: self.tt.as_ref(),
            race_tt: self.race_tt.as_ref(),
            killers: vec![[None, None]; MAX_PLY],
            history: vec![0u32; MOVE_INDEX_SPAN],
            mirror_perm: self.mirror_perm.as_deref(),
            use_ordering: self.use_ordering,
            use_symmetry: self.use_symmetry,
            use_t4: self.use_t4,
            use_footprint: false,
            fp_cache: self.fp_cache.as_ref(),
            fp_in_extraction: false,
            fp_min_depth: self.fp_min_depth,
            fp_margin: self.fp_margin,
            fp_max_dy: self.fp_max_dy,
            fp_defer: self.fp_defer,
            tid: 0,
            stop: &stop,
            nodes: 0,
            live_nodes: &live_nodes,
            flushed_nodes: 0,
            t4_fires: 0,
            t4_cutoffs: 0,
            fp_attempts: 0,
            fp_extractions: 0,
            fp_prunes: 0,
            fp_mask_bits: 0,
            aborted: false,
        };
        let b = self.b;
        let mut solve = |st: &State, d: u32, al: Value, be: Value| -> Option<Value> {
            let v = worker.ab(st, d, 0, al, be);
            if worker.aborted { None } else { Some(v) }
        };
        let res = crate::footprint::extract_win_footprint(b, u0, ceiling, &mut solve);
        self.nodes = worker.nodes;
        res
    }
}

impl<'a> Worker<'a> {
    /// Alpha-beta negamax. Returns the value of `s` for the side to move,
    /// fail-soft within the `(alpha, beta)` window. `ply` is the distance from
    /// the root (used only to index per-ply killer slots; ordering-only).
    ///
    /// LAZY-SMP ABORT: at entry, if the shared `stop` flag is set (a peer thread
    /// already completed the root search), this returns immediately with an
    /// arbitrary placeholder and marks `self.aborted`; the caller discards an
    /// aborted worker's result. The check is at node granularity, so an aborted
    /// worker never lingers and never corrupts the TT (any bound it DID store
    /// before aborting is still a valid bound — soundness is position-local).
    fn ab(&mut self, s: &State, depth: u32, ply: usize, mut alpha: Value, mut beta: Value) -> Value {
        // Cooperative early abort for losing lazy-SMP threads. Checked once per
        // node (cheap relaxed load). Only meaningful with >1 thread; with 1
        // thread `stop` is never set until the single search completes, so this
        // is never taken and behaviour is bit-identical to single-thread.
        if self.stop.load(Ordering::Relaxed) {
            self.aborted = true;
            return Value::Draw; // placeholder; an aborted worker's value is discarded.
        }
        // Profiling: count every internal node entered (instrumentation only).
        self.nodes += 1;
        // Heartbeat publishing: flush the local count to the shared live
        // counter in ~1M-node batches (observability only; one relaxed atomic
        // add per million nodes is unmeasurable).
        if self.nodes - self.flushed_nodes >= 1 << 20 {
            self.live_nodes
                .fetch_add(self.nodes - self.flushed_nodes, Ordering::Relaxed);
            self.flushed_nodes = self.nodes;
        }
        // Terminal: `winner` is the player who just moved (= 1 - turn). If that
        // is the side to move it's a Win, otherwise the side to move has lost.
        if let Some(p) = self.b.winner(s) {
            return if p == s.turn { Value::Win } else { Value::Loss };
        }
        if depth == 0 {
            return Value::Draw;
        }

        // Race short-circuit: with no walls left for either player the position
        // is a pure pawn race, solved exactly by its own retrograde pass. The
        // race value is a pure, context-free function of `State`, so it is
        // memoized in the SHARED bounded race memo across every leaf of this
        // solve (and across threads) — each distinct frozen-wall config is solved
        // once instead of per leaf. See `endgame.rs` for the exactness argument.
        if s.walls_left == [0, 0] {
            let (v, race_nodes) = crate::endgame::race_value(self.b, s, self.race_tt);
            self.nodes += race_nodes;
            return v;
        }

        let alpha0 = alpha;
        // Canonicalize the TT key by the horizontal-mirror representative. The
        // mirror is a value-preserving automorphism (same side to move), so `s`
        // and its mirror have the SAME value at every depth — keying both under
        // the shared representative is exact and roughly halves the TT.
        let canon = if self.use_symmetry {
            canonical(self.b, self.mirror_perm, s)
        } else {
            *s
        };
        // DEPTH-FOLDED key: one entry per CANONICAL position; the remaining
        // search `depth` is NOT in the key (it lives in the entry). The query
        // depth here is `depth`; the entry caps at `u16::MAX` (the solve ceiling
        // is far below that), and `OCCUPIED_BIT` keeps a live key non-zero.
        let key = pack_u128(&canon) | OCCUPIED_BIT;
        let qdepth = depth.min(u16::MAX as u32) as u16;
        // SOUNDNESS OF DEPTH-FOLD REUSE (the correctness-sensitive part). We may
        // reuse a stored bound ONLY when `stored.depth >= qdepth`. This is the
        // standard chess-engine rule, and it is exact on the `Loss < Draw < Win`
        // lattice of this depth-bounded negamax:
        //   * A definitive Win/Loss proven within `stored.depth` plies is a forced
        //     result — it is the TRUE game value and stays exact at ANY query
        //     depth, in particular at the shallower `qdepth <= stored.depth`.
        //   * A `Draw` returned at `stored.depth` means "no decision within
        //     `stored.depth` plies"; for any shallower horizon `qdepth <=
        //     stored.depth` there is likewise no decision (a forced win/loss
        //     reachable within `qdepth` plies would be reachable within
        //     `stored.depth >= qdepth` too), so the value is still `Draw`. Hence a
        //     deeper-or-equal result is always a valid bound for the shallower
        //     query.
        //   * Reusing a SHALLOWER result for a DEEPER query is NOT sound (a deeper
        //     search could turn a shallow `Draw` into a decision), and the
        //     `stored.depth >= qdepth` guard below forbids it: such a hit is a MISS.
        // The bound flags are then applied exactly as in plain alpha-beta. The
        // full u128 key is verified inside `probe`, so a hash collision is a MISS.
        // UNDER SHARING: an entry written by ANY thread is a valid bound for this
        // canonical position (the bound is position-local, independent of the
        // writer), so reusing a peer's entry is as sound as reusing our own.
        if let Some((val, flag, sdepth)) = self.tt.probe(key)
            && sdepth >= qdepth
        {
            match flag {
                Flag::Exact => return val,
                Flag::Lower => {
                    if val > alpha {
                        alpha = val;
                    }
                }
                Flag::Upper => {
                    if val < beta {
                        beta = val;
                    }
                }
            }
            if alpha >= beta {
                return val;
            }
        }

        // THEOREM 4 — one-sided frozen-race bounds, synthesized as a
        // depth-infinity TT-style hit (docs/superpowers/solver-pruning-theorems.md
        // §B.4; proven via Lemma M + race exactness + the refinement property,
        // falsification-validated by exact whole-graph relabeling at 2,121,148
        // states, zero mismatches).
        //
        // At a node where EXACTLY ONE side has exhausted its walls, let
        // `r = race_value(s with walls_left := [0,0])` — the TRUE value (draws
        // included, same side to move) of the race over the current frozen wall
        // config, from the existing exact retrograde machinery (memoized per
        // config in the shared race memo; the `[0,0]` leaves of this subtree
        // would need the same configs anyway, so the solve is amortized).
        //
        //   * Opponent exhausted (mover still holds walls): by Lemma M, zeroing
        //     the MOVER's budget only shrinks the mover's options, so
        //     `V_d(s) >= V_d(frozen) = r` for ALL d — a LOWER bound for the side
        //     to move, uniformly in depth.
        //       - `r = Win` is DECISIVE: `s` is a true forced Win. Return
        //         immediately and synthesize the permanent TT entry
        //         `(Win, Lower, depth = ∞)` — valid at every query depth under
        //         the depth-fold rule, exactly like the `[0,0]` short-circuit.
        //       - `r = Draw` raises alpha to Draw (a Loss return is impossible);
        //         `r = Loss` is vacuous.
        //   * Mover exhausted (opponent still holds walls): dually
        //     `V_d(s) <= r` for ALL d — an UPPER bound.
        //       - `r = Loss` is DECISIVE: a true forced Loss; return immediately
        //         and synthesize `(Loss, Upper, depth = ∞)`.
        //       - `r = Draw` lowers beta to Draw (a Win return is impossible);
        //         `r = Win` is vacuous.
        //
        // The non-decisive Draw bounds narrow the window EXACTLY like a TT probe
        // hit (same flag logic, same `alpha >= beta -> return r` close-out);
        // they are NOT stored, so a later finite-depth exact entry for this
        // position is never blocked by an unimprovable depth-∞ bound. The final
        // store at the end of this node remains flagged against `alpha0`
        // (captured BEFORE this narrowing), matching the TT-narrowing precedent,
        // so no stored bound weakens. Toggling `use_t4` is value-neutral.
        if self.use_t4 && (s.walls_left[0] == 0) != (s.walls_left[1] == 0) {
            let mut frozen = *s;
            frozen.walls_left = [0, 0];
            let (r, race_nodes) = crate::endgame::race_value(self.b, &frozen, self.race_tt);
            self.nodes += race_nodes;
            self.t4_fires += 1;
            if s.walls_left[(1 - s.turn) as usize] == 0 {
                // Opponent exhausted: r is a LOWER bound for the side to move.
                if r == Value::Win {
                    self.tt.store(key, Value::Win, Flag::Lower, u16::MAX);
                    self.t4_cutoffs += 1;
                    return Value::Win;
                }
                if r > alpha {
                    alpha = r;
                }
            } else {
                // Mover exhausted: r is an UPPER bound for the side to move.
                if r == Value::Loss {
                    self.tt.store(key, Value::Loss, Flag::Upper, u16::MAX);
                    self.t4_cutoffs += 1;
                    return Value::Loss;
                }
                if r < beta {
                    beta = r;
                }
            }
            if alpha >= beta {
                // The Draw bound met the window's other side: resolved with no
                // move loop, exactly as a TT probe hit returning `val` would.
                self.t4_cutoffs += 1;
                return r;
            }
        }

        let mut best = Value::Loss;
        let moves = self.ordered_moves(s, ply);
        // THEOREM 1 + COROLLARY 1, Win direction — wall-relevance footprint
        // mustplay pruning (docs/superpowers/solver-pruning-theorems.md §A;
        // falsification-validated: 537,771 complete-graph wall checks plus the
        // targeted T1-T5 families, zero violations).
        //
        // At this node the side to move is Z; `flip(s)` is the SAME position
        // with the opponent Y to move. When `V(flip(s))` is a PROVEN true Win
        // for Y, a Definition-1 certificate P (strategy + all-Z-replies
        // closure + verified rank) exists, and Theorem 1 says any Z wall `w`
        // anchored OUTSIDE the certificate's footprint masks leaves P valid
        // verbatim: `V(apply(s, w)) = Win` for Y, i.e. EXACTLY `Loss` for Z.
        // Skipping such a wall therefore never changes the negamax max (the
        // §A.2 integration rule: Win-direction prunes are exact child values,
        // safe unconditionally — no alpha guard needed). Pawn moves are always
        // searched. The masks come lazily from `footprint_masks` (cache or a
        // fresh build-then-verify extraction; `None` = no certificate = no
        // pruning, always sound).
        let mut fp_masks: Option<Option<(u64, u64)>> = None;
        let mut walls_searched: u32 = 0;
        for m in moves {
            let is_wall = matches!(m, Move::Wall { .. });
            if self.use_footprint
                && let Move::Wall { wc, wr, horiz } = m
            {
                // Resolve the node's masks lazily: cached masks apply from the
                // FIRST wall; a FRESH extraction is deferred until `fp_defer`
                // wall replies have already been searched at this node (an
                // early-cutoff node never pays an extraction; a genuine
                // refutation fan pays once and prunes the rest).
                let masks = match fp_masks {
                    Some(x) => x,
                    None => {
                        let allow_attempt = walls_searched >= self.fp_defer;
                        let (resolved, x) = self.footprint_masks(s, depth, ply, allow_attempt);
                        if self.aborted {
                            return Value::Draw; // placeholder; discarded
                        }
                        if resolved {
                            fp_masks = Some(x);
                        }
                        x
                    }
                };
                if let Some((hm, vm)) = masks {
                    let bit = 1u64 << ((wr as u32) * (self.b.w as u32 - 1) + wc as u32);
                    let inside = if horiz { hm & bit != 0 } else { vm & bit != 0 };
                    if !inside {
                        // Exact child value Loss-for-Z: skip with zero search.
                        self.fp_prunes += 1;
                        continue;
                    }
                }
            }
            let s2 = crate::movegen::apply(self.b, s, m);
            let v = self
                .ab(&s2, depth - 1, ply + 1, beta.negate(), alpha.negate())
                .negate();
            // If a peer signalled stop mid-recursion, the child returned a
            // placeholder; propagate the abort up without polluting `best`.
            if self.aborted {
                return Value::Draw;
            }
            if is_wall {
                walls_searched += 1;
            }
            if v > best {
                best = v;
            }
            if best > alpha {
                alpha = best;
            }
            if alpha >= beta {
                // Beta cutoff: reward this move in the killer (per-ply) and
                // history tables to try it earlier in sibling/future nodes.
                if self.use_ordering {
                    self.record_cutoff(m, ply, depth);
                }
                break;
            }
        }

        // Flag relative to the ORIGINAL window: `alpha0` is alpha captured BEFORE
        // any TT narrowing above; `beta` is the (possibly narrowed) upper bound,
        // matching the prior implementation exactly so no value can change.
        let flag = if best <= alpha0 {
            Flag::Upper
        } else if best >= beta {
            Flag::Lower
        } else {
            Flag::Exact
        };
        // Store under the depth-fold table: one entry per canonical position,
        // tagged with the query depth `qdepth` at which this bound was proven.
        self.tt.store(key, best, flag, qdepth);
        best
    }

    /// Footprint masks for the Z-to-move node `s` (called lazily at the first
    /// wall move of the move loop; `s.walls_left[s.turn] >= 1` is implied).
    /// Returns the forbidden (H, V) anchor masks of a verified Win certificate
    /// for `flip(s)` in `s`'s REAL orientation, or `None` (no pruning).
    ///
    /// Order of business: shared cache (canonical key, masks mirrored back per
    /// G5 when the canonical representative is the mirror image) → re-entrancy
    /// and cost gates → fresh extraction via `footprint::extract_win_footprint`
    /// with this worker's own alpha-beta as the certificate child solver
    /// (full window; decisive results are true forced values — G6).
    fn footprint_masks(
        &mut self,
        s: &State,
        depth: u32,
        ply: usize,
        allow_attempt: bool,
    ) -> (bool, Option<(u64, u64)>) {
        let mut flip = *s;
        flip.turn = 1 - s.turn;
        let canon = canonical(self.b, self.mirror_perm, &flip);
        let mirrored = canon != flip;
        let key = pack_u128(&canon);
        match self.fp_cache.get(key) {
            Some(FpEntry::Masks(h, v)) => {
                let m = if mirrored { self.mirror_masks(h, v) } else { (h, v) };
                return (true, Some(m));
            }
            // A failed attempt is retried only at meaningfully greater depth
            // (failures can be depth-relative; successes never are).
            Some(FpEntry::Failed(d)) if depth <= d.saturating_add(FP_RETRY_MARGIN) => {
                return (true, None);
            }
            _ => {}
        }
        // No NEW attempts while already inside an extraction's certificate
        // solves (cost control; cached masks above are still used). Skipping
        // a prune is always sound.
        if self.fp_in_extraction {
            return (true, None);
        }
        if !allow_attempt {
            // Deferred: the caller may ask again at a later wall move of this
            // node (NOT resolved — nothing is memoized for the node yet).
            return (false, None);
        }
        if depth < self.fp_min_depth {
            return (true, None);
        }
        // Heuristic gate (doc §A.5 "Where to apply"): attempt only when Y is
        // STRICTLY ahead in the raw race — `V(flip(s)) = Win` for Y against a
        // wall-holding Z is implausible otherwise, and the level-race (margin
        // 0) flip-proofs are the deep, expensive ones. Cost heuristic only.
        let y = 1 - s.turn;
        match (self.b.dist_to_goal(s, y), self.b.dist_to_goal(s, s.turn)) {
            (Some(dy), Some(dz))
                if dy <= self.fp_max_dy && (dy as i64) + self.fp_margin <= (dz as i64) => {}
            _ => return (true, None),
        }
        self.fp_attempts += 1;
        self.fp_in_extraction = true;
        let b = self.b;
        let res = {
            let mut solve = |st: &State, d: u32, al: Value, be: Value| -> Option<Value> {
                let v = self.ab(st, d, ply + 1, al, be);
                if self.aborted { None } else { Some(v) }
            };
            crate::footprint::extract_win_footprint(b, &flip, depth, &mut solve)
        };
        self.fp_in_extraction = false;
        if self.aborted {
            return (true, None); // peer finished the root: discard everything
        }
        match res {
            Some((h, v)) => {
                self.fp_extractions += 1;
                self.fp_mask_bits += u64::from(h.count_ones() + v.count_ones());
                let (sh, sv) = if mirrored { self.mirror_masks(h, v) } else { (h, v) };
                self.fp_cache.put(key, FpEntry::Masks(sh, sv));
                (true, Some((h, v)))
            }
            None => {
                self.fp_cache.put(key, FpEntry::Failed(depth));
                (true, None)
            }
        }
    }

    /// Mirror a pair of wall-anchor masks through the horizontal-reflection
    /// permutation (identity on degenerate boards without anchors). H masks
    /// map to H masks and V to V — orientation is preserved by a left-right
    /// flip, exactly as in `mirror_state`.
    fn mirror_masks(&self, h: u64, v: u64) -> (u64, u64) {
        match self.mirror_perm {
            Some(perm) => (permute_bits(h, perm), permute_bits(v, perm)),
            None => (h, v),
        }
    }

    /// Record a beta-cutoff move into the killer (per-ply, up to 2 distinct
    /// moves, most-recent first) and history (cutoff-count, weighted by depth)
    /// tables. ORDERING ONLY — never affects values.
    #[inline]
    fn record_cutoff(&mut self, m: Move, ply: usize, depth: u32) {
        if ply < self.killers.len() {
            let slot = &mut self.killers[ply];
            if slot[0] != Some(m) {
                slot[1] = slot[0];
                slot[0] = Some(m);
            }
        }
        // Deeper cutoffs (more remaining depth) are worth more; saturating to
        // avoid overflow on long runs.
        let idx = move_index(self.b, m);
        self.history[idx] = self.history[idx].saturating_add(depth.saturating_mul(depth));
    }

    /// Order legal moves to maximize alpha-beta cutoffs. The proven distance
    /// advantage `d_opp(s2) - d_self(s2)` (descending) is the ABSOLUTE primary
    /// key: it is an exceptionally strong, position-dependent heuristic on these
    /// boards, and displacing its top move measurably blows up the search (at
    /// W2, making a killer-first ordering primary cost ~9x the nodes). The
    /// global history cutoff-count and the killer moves (a refutation that
    /// caused a cutoff at THIS ply) are therefore applied ONLY as tiebreakers,
    /// refining the order WITHIN a group of moves that the distance heuristic
    /// ranks equal (where the baseline's stable sort left an arbitrary order).
    /// History is the stronger, smoother signal here, so it is the SECONDARY key
    /// and killers only break history ties. This can only help or be neutral —
    /// it never moves a move past one the distance heuristic prefers. Sort is
    /// descending on `(distance, history, killer_rank)`. (Measured: 6x5 W1
    /// 963K->348K nodes, W2 9.45M->6.52M nodes.)
    ///
    /// LAZY-SMP DIVERSITY: among moves the distance heuristic ranks EQUAL and the
    /// ordering heuristics leave tied, worker `tid > 0` applies a deterministic
    /// per-thread rotation of the tied group so different threads explore the
    /// tree in different orders (the lazy-SMP speedup mechanism). `tid == 0` (and
    /// the only worker when `nthreads == 1`) uses the unrotated baseline order,
    /// so single-thread behaviour is bit-identical to before. EXACTNESS: this is
    /// pure reordering — the legal-move SET is unchanged and alpha-beta returns
    /// the identical value under any visitation order, so no value can change.
    fn ordered_moves(&self, s: &State, ply: usize) -> Vec<Move> {
        let mover = s.turn;
        let opp = 1 - mover;
        let big = 4 * (self.b.w as i64 + self.b.h as i64);
        let killers = if self.use_ordering && ply < self.killers.len() {
            self.killers[ply]
        } else {
            [None, None]
        };
        let use_hist = self.use_ordering;
        let mut scored: Vec<(i64, u32, u8, Move)> = crate::movegen::legal_moves(self.b, s)
            .into_iter()
            .map(|m| {
                let s2 = crate::movegen::apply(self.b, s, m);
                let d_self = self.b.dist_to_goal(&s2, mover).map_or(big, |d| d as i64);
                let d_opp = self.b.dist_to_goal(&s2, opp).map_or(big, |d| d as i64);
                // Killer rank: first killer best (2), second (1), none (0).
                let krank = if killers[0] == Some(m) {
                    2
                } else if killers[1] == Some(m) {
                    1
                } else {
                    0
                };
                let hist = if use_hist {
                    self.history[move_index(self.b, m)]
                } else {
                    0
                };
                (d_opp - d_self, hist, krank, m)
            })
            .collect();
        // Descending by (distance score, history, killer rank). Stable, so moves
        // tied on all three keep their `legal_moves` order — identical to the
        // single-thread baseline.
        scored.sort_by(|a, b| b.0.cmp(&a.0).then(b.1.cmp(&a.1)).then(b.2.cmp(&a.2)));
        // Lazy-SMP diversification for helper threads (tid > 0): rotate ONLY
        // WITHIN each maximal run of moves sharing the SAME distance score (the
        // dominant primary key). The distance heuristic is by far the strongest
        // signal here — displacing its top move blows up the search ~9x — so the
        // ordering ACROSS distance-score groups is preserved EXACTLY (the best
        // distance group still leads). Within a tied-distance group the moves are
        // heuristically equivalent, so permuting them is ordering-neutral in
        // quality yet sends each helper down a different first line, warming the
        // shared TT from a different angle. `tid == 0` (and the only worker when
        // `nthreads == 1`) leaves every group unrotated, so single-thread
        // behaviour is bit-identical to the baseline. EXACTNESS: pure reordering
        // — the legal-move SET is unchanged and alpha-beta returns the identical
        // value under any visitation order, so no value can change.
        if self.tid > 0 {
            let mut i = 0;
            while i < scored.len() {
                let dist = scored[i].0;
                let mut j = i + 1;
                while j < scored.len() && scored[j].0 == dist {
                    j += 1;
                }
                let group = &mut scored[i..j];
                if group.len() > 1 {
                    let shift = self.tid % group.len();
                    if shift > 0 {
                        group.rotate_left(shift);
                    }
                }
                i = j;
            }
        }
        scored.into_iter().map(|(_, _, _, m)| m).collect()
    }
}

#[cfg(test)]
mod tests {
    use crate::board::Board;
    use crate::solver::{brute_value, Solver, Value};

    #[test]
    fn solver_matches_bruteforce_3x3() {
        let b = Board::new(3, 3, 1);
        let mut sol = Solver::new(&b);
        assert_eq!(sol.solve(&b.initial()), brute_value(&b, &b.initial(), 14));
    }

    #[test]
    fn three_by_three_is_second_player_win() {
        let b = Board::new(3, 3, 1);
        let mut sol = Solver::new(&b);
        assert_eq!(sol.solve(&b.initial()), Value::Loss); // side-to-move (p0) loses
    }

    #[test]
    fn solver_matches_bruteforce_random_3x3() {
        // walk seeded random 3x3 games; at each non-terminal node, Solver::solve must
        // equal brute_value(depth=14). Use a simple LCG; check >40 nodes.
        let b = Board::new(3, 3, 1);
        let mut checked = 0;
        let mut st = 0x1234u64;
        let mut next = |n: usize| {
            st = st.wrapping_mul(6364136223846793005).wrapping_add(1);
            (st >> 33) as usize % n
        };
        for _ in 0..30 {
            let mut s = b.initial();
            for _ in 0..8 {
                if b.is_terminal(&s) {
                    break;
                }
                let mut sol = Solver::new(&b);
                assert_eq!(sol.solve(&s), brute_value(&b, &s, 14));
                let ms = crate::movegen::legal_moves(&b, &s);
                s = crate::movegen::apply(&b, &s, ms[next(ms.len())]);
                checked += 1;
            }
        }
        assert!(checked > 40);
    }
}
