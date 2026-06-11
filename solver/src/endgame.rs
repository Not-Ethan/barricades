//! Endgame tablebase slice: the *race* over a FROZEN wall configuration.
//!
//! When both players have exhausted their walls (`walls_left == [0, 0]`), no
//! MORE walls can ever be placed, but the walls already on the board stay
//! FROZEN. The game reduces to a pure pawn race over steps only, played on the
//! fixed maze those frozen walls define. This module computes the exact
//! game-theoretic value (Win / Loss / **Draw**) of such a race by EXACT
//! retrograde analysis over the finite pawn-state graph — no depth bound, no
//! panic.
//!
//! ## Frozen-wall races CAN be genuine draws
//!
//! A previous implementation rested on the (false) invariant that a wall-less
//! race is never a true draw and `panic!`d on anything else. That is wrong: the
//! race is "no MORE walls" but FROZEN walls remain, and a frozen maze can let
//! one pawn perpetually body-block the only corridor the other must traverse.
//! Neither side can then force a goal, so the position is a GENUINE DRAW (with
//! legal moves available — not zugzwang). Confirmed repro: `Board::new(7, 5, 4)`
//! with `pawn = [18, 24]`, `h_walls = 0x280240`, `v_walls = 0x500400`,
//! `walls_left = [0, 0]`, `turn = 1` is a stable Draw.
//!
//! ## Why retrograde is exact
//!
//! The race graph for a fixed wall configuration is finite: a node is
//! `(pawn0, pawn1, turn)` with `pawn0 != pawn1`, both on board, and there are at
//! most `(w*h)^2 * 2` of them. Retrograde analysis (backward induction /
//! standard pursuit-game labeling) computes the exact value of EVERY node of a
//! finite graph and handles cycles correctly:
//!
//!   * Terminal/stuck nodes are seeded as `Loss` for the mover (the opponent is
//!     already on its goal, or the mover has no legal step), and the rare
//!     own-goal node as `Win`.
//!   * A node is finalized `Win` as soon as ANY successor is finalized `Loss`
//!     (the mover can step into a position the opponent loses).
//!   * A node is finalized `Loss` only once ALL its successors are finalized
//!     `Win` (every move hands the opponent a win).
//!   * Any node never finalized is a `Draw`: neither side can force a win, i.e.
//!     a perpetual blockade. This is the unique fixpoint of the negamax
//!     equations on the `Loss < Draw < Win` lattice, so it is the exact value.
//!
//! Because the labeling is over the full finite graph with no truncation, no
//! false draw and no missed win is possible, and there is no depth to exhaust —
//! it is `O(nodes + edges)`, fast even on draw-heavy (blockade) mazes where the
//! old iterative-deepening search would deepen to its ceiling and panic.
//!
//! ## Value convention (matches the rest of the solver EXACTLY)
//!
//! For a state `s` with side-to-move `t = s.turn`:
//!   * `winner(s) == Some(t)`   -> `Win`  (own goal; unreachable in normal play).
//!   * `winner(s) == Some(1-t)` -> `Loss` (opponent already on its goal).
//!   * otherwise `value(s) = max over legal_steps of negate(value(child))`
//!     (plain negamax; `Win > Draw > Loss`), and a non-terminal node with NO
//!     legal step is a `Loss` for the mover.
//!
//! ## Bounded, exact, config-granular LRU memo (sharded, thread-safe)
//!
//! The race value of a frozen-wall `State` is a pure, context-free function of
//! that `State` (pawns + frozen walls + turn, with `walls_left == [0, 0]`). One
//! retrograde pass labels EVERY reachable pawn-pair for the current wall
//! configuration at once, so the natural unit of caching is a whole CONFIG: the
//! frozen wall layout `(h_walls, v_walls)` together with the table of every
//! pawn-pair-and-turn value it admits. Subsequent races with the same frozen
//! walls — any pawn pair, any turn — are then instant memo hits.
//!
//! At high wall counts the number of distinct frozen configs explodes and an
//! UNBOUNDED memo grows to many gigabytes, thrashing the allocator. The memo is
//! therefore BOUNDED by `QS_RACE_MB` (default `DEFAULT_RACE_MB`) and evicted at
//! **whole-config granularity in LRU order**: a single retrograde pass fills all
//! ~`(w*h)^2 * 2` pawn-states of one config together, so eviction must drop a
//! config's table as a UNIT (never half a config, which would force re-solving
//! it per-pawn). When inserting a freshly solved config would exceed the cap, we
//! drop least-recently-used configs whole until back under budget.
//!
//! ## Sharding (kill the global-mutex serialization)
//!
//! Live profiling of an 8-worker lazy-SMP solve showed the workers serializing
//! on this cache's previous SINGLE global `Mutex` (`__psynch_mutexwait` 17.8k
//! samples vs 2.4k of real `race_value` work). The memo is therefore SHARDED,
//! mirroring the proven `ShardedTt` pattern of the main table (62 samples under
//! the same profile): a config key `(h_walls, v_walls)` hashes to exactly ONE of
//! `RACE_SHARDS` shards; each shard owns its own map, its own LRU bookkeeping,
//! and `1/RACE_SHARDS` of the byte budget, enforced at whole-config granularity
//! exactly as before. Lookups vastly outnumber inserts, so each shard is an
//! `RwLock`: reads take the SHARED lock and never block each other, and the LRU
//! "touch" on a read is a relaxed store of a monotonic per-shard atomic tick
//! into the config's atomic `last_used` stamp — NO write lock, no list
//! reordering on the read path. Only inserts/evictions take the write lock, and
//! eviction consults the atomic stamps to pick its LRU victims.
//!
//! EXACTNESS-SAFE: this is a PURE cache. Every stored value is the position's
//! exact game-theoretic value; a cap-induced miss simply re-runs the cheap
//! retrograde for that one config and recomputes the identical value. Capping,
//! sharding, and the relaxed (approximate-order) LRU stamps can NEVER change a
//! returned value — only the work done to obtain it. (The retrograde labeling
//! is deterministic and context-free, so a re-solve of an evicted config yields
//! a byte-identical table.)

use crate::board::Board;
use crate::solver::Value;
use crate::state::State;
use rustc_hash::FxHashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::RwLock;

/// Default race-memo budget in MiB when `QS_RACE_MB` is unset/invalid.
pub const DEFAULT_RACE_MB: usize = 1024;

/// The frozen wall layout that identifies a race config: `(h_walls, v_walls)`.
/// `walls_left` is always `[0, 0]` in a race, so it is not part of the key. All
/// pawn-pairs-and-turns over the SAME frozen maze are solved together by one
/// retrograde pass and cached as a unit under this key.
type ConfigKey = (u64, u64);

/// One race config's solved table: every `(pawn0, pawn1, turn)` reachable in the
/// frozen maze mapped to its exact game value. Built whole by a single
/// retrograde pass and inserted/evicted as a UNIT.
type ConfigTable = FxHashMap<(u8, u8, u8), Value>;

/// Estimated heap bytes a `ConfigTable` of `n` entries occupies, for the LRU
/// budget accounting. The map is an `FxHashMap<(u8,u8,u8), Value>`: a hashbrown
/// SwissTable storing a 4-byte `(key,value)` pair plus 1 control byte per slot,
/// at up to 7/8 load and with the bucket array rounded up to a power of two
/// (worst case ~2x). That works out to roughly 8-12 real bytes per live entry;
/// we charge 16 — a modest safety margin over the realistic figure so the cap
/// is a slight UNDER-estimate of how much fits (it never blows the RSS budget)
/// without the gross 3x over-charge that an earlier 32-byte figure used (which
/// evicted far too eagerly and thrashed re-solves). Plus a small fixed per-config
/// overhead. Exactness-neutral: the estimate only governs WHEN we evict, never
/// any value.
#[inline]
fn config_bytes(n: usize) -> usize {
    n.saturating_mul(16).saturating_add(128)
}

/// Number of race-memo shards. 16 (power of two; selection is a mask).
///
/// Why 16 and not 32: unlike the main `ShardedTt` (32 Mutex shards, where every
/// PROBE takes an exclusive lock), race reads here take a SHARED `RwLock` read
/// lock, so readers never block readers regardless of shard count. Shards only
/// need to (a) spread the rare insert/evict write-lock stalls and (b) split the
/// cache-line traffic on the per-shard LRU clock atomics. The default (64)
/// keeps same-shard write collisions unlikely up to ~32 lazy-SMP workers
/// (cloud-class CPU pods); shards are cheap structs, so the overhead at 8
/// local threads is negligible. Env `QS_RACE_SHARDS` overrides (rounded up to
/// a power of two) — lower it (e.g. 16) if running tiny budgets where coarser
/// per-shard slices retain more whole configs before evicting.
fn race_shards() -> usize {
    std::env::var("QS_RACE_SHARDS")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .map(|n| n.clamp(1, 1024).next_power_of_two())
        .unwrap_or(64)
}

/// Bounded, exact, config-granular LRU race memo, SHARDED by config key, with
/// interior mutability so it can be shared (`&self`) across the parallel search
/// threads. Each shard's `RwLock` is held only for the (fast) map lookup or
/// insert/evict bookkeeping — never across the retrograde computation itself:
/// threads compute their retrograde passes in parallel and only briefly lock
/// one shard to publish/read the result.
pub struct RaceTt {
    shards: Vec<Shard>,
    /// Per-shard byte budget: `QS_RACE_MB` (in bytes) split evenly across
    /// `RACE_SHARDS`. Each shard enforces its slice independently at
    /// whole-config granularity.
    shard_cap_bytes: usize,
}

struct Shard {
    /// Monotonic per-shard LRU clock. Ticked with a relaxed `fetch_add` on
    /// every touch (read hit or insert) WITHOUT the write lock; per-shard (not
    /// global) so readers of different shards never contend on one cache line.
    /// Relaxed ordering gives an approximate-but-monotonic recency order, which
    /// is all LRU eviction needs — exactness-neutral by construction.
    clock: AtomicU64,
    inner: RwLock<ShardInner>,
}

struct ShardInner {
    /// Solved config tables keyed by frozen wall layout. Each value carries the
    /// table plus its atomic last-use LRU stamp and accounted byte size.
    configs: FxHashMap<ConfigKey, Slot>,
    /// Sum of every resident config's accounted `bytes` (kept in sync on
    /// insert/evict), compared against the shard's budget slice.
    total_bytes: usize,
}

struct Slot {
    table: ConfigTable,
    /// Last-touch tick from the shard clock. Written with a RELAXED store under
    /// the shard's READ lock on every hit (the LRU "touch" needs no write
    /// lock); read by eviction under the write lock. Races between concurrent
    /// readers just keep one of several near-identical recent ticks — the
    /// resulting LRU order is approximate, never the values.
    last_used: AtomicU64,
    bytes: usize,
}

impl Default for RaceTt {
    fn default() -> Self {
        let mb = std::env::var("QS_RACE_MB")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .filter(|&m| m > 0)
            .unwrap_or(DEFAULT_RACE_MB);
        RaceTt::with_cap_mb(mb)
    }
}

impl RaceTt {
    /// Build a race memo with an explicit TOTAL cap in MiB, split evenly across
    /// the shards (test hook for the tiny-cap eviction-stress gate; `Default`
    /// funnels `QS_RACE_MB` here).
    pub fn with_cap_mb(mb: usize) -> RaceTt {
        let nshards = race_shards();
        let cap_bytes = mb.max(1).saturating_mul(1024 * 1024);
        let shard_cap_bytes = (cap_bytes / nshards).max(1);
        let shards = (0..nshards)
            .map(|_| Shard {
                clock: AtomicU64::new(0),
                inner: RwLock::new(ShardInner {
                    configs: FxHashMap::default(),
                    total_bytes: 0,
                }),
            })
            .collect();
        RaceTt {
            shards,
            shard_cap_bytes,
        }
    }

    /// The shard owning config key `ck`. A splitmix64-style mix of both wall
    /// words, masked to the (power-of-two) shard count, so every config lives
    /// in exactly ONE shard and similar wall layouts spread evenly.
    #[inline]
    fn shard(&self, ck: ConfigKey) -> &Shard {
        let mut x = ck
            .0
            .wrapping_mul(0x9e37_79b9_7f4a_7c15)
            .wrapping_add(ck.1.wrapping_mul(0xbf58_476d_1ce4_e5b9));
        x ^= x >> 30;
        x = x.wrapping_mul(0x94d0_49bb_1331_11eb);
        x ^= x >> 31;
        // shards.len() is a power of two by construction (race_shards()).
        &self.shards[(x as usize) & (self.shards.len() - 1)]
    }

    /// Total number of cached pawn-state entries across all resident configs of
    /// all shards (reporting/tests only). Counts entries, not configs, matching
    /// the old flat-memo semantics so `race_tt_len() > 0` still means "memo
    /// populated".
    pub fn len(&self) -> usize {
        self.shards
            .iter()
            .map(|sh| {
                let g = sh.inner.read().expect("race memo shard lock poisoned");
                g.configs.values().map(|s| s.table.len()).sum::<usize>()
            })
            .sum()
    }

    /// Whether the memo holds no entries.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Number of resident (whole) configs across all shards (reporting/tests).
    pub fn config_count(&self) -> usize {
        self.shards
            .iter()
            .map(|sh| {
                sh.inner
                    .read()
                    .expect("race memo shard lock poisoned")
                    .configs
                    .len()
            })
            .sum()
    }

    /// Look up the exact value of race state `s` if its frozen-wall config is
    /// resident. Takes only the owning shard's READ lock — concurrent lookups
    /// (the dominant operation) never block each other. A hit bumps the
    /// config's recency via a relaxed atomic tick store (the LRU "touch"
    /// without a write lock). Returns `None` (a cache miss) when the config is
    /// absent — the caller then solves it.
    fn get(&self, s: &State) -> Option<Value> {
        let ck: ConfigKey = (s.h_walls, s.v_walls);
        let sh = self.shard(ck);
        let g = sh.inner.read().expect("race memo shard lock poisoned");
        let slot = g.configs.get(&ck)?;
        let tick = sh.clock.fetch_add(1, Ordering::Relaxed) + 1;
        slot.last_used.store(tick, Ordering::Relaxed);
        slot.table.get(&(s.pawn[0], s.pawn[1], s.turn)).copied()
    }

    /// Insert a freshly solved config table (built by one retrograde pass over
    /// the frozen maze `ck`) into its owning shard, under that shard's WRITE
    /// lock. Stamps it most-recently-used, updates the shard's byte total, and
    /// evicts the shard's least-recently-used configs WHOLE until back under
    /// the shard's budget slice. Exactness-neutral: this is a pure cache
    /// publish.
    fn insert_config(&self, ck: ConfigKey, table: ConfigTable) {
        let sh = self.shard(ck);
        let bytes = config_bytes(table.len());
        let tick = sh.clock.fetch_add(1, Ordering::Relaxed) + 1;
        let mut g = sh.inner.write().expect("race memo shard lock poisoned");
        // If this config was already resident (a concurrent thread solved it
        // too, or a re-solve after eviction), replace it and adjust the total.
        // The replacement table is byte-identical (deterministic retrograde),
        // so which copy survives is value-irrelevant.
        if let Some(old) = g.configs.insert(
            ck,
            Slot {
                table,
                last_used: AtomicU64::new(tick),
                bytes,
            },
        ) {
            g.total_bytes = g.total_bytes.saturating_sub(old.bytes);
        }
        g.total_bytes = g.total_bytes.saturating_add(bytes);
        ShardInner::evict_to_cap(&mut g, self.shard_cap_bytes);
    }
}

impl ShardInner {
    /// Evict this shard's least-recently-used configs WHOLE until
    /// `total_bytes <= cap_bytes` (the shard's slice of the budget). Never
    /// evicts below one config (a single config must always fit so a
    /// just-solved query can be served — the cap is advisory,
    /// exactness-neutral). Runs under the shard's write lock, so the relaxed
    /// loads of the per-config atomic stamps see settled values (no concurrent
    /// readers hold the lock).
    fn evict_to_cap(g: &mut ShardInner, cap_bytes: usize) {
        while g.total_bytes > cap_bytes && g.configs.len() > 1 {
            // Find the config with the smallest last_used stamp (the LRU one).
            let victim = g
                .configs
                .iter()
                .min_by_key(|(_, slot)| slot.last_used.load(Ordering::Relaxed))
                .map(|(k, _)| *k);
            match victim {
                Some(k) => {
                    if let Some(old) = g.configs.remove(&k) {
                        g.total_bytes = g.total_bytes.saturating_sub(old.bytes);
                    }
                }
                None => break,
            }
        }
    }
}

/// A race-graph node: the two pawn cell indices plus the side to move. The
/// frozen wall configuration is fixed for the whole retrograde pass, so it is
/// NOT part of the node key (it lives in the `Board` + the wall bits we carry
/// in the working `State`).
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct Node {
    pawn: [u8; 2],
    turn: u8,
}

/// Final game-theoretic label assigned by the retrograde pass.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Label {
    Win,
    Loss,
}

/// Exact game value of a frozen-wall race for the side to move, paired with the
/// number of race nodes visited (for profiling; the value is unaffected).
///
/// Computed by EXACT retrograde (backward-induction) labeling over the finite
/// `(pawn0, pawn1, turn)` graph of the current frozen wall configuration. Win,
/// Loss, AND Draw are all returned exactly: there is no depth bound, no ceiling,
/// and no panic — a perpetual blockade is labeled `Draw`, never mis-resolved.
///
/// One pass labels every pawn pair reachable (forward) from `s`; ALL of them are
/// cached as a UNIT into the bounded config-granular LRU memo `tt`, so any later
/// race with the same frozen walls is an instant hit (until/unless that whole
/// config is LRU-evicted, in which case it is simply re-solved — the value is
/// unchanged because the retrograde labeling is a deterministic pure function of
/// the frozen maze).
///
/// `tt` is taken by shared reference (`&RaceTt`) and uses interior mutability,
/// so the same memo is shared across the parallel search threads. Only the one
/// shard owning this config is locked, and only for the brief lookup (shared
/// read lock) or publish (write lock) — never across this retrograde pass.
pub fn race_value(b: &Board, s: &State, tt: &RaceTt) -> (Value, u64) {
    debug_assert_eq!(
        s.walls_left,
        [0, 0],
        "race_value called on a non-race state (walls remain): {:?}",
        s.walls_left
    );

    // Fast path: this config is resident (e.g. a sibling leaf solved it).
    if let Some(v) = tt.get(s) {
        return (v, 0);
    }

    let query = Node {
        pawn: s.pawn,
        turn: s.turn,
    };

    // A reusable working state carrying the FROZEN wall bits; only `pawn`/`turn`
    // change as we explore. `walls_left` stays `[0, 0]` so movegen never offers
    // a wall move (and `legal_steps` ignores `walls_left` anyway).
    let mut work = *s;

    // ---- 1. Enumerate the forward-reachable component and build edges. ----
    //
    // `index[node] = id`. Successor edges are stored in CSR form — one flat
    // `succ_edges` arena plus a per-node `(start, end)` range — rather than a
    // `Vec<u32>` per node: a node's successors are all emitted contiguously
    // while it is expanded, so the flat layout is natural and removes the
    // O(nodes) small-Vec allocations that showed up as malloc/free churn in
    // live profiles of `race_value`. Predecessor edges are built afterwards by
    // a two-pass counting transpose (also CSR, also O(1) allocations).
    let mut index: FxHashMap<Node, u32> = FxHashMap::default();
    let mut nodes: Vec<Node> = Vec::new();
    // Per-node seed label (Win/Loss) if it is a terminal/stuck node, else None.
    let mut seed: Vec<Option<Label>> = Vec::new();
    // CSR successors: `succ_edges[succ_range[id].0 .. succ_range[id].1]` are
    // the successor node ids of `id` (set when `id` is expanded; terminal and
    // stuck nodes keep the empty `(0, 0)` range).
    let mut succ_edges: Vec<u32> = Vec::new();
    let mut succ_range: Vec<(u32, u32)> = Vec::new();

    let mut visited = 0u64; // profiling: race nodes touched.

    // Intern a node, allocating its id + parallel rows on first sight.
    let intern = |nodes: &mut Vec<Node>,
                  seed: &mut Vec<Option<Label>>,
                  succ_range: &mut Vec<(u32, u32)>,
                  index: &mut FxHashMap<Node, u32>,
                  n: Node|
     -> u32 {
        if let Some(&id) = index.get(&n) {
            return id;
        }
        let id = nodes.len() as u32;
        nodes.push(n);
        seed.push(None);
        succ_range.push((0, 0));
        index.insert(n, id);
        id
    };

    let start = intern(&mut nodes, &mut seed, &mut succ_range, &mut index, query);

    // BFS/DFS the reachable component, recording successor edges.
    let mut stack: Vec<u32> = vec![start];
    // `expanded[id]` guards against re-expanding a node (its edges are built
    // exactly once). A node can be interned (as a successor) before it is
    // expanded, so this is separate from membership in `index`.
    let mut expanded: Vec<bool> = vec![false];
    // Reused step buffer (max 5 destinations) — one allocation for the whole
    // pass instead of one `Vec` per expanded node.
    let mut steps: Vec<u8> = Vec::with_capacity(8);
    while let Some(id) = stack.pop() {
        if expanded[id as usize] {
            continue;
        }
        expanded[id as usize] = true;
        visited += 1;

        let node = nodes[id as usize];
        work.pawn = node.pawn;
        work.turn = node.turn;

        let t = node.turn;
        // Terminal handling (matches the solver's convention exactly):
        //  - own goal on the node's turn -> Win (rare; e.g. a query already on
        //    its goal). It has no race successors that matter.
        //  - opponent already on its goal -> Loss for the mover.
        match b.winner(&work) {
            Some(p) if p == t => {
                seed[id as usize] = Some(Label::Win);
                continue;
            }
            Some(_) => {
                seed[id as usize] = Some(Label::Loss);
                continue;
            }
            None => {}
        }

        crate::movegen::legal_steps_into(b, &work, &mut steps);
        if steps.is_empty() {
            // Non-terminal but stuck: a Loss for the mover (matches the solver —
            // a no-step node never improves past the initial Loss).
            seed[id as usize] = Some(Label::Loss);
            continue;
        }

        // Build successor edges. A step moves the mover's pawn and flips turn.
        let lo = succ_edges.len() as u32;
        for &dest in &steps {
            let mut child = node;
            child.pawn[t as usize] = dest;
            child.turn = 1 - t;
            let cid = intern(&mut nodes, &mut seed, &mut succ_range, &mut index, child);
            // `expanded` must stay parallel to `nodes`.
            if cid as usize >= expanded.len() {
                expanded.resize(nodes.len(), false);
            }
            succ_edges.push(cid);
            if !expanded[cid as usize] {
                stack.push(cid);
            }
        }
        succ_range[id as usize] = (lo, succ_edges.len() as u32);
    }

    let n = nodes.len();
    // `succ_remaining[id]` starts at the successor count; each time a successor
    // is finalized WIN we decrement it. Reaching 0 means ALL successors are WIN
    // -> the node is a LOSS for its mover.
    let mut succ_remaining: Vec<u32> = succ_range.iter().map(|&(lo, hi)| hi - lo).collect();

    // Predecessor CSR by counting transpose of the successor edges:
    // `pred_edges[pred_start[id] .. pred_start[id + 1]]` are the predecessor
    // node ids of `id`. Exactly the same edge multiset the per-node `Vec`s
    // held before, in flat form.
    let mut pred_start: Vec<u32> = vec![0; n + 1];
    for &c in &succ_edges {
        pred_start[c as usize + 1] += 1;
    }
    for i in 0..n {
        pred_start[i + 1] += pred_start[i];
    }
    let mut pred_edges: Vec<u32> = vec![0; succ_edges.len()];
    let mut cursor: Vec<u32> = pred_start[..n].to_vec();
    for (id, &(lo, hi)) in succ_range.iter().enumerate() {
        for &c in &succ_edges[lo as usize..hi as usize] {
            let slot = cursor[c as usize] as usize;
            pred_edges[slot] = id as u32;
            cursor[c as usize] += 1;
        }
    }

    // ---- 2. Initialize the worklist from seeded terminal/stuck nodes. ----
    let mut label: Vec<Option<Label>> = vec![None; n];
    let mut queue: Vec<u32> = Vec::new();
    for id in 0..n {
        if let Some(l) = seed[id] {
            label[id] = Some(l);
            queue.push(id as u32);
        }
    }

    // ---- 3. Propagate backward to fixpoint (standard pursuit retrograde). ----
    //
    //  - A node finalized LOSS makes EVERY predecessor a WIN (the predecessor's
    //    mover can step into this losing-for-the-opponent node).
    //  - A node finalized WIN decrements each predecessor's remaining counter;
    //    when a predecessor's counter hits 0 (all successors are WIN) it is a
    //    LOSS.
    let mut qi = 0usize;
    while qi < queue.len() {
        let id = queue[qi] as usize;
        qi += 1;
        let preds = &pred_edges[pred_start[id] as usize..pred_start[id + 1] as usize];
        match label[id].expect("queued node must be labeled") {
            Label::Loss => {
                // Predecessors become WIN.
                for &p in preds {
                    let p = p as usize;
                    if label[p].is_none() {
                        label[p] = Some(Label::Win);
                        queue.push(p as u32);
                    }
                }
            }
            Label::Win => {
                // Each predecessor loses one not-yet-WIN successor.
                for &p in preds {
                    let p = p as usize;
                    if label[p].is_some() {
                        continue;
                    }
                    succ_remaining[p] -= 1;
                    if succ_remaining[p] == 0 {
                        label[p] = Some(Label::Loss);
                        queue.push(p as u32);
                    }
                }
            }
        }
    }

    // ---- 4 & 5. Residue = Draw; build the WHOLE config table and publish it. ----
    //
    // Caching all reachable pawn pairs (not just the query) is what makes a
    // later race with the same frozen walls an instant memo hit. The whole table
    // is inserted as a UNIT under the frozen-wall config key, so LRU eviction
    // drops a config atomically (never half of one).
    let mut query_value: Option<Value> = None;
    let mut table: ConfigTable = FxHashMap::default();
    table.reserve(n);
    for id in 0..n {
        let node = nodes[id];
        let v = match label[id] {
            Some(Label::Win) => Value::Win,
            Some(Label::Loss) => Value::Loss,
            None => Value::Draw, // never finalized -> perpetual blockade.
        };
        table.insert((node.pawn[0], node.pawn[1], node.turn), v);
        if node == query {
            query_value = Some(v);
        }
    }

    // Sanity: retrograde labels every node of the finite graph (Win/Loss/Draw),
    // so the query — which is in the enumerated component by construction — is
    // always assigned a value. This assert can never fire on a correct pass.
    let v = query_value.expect(
        "retrograde pass must assign the query node a value (Win/Loss/Draw); \
         missing assignment indicates a bug in race enumeration",
    );

    // Publish the whole config table (with LRU eviction if over the cap).
    tt.insert_config((s.h_walls, s.v_walls), table);

    (v, visited)
}
