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
//! ## Bounded, exact, config-granular LRU memo (thread-safe)
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
//! EXACTNESS-SAFE: this is a PURE cache. Every stored value is the position's
//! exact game-theoretic value; a cap-induced miss simply re-runs the cheap
//! retrograde for that one config and recomputes the identical value. Capping
//! can NEVER change a returned value — only the work done to obtain it. (The
//! retrograde labeling is deterministic and context-free, so a re-solve of an
//! evicted config yields a byte-identical table.)

use crate::board::Board;
use crate::solver::Value;
use crate::state::State;
use rustc_hash::FxHashMap;
use std::sync::Mutex;

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

/// Bounded, exact, config-granular LRU race memo with interior mutability so it
/// can be shared (`&self`) across the parallel search threads. All access goes
/// through a single `Mutex` guarding the map, the LRU clock, and the byte total.
///
/// The mutex is held only for the (fast) map lookups/inserts and eviction
/// bookkeeping — never across the retrograde computation itself, so contention
/// stays low: threads compute their retrograde passes in parallel and only
/// briefly lock to publish/read the result.
pub struct RaceTt {
    inner: Mutex<Inner>,
}

struct Inner {
    /// Solved config tables keyed by frozen wall layout. Each value carries the
    /// table plus its last-use LRU stamp and accounted byte size.
    configs: FxHashMap<ConfigKey, Slot>,
    /// Monotonic LRU clock; every touch (hit or insert) stamps the config with
    /// the next tick. Eviction drops the smallest-stamp (oldest) configs first.
    clock: u64,
    /// Sum of every resident config's accounted `bytes` (kept in sync on
    /// insert/evict), compared against `cap_bytes`.
    total_bytes: usize,
    /// Hard budget in bytes (from `QS_RACE_MB`). When `total_bytes` would exceed
    /// this after an insert, LRU configs are evicted whole until back under it.
    cap_bytes: usize,
}

struct Slot {
    table: ConfigTable,
    last_used: u64,
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
    /// Build a race memo with an explicit cap in MiB (test hook for the
    /// tiny-cap eviction-stress gate; `Default` funnels `QS_RACE_MB` here).
    pub fn with_cap_mb(mb: usize) -> RaceTt {
        let cap_bytes = mb.max(1).saturating_mul(1024 * 1024);
        RaceTt {
            inner: Mutex::new(Inner {
                configs: FxHashMap::default(),
                clock: 0,
                total_bytes: 0,
                cap_bytes,
            }),
        }
    }

    /// Total number of cached pawn-state entries across all resident configs
    /// (reporting/tests only). Counts entries, not configs, matching the old
    /// flat-memo semantics so `race_tt_len() > 0` still means "memo populated".
    pub fn len(&self) -> usize {
        let g = self.inner.lock().expect("race memo mutex poisoned");
        g.configs.values().map(|s| s.table.len()).sum()
    }

    /// Whether the memo holds no entries.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Number of resident (whole) configs (reporting/tests).
    pub fn config_count(&self) -> usize {
        self.inner.lock().expect("race memo mutex poisoned").configs.len()
    }

    /// Look up the exact value of race state `s` if its frozen-wall config is
    /// resident. A hit bumps the config to most-recently-used. Returns `None`
    /// (a cache miss) when the config is absent — the caller then solves it.
    fn get(&self, s: &State) -> Option<Value> {
        let mut g = self.inner.lock().expect("race memo mutex poisoned");
        let ck: ConfigKey = (s.h_walls, s.v_walls);
        let tick = {
            g.clock += 1;
            g.clock
        };
        let pk = (s.pawn[0], s.pawn[1], s.turn);
        if let Some(slot) = g.configs.get_mut(&ck) {
            slot.last_used = tick;
            slot.table.get(&pk).copied()
        } else {
            None
        }
    }

    /// Insert a freshly solved config table (built by one retrograde pass over
    /// the frozen maze `ck`). Stamps it most-recently-used, updates the byte
    /// total, and evicts least-recently-used configs WHOLE until back under the
    /// cap. Exactness-neutral: this is a pure cache publish.
    fn insert_config(&self, ck: ConfigKey, table: ConfigTable) {
        let mut g = self.inner.lock().expect("race memo mutex poisoned");
        let bytes = config_bytes(table.len());
        let tick = {
            g.clock += 1;
            g.clock
        };
        // If this config was already resident (a concurrent thread solved it
        // too, or a re-solve after eviction), replace it and adjust the total.
        if let Some(old) = g.configs.insert(
            ck,
            Slot {
                table,
                last_used: tick,
                bytes,
            },
        ) {
            g.total_bytes = g.total_bytes.saturating_sub(old.bytes);
        }
        g.total_bytes = g.total_bytes.saturating_add(bytes);
        Inner::evict_to_cap(&mut g);
    }
}

impl Inner {
    /// Evict least-recently-used configs WHOLE until `total_bytes <= cap_bytes`.
    /// Never evicts below one config (a single config must always fit so a
    /// just-solved query can be served — the cap is advisory, exactness-neutral).
    fn evict_to_cap(g: &mut Inner) {
        while g.total_bytes > g.cap_bytes && g.configs.len() > 1 {
            // Find the config with the smallest last_used stamp (the LRU one).
            let victim = g
                .configs
                .iter()
                .min_by_key(|(_, slot)| slot.last_used)
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
/// so the same memo is shared across the parallel search threads. The mutex is
/// held only for the brief lookup/publish, never across this retrograde pass.
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
    // `index[node] = id`; `succ[id]` = successor node ids; `pred[id]` =
    // predecessor node ids; `succ_remaining[id]` = count of not-yet-WIN
    // successors (the retrograde "all children are wins" counter).
    let mut index: FxHashMap<Node, u32> = FxHashMap::default();
    let mut nodes: Vec<Node> = Vec::new();
    let mut succ: Vec<Vec<u32>> = Vec::new();
    let mut pred: Vec<Vec<u32>> = Vec::new();
    // Per-node seed label (Win/Loss) if it is a terminal/stuck node, else None.
    let mut seed: Vec<Option<Label>> = Vec::new();

    let mut visited = 0u64; // profiling: race nodes touched.

    // Intern a node, allocating its id + parallel rows on first sight.
    let intern = |nodes: &mut Vec<Node>,
                      succ: &mut Vec<Vec<u32>>,
                      pred: &mut Vec<Vec<u32>>,
                      seed: &mut Vec<Option<Label>>,
                      index: &mut FxHashMap<Node, u32>,
                      n: Node|
     -> u32 {
        if let Some(&id) = index.get(&n) {
            return id;
        }
        let id = nodes.len() as u32;
        nodes.push(n);
        succ.push(Vec::new());
        pred.push(Vec::new());
        seed.push(None);
        index.insert(n, id);
        id
    };

    let start = intern(
        &mut nodes,
        &mut succ,
        &mut pred,
        &mut seed,
        &mut index,
        query,
    );

    // BFS/DFS the reachable component, recording successor + predecessor edges.
    let mut stack: Vec<u32> = vec![start];
    // `expanded[id]` guards against re-expanding a node (its edges are built
    // exactly once). A node can be interned (as a successor) before it is
    // expanded, so this is separate from membership in `index`.
    let mut expanded: Vec<bool> = vec![false];
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

        let steps = crate::movegen::legal_steps(b, &work);
        if steps.is_empty() {
            // Non-terminal but stuck: a Loss for the mover (matches the solver —
            // a no-step node never improves past the initial Loss).
            seed[id as usize] = Some(Label::Loss);
            continue;
        }

        // Build successor edges. A step moves the mover's pawn and flips turn.
        for dest in steps {
            let mut child = node;
            child.pawn[t as usize] = dest;
            child.turn = 1 - t;
            let cid = intern(
                &mut nodes,
                &mut succ,
                &mut pred,
                &mut seed,
                &mut index,
                child,
            );
            // `expanded` must stay parallel to `nodes`.
            if cid as usize >= expanded.len() {
                expanded.resize(nodes.len(), false);
            }
            succ[id as usize].push(cid);
            pred[cid as usize].push(id);
            if !expanded[cid as usize] {
                stack.push(cid);
            }
        }
    }

    let n = nodes.len();
    // `succ_remaining[id]` starts at the successor count; each time a successor
    // is finalized WIN we decrement it. Reaching 0 means ALL successors are WIN
    // -> the node is a LOSS for its mover.
    let mut succ_remaining: Vec<u32> = succ.iter().map(|v| v.len() as u32).collect();

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
        match label[id].expect("queued node must be labeled") {
            Label::Loss => {
                // Predecessors become WIN.
                let preds = std::mem::take(&mut pred[id]);
                for &p in &preds {
                    let p = p as usize;
                    if label[p].is_none() {
                        label[p] = Some(Label::Win);
                        queue.push(p as u32);
                    }
                }
                pred[id] = preds;
            }
            Label::Win => {
                // Each predecessor loses one not-yet-WIN successor.
                let preds = std::mem::take(&mut pred[id]);
                for &p in &preds {
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
                pred[id] = preds;
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
