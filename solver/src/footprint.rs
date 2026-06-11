//! THEOREM 1 (Wall-Insertion Invariance), Win direction — wall-relevance
//! footprint ("mustplay") extraction and caching.
//! Spec: `docs/superpowers/solver-pruning-theorems.md` §A (final,
//! falsification-validated form, including the §A.5 amendment).
//!
//! Given a position `u0 = flip(s)` (the certificate owner `Y` to move) whose
//! TRUE value is a Win for `Y`, this module builds an explicit Definition-1
//! certificate closure — a designated winning move `σ(u)` at every Y-node and
//! ALL legal replies (walls included) at every Z-node — and compiles its
//! Definition-2 footprint `R(P)` into two u64 wall-anchor masks (forbidden
//! H / V anchors). By Theorem 1 + Corollary 1, any Z wall whose anchor is
//! OUTSIDE both masks cannot change the value: placed from `s` (Z to move) it
//! yields exactly `Win` for Y (= `Loss` for Z), so the alpha-beta refutation
//! loop may SKIP it. Win-direction prunes are EXACT child values — skipping
//! never changes the max (§A.2 integration rule; safe unconditionally).
//!
//! The FULL footprint is mandatory (every ablation produced violations):
//!   (A) Y-move edges: the unblocked edges certifying `σ(u)`'s legality
//!       (step: 1 edge; straight/diagonal jump: mover→opp and opp→landing);
//!   (B) Z anti-growth (the jump trap): at every Z-node with the pawns
//!       orthogonally adjacent and the straight-jump path open, the landing
//!       edge — blocking it is the UNIQUE mechanism by which a wall insertion
//!       can ADD a pawn move (straight→diagonal conversion);
//!   (C) anti-pre-block: `Conflict(x)` for every Y wall move `σ(u) = x`;
//!   (D) legality witnesses: one goal path per player in `apply(u, x)` for
//!       every such placement (both exist since `x` was legal).
//!
//! RANK (§A.5 amendment — the original TT-hot `dtw` rank is UNSOUND under the
//! depth-folded TT and is NOT used): this module uses the validated
//! BUILD-THEN-VERIFY scheme (§A.5 (b)) uniformly — build the closure with a
//! greedy win-preserving `σ` (steps before walls, nearer-goal steps first),
//! then verify well-foundedness post-hoc by checking the closure graph is
//! acyclic (Kahn); a valid Definition-1 rank exists iff it is (the removal
//! order IS the rank). On failure, locally re-pick `σ` at cycle-participating
//! Y-nodes (K-boosting) and re-verify, up to a small iteration cap. Abort =
//! sound no-prune fallback. Race regions (`walls_left == [0, 0]`) flow through
//! the same scan: their certificate queries resolve via the engine's EXACT
//! race short-circuit (retrograde-labeled, memoized per frozen config), and
//! the same global acyclicity check validates their rank.
//!
//! Mirror discipline (G5): extraction always runs in the REAL orientation of
//! `flip(s)`; only the CACHE keys on the canonical (mirror-folded)
//! representative, and cached masks are bit-permuted back when the canonical
//! representative is the mirrored state. TT-bound discipline (G6): every
//! certificate value used here is either a decisive full-window `ab` result
//! (true forced value) or an exact race-retrograde label — never a TT bound
//! read out as a strategy.

use crate::board::Board;
use crate::solver::Value;
use crate::state::{Move, State};
use rustc_hash::FxHashMap;
use std::sync::Mutex;

/// Closure-node budget per extraction attempt (doc §A.5: `N_nodes ≈ 10⁴`;
/// 2x headroom because the verified T5.1 falsification target needs ~10.6K
/// closure nodes). Exceeding it aborts the attempt — a sound no-prune
/// fallback.
const FP_NODE_BUDGET: usize = 20_000;

/// Budget of certificate child solves (full-window `ab` calls) per attempt.
/// Purely a cost guard; exceeding it aborts the attempt (no prune).
const FP_SOLVE_BUDGET: usize = 50_000;

/// Maximum build-then-verify repair rounds (K-boost iterations). The
/// falsification phase converged in ≤ 4 on every tested instance; we allow a
/// little slack. Exceeding it aborts (no prune).
const FP_MAX_ROUNDS: usize = 6;

/// Hard cap on cached footprint entries (cost guard only; inserts stop at the
/// cap and extraction simply re-runs on demand — exactness-neutral).
const FP_CACHE_CAP: usize = 1 << 22;

/// One cached per-position outcome: a successful certificate's anchor masks
/// (valid at EVERY depth — the certificate is a true-value object), or the
/// deepest remaining-depth at which an attempt failed (failures can be
/// depth-relative — e.g. the `flip(s)` win was not provable within the
/// remaining depth — so a sufficiently deeper query retries).
#[derive(Clone, Copy)]
pub(crate) enum FpEntry {
    /// Forbidden (H, V) anchor masks in the CANONICAL orientation.
    Masks(u64, u64),
    /// An attempt at this remaining depth failed; retry only when meaningfully
    /// deeper (see `FP_RETRY_MARGIN` in `solver.rs`).
    Failed(u32),
}

/// Thread-safe footprint cache, shared across the lazy-SMP workers exactly
/// like the race memo. Keyed by the `pack_u128` of the CANONICAL `flip(s)`;
/// values store canonical-orientation masks (mirrored back by the caller when
/// the canonical representative is the mirror image — G5).
///
/// EXACTNESS: a cached `Masks` entry is a compiled true-Win certificate for
/// its position — a pure function of the position, valid at every depth and
/// for every thread. `Failed` entries only suppress re-attempts (cost), never
/// license a prune. Capacity capping only disables further inserts.
pub struct FootprintCache {
    inner: Mutex<FxHashMap<u128, FpEntry>>,
}

impl Default for FootprintCache {
    fn default() -> Self {
        FootprintCache {
            inner: Mutex::new(FxHashMap::default()),
        }
    }
}

impl FootprintCache {
    pub(crate) fn get(&self, key: u128) -> Option<FpEntry> {
        self.inner
            .lock()
            .expect("footprint cache mutex poisoned")
            .get(&key)
            .copied()
    }

    pub(crate) fn put(&self, key: u128, entry: FpEntry) {
        let mut g = self.inner.lock().expect("footprint cache mutex poisoned");
        if g.len() >= FP_CACHE_CAP && !g.contains_key(&key) {
            return; // cost cap; exactness-neutral (only forces re-attempts)
        }
        match (g.get(&key), &entry) {
            // Never let a depth-relative failure clobber a valid certificate.
            (Some(FpEntry::Masks(_, _)), FpEntry::Failed(_)) => {}
            // Keep the DEEPEST failure depth (governs the retry heuristic).
            (Some(FpEntry::Failed(old)), FpEntry::Failed(new)) => {
                let d = (*old).max(*new);
                g.insert(key, FpEntry::Failed(d));
            }
            _ => {
                g.insert(key, entry);
            }
        }
    }

    /// Number of cached positions (reporting only).
    pub fn len(&self) -> usize {
        self.inner.lock().expect("footprint cache mutex poisoned").len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

// ---------------------------------------------------------------------------
// Geometry helpers: edges -> blocking anchors, conflict sets, witness paths.
// ---------------------------------------------------------------------------

const DIRS: [(i16, i16); 4] = [(0, 1), (0, -1), (1, 0), (-1, 0)];

#[inline]
fn on_board(b: &Board, c: i16, r: i16) -> bool {
    c >= 0 && r >= 0 && c < b.w as i16 && r < b.h as i16
}

#[inline]
fn anchor_in_range(b: &Board, wc: i16, wr: i16) -> bool {
    wc >= 0 && wr >= 0 && wc < b.w as i16 - 1 && wr < b.h as i16 - 1
}

/// OR the ≤ 2 anchors that can block the grid edge between orthogonally
/// adjacent cells `a`/`b` into the (H, V) masks. North edge `(c,r)-(c,r+1)`:
/// `H(c,r)`, `H(c-1,r)`. East edge `(c,r)-(c+1,r)`: `V(c,r)`, `V(c,r-1)`.
/// (Exactly the read set of `Board::step_blocked`.) Returns false if the
/// cells are not orthogonally adjacent (defensive — aborts the extraction).
fn add_edge_anchors(b: &Board, a: u8, d: u8, h_mask: &mut u64, v_mask: &mut u64) -> bool {
    let (ac, ar) = b.cr(a);
    let (dc, dr) = b.cr(d);
    let (ac, ar, dc, dr) = (ac as i16, ar as i16, dc as i16, dr as i16);
    let mut put_h = |wc: i16, wr: i16| {
        if anchor_in_range(b, wc, wr) {
            *h_mask |= 1u64 << b.hbit(wc as u8, wr as u8);
        }
    };
    if ac == dc && (ar - dr).abs() == 1 {
        let r = ar.min(dr); // north edge at (ac, r)
        put_h(ac, r);
        put_h(ac - 1, r);
        return true;
    }
    let mut put_v = |wc: i16, wr: i16| {
        if anchor_in_range(b, wc, wr) {
            *v_mask |= 1u64 << b.vbit(wc as u8, wr as u8);
        }
    };
    if ar == dr && (ac - dc).abs() == 1 {
        let c = ac.min(dc); // east edge at (c, ar)
        put_v(c, ar);
        put_v(c, ar - 1);
        return true;
    }
    false
}

/// Component (A): the edges whose UNBLOCKED status certifies the legality of
/// the pawn move `mover -> dest` (with the opponent at `opp`):
///   * plain step: the single edge mover–dest;
///   * straight jump: mover–opp and opp–dest;
///   * diagonal jump: mover–opp and opp–dest (the blocked straight edge that
///     ENABLES the diagonal is deliberately NOT protected — blocking is
///     monotone under insertion).
///
/// Returns false on an unclassifiable geometry (defensive abort).
fn add_step_move_edges(
    b: &Board,
    mover: u8,
    opp: u8,
    dest: u8,
    h_mask: &mut u64,
    v_mask: &mut u64,
) -> bool {
    let (mc, mr) = b.cr(mover);
    let (dc, dr) = b.cr(dest);
    let (mc, mr, dc, dr) = (mc as i16, mr as i16, dc as i16, dr as i16);
    let (oc, or) = b.cr(opp);
    let (oc, or) = (oc as i16, or as i16);
    let man = (mc - dc).abs() + (mr - dr).abs();
    if man == 1 {
        // Plain step.
        return add_edge_anchors(b, mover, dest, h_mask, v_mask);
    }
    // Jump: the opponent must sit orthogonally adjacent to BOTH cells.
    if (mc - oc).abs() + (mr - or).abs() != 1 || (dc - oc).abs() + (dr - or).abs() != 1 {
        return false;
    }
    add_edge_anchors(b, mover, opp, h_mask, v_mask)
        && add_edge_anchors(b, opp, dest, h_mask, v_mask)
}

/// Component (C): `Conflict(x)` — the anchor slots whose occupation would
/// make the wall `x` ILLEGAL to place later (exactly `movegen::overlaps`):
/// `Conflict(H@(c,r)) = {H(c−1,r), H(c,r), H(c+1,r), V(c,r)}`,
/// `Conflict(V@(c,r)) = {V(c,r−1), V(c,r), V(c,r+1), H(c,r)}`, clipped.
fn add_conflict_slots(b: &Board, wc: u8, wr: u8, horiz: bool, h_mask: &mut u64, v_mask: &mut u64) {
    let (c, r) = (wc as i16, wr as i16);
    let put = |mask: &mut u64, wc: i16, wr: i16| {
        if anchor_in_range(b, wc, wr) {
            *mask |= 1u64 << ((wr as u8) * (b.w - 1) + wc as u8);
        }
    };
    if horiz {
        put(h_mask, c - 1, r);
        put(h_mask, c, r);
        put(h_mask, c + 1, r);
        put(v_mask, c, r);
    } else {
        put(v_mask, c, r - 1);
        put(v_mask, c, r);
        put(v_mask, c, r + 1);
        put(h_mask, c, r);
    }
}

/// Component (D): one shortest goal path for `player` in `s` (BFS with parent
/// pointers, same blocking predicate as the engine), all its edges ORed into
/// the masks. Returns false when no path exists (cannot happen after a LEGAL
/// placement — defensive abort).
fn add_goal_path_edges(
    b: &Board,
    s: &State,
    player: u8,
    h_mask: &mut u64,
    v_mask: &mut u64,
) -> bool {
    let goal = b.goal_row(player);
    let start = s.pawn[player as usize];
    let (_, sr) = b.cr(start);
    if sr == goal {
        return true; // already on the goal row: empty path
    }
    let ncells = (b.w as usize) * (b.h as usize);
    let mut parent: Vec<u8> = vec![u8::MAX; ncells];
    let mut seen: u64 = 1u64 << start;
    let mut frontier: Vec<u8> = vec![start];
    let mut hit: Option<u8> = None;
    'bfs: while !frontier.is_empty() {
        let mut next: Vec<u8> = Vec::new();
        for &cell in &frontier {
            let (c, r) = b.cr(cell);
            let (c, r) = (c as i16, r as i16);
            for &(dc, dr) in &DIRS {
                let (nc, nr) = (c + dc, r + dr);
                if !on_board(b, nc, nr) || b.step_blocked(s, c, r, dc, dr) {
                    continue;
                }
                let nidx = b.idx(nc as u8, nr as u8);
                if seen & (1u64 << nidx) != 0 {
                    continue;
                }
                seen |= 1u64 << nidx;
                parent[nidx as usize] = cell;
                if nr as u8 == goal {
                    hit = Some(nidx);
                    break 'bfs;
                }
                next.push(nidx);
            }
        }
        frontier = next;
    }
    let Some(mut cur) = hit else { return false };
    while cur != start {
        let p = parent[cur as usize];
        debug_assert_ne!(p, u8::MAX);
        if !add_edge_anchors(b, p, cur, h_mask, v_mask) {
            return false;
        }
        cur = p;
    }
    true
}

/// Component (B): at a Z-node, when the pawns are orthogonally adjacent
/// (Z at `m`, Y at `o = m + d`), the straight-jump edge `m–o` is open, the
/// landing cell `o + d` is on-board and the landing edge `o–(o+d)` is open,
/// protect the LANDING edge: blocking it is the unique additive transition
/// (straight jump → diagonal jumps), the only way an inserted wall can hand
/// Z a NEW move.
fn add_jump_trap_edges(b: &Board, s: &State, z: u8, h_mask: &mut u64, v_mask: &mut u64) {
    let m = s.pawn[z as usize];
    let o = s.pawn[(1 - z) as usize];
    let (mc, mr) = b.cr(m);
    let (oc, or) = b.cr(o);
    let (mc, mr, oc, or) = (mc as i16, mr as i16, oc as i16, or as i16);
    let (dc, dr) = (oc - mc, or - mr);
    if dc.abs() + dr.abs() != 1 {
        return; // not orthogonally adjacent
    }
    if b.step_blocked(s, mc, mr, dc, dr) {
        return; // approach edge already blocked: no straight jump to convert
    }
    let (lc, lr) = (oc + dc, or + dr);
    if !on_board(b, lc, lr) || b.step_blocked(s, oc, or, dc, dr) {
        return; // straight landing off-board/blocked: diagonals already legal
    }
    let land = b.idx(lc as u8, lr as u8);
    let ok = add_edge_anchors(b, o, land, h_mask, v_mask);
    debug_assert!(ok);
}

// ---------------------------------------------------------------------------
// Certificate closure: build-then-verify with K-boost repair (§A.5 scheme (b)).
// ---------------------------------------------------------------------------

/// Shared mutable context across the repair rounds of one extraction attempt.
/// `solve` is the worker-supplied certificate solver: an alpha-beta search of
/// the given state at the given remaining depth over the given `(alpha,
/// beta)` window. The extraction only ever issues the two cheap one-sided
/// queries below, and only trusts the lattice-EXTREME results, which fail-soft
/// alpha-beta proves exactly (G6 — never a TT bound read as a strategy):
///   * window `(Loss, Draw)` returning `Loss` ⟺ `V_d(child) = Loss` — a TRUE
///     forced loss (decisive results are depth-uniform true values);
///   * window `(Draw, Win)` returning `Win`  ⟺ `V_d(u0) = Win` — a TRUE win.
///
/// `solve` returns `None` on the lazy-SMP abort; the extraction unwinds and
/// nothing is cached.
struct Ctx<'b, F> {
    b: &'b Board,
    solve: F,
    /// Proven-true-Loss memo (depth-uniform — a decisive value never flips).
    loss: FxHashMap<State, ()>,
    /// Remaining certificate-solve budget across all rounds.
    solves_left: usize,
    /// Lazy-SMP abort observed (propagated out; nothing is cached).
    aborted: bool,
}

impl<F: FnMut(&State, u32, Value, Value) -> Option<Value>> Ctx<'_, F> {
    /// Whether `child` is a PROVEN true Loss for its side to move within
    /// `depth` plies (the σ-witness test). `Some(false)` = not proven at this
    /// budget (Win, Draw, or undecided — not usable as σ, nothing more);
    /// `None` = the attempt must abort (lazy-SMP stop or solve budget).
    fn child_is_loss(&mut self, child: &State, depth: u32) -> Option<bool> {
        if self.loss.contains_key(child) {
            return Some(true);
        }
        if self.solves_left == 0 {
            return None;
        }
        self.solves_left -= 1;
        match (self.solve)(child, depth, Value::Loss, Value::Draw) {
            None => {
                self.aborted = true;
                None
            }
            Some(Value::Loss) => {
                self.loss.insert(*child, ());
                Some(true)
            }
            Some(_) => Some(false),
        }
    }

    /// Whether `u0` is a PROVEN true Win for its side to move within `depth`
    /// plies (the certificate root test). `None` = abort.
    fn root_is_win(&mut self, u0: &State, depth: u32) -> Option<bool> {
        if self.solves_left == 0 {
            return None;
        }
        self.solves_left -= 1;
        match (self.solve)(u0, depth, Value::Draw, Value::Win) {
            None => {
                self.aborted = true;
                None
            }
            Some(v) => Some(v == Value::Win),
        }
    }

}

/// One built closure (per repair round).
struct Closure {
    states: Vec<State>,
    index: FxHashMap<State, u32>,
    /// Out-edges: the single `σ` edge at Y-nodes, the full legal-move fan at
    /// Z-nodes, nothing at terminals/stuck leaves.
    edges: Vec<Vec<u32>>,
    /// Y-nodes (certificate owner to move).
    is_y: Vec<bool>,
    /// Per WALL-PHASE Y-node: how many winning moves the σ-scan had found
    /// when it stopped (used by K-boost to know whether an alternative may
    /// exist), and whether the scan already exhausted the candidate list.
    exhausted: Vec<bool>,
    h_mask: u64,
    v_mask: u64,
}

/// Build the certificate closure from `u0` (Y to move, TRUE value Win),
/// choosing `σ` at wall-phase Y-nodes by a greedy scan over
/// distance-preferred candidates with per-state pick-index `overrides`
/// (K-boost state); race regions flow through the same scan via the engine's
/// exact race short-circuit. Accumulates the FULL Definition-2 footprint
/// (components A-D). Returns `None` on any budget/certificate failure (sound
/// no-prune).
fn build_closure<F: FnMut(&State, u32, Value, Value) -> Option<Value>>(
    ctx: &mut Ctx<'_, F>,
    u0: &State,
    depth: u32,
    overrides: &FxHashMap<State, usize>,
) -> Option<Closure> {
    let b = ctx.b;
    let y = u0.turn;
    let mut cl = Closure {
        states: Vec::new(),
        index: FxHashMap::default(),
        edges: Vec::new(),
        is_y: Vec::new(),
        exhausted: Vec::new(),
        h_mask: 0,
        v_mask: 0,
    };
    // Worklist of (node id, remaining-depth budget for its child solves).
    let mut work: Vec<(u32, u32)> = Vec::new();
    let intern = |cl: &mut Closure, s: &State| -> (u32, bool) {
        if let Some(&id) = cl.index.get(s) {
            return (id, false);
        }
        let id = cl.states.len() as u32;
        cl.states.push(*s);
        cl.index.insert(*s, id);
        cl.edges.push(Vec::new());
        cl.is_y.push(s.turn == y);
        cl.exhausted.push(false);
        (id, true)
    };
    let (root, _) = intern(&mut cl, u0);
    work.push((root, depth));

    while let Some((id, d)) = work.pop() {
        if cl.states.len() > FP_NODE_BUDGET {
            return None; // budget abort: sound no-prune fallback
        }
        let u = cl.states[id as usize];
        // Terminals: every certificate terminal must be a Y win.
        if let Some(p) = b.winner(&u) {
            if p != y {
                return None; // certificate broken (defensive; cannot happen)
            }
            continue;
        }
        if u.turn == y {
            // ---- Y-node: designate a winning move σ(u) by a greedy scan ----
            // over distance-preferred candidates, with the K-boost override
            // index for this state (default 0 = first winning move found).
            // Steps are tried before walls and nearer-goal steps first (doc:
            // "prefer steps over walls, shortest wins => small R"). Race
            // nodes (`walls_left == [0, 0]`) go through the SAME scan — their
            // child queries hit the engine's exact race short-circuit (memoized
            // per frozen config in the shared race memo, so they are near-free)
            // and the global build-then-verify rank check covers them (doc
            // §A.5 scheme (b)) exactly like the wall phase.
            if d == 0 {
                return None; // cannot certify children within the budget
            }
            let mut candidates: Vec<(i64, Move)> = Vec::new();
            for (i, m) in crate::movegen::legal_moves(b, &u).into_iter().enumerate() {
                let key = match m {
                    Move::Step(dest) => {
                        let mut t = u;
                        t.pawn[y as usize] = dest;
                        let dist = b.dist_to_goal(&t, y).map_or(i64::MAX / 2, |x| x as i64);
                        dist * 256 + i as i64
                    }
                    // Walls strictly after every step, in movegen order.
                    Move::Wall { .. } => (i64::MAX / 4) + i as i64,
                };
                candidates.push((key, m));
            }
            candidates.sort_by_key(|&(k, _)| k);
            let want = overrides.get(&u).copied().unwrap_or(0);
            let mut found: Option<Move> = None; // the `want`-th winning move
            let mut last_win: Option<Move> = None;
            let mut wins_seen = 0usize;
            for &(_, m) in &candidates {
                let child = crate::movegen::apply(b, &u, m);
                match ctx.child_is_loss(&child, d - 1) {
                    Some(true) => {
                        last_win = Some(m);
                        if wins_seen == want {
                            found = Some(m);
                            break;
                        }
                        wins_seen += 1;
                    }
                    Some(false) => {}
                    None => return None, // abort (lazy-SMP stop) or budget
                }
            }
            let sigma = match found {
                Some(m) => m,
                None => {
                    // Fewer than `want+1` winning moves exist: fall back to
                    // the deepest alternative and mark the node exhausted so
                    // K-boost stops advancing it.
                    cl.exhausted[id as usize] = true;
                    last_win? // no winning move at all: no certificate
                }
            };
            // Footprint components for σ(u).
            match sigma {
                Move::Step(dest) => {
                    if !add_step_move_edges(
                        b,
                        u.pawn[y as usize],
                        u.pawn[(1 - y) as usize],
                        dest,
                        &mut cl.h_mask,
                        &mut cl.v_mask,
                    ) {
                        return None;
                    }
                }
                Move::Wall { wc, wr, horiz } => {
                    // (C) anti-pre-block: the wall's conflict slots.
                    add_conflict_slots(b, wc, wr, horiz, &mut cl.h_mask, &mut cl.v_mask);
                    // (D) legality witnesses: one goal path per player AFTER
                    // the placement (both exist — the wall was legal).
                    let t = crate::movegen::apply(b, &u, sigma);
                    if !add_goal_path_edges(b, &t, 0, &mut cl.h_mask, &mut cl.v_mask)
                        || !add_goal_path_edges(b, &t, 1, &mut cl.h_mask, &mut cl.v_mask)
                    {
                        return None;
                    }
                }
            }
            let child = crate::movegen::apply(b, &u, sigma);
            let (cid, fresh) = intern(&mut cl, &child);
            cl.edges[id as usize].push(cid);
            if fresh {
                work.push((cid, child_depth(&child, d)));
            }
        } else {
            // ---- Z-node: (B) jump-trap edge, then EVERY legal reply. ----
            add_jump_trap_edges(b, &u, u.turn, &mut cl.h_mask, &mut cl.v_mask);
            let moves = crate::movegen::legal_moves(b, &u);
            // Z stuck (non-terminal, no moves) is a Y-win leaf.
            for m in moves {
                let child = crate::movegen::apply(b, &u, m);
                let (cid, fresh) = intern(&mut cl, &child);
                cl.edges[id as usize].push(cid);
                if fresh {
                    work.push((cid, child_depth(&child, d)));
                }
            }
        }
    }
    Some(cl)
}

/// Remaining-depth budget for a closure child. Race states (`walls_left ==
/// [0, 0]`) get a small FLOOR instead of the decrement: their certificate
/// queries resolve through the engine's exact, depth-free race short-circuit
/// (any query depth >= 1), so the depth budget must never starve a long
/// pure-race tail. The budget is pure cost bookkeeping — never soundness.
#[inline]
fn child_depth(child: &State, d: u32) -> u32 {
    if child.walls_left == [0, 0] {
        2
    } else {
        d.saturating_sub(1)
    }
}

/// Kahn's algorithm from sinks: returns the set of nodes NOT removable, i.e.
/// the nodes that lie on or reach a cycle (empty ⟺ the closure graph is a
/// DAG ⟺ a valid Definition-1 rank exists; the removal order IS the rank).
fn kahn_leftover(cl: &Closure) -> Vec<bool> {
    let n = cl.states.len();
    let mut outdeg: Vec<u32> = cl.edges.iter().map(|e| e.len() as u32).collect();
    let mut rev: Vec<Vec<u32>> = vec![Vec::new(); n];
    for (u, es) in cl.edges.iter().enumerate() {
        for &v in es {
            rev[v as usize].push(u as u32);
        }
    }
    let mut queue: Vec<u32> = (0..n as u32).filter(|&i| outdeg[i as usize] == 0).collect();
    let mut removed = vec![false; n];
    let mut qi = 0;
    while qi < queue.len() {
        let v = queue[qi] as usize;
        qi += 1;
        removed[v] = true;
        for &u in &rev[v] {
            let u = u as usize;
            outdeg[u] -= 1;
            if outdeg[u] == 0 {
                queue.push(u as u32);
            }
        }
    }
    removed.iter().map(|&r| !r).collect()
}

/// Iterative Tarjan SCC restricted to the `alive` subgraph; returns for each
/// node whether it belongs to a NON-TRIVIAL SCC (size ≥ 2; self-loops cannot
/// occur — every move changes the state). These are the cycle participants
/// K-boost re-picks σ at.
fn cyclic_nodes(cl: &Closure, alive: &[bool]) -> Vec<bool> {
    let n = cl.states.len();
    let mut index_of: Vec<i64> = vec![-1; n];
    let mut low: Vec<u32> = vec![0; n];
    let mut on_stack = vec![false; n];
    let mut stack: Vec<u32> = Vec::new();
    let mut next_index: u32 = 0;
    let mut in_cycle = vec![false; n];
    // Explicit DFS stack: (node, child-iterator position).
    let mut call: Vec<(u32, usize)> = Vec::new();
    for s in 0..n {
        if !alive[s] || index_of[s] >= 0 {
            continue;
        }
        call.push((s as u32, 0));
        index_of[s] = next_index as i64;
        low[s] = next_index;
        next_index += 1;
        stack.push(s as u32);
        on_stack[s] = true;
        while !call.is_empty() {
            let (u, ci) = {
                let top = call.last().expect("checked non-empty");
                (top.0 as usize, top.1)
            };
            // Scan u's remaining out-edges for the next unvisited live child.
            let mut ci2 = ci;
            let mut descend: Option<usize> = None;
            while ci2 < cl.edges[u].len() {
                let v = cl.edges[u][ci2] as usize;
                ci2 += 1;
                if !alive[v] {
                    continue;
                }
                if index_of[v] < 0 {
                    descend = Some(v);
                    break;
                } else if on_stack[v] {
                    low[u] = low[u].min(index_of[v] as u32);
                }
            }
            call.last_mut().expect("checked non-empty").1 = ci2;
            if let Some(v) = descend {
                index_of[v] = next_index as i64;
                low[v] = next_index;
                next_index += 1;
                stack.push(v as u32);
                on_stack[v] = true;
                call.push((v as u32, 0));
                continue;
            }
            // u finished: pop, fold lowlink into parent, emit SCC at root.
            call.pop();
            if let Some(&(p, _)) = call.last() {
                let p = p as usize;
                low[p] = low[p].min(low[u]);
            }
            if low[u] == index_of[u] as u32 {
                let mut comp: Vec<u32> = Vec::new();
                loop {
                    let w = stack.pop().expect("Tarjan stack underflow");
                    on_stack[w as usize] = false;
                    comp.push(w);
                    if w as usize == u {
                        break;
                    }
                }
                if comp.len() >= 2 {
                    for w in comp {
                        in_cycle[w as usize] = true;
                    }
                }
            }
        }
    }
    in_cycle
}

/// Extract the Win-direction footprint of `u0 = flip(s)` (Y to move): verify
/// `V(u0) = Win` (true value, full-window decisive), build the certificate
/// closure, verify the rank (build-then-verify with K-boost repair), and
/// compile the Definition-2 footprint into forbidden (H, V) anchor masks in
/// the REAL orientation of `u0`. `None` = no certificate at this budget
/// (sound: the caller prunes nothing).
pub(crate) fn extract_win_footprint<F: FnMut(&State, u32, Value, Value) -> Option<Value>>(
    b: &Board,
    u0: &State,
    depth: u32,
    solve: F,
) -> Option<(u64, u64)> {
    debug_assert!(b.winner(u0).is_none());
    let depth = if u0.walls_left == [0, 0] { 2 } else { depth };
    let mut ctx = Ctx {
        b,
        solve,
        loss: FxHashMap::default(),
        solves_left: FP_SOLVE_BUDGET,
        aborted: false,
    };
    // Root certificate value: must be a TRUE Win for Y (a decisive alpha-beta
    // proof — for a race root this resolves through the exact race
    // short-circuit; G6 — never a TT bound read as a strategy).
    if ctx.root_is_win(u0, depth) != Some(true) {
        return None;
    }

    let mut overrides: FxHashMap<State, usize> = FxHashMap::default();
    for _round in 0..FP_MAX_ROUNDS {
        let cl = build_closure(&mut ctx, u0, depth, &overrides)?;
        let leftover = kahn_leftover(&cl);
        if leftover.iter().all(|&x| !x) {
            // Acyclic: a valid rank exists (the Kahn removal order). Compile.
            return Some((cl.h_mask, cl.v_mask));
        }
        // K-boost: advance σ at every non-exhausted Y-node participating in a
        // cycle. If none can advance, the repair is stuck: abort (no prune).
        let in_cycle = cyclic_nodes(&cl, &leftover);
        let mut boosted = false;
        for (id, &cyc) in in_cycle.iter().enumerate() {
            if cyc && cl.is_y[id] && !cl.exhausted[id] {
                *overrides.entry(cl.states[id]).or_insert(0) += 1;
                boosted = true;
            }
        }
        if !boosted {
            return None;
        }
    }
    None
}
