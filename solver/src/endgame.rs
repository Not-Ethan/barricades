//! Endgame tablebase slice: the wall-less *race*.
//!
//! When both players have exhausted their walls, the game reduces to a pure
//! pawn race over steps only. With no walls left to place, neither path length
//! can ever increase, so the race always resolves to a Win or Loss for the side
//! to move — it can never be a Draw. This is the first endgame-tablebase slice;
//! a `k`-wall generalization is deferred to Phase 1.
//!
//! ## Module invariant: a wall-less race is never a true draw
//!
//! Walls are frozen, so each player's shortest-path distance to goal is fixed
//! and can only DECREASE as they advance — and `legal_walls` guarantees every
//! reachable position keeps both pawns connected to their goal. Therefore one
//! player always has a finite, strictly-progressing (acyclic) winning line:
//! racing straight along a shortest path, whose length never grows. By optimal
//! play exactly one side reaches its goal first, so the game value of every
//! reachable wall-less race is Win or Loss — never Draw. A `Draw` escaping this
//! module is therefore impossible; if one ever did it would be a bug, so we
//! `panic!` instead of returning it (belt-and-suspenders).
//!
//! ## Exactness under a depth bound — Win/Loss are bound-independent
//!
//! The wall-less step graph is cyclic (pawns can shuffle), so an unbounded DFS
//! would not terminate. We terminate it with a depth bound. The crucial fact is
//! that on the `Loss < Draw < Win` lattice the depth floor can only ever taint a
//! result *up to `Draw`* — it can never flip a true `Win` or `Loss`:
//!
//!   * The floor (depth 0) returns `Draw`.
//!   * A node resolves to `Win` only via a child whose value is `Loss`
//!     (`negate(Loss) = Win`). A floor-truncated node returns `Draw`, never
//!     `Loss`, so a `Win` is never certified by a truncated line.
//!   * A node resolves to `Loss` only when EVERY child has value `Win`
//!     (`negate(Win) = Loss` is the max). A floor-truncated child returns `Draw`,
//!     and `negate(Draw) = Draw > Loss`, which would lift the node off `Loss`.
//!     So a `Loss` likewise can have no truncated child.
//!
//! Hence **any `Win` or `Loss` returned by the search is the exact, bound-
//! independent game value**, and the only thing a too-small bound can produce is
//! a spurious `Draw`. There is no tight analytic bound on the proof depth (a
//! twisty frozen maze can force a long non-repeating proof line — a fixed bound
//! that was too small here silently returned `Draw` under the old code). So we
//! **iteratively deepen** the bound (doubling) until the top-level result is a
//! definitive `Win`/`Loss`; the first such result is exact and we return it.
//! Clean Win/Loss memo entries are depth-independent, so the `tt` is reused
//! across deepening iterations — no clean work is repeated. A hard ceiling of
//! `2*(w*h)^2` (one more than the longest possible non-repeating proof line over
//! the `(w*h)^2 * 2` race states) bounds deepening; reaching it without a
//! definitive result is impossible for a reachable race, so `race_value`
//! `panic!`s — the loud guard that makes a silent wrong `Draw` impossible.
//!
//! ## Persistent, exact memo
//!
//! The race value of a wall-less `State` (pawns + frozen walls + turn, with
//! `walls_left == [0, 0]`) is a **pure, context-free function of that State**.
//! That makes a memo keyed on the bare `State` and persisted across every race
//! leaf of a `solve()` call sound — provided we only ever store EXACT values.
//!
//! We store only "depth-clean" resolved Win/Loss: a value is depth-clean iff its
//! proof never bottomed out at the depth-0 floor anywhere in its subtree. As
//! argued above every `Win`/`Loss` is automatically depth-clean (the floor only
//! yields `Draw`), so in practice every reachable race resolves cleanly and is
//! memoized. A depth-clean value used no depth information, so it is identical at
//! any depth — hence safe to key on the bare `State` and reuse across leaves.
//! Values that touched the floor (necessarily `Draw`) are never persisted.

use crate::solver::Value;
use crate::state::State;
use crate::board::Board;
use rustc_hash::FxHashMap;

/// Persistent, exact race memo keyed on the bare (walls-frozen) `State`.
/// Every stored value is the position's EXACT, depth-independent game-theoretic
/// value, so it is sound to reuse across any race leaf within a `solve()` call.
pub type RaceTt = FxHashMap<State, Value>;

/// Absolute upper bound on the negamax resolution depth of a wall-less race.
///
/// A definitive Win/Loss is proven along a *simple* (non-repeating) line of race
/// states; once a line is longer than the number of distinct race states it must
/// repeat one, so no proof line need exceed `|reachable race states| - 1` plies.
/// The race state space is `pawn0 x pawn1 x turn`, i.e. at most `(w*h)^2 * 2`
/// states. We use `2 * (w*h)^2` as the hard ceiling: every reachable wall-less
/// race resolves to Win/Loss at or below it. Iterative deepening (below) almost
/// always resolves far sooner; this ceiling is only the loud-failure backstop.
#[inline]
fn race_depth_ceiling(b: &Board) -> u32 {
    let cells = b.w as u32 * b.h as u32;
    2 * cells * cells
}

/// Exact game value of a wall-less position for the side to move, paired with
/// the number of race nodes visited (for profiling; the value is unaffected).
///
/// **Iterative-deepening** full-window negamax over **steps only** (walls are
/// exhausted) with a PERSISTENT `State`-keyed memo of exact values (`tt`).
///
/// On the `Loss < Draw < Win` lattice the depth floor can only ever taint a
/// result up to `Draw`; it can NEVER flip a true `Win`/`Loss` (a node is `Win`
/// only via a `Loss` child, and a floored node returns `Draw`, never `Loss`; a
/// node is `Loss` only when every child is `Win`, and a floored `Draw` child
/// would lift it off `Loss`). Hence any `Win`/`Loss` the search returns is the
/// exact, depth-independent game value, and the only effect of too small a bound
/// is a spurious `Draw`. We therefore deepen the bound until the top-level result
/// is a definitive `Win`/`Loss`: the first such result is exact. Clean Win/Loss
/// memo entries are depth-independent, so the `tt` is reused across deepening
/// iterations (and across leaves) — no clean work is repeated. Most leaves
/// resolve at the smallest bound; only twisty frozen mazes deepen.
///
/// # Panics
/// Panics if deepening reaches `race_depth_ceiling` without a definitive
/// Win/Loss. A wall-less race is never a true draw (see module docs) and always
/// resolves within that ceiling, so reaching it can only mean the search is
/// unsound — we fail loudly rather than let a wrong value escape.
pub fn race_value(b: &Board, s: &State, tt: &mut RaceTt) -> (Value, u64) {
    debug_assert_eq!(
        s.walls_left,
        [0, 0],
        "race_value called on a non-race state (walls remain): {:?}",
        s.walls_left
    );
    let ceiling = race_depth_ceiling(b);
    let mut nodes: u64 = 0;
    // Start generous-but-small and double until definitive. The initial bound
    // covers ordinary races (the leading pawn's monotone path plus slack); twisty
    // mazes trigger one or more doublings.
    let mut bound = 2 * (b.w as u32 + b.h as u32);
    loop {
        let (v, _clean) = race_nega(b, s, bound, tt, &mut nodes);
        if v == Value::Win || v == Value::Loss {
            return (v, nodes);
        }
        assert!(
            bound < ceiling,
            "BUG: wall-less race still unresolved (=Draw) at depth bound {} \
             (ceiling {}). A wall-less race is never a true draw, so failing to \
             resolve within the ceiling means the race search is unsound. \
             pawns={:?} h_walls={:#x} v_walls={:#x} turn={}",
            bound,
            ceiling,
            s.pawn,
            s.h_walls,
            s.v_walls,
            s.turn,
        );
        bound = (bound * 2).min(ceiling);
    }
}

/// Depth-bounded full-window negamax over pawn steps only, with an EXACT
/// persistent `State`-keyed memo.
///
/// Returns `(value, depth_clean)`. `depth_clean` is `true` iff the returned
/// value's proof never reached the depth-0 floor — i.e. the value equals the
/// unbounded game value and is therefore exact and depth-independent. Only
/// depth-clean Win/Loss results are persisted (see module docs). `nodes`
/// accumulates internal nodes entered (profiling only).
fn race_nega(
    b: &Board,
    s: &State,
    depth: u32,
    tt: &mut RaceTt,
    nodes: &mut u64,
) -> (Value, bool) {
    *nodes += 1;
    // Terminal: `winner` is the player who just moved (= 1 - turn); if that is
    // the side to move it's a Win, else a Loss. Terminals are depth-clean.
    if let Some(p) = b.winner(s) {
        let v = if p == s.turn { Value::Win } else { Value::Loss };
        return (v, true);
    }
    if depth == 0 {
        // Cyclic-graph cutoff. NOT depth-clean: this line was truncated, so its
        // `Draw` carries no proof and must never be persisted. At a small bound
        // this fires on twisty mazes and propagates a `Draw` up to `race_value`,
        // which then deepens the bound and retries; only if the hard ceiling is
        // reached without resolution does `race_value` panic.
        return (Value::Draw, false);
    }

    // Persistent exact memo: a hit is the position's true, depth-independent
    // game value (only depth-clean results are stored), so it is exact at any
    // depth and is itself depth-clean.
    if let Some(&v) = tt.get(s) {
        return (v, true);
    }

    let mut best = Value::Loss;
    // `clean` tracks whether the proof of `best` avoided the depth floor.
    //  - A `Win` is proven by the single child that achieves it: clean iff that
    //    child is clean.
    //  - A `Loss`/`Draw` is proven by ALL children (none beat it): clean iff
    //    every explored child is clean.
    let mut all_children_clean = true;
    let mut win_child_clean = false;
    let mut found_win = false;
    // Steps only — walls are exhausted in the race endgame.
    for dest in crate::movegen::legal_steps(b, s) {
        let s2 = crate::movegen::apply(b, s, crate::state::Move::Step(dest));
        let (cv, cclean) = race_nega(b, &s2, depth - 1, tt, nodes);
        let v = cv.negate();
        all_children_clean &= cclean;
        if v > best {
            best = v;
        }
        // Full-window negamax: the only sound early break is on the lattice
        // maximum (`Win`), which cannot be improved upon, so this break is
        // exact. Record the winning child's cleanliness; its proof alone
        // certifies the Win.
        if best == Value::Win {
            found_win = true;
            win_child_clean = cclean;
            break;
        }
    }

    // Reconcile cleanliness with the resolved value:
    //  - Win: certified by the single winning child -> clean iff that child is.
    //  - Loss/Draw: certified by all children failing to beat it -> clean iff
    //    every explored child was clean.
    let clean = if found_win { win_child_clean } else { all_children_clean };

    // Persist ONLY depth-clean, definitive (exact, depth-independent) results.
    if clean && (best == Value::Win || best == Value::Loss) {
        tt.insert(*s, best);
    }
    (best, clean)
}
