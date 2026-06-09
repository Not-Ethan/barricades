//! Endgame tablebase slice: the wall-less *race*.
//!
//! When both players have exhausted their walls, the game reduces to a pure
//! pawn race over steps only. With no walls left to place, neither path length
//! can ever increase, so the race always resolves to a Win or Loss for the side
//! to move within a small step bound — it can never stay a Draw. This is the
//! first endgame-tablebase slice; a `k`-wall generalization is deferred to
//! Phase 1.
//!
//! ## Persistent, exact memo
//!
//! The race value of a wall-less `State` (pawns + frozen walls + turn, with
//! `walls_left == [0, 0]`) is a **pure, context-free function of that State**:
//! no wall can ever be placed, so the reachable game graph from `s` — and hence
//! its game-theoretic value — depends on nothing but `s`. That makes a memo
//! keyed on the bare `State` and **persisted across every race leaf of a single
//! `solve()` call** sound: identical race positions reached via different move
//! orders are solved exactly once.
//!
//! Soundness hinges on storing only EXACT values. Two safeguards together
//! guarantee that:
//!
//!  1. **Full-window negamax, no alpha-beta.** The only early break is on the
//!     lattice maximum `Win`, which cannot be improved upon, so the break never
//!     hides a better child — every computed value is exact w.r.t. the explored
//!     (depth-bounded) tree.
//!  2. **Depth-clean persistence.** The depth bound `2*(w+h)` exists only to
//!     terminate the *cyclic* wall-less graph; pawns can shuffle back and forth,
//!     so the graph has cycles and an unbounded DFS would not terminate. A value
//!     is "depth-clean" iff its proof never bottomed out at the depth-0 `Draw`
//!     floor anywhere in its subtree. A depth-clean Win/Loss is identical to the
//!     unbounded game value (no line was truncated), hence exact and
//!     depth-independent — so it is safe to key on the bare `State` and reuse at
//!     any depth. Values that *did* touch the floor are returned to the caller
//!     but NOT persisted, preventing a depth-pressured (possibly non-exact)
//!     result from poisoning the persistent memo. By the bound's generous
//!     construction the floor is never reached for a reachable race, so in
//!     practice every race result is depth-clean and gets memoized.

use crate::solver::Value;
use crate::state::State;
use crate::board::Board;
use rustc_hash::FxHashMap;

/// Persistent, exact race memo keyed on the bare (walls-frozen) `State`.
/// Every stored value is the position's EXACT, depth-independent
/// game-theoretic value, so it is sound to reuse across any race leaf within a
/// single `solve()` call.
pub type RaceTt = FxHashMap<State, Value>;

/// Exact game value of a wall-less position for the side to move, paired with
/// the number of race nodes visited (for profiling; the value is unaffected).
///
/// Plain depth-bounded negamax over **steps only** (walls are exhausted) with a
/// PERSISTENT `State`-keyed memo of exact values (`tt`). The bound `2*(w+h)` is
/// generous enough that every wall-less race resolves to Win/Loss within it (an
/// upper bound on the longest sensible race on a `w x h` board), so the depth-0
/// `Draw` fallback is never the final answer for a reachable race.
pub fn race_value(b: &Board, s: &State, tt: &mut RaceTt) -> (Value, u64) {
    let bound = 2 * (b.w as u32 + b.h as u32);
    let mut nodes: u64 = 0;
    let (v, _clean) = race_nega(b, s, bound, tt, &mut nodes);
    (v, nodes)
}

/// Plain (full-window) negamax over pawn steps only, with an EXACT persistent
/// `State`-keyed memo.
///
/// Returns `(value, depth_clean)`. `depth_clean` is `true` iff the returned
/// value's proof never reached the depth-0 `Draw` floor — i.e. the value equals
/// the unbounded game value and is therefore exact and depth-independent. Only
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
        // `Draw` carries no proof and must never be persisted.
        return (Value::Draw, false);
    }

    // Persistent exact memo: a hit is the position's true, depth-independent
    // game value (only depth-clean results are ever stored), so it is exact at
    // any depth and is itself depth-clean.
    if let Some(&v) = tt.get(s) {
        return (v, true);
    }

    let mut best = Value::Loss;
    // `clean` tracks whether the proof of `best` avoided the depth floor.
    // - A `Win` is proven by the single child that achieves it: clean iff that
    //   child is clean.
    // - A `Loss`/`Draw` is proven by ALL children (none beat it): clean iff
    //   every explored child is clean.
    // We accumulate child-cleanliness and reconcile with `best` after the loop.
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
    //    every explored child was clean (no truncated line could have hidden a
    //    better move).
    let clean = if found_win { win_child_clean } else { all_children_clean };

    // Persist ONLY depth-clean, definitive (exact, depth-independent) results.
    if clean && (best == Value::Win || best == Value::Loss) {
        tt.insert(*s, best);
    }
    (best, clean)
}
