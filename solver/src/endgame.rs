//! Endgame tablebase slice: the wall-less *race*.
//!
//! When both players have exhausted their walls, the game reduces to a pure
//! pawn race over steps only. With no walls left to place, neither path length
//! can ever increase, so the race always resolves to a Win or Loss for the side
//! to move within a small step bound — it can never stay a Draw. This is the
//! first endgame-tablebase slice; a `k`-wall generalization is deferred to
//! Phase 1.

use crate::solver::Value;
use crate::state::State;
use crate::board::Board;
use rustc_hash::FxHashMap;

/// Exact game value of a wall-less position for the side to move.
///
/// Depth-bounded negamax over **steps only** (walls are exhausted) with a
/// `(State, depth)` memo of exact values. The bound `2*(w+h)` is generous
/// enough that every wall-less race resolves to Win/Loss within it (an upper
/// bound on the longest sensible race on a `w x h` board), so the depth-0
/// `Draw` fallback is never the final answer for a reachable race.
pub fn race_value(b: &Board, s: &State) -> Value {
    let bound = 2 * (b.w as u32 + b.h as u32);
    let mut memo: FxHashMap<(State, u32), Value> = FxHashMap::default();
    race_ab(b, s, bound, Value::Loss, Value::Win, &mut memo)
}

/// Alpha-beta negamax over pawn steps only, with an exact-value memo.
fn race_ab(
    b: &Board,
    s: &State,
    depth: u32,
    mut alpha: Value,
    beta: Value,
    memo: &mut FxHashMap<(State, u32), Value>,
) -> Value {
    // Terminal: `winner` is the player who just moved (= 1 - turn); if that is
    // the side to move it's a Win, else a Loss.
    if let Some(p) = b.winner(s) {
        return if p == s.turn { Value::Win } else { Value::Loss };
    }
    if depth == 0 {
        return Value::Draw;
    }

    let key = (*s, depth);
    if let Some(&v) = memo.get(&key) {
        return v;
    }

    let mut best = Value::Loss;
    // Steps only — walls are exhausted in the race endgame.
    for dest in crate::movegen::legal_steps(b, s) {
        let s2 = crate::movegen::apply(b, s, crate::state::Move::Step(dest));
        let v = race_ab(b, &s2, depth - 1, beta.negate(), alpha.negate(), memo).negate();
        if v > best {
            best = v;
        }
        if best > alpha {
            alpha = best;
        }
        if alpha >= beta {
            break;
        }
        if best == Value::Win {
            break;
        }
    }

    // Memoize only full-window (exact) resolutions to keep the memo sound: a
    // value cut by a narrow window may not be exact. The race is called from
    // `race_value` with the full window, and recursive calls inherit windows;
    // to stay simple and correct we only cache when the window was not
    // narrowed below by a fail-high (best < beta) and not pruned by alpha. In
    // practice the race resolves to Win/Loss; we cache those exact results.
    if best == Value::Win || best == Value::Loss {
        memo.insert(key, best);
    }
    best
}
