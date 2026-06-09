//! Exact full-game solver: depth-bounded negamax with alpha-beta, a
//! bound-flagged transposition table, move ordering, and a wall-less race
//! short-circuit. Returns the game-theoretic value for the side to move.
//!
//! Mirrors `smallboard/solver.py` (the reference Python solver) but specialized
//! to the three-valued `Value` lattice and the Rust engine.

use crate::board::Board;
use crate::endgame::RaceTt;
use crate::state::{Move, State};
use rustc_hash::FxHashMap;

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

/// The optimized exact solver. Borrows a `Board` and owns a transposition table
/// keyed on `(state, depth)`.
pub struct Solver<'a> {
    b: &'a Board,
    tt: FxHashMap<(State, u32), (Value, Flag)>,
    /// PERSISTENT, exact race endgame memo keyed on the bare (walls-frozen)
    /// `State`. Every entry is the position's exact game-theoretic value, so it
    /// is sound to reuse across every walls-exhausted leaf within a `solve()`
    /// call: each distinct wall-less race position is solved exactly once
    /// instead of being re-derived per leaf. See `endgame.rs` for the soundness
    /// argument. Survives across `solve()` calls on the same `Solver` (extra
    /// reuse; values stay valid because the race value is a pure function of
    /// `State`, independent of the surrounding board's wall count).
    race_tt: RaceTt,
    /// Profiling counter: total internal nodes visited. Counts every `ab(...)`
    /// entry (main alpha-beta search) plus every wall-less race node entered via
    /// `race_value`. Instrumentation only — does not affect search results.
    pub nodes: u64,
}

impl<'a> Solver<'a> {
    pub fn new(b: &'a Board) -> Solver<'a> {
        Solver {
            b,
            tt: FxHashMap::default(),
            race_tt: RaceTt::default(),
            nodes: 0,
        }
    }

    /// Number of entries currently in the persistent race endgame memo.
    pub fn race_tt_len(&self) -> usize {
        self.race_tt.len()
    }

    /// Number of entries currently in the transposition table.
    pub fn tt_len(&self) -> usize {
        self.tt.len()
    }

    /// Rough lower-bound estimate of the transposition table's memory footprint
    /// in bytes: entry count times the per-entry key+value size
    /// (`(State, u32)` key plus `(Value, Flag)` value). Ignores `HashMap`
    /// overhead and load factor, so it under-counts true RSS — it is an estimate
    /// of the dominant TT memory only, not real resident size.
    pub fn tt_bytes(&self) -> usize {
        let per_entry = size_of::<(State, u32)>() + size_of::<(Value, Flag)>();
        self.tt.len() * per_entry
    }

    /// Solve `s` to its game-theoretic value for the side to move.
    ///
    /// This is a **single** alpha-beta pass at a generous fixed depth bound
    /// `ceiling = 4*(w+h) + 2*walls + 8`. Rationale: a forced Win/Loss proven
    /// within the bound is final (alpha-beta over the full `(Loss, Win)` window
    /// never mis-proves a forced result), so only `Draw` is depth-limited. For
    /// the validation boards the bound is generous enough that every Win/Loss
    /// board resolves and the one true draw (8x3) stays `Draw` at every depth.
    ///
    /// NOTE (Phase 1): iterative deepening plus retrograde draw-proving for
    /// novel boards (to distinguish a genuine draw from a not-yet-resolved line)
    /// is deferred; this single deep pass is sufficient for the current
    /// validation set.
    pub fn solve(&mut self, s: &State) -> Value {
        let w = self.b.w as u32;
        let h = self.b.h as u32;
        let walls = self.b.walls as u32;
        let ceiling = 4 * (w + h) + 2 * walls + 8;
        self.ab(s, ceiling, Value::Loss, Value::Win)
    }

    /// Alpha-beta negamax. Returns the value of `s` for the side to move,
    /// fail-soft within the `(alpha, beta)` window.
    fn ab(&mut self, s: &State, depth: u32, mut alpha: Value, mut beta: Value) -> Value {
        // Profiling: count every internal node entered (instrumentation only).
        self.nodes += 1;
        // Terminal: `winner` is the player who just moved (= 1 - turn). If that
        // is the side to move it's a Win, otherwise the side to move has lost.
        if let Some(p) = self.b.winner(s) {
            return if p == s.turn { Value::Win } else { Value::Loss };
        }
        if depth == 0 {
            return Value::Draw;
        }

        // Race short-circuit: with no walls left for either player the position
        // is a pure pawn race, solved exactly by its own bounded negamax. The
        // race value is a pure, context-free function of `State`, so it is
        // memoized PERSISTENTLY in `race_tt` across every leaf of this solve —
        // each distinct race position is solved exactly once instead of being
        // re-derived per leaf. See `endgame.rs` for the exactness argument.
        if s.walls_left == [0, 0] {
            let (v, race_nodes) = crate::endgame::race_value(self.b, s, &mut self.race_tt);
            self.nodes += race_nodes;
            return v;
        }

        let alpha0 = alpha;
        let key = (*s, depth);
        if let Some(&(val, flag)) = self.tt.get(&key) {
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

        let mut best = Value::Loss;
        for m in self.ordered_moves(s) {
            let s2 = crate::movegen::apply(self.b, s, m);
            let v = self.ab(&s2, depth - 1, beta.negate(), alpha.negate()).negate();
            if v > best {
                best = v;
            }
            if best > alpha {
                alpha = best;
            }
            if alpha >= beta {
                break;
            }
        }

        let flag = if best <= alpha0 {
            Flag::Upper
        } else if best >= beta {
            Flag::Lower
        } else {
            Flag::Exact
        };
        self.tt.insert(key, (best, flag));
        best
    }

    /// Order legal moves by the mover's resulting shortest-path advantage:
    /// `score = d_opp(s2) - d_self(s2)`, descending. A larger score means the
    /// move leaves the mover closer to goal than the opponent, so it's tried
    /// first to maximize alpha-beta cutoffs. Mirrors
    /// `smallboard/solver.py::_ordered_moves`.
    fn ordered_moves(&self, s: &State) -> Vec<Move> {
        let mover = s.turn;
        let opp = 1 - mover;
        let big = 4 * (self.b.w as i64 + self.b.h as i64);
        let mut scored: Vec<(i64, Move)> = crate::movegen::legal_moves(self.b, s)
            .into_iter()
            .map(|m| {
                let s2 = crate::movegen::apply(self.b, s, m);
                let d_self = self.b.dist_to_goal(&s2, mover).map_or(big, |d| d as i64);
                let d_opp = self.b.dist_to_goal(&s2, opp).map_or(big, |d| d as i64);
                (d_opp - d_self, m)
            })
            .collect();
        scored.sort_by(|a, b| b.0.cmp(&a.0));
        scored.into_iter().map(|(_, m)| m).collect()
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
