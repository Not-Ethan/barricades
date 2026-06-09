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

/// Maximum search ply we keep killer slots for. The `solve()` depth ceiling for
/// the validation boards is well under this, and any deeper ply simply skips the
/// killer heuristic (it only affects ordering, never values).
const MAX_PLY: usize = 256;

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
    /// Killer moves: up to two moves per search ply that previously caused a
    /// beta cutoff. Tried first (after the TT move would be, but we have no
    /// separate TT-move slot) to maximize cutoffs. ORDERING ONLY — never changes
    /// the legal-move set or any value.
    killers: Vec<[Option<Move>; 2]>,
    /// History heuristic: per-move cumulative beta-cutoff count, indexed by the
    /// dense `move_index`. Biases ordering globally. ORDERING ONLY.
    history: Vec<u32>,
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
    /// Profiling counter: total internal nodes visited. Counts every `ab(...)`
    /// entry (main alpha-beta search) plus every wall-less race node entered via
    /// `race_value`. Instrumentation only — does not affect search results.
    pub nodes: u64,
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

/// Build the horizontal-mirror permutation of the wall-anchor bit layout.
///
/// The anchor grid is `(w-1)` columns by `(h-1)` rows; anchor `(wc, wr)` lives
/// at bit `wr*(w-1)+wc` in BOTH `h_walls` and `v_walls`. A horizontal board
/// reflection maps column `wc -> w-2-wc` (the mirror of an index in `0..w-1`),
/// leaving the row unchanged. The returned `perm[src_bit] = dst_bit` is applied
/// identically to the horizontal and vertical anchor bitsets (a horizontal wall
/// reflects to a horizontal wall, a vertical to a vertical — orientation is
/// preserved under a left-right flip). Returns `None` when `w < 2` (no anchor
/// columns exist; the mirror is the identity and there is nothing to permute).
fn build_mirror_perm(b: &Board) -> Option<Vec<u8>> {
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

/// The canonical (orientation-folding) representative of `s`: the
/// lexicographically-smaller of `s` and its horizontal mirror, by `pack_key`.
/// Because mirror is value-preserving with the same side to move, `s` and its
/// mirror share a game value, so keying the TT on the representative is exact.
#[inline]
fn canonical(b: &Board, perm: Option<&[u8]>, s: &State) -> State {
    let m = mirror_state(b, perm, s);
    if pack_key(&m) < pack_key(s) {
        m
    } else {
        *s
    }
}

impl<'a> Solver<'a> {
    pub fn new(b: &'a Board) -> Solver<'a> {
        Solver {
            b,
            tt: FxHashMap::default(),
            race_tt: RaceTt::default(),
            killers: vec![[None, None]; MAX_PLY],
            history: vec![0u32; MOVE_INDEX_SPAN],
            mirror_perm: build_mirror_perm(b),
            use_ordering: true,
            use_symmetry: true,
            nodes: 0,
        }
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
        // Reset ordering heuristics per solve (they only affect speed, but a
        // fresh start keeps behaviour reproducible across calls).
        for k in self.killers.iter_mut() {
            *k = [None, None];
        }
        for h in self.history.iter_mut() {
            *h = 0;
        }
        self.ab(s, ceiling, 0, Value::Loss, Value::Win)
    }

    /// Alpha-beta negamax. Returns the value of `s` for the side to move,
    /// fail-soft within the `(alpha, beta)` window. `ply` is the distance from
    /// the root (used only to index per-ply killer slots; ordering-only).
    fn ab(&mut self, s: &State, depth: u32, ply: usize, mut alpha: Value, mut beta: Value) -> Value {
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
        // Canonicalize the TT key by the horizontal-mirror representative. The
        // mirror is a value-preserving automorphism (same side to move), so `s`
        // and its mirror have the SAME value at every depth — keying both under
        // the shared representative is exact and roughly halves the TT.
        let canon = if self.use_symmetry {
            canonical(self.b, self.mirror_perm.as_deref(), s)
        } else {
            *s
        };
        let key = (canon, depth);
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
        let moves = self.ordered_moves(s, ply);
        for m in moves {
            let s2 = crate::movegen::apply(self.b, s, m);
            let v = self
                .ab(&s2, depth - 1, ply + 1, beta.negate(), alpha.negate())
                .negate();
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
    /// EXACTNESS: this is pure reordering. The legal-move SET is unchanged, and
    /// alpha-beta returns the identical value under any visitation order, so no
    /// value can change — zero exactness risk.
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
        // Descending by (distance score, history, killer rank).
        scored.sort_by(|a, b| b.0.cmp(&a.0).then(b.1.cmp(&a.1)).then(b.2.cmp(&a.2)));
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
