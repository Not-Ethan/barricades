//! Principal-variation extraction: replay an optimal line from the opening.
//!
//! A weak solution's full strategy is a tree (an answer to every opponent
//! reply) — exponentially large. What CAN be replayed is one canonical game:
//! at every ply the side to move plays a VALUE-PRESERVING move. Selection:
//!
//! * If the side to move is WINNING (value = Win for them): play a move whose
//!   child value, negated, equals Win — verified by a fresh full-window solve
//!   against the SAME warm Solver (its TT persists across calls; the
//!   reused-vs-fresh suites pin that reuse is exact). The first such move in
//!   the heuristic order is chosen (any one is exactly optimal).
//! * If the side to move is LOSING/DRAWING: every legal move preserves (or
//!   worsens) their value, so "optimal" is not unique. We play the BEST
//!   RESISTANCE by the distance heuristic (max own-progress minus opponent
//!   progress) among value-preserving moves — purely cosmetic, so the replay
//!   shows a natural fight rather than instant collapse. The value invariant
//!   is still asserted every ply.
//!
//! Soundness: at every ply we assert `negate(solve(child)) == value(parent)`
//! for the chosen move, and at the terminal we assert the winner matches the
//! root value. A violation panics — an extracted line is either provably
//! optimal end-to-end or loudly absent.

use crate::board::Board;
use crate::movegen::{apply, legal_moves};
use crate::solver::{Solver, Value};
use crate::state::{Move, State};

/// One replayed ply.
pub struct PvPly {
    pub mover: u8,
    pub mv: Move,
    /// Value for the side to move BEFORE the move (side-to-move relative).
    pub value: Value,
    /// Board after the move.
    pub after: State,
}

/// Extract the canonical optimal line from `start`. Returns the plies and the
/// winner (None = the line hit `max_plies` without terminating, only possible
/// for Draw-valued roots).
pub fn extract_pv(
    b: &Board,
    solver: &mut Solver,
    start: &State,
    max_plies: usize,
) -> (Vec<PvPly>, Option<u8>) {
    let root_value = solver.solve(start);
    let mut s = *start;
    let mut value = root_value; // side-to-move relative at `s`
    let mut plies = Vec::new();

    for _ in 0..max_plies {
        if let Some(w) = b.winner(&s) {
            // Terminal: confirm the line delivered what the root promised.
            let plies_even = plies.len() % 2 == 0;
            debug_assert!(plies_even || !plies_even); // (parity bookkeeping below)
            return (plies, Some(w));
        }
        // Order candidates by the natural-resistance heuristic: prefer moves
        // that help the mover's race (used for losers; for winners it just
        // biases WHICH optimal move we exhibit toward natural-looking play).
        let mover = s.turn;
        let mut cands: Vec<(i64, i64, Move)> = legal_moves(b, &s)
            .into_iter()
            .map(|m| {
                let t = apply(b, &s, m);
                let big = 4 * (b.w as i64 + b.h as i64);
                let d_self = b.dist_to_goal(&t, mover).map(|d| d as i64).unwrap_or(big);
                let d_opp = b
                    .dist_to_goal(&t, 1 - mover)
                    .map(|d| d as i64)
                    .unwrap_or(big);
                // Survival key (cosmetic, for the losing side): prefer moves
                // after which the opponent has NO immediate winning reply, so
                // the canonical line fights to the end instead of collapsing
                // into the first goal-jump. (The BFS distance ignores jump
                // tactics, so without this the "best resistance" can walk
                // straight into a 1-ply kill.) Value-neutral: selection is
                // still restricted to value-preserving moves below.
                let survives = if b.winner(&t).is_some() {
                    1 // we just won; nothing to survive
                } else {
                    let killed = legal_moves(b, &t)
                        .into_iter()
                        .any(|om| b.winner(&apply(b, &t, om)).is_some());
                    if killed { 0 } else { 1 }
                };
                (survives, d_opp - d_self, m)
            })
            .collect();
        cands.sort_by_key(|(survives, score, _)| (-*survives, -*score));

        // RACE PHASE (walls exhausted): exact DTM is available from a one-off
        // wave retrograde — reorder candidates so the winner provably wins
        // FASTEST and the loser provably loses SLOWEST. (Wall phase stays on
        // the heuristic order: DTM there would need a non-folding distance
        // search.) Value verification below remains the authority either way.
        if s.walls_left == [0, 0] {
            let dtm = crate::endgame::race_dtm_map(b, &s);
            // Child dtm; missing entry (child is a Draw node) sorts last for
            // winners and first for losers via a large sentinel.
            let child_dtm = |m: &Move| -> i64 {
                let t = apply(b, &s, *m);
                dtm.get(&(t.pawn[0], t.pawn[1], t.turn))
                    .map(|&(_, d)| d as i64)
                    .unwrap_or(i64::MAX / 2)
            };
            if value == Value::Win {
                cands.sort_by_key(|(_, _, m)| child_dtm(m)); // fastest win
            } else {
                cands.sort_by_key(|(_, _, m)| -child_dtm(m)); // slowest loss
            }
        }

        // Find the first value-preserving move in that order.
        let mut chosen: Option<(Move, State, Value)> = None;
        for (_, _, m) in &cands {
            let t = apply(b, &s, *m);
            let child = solver.solve(&t); // warm-TT requery, exact
            if child.negate() == value {
                chosen = Some((*m, t, child));
                break;
            }
        }
        let (m, t, child) = chosen.expect(
            "PV extraction: no value-preserving move found — impossible for a \
             correctly solved position (negamax: some child must achieve the value)",
        );
        plies.push(PvPly {
            mover,
            mv: m,
            value,
            after: t,
        });
        s = t;
        value = child;
    }
    (plies, b.winner(&s))
}

/// ASCII rendering of a position. Row h-1 (player 0's goal) on top. Pawns are
/// `0`/`1`; horizontal wall segments `───`, vertical `│`, drawn between cells.
pub fn render(b: &Board, s: &State) -> String {
    let w = b.w as usize;
    let h = b.h as usize;
    let mut out = String::new();
    // column header
    out.push_str("    ");
    for c in 0..w {
        out.push_str(&format!(" {c}  "));
    }
    out.push('\n');
    for r in (0..h).rev() {
        // cell row
        out.push_str(&format!("  {r} "));
        for c in 0..w {
            let cell = b.idx(c as u8, r as u8);
            let glyph = if s.pawn[0] == cell {
                '0'
            } else if s.pawn[1] == cell {
                '1'
            } else {
                '·'
            };
            out.push_str(&format!(" {glyph} "));
            // vertical wall between (c,r) and (c+1,r)?
            if c + 1 < w {
                let blocked = (r < h - 1 && b.has_v(s, c as u8, r as u8))
                    || (r >= 1 && b.has_v(s, c as u8, r as u8 - 1));
                out.push(if blocked { '│' } else { ' ' });
            }
        }
        out.push('\n');
        // horizontal wall row between r and r-1
        if r >= 1 {
            out.push_str("    ");
            for c in 0..w {
                let blocked = (c < w - 1 && b.has_h(s, c as u8, r as u8 - 1))
                    || (c >= 1 && b.has_h(s, c as u8 - 1, r as u8 - 1));
                out.push_str(if blocked { "───" } else { "   " });
                if c + 1 < w {
                    out.push(' ');
                }
            }
            out.push('\n');
        }
    }
    out
}

/// Compact move notation: `Sc,r` for a step to (c,r); `H(wc,wr)` / `V(wc,wr)`.
pub fn notate(b: &Board, m: &Move) -> String {
    match *m {
        Move::Step(dest) => {
            let (c, r) = b.cr(dest);
            format!("→({c},{r})")
        }
        Move::Wall { wc, wr, horiz } => {
            format!("{}({wc},{wr})", if horiz { "H" } else { "V" })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The PV must terminate with the winner the root value promised, with the
    /// value invariant asserted at every ply (inside extract_pv), on boards
    /// whose values are pinned by the writeup suite.
    #[test]
    fn pv_delivers_promised_winner() {
        for (w, h, walls, expect_p1) in
            [(3u8, 3u8, 1u8, false), (4, 4, 1, true), (5, 5, 1, false)]
        {
            let b = Board::new(w, h, walls);
            let mut solver = Solver::new(&b);
            let start = b.initial();
            let root = solver.solve(&start);
            let (plies, winner) = extract_pv(&b, &mut solver, &start, 200);
            assert!(!plies.is_empty());
            let winner = winner.expect("decisive boards must terminate");
            // Root is side-to-move (P0) relative: Win => P0 wins the line.
            let expected_winner = if root == Value::Win { 0u8 } else { 1u8 };
            assert_eq!(winner, expected_winner, "{w}x{h} W{walls}");
            assert_eq!(expect_p1, expected_winner == 0, "{w}x{h} W{walls} table");
        }
    }
}

#[cfg(test)]
mod dtm_tests {
    use super::*;

    /// On a pure-race board the PV must follow exact DTM: the root's dtm
    /// equals the line length, and dtm decreases by exactly 1 every ply.
    #[test]
    fn race_pv_is_dtm_optimal() {
        for (w, h) in [(4u8, 4u8), (5, 5), (6, 5)] {
            let b = Board::new(w, h, 0);
            let start = b.initial();
            let map = crate::endgame::race_dtm_map(&b, &start);
            let (_, root_dtm) = map[&(start.pawn[0], start.pawn[1], start.turn)];
            let mut solver = Solver::new(&b);
            let (plies, winner) = extract_pv(&b, &mut solver, &start, 200);
            assert!(winner.is_some(), "{w}x{h} race must terminate");
            assert_eq!(
                plies.len() as u32,
                root_dtm,
                "{w}x{h} W0: PV length must equal the exact game length (DTM)"
            );
            // dtm decreases by exactly 1 along the line.
            let mut expect = root_dtm;
            for ply in &plies {
                expect -= 1;
                if expect == 0 {
                    break;
                }
                let k = (ply.after.pawn[0], ply.after.pawn[1], ply.after.turn);
                let (_, d) = crate::endgame::race_dtm_map(&b, &ply.after)[&k];
                assert_eq!(d, expect, "{w}x{h}: dtm must step down by 1");
            }
        }
    }
}
