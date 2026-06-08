use std::collections::HashMap;

use crate::movegen::legal_steps;
use crate::state::{apply_move, winner, GameState, Move};

const RACE_PLY_BOUND: u32 = 36; // 4*N: ample for either pawn to reach its goal

fn pawn_moves(s: &GameState) -> Vec<Move> {
    legal_steps(s).into_iter().map(|(c, r)| Move::Step { c, r }).collect()
}

fn negamax(s: &GameState, depth: u32,
           memo: &mut HashMap<((u8, u8), (u8, u8), u8, u32), i32>) -> i32 {
    if let Some(w) = winner(s) {
        return if w == s.turn as usize { 1 } else { -1 };
    }
    if depth == 0 {
        return 0; // draw at the ply bound
    }
    let key = (s.pawns[0], s.pawns[1], s.turn, depth);
    if let Some(&v) = memo.get(&key) {
        return v;
    }
    let mut best = i32::MIN;
    for m in pawn_moves(s) {
        let v = -negamax(&apply_move(s, &m), depth - 1, memo);
        if v > best {
            best = v;
        }
        if best == 1 {
            break; // cannot beat a forced win
        }
    }
    let v = if best == i32::MIN { -1 } else { best }; // no moves => stuck => loss
    memo.insert(key, v);
    v
}

/// Exact value for the side to move in a frozen-wall race. Precondition:
/// walls_left == (0,0) and not terminal. +1 win / -1 loss / 0 draw-at-bound,
/// plus the optimal move. Depth-bounded + memoized => total and fast.
pub fn solve_race(s: &GameState) -> (i32, Move) {
    let mut memo = HashMap::new();
    let mut best_val = i32::MIN;
    let moves = pawn_moves(s);
    let mut best = moves[0];
    for m in &moves {
        let v = -negamax(&apply_move(s, m), RACE_PLY_BOUND - 1, &mut memo);
        if v > best_val {
            best_val = v;
            best = *m;
        }
        if best_val == 1 {
            break;
        }
    }
    (best_val, best)
}
