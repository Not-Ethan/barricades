//! THROWAWAY adversarial audit (jumps-movegen dimension). Delete after run.
//! Clean-room references re-implemented here, deliberately NOT sharing the
//! production blocking/step logic, to catch jump + wall-legality bugs.

use quoridor_solver::board::Board;
use quoridor_solver::movegen::{
    apply, legal_moves, legal_steps, legal_walls, legal_walls_bruteforce,
};
use quoridor_solver::solver::{brute_value, Solver};
use quoridor_solver::state::{Move, State};
use std::collections::{HashMap, HashSet};

// ---------- clean-room geometry (independent of production code) ----------

#[derive(Clone)]
struct Ref {
    w: i32,
    h: i32,
}

impl Ref {
    fn on(&self, c: i32, r: i32) -> bool {
        c >= 0 && r >= 0 && c < self.w && r < self.h
    }
    fn idx(&self, c: i32, r: i32) -> u8 {
        (r * self.w + c) as u8
    }
    fn cr(&self, i: u8) -> (i32, i32) {
        ((i as i32) % self.w, (i as i32) / self.w)
    }
    // Does h_wall anchor (wc,wr) exist in set? wc in 0..w-1, wr in 0..h-1.
    fn has_h(&self, hw: &HashSet<(i32, i32)>, wc: i32, wr: i32) -> bool {
        if wc < 0 || wr < 0 || wc >= self.w - 1 || wr >= self.h - 1 {
            return false;
        }
        hw.contains(&(wc, wr))
    }
    fn has_v(&self, vw: &HashSet<(i32, i32)>, wc: i32, wr: i32) -> bool {
        if wc < 0 || wr < 0 || wc >= self.w - 1 || wr >= self.h - 1 {
            return false;
        }
        vw.contains(&(wc, wr))
    }
    // Clean-room blocking, derived from first principles (edge between cell (c,r)
    // and its neighbor in dir (dc,dr)).
    fn blocked(
        &self,
        hw: &HashSet<(i32, i32)>,
        vw: &HashSet<(i32, i32)>,
        c: i32,
        r: i32,
        dc: i32,
        dr: i32,
    ) -> bool {
        // North edge between row r and r+1, at column c: blocked if a horizontal
        // wall sits on that edge. A horizontal wall anchored at (wc,wr) occupies
        // the edge between rows wr and wr+1 spanning columns wc and wc+1.
        // So the north edge of (c,r) is covered by anchor (c,r) or (c-1,r).
        if dr == 1 {
            self.has_h(hw, c, r) || self.has_h(hw, c - 1, r)
        } else if dr == -1 {
            // south edge of (c,r) = north edge of (c,r-1): anchors (c,r-1),(c-1,r-1)
            self.has_h(hw, c, r - 1) || self.has_h(hw, c - 1, r - 1)
        } else if dc == 1 {
            // east edge of (c,r): vertical wall between cols c and c+1 spanning
            // rows r and r+1 -> anchor (c,r) or (c,r-1)
            self.has_v(vw, c, r) || self.has_v(vw, c, r - 1)
        } else {
            // west edge of (c,r) = east edge of (c-1,r): anchors (c-1,r),(c-1,r-1)
            self.has_v(vw, c - 1, r) || self.has_v(vw, c - 1, r - 1)
        }
    }

    // Clean-room legal_steps returning a sorted set of destination cell indices.
    fn legal_steps(
        &self,
        hw: &HashSet<(i32, i32)>,
        vw: &HashSet<(i32, i32)>,
        me: (i32, i32),
        opp: (i32, i32),
    ) -> Vec<u8> {
        let dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)];
        let mut out = Vec::new();
        for &(dx, dy) in &dirs {
            let (ac, ar) = (me.0 + dx, me.1 + dy);
            if !self.on(ac, ar) || self.blocked(hw, vw, me.0, me.1, dx, dy) {
                continue;
            }
            if (ac, ar) != opp {
                out.push(self.idx(ac, ar));
                continue;
            }
            // jump
            let (lc, lr) = (opp.0 + dx, opp.1 + dy);
            if self.on(lc, lr) && !self.blocked(hw, vw, opp.0, opp.1, dx, dy) {
                out.push(self.idx(lc, lr));
            } else {
                for &(px, py) in &dirs {
                    if (px, py) == (dx, dy) || (px, py) == (-dx, -dy) {
                        continue;
                    }
                    let (gc, gr) = (opp.0 + px, opp.1 + py);
                    if self.on(gc, gr) && !self.blocked(hw, vw, opp.0, opp.1, px, py) {
                        out.push(self.idx(gc, gr));
                    }
                }
            }
        }
        out.sort();
        out.dedup();
        out
    }

    // Clean-room BFS path existence.
    fn has_path(
        &self,
        hw: &HashSet<(i32, i32)>,
        vw: &HashSet<(i32, i32)>,
        start: (i32, i32),
        goal_row: i32,
    ) -> bool {
        if start.1 == goal_row {
            return true;
        }
        let dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)];
        let mut seen = HashSet::new();
        seen.insert(start);
        let mut frontier = vec![start];
        while !frontier.is_empty() {
            let mut next = Vec::new();
            for &(c, r) in &frontier {
                for &(dc, dr) in &dirs {
                    let (nc, nr) = (c + dc, r + dr);
                    if !self.on(nc, nr) || self.blocked(hw, vw, c, r, dc, dr) {
                        continue;
                    }
                    if seen.contains(&(nc, nr)) {
                        continue;
                    }
                    if nr == goal_row {
                        return true;
                    }
                    seen.insert((nc, nr));
                    next.push((nc, nr));
                }
            }
            frontier = next;
        }
        false
    }

    // Clean-room legal walls (full path check, no fast path). Returns
    // sorted (orient,wc,wr) tuples. orient: true=H.
    fn legal_walls(
        &self,
        hw: &HashSet<(i32, i32)>,
        vw: &HashSet<(i32, i32)>,
        p0: (i32, i32),
        p1: (i32, i32),
        walls_left_turn: u8,
    ) -> Vec<(bool, i32, i32)> {
        if walls_left_turn == 0 {
            return Vec::new();
        }
        let mut out = Vec::new();
        for &horiz in &[true, false] {
            for wc in 0..self.w - 1 {
                for wr in 0..self.h - 1 {
                    // overlap test
                    let overlaps = if horiz {
                        self.has_h(hw, wc, wr)
                            || self.has_h(hw, wc - 1, wr)
                            || self.has_h(hw, wc + 1, wr)
                            || self.has_v(vw, wc, wr)
                    } else {
                        self.has_v(vw, wc, wr)
                            || self.has_v(vw, wc, wr - 1)
                            || self.has_v(vw, wc, wr + 1)
                            || self.has_h(hw, wc, wr)
                    };
                    if overlaps {
                        continue;
                    }
                    let mut hw2 = hw.clone();
                    let mut vw2 = vw.clone();
                    if horiz {
                        hw2.insert((wc, wr));
                    } else {
                        vw2.insert((wc, wr));
                    }
                    if self.has_path(&hw2, &vw2, p0, self.h - 1)
                        && self.has_path(&hw2, &vw2, p1, 0)
                    {
                        out.push((horiz, wc, wr));
                    }
                }
            }
        }
        out.sort();
        out
    }
}

// Convert a production State's wall bitsets into coordinate sets via Board.
fn wall_sets(b: &Board, s: &State) -> (HashSet<(i32, i32)>, HashSet<(i32, i32)>) {
    let mut hw = HashSet::new();
    let mut vw = HashSet::new();
    for wc in 0..b.w - 1 {
        for wr in 0..b.h - 1 {
            if b.has_h(s, wc, wr) {
                hw.insert((wc as i32, wr as i32));
            }
            if b.has_v(s, wc, wr) {
                vw.insert((wc as i32, wr as i32));
            }
        }
    }
    (hw, vw)
}

fn prod_steps_sorted(b: &Board, s: &State) -> Vec<u8> {
    let mut v = legal_steps(b, s);
    v.sort();
    v.dedup();
    v
}

fn prod_walls_sorted(b: &Board, s: &State) -> Vec<(bool, i32, i32)> {
    let mut v: Vec<(bool, i32, i32)> = legal_walls(b, s)
        .into_iter()
        .map(|m| match m {
            Move::Wall { wc, wr, horiz } => (horiz, wc as i32, wr as i32),
            _ => unreachable!(),
        })
        .collect();
    v.sort();
    v
}

fn prod_walls_bf_sorted(b: &Board, s: &State) -> Vec<(bool, i32, i32)> {
    let mut v: Vec<(bool, i32, i32)> = legal_walls_bruteforce(b, s)
        .into_iter()
        .map(|m| match m {
            Move::Wall { wc, wr, horiz } => (horiz, wc as i32, wr as i32),
            _ => unreachable!(),
        })
        .collect();
    v.sort();
    v
}

// Exhaustively BFS the reachable state graph from initial, up to a cap, checking
// at every node: (a) production steps == clean-room steps, (b) fast-path walls
// == clean-room walls, (c) fast-path walls == brute-force walls.
fn exhaustive_check(w: u8, h: u8, walls: u8, cap: usize) -> usize {
    let b = Board::new(w, h, walls);
    let rf = Ref { w: w as i32, h: h as i32 };
    let mut seen: HashSet<State> = HashSet::new();
    let mut stack: Vec<State> = vec![b.initial()];
    seen.insert(b.initial());
    let mut checked = 0usize;
    while let Some(s) = stack.pop() {
        if b.is_terminal(&s) {
            continue;
        }
        let (hw, vw) = wall_sets(&b, &s);
        let me = b.cr(s.pawn[s.turn as usize]);
        let opp = b.cr(s.pawn[(1 - s.turn) as usize]);
        let me = (me.0 as i32, me.1 as i32);
        let opp = (opp.0 as i32, opp.1 as i32);

        // (a) steps
        let ps = prod_steps_sorted(&b, &s);
        let rs = rf.legal_steps(&hw, &vw, me, opp);
        assert_eq!(
            ps, rs,
            "STEP mismatch w={w} h={h} pawns={:?} turn={} hw={:?} vw={:?}\nprod={:?} ref={:?}",
            s.pawn, s.turn, hw, vw, ps, rs
        );

        // (b)+(c) walls
        let p0 = b.cr(s.pawn[0]);
        let p1 = b.cr(s.pawn[1]);
        let p0 = (p0.0 as i32, p0.1 as i32);
        let p1 = (p1.0 as i32, p1.1 as i32);
        let pw = prod_walls_sorted(&b, &s);
        let rw = rf.legal_walls(&hw, &vw, p0, p1, s.walls_left[s.turn as usize]);
        let bw = prod_walls_bf_sorted(&b, &s);
        assert_eq!(
            pw, rw,
            "WALL(fast vs ref) mismatch w={w} h={h} pawns={:?} turn={} hw={:?} vw={:?}",
            s.pawn, s.turn, hw, vw
        );
        assert_eq!(
            pw, bw,
            "WALL(fast vs brute) mismatch w={w} h={h} pawns={:?} turn={} hw={:?} vw={:?}",
            s.pawn, s.turn, hw, vw
        );

        checked += 1;

        for m in legal_moves(&b, &s) {
            let s2 = apply(&b, &s, m);
            if seen.len() < cap && !seen.contains(&s2) {
                seen.insert(s2);
                stack.push(s2);
            }
        }
        if seen.len() >= cap {
            // keep draining stack but stop adding; ensures we cover a big chunk
        }
    }
    checked
}

#[test]
fn audit_exhaustive_rect_movegen() {
    // Rectangular boards — NOT covered by the square-only smallboard fixture.
    let mut total = 0;
    total += exhaustive_check(4, 3, 1, 200_000);
    total += exhaustive_check(3, 4, 1, 200_000);
    total += exhaustive_check(5, 3, 2, 400_000);
    total += exhaustive_check(3, 5, 2, 400_000);
    total += exhaustive_check(4, 4, 2, 400_000);
    total += exhaustive_check(6, 3, 1, 300_000);
    eprintln!("audit_exhaustive_rect_movegen checked {total} nodes");
    assert!(total > 10_000);
}

// Hand-built adversarial jump scenarios at corners/edges with walls.
fn jump_scan(W: u8, H: u8) -> usize {
    // pawns adjacent, with up to 2 walls forcing diagonal jumps.
    let b = Board::new(W, H, 5);
    let rf = Ref { w: W as i32, h: H as i32 };
    // Enumerate every pawn placement (distinct) x every wall subset of small size
    // is too big; instead enumerate all adjacency configs with up to 2 walls.
    let all_anchors: Vec<(bool, u8, u8)> = {
        let mut v = Vec::new();
        for horiz in [true, false] {
            for wc in 0..b.w - 1 {
                for wr in 0..b.h - 1 {
                    v.push((horiz, wc, wr));
                }
            }
        }
        v
    };
    let mut checked = 0usize;
    for p0c in 0..W {
        for p0r in 0..H {
            for p1c in 0..W {
                for p1r in 0..H {
                    if (p0c, p0r) == (p1c, p1r) {
                        continue;
                    }
                    // only adjacency-relevant: skip if not orthogonally adjacent
                    let adj =
                        (p0c.abs_diff(p1c) + p0r.abs_diff(p1r)) == 1;
                    if !adj {
                        continue;
                    }
                    // up to 2 walls from the anchor list
                    for i in 0..all_anchors.len() {
                        for j in i..all_anchors.len() {
                            let mut hw = HashSet::new();
                            let mut vw = HashSet::new();
                            let mut s = b.initial();
                            s.h_walls = 0;
                            s.v_walls = 0;
                            let mut ok = true;
                            for &(horiz, wc, wr) in [all_anchors[i], all_anchors[j]].iter() {
                                if horiz {
                                    s.h_walls |= 1u64 << b.hbit(wc, wr);
                                    hw.insert((wc as i32, wr as i32));
                                } else {
                                    s.v_walls |= 1u64 << b.vbit(wc, wr);
                                    vw.insert((wc as i32, wr as i32));
                                }
                            }
                            let _ = ok;
                            ok = true;
                            let _ = ok;
                            for turn in 0..2u8 {
                                s.pawn = [b.idx(p0c, p0r), b.idx(p1c, p1r)];
                                s.turn = turn;
                                if b.is_terminal(&s) {
                                    continue;
                                }
                                let me = if turn == 0 {
                                    (p0c as i32, p0r as i32)
                                } else {
                                    (p1c as i32, p1r as i32)
                                };
                                let opp = if turn == 0 {
                                    (p1c as i32, p1r as i32)
                                } else {
                                    (p0c as i32, p0r as i32)
                                };
                                let ps = prod_steps_sorted(&b, &s);
                                let rs = rf.legal_steps(&hw, &vw, me, opp);
                                assert_eq!(
                                    ps, rs,
                                    "JUMP mismatch me={:?} opp={:?} turn={} hw={:?} vw={:?}\nprod={:?} ref={:?}",
                                    me, opp, turn, hw, vw, ps, rs
                                );
                                checked += 1;
                            }
                        }
                    }
                }
            }
        }
    }
    checked
}

#[test]
fn audit_jump_corner_edge_cases() {
    let mut total = 0;
    total += jump_scan(3, 3);
    total += jump_scan(4, 3);
    total += jump_scan(3, 4);
    total += jump_scan(4, 4);
    total += jump_scan(5, 3);
    eprintln!("audit_jump_corner_edge_cases checked {total}");
    assert!(total > 5000);
}

// Adversarial wall-legality: exhaustively enumerate ALL wall subsets up to
// size 3 on small boards with pawns parked in/near corners, and for every
// resulting position assert fast-path legal_walls == brute-force legal_walls.
// This is the "boxing a pawn near the edge" stress: small boards make a single
// new wall capable of sealing a pawn's last escape, so any unsound `false`
// from needs_path_check would admit an illegal trapping wall here.
fn box_scan(W: u8, H: u8, max_walls: usize) -> usize {
    let b = Board::new(W, H, 5);
    let anchors: Vec<(bool, u8, u8)> = {
        let mut v = Vec::new();
        for horiz in [true, false] {
            for wc in 0..W - 1 {
                for wr in 0..H - 1 {
                    v.push((horiz, wc, wr));
                }
            }
        }
        v
    };
    let n = anchors.len();
    let mut checked = 0usize;
    // pawn placements: park p0 and p1 at assorted corner/edge cells.
    let pawn_spots: Vec<(u8, u8)> = vec![
        (0, 0),
        (W - 1, 0),
        (0, H - 1),
        (W - 1, H - 1),
        (W / 2, 0),
        (W / 2, H - 1),
        (0, H / 2),
        (W - 1, H / 2),
    ];
    // Enumerate wall subsets up to max_walls. We don't require the subset to be
    // mutually legal (overlapping anchors just get filtered by `overlaps` in
    // both fast and brute paths identically); we only need a valid bitset.
    // Use combinations via nested indices for sizes 1..=max_walls.
    let mut subsets: Vec<Vec<usize>> = vec![vec![]];
    if max_walls >= 1 {
        for i in 0..n {
            subsets.push(vec![i]);
            if max_walls >= 2 {
                for j in i + 1..n {
                    subsets.push(vec![i, j]);
                    if max_walls >= 3 {
                        for k in j + 1..n {
                            subsets.push(vec![i, j, k]);
                        }
                    }
                }
            }
        }
    }
    for subset in &subsets {
        let mut h_walls = 0u64;
        let mut v_walls = 0u64;
        let mut conflict = false;
        for &ix in subset {
            let (horiz, wc, wr) = anchors[ix];
            if horiz {
                let bit = 1u64 << b.hbit(wc, wr);
                h_walls |= bit;
            } else {
                let bit = 1u64 << b.vbit(wc, wr);
                v_walls |= bit;
            }
        }
        let _ = conflict;
        for &p0 in &pawn_spots {
            for &p1 in &pawn_spots {
                if p0 == p1 {
                    continue;
                }
                for turn in 0..2u8 {
                    let s = State {
                        pawn: [b.idx(p0.0, p0.1), b.idx(p1.0, p1.1)],
                        h_walls,
                        v_walls,
                        walls_left: [3, 3],
                        turn,
                    };
                    if b.is_terminal(&s) {
                        continue;
                    }
                    // If the *existing* walls already disconnect someone, the
                    // position is unreachable in real play, but legal_walls must
                    // still agree fast-vs-brute; include it anyway.
                    let fast = prod_walls_sorted(&b, &s);
                    let brute = prod_walls_bf_sorted(&b, &s);
                    assert_eq!(
                        fast, brute,
                        "BOX fast!=brute W={W} H={H} pawns={:?} turn={} hw={:#x} vw={:#x}",
                        s.pawn, turn, h_walls, v_walls
                    );
                    checked += 1;
                }
            }
        }
    }
    checked
}

#[test]
fn audit_box_pawn_near_edge() {
    let mut total = 0;
    total += box_scan(3, 3, 3);
    total += box_scan(4, 3, 2);
    total += box_scan(3, 4, 2);
    total += box_scan(4, 4, 2);
    eprintln!("audit_box_pawn_near_edge checked {total}");
    assert!(total > 50_000);
}

// Solver vs unpruned oracle on small rectangular boards, REUSING one Solver
// across many positions (reuse is the stated risk).
#[test]
fn audit_solver_vs_brute_reused_rect() {
    for &(w, h, walls, depth) in &[
        (3u8, 4u8, 1u8, 16u32),
        (4, 3, 1, 16),
        (4, 4, 1, 18),
        (3, 3, 2, 16),
        (5, 3, 1, 18),
        (3, 5, 1, 18),
    ] {
        let b = Board::new(w, h, walls);
        let mut sol = Solver::new(&b); // REUSED across all positions below
        // BFS a bounded set of reachable positions and compare.
        let mut seen: HashSet<State> = HashSet::new();
        let mut stack = vec![b.initial()];
        seen.insert(b.initial());
        let mut checked = 0usize;
        let cap = 4000usize;
        while let Some(s) = stack.pop() {
            if b.is_terminal(&s) {
                continue;
            }
            let got = sol.solve(&s);
            // Depth-stability gate: only trust the oracle where it is saturated
            // (same answer at depth and depth+2), so a shallow `Draw` floor can
            // never masquerade as a real mismatch.
            let want = brute_value(&b, &s, depth);
            let want2 = brute_value(&b, &s, depth + 2);
            if want == want2 {
                assert_eq!(
                    got, want,
                    "SOLVE mismatch w={w} h={h} walls={walls} pawns={:?} turn={} wl={:?} hw={:#x} vw={:#x}",
                    s.pawn, s.turn, s.walls_left, s.h_walls, s.v_walls
                );
                checked += 1;
            }
            if checked >= 1500 {
                break;
            }
            for m in legal_moves(&b, &s) {
                let s2 = apply(&b, &s, m);
                if seen.len() < cap && !seen.contains(&s2) {
                    seen.insert(s2);
                    stack.push(s2);
                }
            }
        }
        eprintln!("audit_solver_vs_brute_reused_rect w={w} h={h} walls={walls} checked {checked}");
        assert!(checked > 20);
    }
}
