use rayon::prelude::*;

use crate::bitboard::bfs_dist;
use crate::encoding::encode_planes;
use crate::endgame::solve_race;
use crate::mcts::{Leaf, Tree};
use crate::state::{apply_move, initial_state, is_terminal, winner, GameState};

const N_FEATS: usize = 4;

pub struct Example {
    pub planes: Vec<f32>,
    pub pi: Vec<f32>,
    pub z: f32,
    pub feats: [f32; N_FEATS],
}

struct Pending {
    planes: Vec<f32>,
}

struct Record {
    planes: Vec<f32>,
    pi: Vec<f32>,
    player: usize,
    feats: [f32; N_FEATS],
}

enum Phase {
    AwaitingEval,
    ReadyToMove,
}

struct Slot {
    game: GameState,
    tree: Tree,
    sims_done: u32,
    ply: u32,
    phase: Phase,
    records: Vec<Record>,
    active: bool,
    pending: Option<Pending>,
    forced_outcome: Option<Option<usize>>,
}

#[derive(Clone, Copy)]
pub struct Config {
    pub sims: u32,
    pub c_puct: f64,
    pub dirichlet_alpha: f64,
    pub dirichlet_eps: f64,
    pub temp_moves: u32,
    pub max_plies: u32,
    pub carryover: bool,
    pub endgame_solve: bool,
}

pub struct SelfPlayPool {
    slots: Vec<Slot>,
    cfg: Config,
    next_seed: u64,
    launched: u32,
    total_games: u32,
    finished: u32,
    out_examples: Vec<Example>,
    last_pending: Vec<usize>,
    solved_games: u32,
}

fn features(g: &GameState) -> [f32; N_FEATS] {
    let mover = g.turn as usize;
    let d_self = bfs_dist(g, mover).unwrap_or(1000) as f32;
    let d_opp = bfs_dist(g, 1 - mover).unwrap_or(1000) as f32;
    [d_opp - d_self, g.walls_left[mover] as f32, g.walls_left[1 - mover] as f32, 0.0]
}

impl SelfPlayPool {
    pub fn new(n_games: u32, total_games: u32, cfg: Config, seed: u64) -> SelfPlayPool {
        let mut next_seed = seed;
        let mut slots = Vec::with_capacity(n_games as usize);
        let mut launched = 0u32;
        for _ in 0..n_games.min(total_games) {
            let g = initial_state();
            let mut tree = Tree::new(g, cfg.c_puct, next_seed);
            tree.set_endgame_solve(cfg.endgame_solve);
            slots.push(Slot {
                game: g,
                tree,
                sims_done: 0,
                ply: 0,
                phase: Phase::AwaitingEval,
                records: Vec::new(),
                active: true,
                pending: None,
                forced_outcome: None,
            });
            next_seed = next_seed.wrapping_add(1);
            launched += 1;
        }
        SelfPlayPool {
            slots, cfg, next_seed, launched, total_games,
            finished: 0, out_examples: Vec::new(), last_pending: Vec::new(),
            solved_games: 0,
        }
    }

    fn refill(&mut self, i: usize) {
        if self.launched < self.total_games {
            let g = initial_state();
            let seed = self.next_seed;
            self.next_seed = self.next_seed.wrapping_add(1);
            self.slots[i].game = g;
            self.slots[i].tree = Tree::new(g, self.cfg.c_puct, seed);
            self.slots[i].tree.set_endgame_solve(self.cfg.endgame_solve);
            self.slots[i].sims_done = 0;
            self.slots[i].ply = 0;
            self.slots[i].phase = Phase::AwaitingEval;
            self.slots[i].records.clear();
            self.slots[i].active = true;
            self.slots[i].pending = None;
            self.slots[i].forced_outcome = None;
            self.launched += 1;
        } else {
            self.slots[i].active = false;
            self.slots[i].pending = None;
        }
    }

    fn commit_move(&mut self, i: usize) -> bool {
        let cfg = self.cfg;
        // Take a seed up front to avoid borrowing self while holding &mut self.slots[i].
        let seed = self.next_seed;
        self.next_seed = self.next_seed.wrapping_add(1);
        let slot = &mut self.slots[i];
        let temp = if slot.ply < cfg.temp_moves { 1.0 } else { 0.0 };
        let (mv, pi) = slot
            .tree
            .best_move(temp)
            .expect("pool: best_move on a non-terminal searched root must have children");
        let pre = slot.game;
        let mut planes = vec![0f32; 6 * 81];
        encode_planes(&pre, &mut planes);
        slot.records.push(Record { planes, pi: pi.to_vec(), player: pre.turn as usize, feats: features(&pre) });
        let next = apply_move(&pre, &mv);
        slot.game = next;
        slot.ply += 1;
        let ply = slot.ply;
        if is_terminal(&next) {
            return false; // natural win -> finalize uses winner(next)
        }
        if cfg.endgame_solve && next.walls_left == [0, 0] {
            let (val_mover, _) = solve_race(&next);
            let w = if val_mover > 0 {
                Some(next.turn as usize)
            } else if val_mover < 0 {
                Some(1 - next.turn as usize)
            } else {
                None // draw at bound
            };
            // `slot` (a &mut self.slots[i] borrow) is no longer used past this
            // point; compute w into a local, then write through self.* and
            // return immediately (re-borrowing self.slots[i] is fine here).
            self.slots[i].forced_outcome = Some(w);
            self.solved_games += 1;
            return false; // truncate: race is decided
        }
        if ply >= cfg.max_plies {
            return false; // cap -> draw
        }
        let slot = &mut self.slots[i];
        if cfg.carryover {
            slot.tree.advance(mv);
            slot.sims_done = slot.tree.root_visits().min(cfg.sims);
            // Root is already expanded under carryover, so the feed-time
            // "sims_done==0" noise trigger won't fire -> apply noise now.
            // apply_root_noise is a no-op on an unexpanded root (then the feed
            // trigger handles it) and idempotent.
            if cfg.dirichlet_alpha > 0.0 && slot.tree.root_expanded() {
                slot.tree.apply_root_noise(cfg.dirichlet_alpha, cfg.dirichlet_eps);
            }
        } else {
            slot.tree = Tree::new(next, cfg.c_puct, seed);
            slot.tree.set_endgame_solve(cfg.endgame_solve);
            slot.sims_done = 0;
        }
        slot.phase = Phase::AwaitingEval;
        true
    }

    fn finalize(&mut self, i: usize) {
        let w = match self.slots[i].forced_outcome {
            Some(fw) => fw,
            None => winner(&self.slots[i].game),
        };
        let n = self.slots[i].records.len();
        let recs = std::mem::take(&mut self.slots[i].records);
        for (k, rec) in recs.into_iter().enumerate() {
            let z = match w {
                None => 0.0,
                Some(win) => if win == rec.player { 1.0 } else { -1.0 },
            };
            let mut feats = rec.feats;
            feats[3] = (n - k) as f32;
            self.out_examples.push(Example { planes: rec.planes, pi: rec.pi, z, feats });
        }
        self.finished += 1;
    }

    pub fn step(&mut self) -> (Vec<f32>, usize) {
        let n = self.slots.len();
        for i in 0..n {
            if !self.slots[i].active {
                continue;
            }
            if matches!(self.slots[i].phase, Phase::ReadyToMove) {
                let alive = self.commit_move(i);
                if !alive {
                    self.finalize(i);
                    self.refill(i);
                }
            }
        }
        let sims = self.cfg.sims;
        self.slots.par_iter_mut().for_each(|slot| {
            slot.pending = None;
            if !slot.active {
                return;
            }
            let mut buf = vec![0f32; 6 * 81];
            loop {
                match slot.tree.prepare_leaf(&mut buf) {
                    Leaf::Parked => {
                        slot.phase = Phase::AwaitingEval;
                        slot.pending = Some(Pending { planes: buf });
                        break;
                    }
                    Leaf::Terminal => {
                        slot.sims_done += 1;
                        if slot.sims_done >= sims {
                            slot.phase = Phase::ReadyToMove;
                            break;
                        }
                    }
                }
            }
        });
        self.last_pending.clear();
        let mut out = Vec::new();
        for i in 0..n {
            if let Some(p) = self.slots[i].pending.take() {
                out.extend_from_slice(&p.planes);
                self.last_pending.push(i);
            }
        }
        let m = self.last_pending.len();
        (out, m)
    }

    pub fn feed(&mut self, policy: &[f32], value: &[f32]) {
        let pending = self.last_pending.clone();
        let sims = self.cfg.sims;
        let (alpha, eps) = (self.cfg.dirichlet_alpha, self.cfg.dirichlet_eps);
        for (row, &i) in pending.iter().enumerate() {
            let pol = &policy[row * 140..row * 140 + 140];
            let v = value[row] as f64;
            let slot = &mut self.slots[i];
            slot.tree.receive(pol, v);
            if slot.sims_done == 0 && alpha > 0.0 {
                slot.tree.apply_root_noise(alpha, eps);
            }
            slot.sims_done += 1;
            if slot.sims_done >= sims {
                slot.phase = Phase::ReadyToMove;
            }
        }
    }

    pub fn drain(&mut self) -> Vec<Example> {
        std::mem::take(&mut self.out_examples)
    }

    pub fn pending_len(&self) -> usize {
        self.last_pending.len()
    }

    pub fn games_remaining(&self) -> u32 {
        self.total_games - self.finished
    }

    pub fn active(&self) -> usize {
        self.slots.iter().filter(|s| s.active).count()
    }

    pub fn games_solved(&self) -> u32 {
        self.solved_games
    }
}
