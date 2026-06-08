use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

use crate::bitboard::bfs_dist;
use crate::encoding::{move_to_action, N_ACTIONS};
use crate::movegen::legal_moves;
use crate::state::{apply_move, is_terminal, winner, GameState, Move};

const HEUR_SCALE: f64 = 5.0;

struct Node {
    state: GameState,
    parent: i32,
    mv: Option<Move>,
    prior: f32,
    children: Vec<u32>,
    n: u32,
    w: f64,
    expanded: bool,
}

pub enum Leaf {
    Parked,
    Terminal,
}

pub struct Tree {
    nodes: Vec<Node>,
    root: u32,
    root_player: usize,
    c_puct: f64,
    parked: Option<usize>,
    noised: bool,
    rng: StdRng,
}

impl Tree {
    pub fn new(state: GameState, c_puct: f64, seed: u64) -> Tree {
        let root = Node {
            state,
            parent: -1,
            mv: None,
            prior: 0.0,
            children: Vec::new(),
            n: 0,
            w: 0.0,
            expanded: false,
        };
        Tree {
            nodes: vec![root],
            root: 0,
            root_player: state.turn as usize,
            c_puct,
            parked: None,
            noised: false,
            rng: StdRng::seed_from_u64(seed),
        }
    }

    fn select_child(&self, node: usize) -> usize {
        let sqrt_n = (self.nodes[node].n as f64).sqrt();
        let parent_turn = self.nodes[node].state.turn as usize;
        let mut best = self.nodes[node].children[0] as usize;
        let mut best_score = f64::NEG_INFINITY;
        for &ci in &self.nodes[node].children {
            let ch = &self.nodes[ci as usize];
            let mut q = if ch.n > 0 { ch.w / ch.n as f64 } else { 0.0 };
            if parent_turn != self.root_player {
                q = -q;
            }
            let u = self.c_puct * ch.prior as f64 * sqrt_n / (1.0 + ch.n as f64);
            let score = q + u;
            if score > best_score {
                best_score = score;
                best = ci as usize;
            }
        }
        best
    }

    fn backup(&mut self, mut node: usize, v: f64) {
        loop {
            self.nodes[node].n += 1;
            self.nodes[node].w += v;
            let p = self.nodes[node].parent;
            if p < 0 {
                break;
            }
            node = p as usize;
        }
    }

    fn expand_with_policy(&mut self, node: usize, policy: &[f32], value: f64) {
        let st = self.nodes[node].state;
        let legal = legal_moves(&st);
        let mut sum = 0f32;
        let mut probs = Vec::with_capacity(legal.len());
        for m in &legal {
            let p = policy[move_to_action(m, &st)];
            sum += p;
            probs.push(p);
        }
        for (i, m) in legal.iter().enumerate() {
            let prior = if sum > 0.0 { probs[i] / sum } else { 1.0 / legal.len() as f32 };
            let child = Node {
                state: apply_move(&st, m),
                parent: node as i32,
                mv: Some(*m),
                prior,
                children: Vec::new(),
                n: 0,
                w: 0.0,
                expanded: false,
            };
            let idx = self.nodes.len() as u32;
            self.nodes.push(child);
            self.nodes[node].children.push(idx);
        }
        self.nodes[node].expanded = true;
        let v = if st.turn as usize == self.root_player { value } else { -value };
        self.backup(node, v);
    }

    pub fn prepare_leaf(&mut self, planes_out: &mut [f32]) -> Leaf {
        let mut node = self.root as usize;
        while self.nodes[node].expanded && !is_terminal(&self.nodes[node].state) {
            node = self.select_child(node);
        }
        if is_terminal(&self.nodes[node].state) {
            let w = winner(&self.nodes[node].state).unwrap();
            let v = if w == self.root_player { 1.0 } else { -1.0 };
            self.backup(node, v);
            return Leaf::Terminal;
        }
        crate::encoding::encode_planes(&self.nodes[node].state, planes_out);
        self.parked = Some(node);
        Leaf::Parked
    }

    pub fn receive(&mut self, policy: &[f32], value: f64) {
        let node = self.parked.take().expect("receive() called without a preceding parked leaf");
        self.expand_with_policy(node, policy, value);
    }

    fn heuristic_value(s: &GameState) -> f64 {
        if let Some(w) = winner(s) {
            return if w == s.turn as usize { 1.0 } else { -1.0 };
        }
        let d_self = bfs_dist(s, s.turn as usize).unwrap_or(1000) as f64;
        let d_opp = bfs_dist(s, 1 - s.turn as usize).unwrap_or(1000) as f64;
        ((d_opp - d_self) / HEUR_SCALE).tanh()
    }

    pub fn run_heuristic(&mut self, sims: u32) -> Option<Move> {
        let uniform = vec![1.0f32 / N_ACTIONS as f32; N_ACTIONS];
        let mut evals = 0u32;
        let mut guard = 0u32;
        let cap = sims * 8 + 64;
        while evals < sims && guard < cap {
            guard += 1;
            let mut node = self.root as usize;
            while self.nodes[node].expanded && !is_terminal(&self.nodes[node].state) {
                node = self.select_child(node);
            }
            if is_terminal(&self.nodes[node].state) {
                let w = winner(&self.nodes[node].state).unwrap();
                let v = if w == self.root_player { 1.0 } else { -1.0 };
                self.backup(node, v);
                continue;
            }
            let val = Tree::heuristic_value(&self.nodes[node].state);
            self.expand_with_policy(node, &uniform, val);
            evals += 1;
        }
        self.best_move(0.0).map(|(mv, _)| mv)
    }

    pub fn apply_root_noise(&mut self, alpha: f64, eps: f64) {
        if self.noised || !self.nodes[self.root as usize].expanded {
            return;
        }
        use rand_distr::{Distribution, Gamma};
        let kids: Vec<u32> = self.nodes[self.root as usize].children.clone();
        if kids.is_empty() {
            return;
        }
        let gamma = match Gamma::new(alpha, 1.0) {
            Ok(g) => g,
            Err(_) => return, // invalid alpha (<=0, NaN): noise is optional, no-op
        };
        let mut g: Vec<f64> = (0..kids.len()).map(|_| gamma.sample(&mut self.rng)).collect();
        let tot: f64 = g.iter().sum::<f64>().max(1e-12);
        for x in g.iter_mut() {
            *x /= tot;
        }
        for (i, &ci) in kids.iter().enumerate() {
            let p = self.nodes[ci as usize].prior as f64;
            self.nodes[ci as usize].prior = ((1.0 - eps) * p + eps * g[i]) as f32;
        }
        self.noised = true;
    }

    pub fn best_move(&mut self, temp: f64) -> Option<(Move, [f32; N_ACTIONS])> {
        let kids = self.nodes[self.root as usize].children.clone();
        if kids.is_empty() {
            return None;
        }
        let root_state = self.nodes[self.root as usize].state;
        let mut pi = [0f32; N_ACTIONS];
        let total: u32 = kids.iter().map(|&c| self.nodes[c as usize].n).sum();
        if total > 0 {
            for &c in &kids {
                let a = move_to_action(self.nodes[c as usize].mv.as_ref().unwrap(), &root_state);
                pi[a] = self.nodes[c as usize].n as f32 / total as f32;
            }
        } else {
            let u = 1.0 / kids.len() as f32;
            for &c in &kids {
                let a = move_to_action(self.nodes[c as usize].mv.as_ref().unwrap(), &root_state);
                pi[a] = u;
            }
        }
        let chosen = if temp == 0.0 {
            let top = kids.iter().map(|&c| self.nodes[c as usize].n).max().unwrap();
            let winners: Vec<u32> = kids
                .iter()
                .cloned()
                .filter(|&c| self.nodes[c as usize].n == top)
                .collect();
            winners[self.rng.random_range(0..winners.len())]
        } else {
            let r: f32 = self.rng.random::<f32>() * total.max(1) as f32;
            let mut acc = 0f32;
            let mut pick = kids[0];
            for &c in &kids {
                acc += self.nodes[c as usize].n as f32;
                if acc >= r {
                    pick = c;
                    break;
                }
            }
            pick
        };
        Some((self.nodes[chosen as usize].mv.unwrap(), pi))
    }
}
