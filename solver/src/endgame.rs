//! Endgame tablebase slice: the *race* over a FROZEN wall configuration.
//!
//! When both players have exhausted their walls (`walls_left == [0, 0]`), no
//! MORE walls can ever be placed, but the walls already on the board stay
//! FROZEN. The game reduces to a pure pawn race over steps only, played on the
//! fixed maze those frozen walls define. This module computes the exact
//! game-theoretic value (Win / Loss / **Draw**) of such a race by EXACT
//! retrograde analysis over the finite pawn-state graph — no depth bound, no
//! panic.
//!
//! ## Frozen-wall races CAN be genuine draws
//!
//! A previous implementation rested on the (false) invariant that a wall-less
//! race is never a true draw and `panic!`d on anything else. That is wrong: the
//! race is "no MORE walls" but FROZEN walls remain, and a frozen maze can let
//! one pawn perpetually body-block the only corridor the other must traverse.
//! Neither side can then force a goal, so the position is a GENUINE DRAW (with
//! legal moves available — not zugzwang). Confirmed repro: `Board::new(7, 5, 4)`
//! with `pawn = [18, 24]`, `h_walls = 0x280240`, `v_walls = 0x500400`,
//! `walls_left = [0, 0]`, `turn = 1` is a stable Draw.
//!
//! ## Why retrograde is exact
//!
//! The race graph for a fixed wall configuration is finite: a node is
//! `(pawn0, pawn1, turn)` with `pawn0 != pawn1`, both on board, and there are at
//! most `(w*h)^2 * 2` of them. Retrograde analysis (backward induction /
//! standard pursuit-game labeling) computes the exact value of EVERY node of a
//! finite graph and handles cycles correctly:
//!
//!   * Terminal/stuck nodes are seeded as `Loss` for the mover (the opponent is
//!     already on its goal, or the mover has no legal step), and the rare
//!     own-goal node as `Win`.
//!   * A node is finalized `Win` as soon as ANY successor is finalized `Loss`
//!     (the mover can step into a position the opponent loses).
//!   * A node is finalized `Loss` only once ALL its successors are finalized
//!     `Win` (every move hands the opponent a win).
//!   * Any node never finalized is a `Draw`: neither side can force a win, i.e.
//!     a perpetual blockade. This is the unique fixpoint of the negamax
//!     equations on the `Loss < Draw < Win` lattice, so it is the exact value.
//!
//! Because the labeling is over the full finite graph with no truncation, no
//! false draw and no missed win is possible, and there is no depth to exhaust —
//! it is `O(nodes + edges)`, fast even on draw-heavy (blockade) mazes where the
//! old iterative-deepening search would deepen to its ceiling and panic.
//!
//! ## Value convention (matches the rest of the solver EXACTLY)
//!
//! For a state `s` with side-to-move `t = s.turn`:
//!   * `winner(s) == Some(t)`   -> `Win`  (own goal; unreachable in normal play).
//!   * `winner(s) == Some(1-t)` -> `Loss` (opponent already on its goal).
//!   * otherwise `value(s) = max over legal_steps of negate(value(child))`
//!     (plain negamax; `Win > Draw > Loss`), and a non-terminal node with NO
//!     legal step is a `Loss` for the mover.
//!
//! ## Persistent, exact memo
//!
//! The race value of a frozen-wall `State` is a pure, context-free function of
//! that `State` (pawns + frozen walls + turn, with `walls_left == [0, 0]`). One
//! retrograde pass labels EVERY reachable pawn-pair for the current wall
//! configuration at once, so we cache them ALL into the persistent `State`-keyed
//! memo (`tt`). Subsequent races with the same frozen walls — any pawn pair, any
//! turn — are then instant memo hits. This is exactly the `t = 0/1` slice of the
//! future `k`-wall endgame tablebase.

use crate::board::Board;
use crate::solver::Value;
use crate::state::State;
use rustc_hash::FxHashMap;

/// Persistent, exact race memo keyed on the bare (walls-frozen) `State`.
/// Every stored value is the position's EXACT game-theoretic value, so it is
/// sound to reuse across any race leaf within a `solve()` call (and across
/// `solve()` calls on the same `Solver`).
pub type RaceTt = FxHashMap<State, Value>;

/// A race-graph node: the two pawn cell indices plus the side to move. The
/// frozen wall configuration is fixed for the whole retrograde pass, so it is
/// NOT part of the node key (it lives in the `Board` + the wall bits we carry
/// in the working `State`).
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct Node {
    pawn: [u8; 2],
    turn: u8,
}

/// Final game-theoretic label assigned by the retrograde pass.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Label {
    Win,
    Loss,
}

/// Exact game value of a frozen-wall race for the side to move, paired with the
/// number of race nodes visited (for profiling; the value is unaffected).
///
/// Computed by EXACT retrograde (backward-induction) labeling over the finite
/// `(pawn0, pawn1, turn)` graph of the current frozen wall configuration. Win,
/// Loss, AND Draw are all returned exactly: there is no depth bound, no ceiling,
/// and no panic — a perpetual blockade is labeled `Draw`, never mis-resolved.
///
/// One pass labels every pawn pair reachable (forward) from `s`, and ALL of
/// them are cached into the persistent `State`-keyed memo `tt`, so any later
/// race with the same frozen walls is an instant hit.
pub fn race_value(b: &Board, s: &State, tt: &mut RaceTt) -> (Value, u64) {
    debug_assert_eq!(
        s.walls_left,
        [0, 0],
        "race_value called on a non-race state (walls remain): {:?}",
        s.walls_left
    );

    // Fast path: already memoized (e.g. a sibling leaf solved this wall config).
    let query = Node {
        pawn: s.pawn,
        turn: s.turn,
    };
    if let Some(&v) = tt.get(s) {
        return (v, 0);
    }

    // A reusable working state carrying the FROZEN wall bits; only `pawn`/`turn`
    // change as we explore. `walls_left` stays `[0, 0]` so movegen never offers
    // a wall move (and `legal_steps` ignores `walls_left` anyway).
    let mut work = *s;

    // ---- 1. Enumerate the forward-reachable component and build edges. ----
    //
    // `index[node] = id`; `succ[id]` = successor node ids; `pred[id]` =
    // predecessor node ids; `succ_remaining[id]` = count of not-yet-WIN
    // successors (the retrograde "all children are wins" counter).
    let mut index: FxHashMap<Node, u32> = FxHashMap::default();
    let mut nodes: Vec<Node> = Vec::new();
    let mut succ: Vec<Vec<u32>> = Vec::new();
    let mut pred: Vec<Vec<u32>> = Vec::new();
    // Per-node seed label (Win/Loss) if it is a terminal/stuck node, else None.
    let mut seed: Vec<Option<Label>> = Vec::new();

    let mut visited = 0u64; // profiling: race nodes touched.

    // Intern a node, allocating its id + parallel rows on first sight.
    let intern = |nodes: &mut Vec<Node>,
                      succ: &mut Vec<Vec<u32>>,
                      pred: &mut Vec<Vec<u32>>,
                      seed: &mut Vec<Option<Label>>,
                      index: &mut FxHashMap<Node, u32>,
                      n: Node|
     -> u32 {
        if let Some(&id) = index.get(&n) {
            return id;
        }
        let id = nodes.len() as u32;
        nodes.push(n);
        succ.push(Vec::new());
        pred.push(Vec::new());
        seed.push(None);
        index.insert(n, id);
        id
    };

    let start = intern(
        &mut nodes,
        &mut succ,
        &mut pred,
        &mut seed,
        &mut index,
        query,
    );

    // BFS/DFS the reachable component, recording successor + predecessor edges.
    let mut stack: Vec<u32> = vec![start];
    // `expanded[id]` guards against re-expanding a node (its edges are built
    // exactly once). A node can be interned (as a successor) before it is
    // expanded, so this is separate from membership in `index`.
    let mut expanded: Vec<bool> = vec![false];
    while let Some(id) = stack.pop() {
        if expanded[id as usize] {
            continue;
        }
        expanded[id as usize] = true;
        visited += 1;

        let node = nodes[id as usize];
        work.pawn = node.pawn;
        work.turn = node.turn;

        let t = node.turn;
        // Terminal handling (matches the solver's convention exactly):
        //  - own goal on the node's turn -> Win (rare; e.g. a query already on
        //    its goal). It has no race successors that matter.
        //  - opponent already on its goal -> Loss for the mover.
        match b.winner(&work) {
            Some(p) if p == t => {
                seed[id as usize] = Some(Label::Win);
                continue;
            }
            Some(_) => {
                seed[id as usize] = Some(Label::Loss);
                continue;
            }
            None => {}
        }

        let steps = crate::movegen::legal_steps(b, &work);
        if steps.is_empty() {
            // Non-terminal but stuck: a Loss for the mover (matches the solver —
            // a no-step node never improves past the initial Loss).
            seed[id as usize] = Some(Label::Loss);
            continue;
        }

        // Build successor edges. A step moves the mover's pawn and flips turn.
        for dest in steps {
            let mut child = node;
            child.pawn[t as usize] = dest;
            child.turn = 1 - t;
            let cid = intern(
                &mut nodes,
                &mut succ,
                &mut pred,
                &mut seed,
                &mut index,
                child,
            );
            // `expanded` must stay parallel to `nodes`.
            if cid as usize >= expanded.len() {
                expanded.resize(nodes.len(), false);
            }
            succ[id as usize].push(cid);
            pred[cid as usize].push(id);
            if !expanded[cid as usize] {
                stack.push(cid);
            }
        }
    }

    let n = nodes.len();
    // `succ_remaining[id]` starts at the successor count; each time a successor
    // is finalized WIN we decrement it. Reaching 0 means ALL successors are WIN
    // -> the node is a LOSS for its mover.
    let mut succ_remaining: Vec<u32> = succ.iter().map(|v| v.len() as u32).collect();

    // ---- 2. Initialize the worklist from seeded terminal/stuck nodes. ----
    let mut label: Vec<Option<Label>> = vec![None; n];
    let mut queue: Vec<u32> = Vec::new();
    for id in 0..n {
        if let Some(l) = seed[id] {
            label[id] = Some(l);
            queue.push(id as u32);
        }
    }

    // ---- 3. Propagate backward to fixpoint (standard pursuit retrograde). ----
    //
    //  - A node finalized LOSS makes EVERY predecessor a WIN (the predecessor's
    //    mover can step into this losing-for-the-opponent node).
    //  - A node finalized WIN decrements each predecessor's remaining counter;
    //    when a predecessor's counter hits 0 (all successors are WIN) it is a
    //    LOSS.
    let mut qi = 0usize;
    while qi < queue.len() {
        let id = queue[qi] as usize;
        qi += 1;
        match label[id].expect("queued node must be labeled") {
            Label::Loss => {
                // Predecessors become WIN.
                let preds = std::mem::take(&mut pred[id]);
                for &p in &preds {
                    let p = p as usize;
                    if label[p].is_none() {
                        label[p] = Some(Label::Win);
                        queue.push(p as u32);
                    }
                }
                pred[id] = preds;
            }
            Label::Win => {
                // Each predecessor loses one not-yet-WIN successor.
                let preds = std::mem::take(&mut pred[id]);
                for &p in &preds {
                    let p = p as usize;
                    if label[p].is_some() {
                        continue;
                    }
                    succ_remaining[p] -= 1;
                    if succ_remaining[p] == 0 {
                        label[p] = Some(Label::Loss);
                        queue.push(p as u32);
                    }
                }
                pred[id] = preds;
            }
        }
    }

    // ---- 4 & 5. Residue = Draw; cache EVERY node's exact value into `tt`. ----
    //
    // Caching all reachable pawn pairs (not just the query) is what makes a
    // later race with the same frozen walls an instant memo hit.
    let mut query_value: Option<Value> = None;
    for id in 0..n {
        let node = nodes[id];
        let v = match label[id] {
            Some(Label::Win) => Value::Win,
            Some(Label::Loss) => Value::Loss,
            None => Value::Draw, // never finalized -> perpetual blockade.
        };
        let key = State {
            pawn: node.pawn,
            h_walls: s.h_walls,
            v_walls: s.v_walls,
            walls_left: [0, 0],
            turn: node.turn,
        };
        tt.insert(key, v);
        if node == query {
            query_value = Some(v);
        }
    }

    // Sanity: retrograde labels every node of the finite graph (Win/Loss/Draw),
    // so the query — which is in the enumerated component by construction — is
    // always assigned a value. This assert can never fire on a correct pass.
    let v = query_value.expect(
        "retrograde pass must assign the query node a value (Win/Loss/Draw); \
         missing assignment indicates a bug in race enumeration",
    );
    (v, visited)
}
