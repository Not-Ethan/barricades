use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyModule;

use crate::bitboard::bfs_dist;
use crate::encoding::{action_to_move, encode_planes, move_to_action};
use crate::mcts::{Leaf, Tree as CoreTree};
use crate::movegen::{is_blocked, legal_moves};
use crate::selfplay::{Config, SelfPlayPool as CorePool};
use crate::state::{apply_move, is_terminal, winner, GameState, Move};
use numpy::{IntoPyArray, PyArray3, PyArray4, PyReadonlyArray1, PyReadonlyArray2};

pub fn parse_state(state: &Bound<'_, PyAny>) -> PyResult<GameState> {
    let pawns: ((i32, i32), (i32, i32)) = state.get_item(0)?.extract()?;
    let h: Vec<(i32, i32)> = state.get_item(1)?.extract()?;
    let v: Vec<(i32, i32)> = state.get_item(2)?.extract()?;
    let wl: (u8, u8) = state.get_item(3)?.extract()?;
    let turn: u8 = state.get_item(4)?.extract()?;
    let mut g = GameState {
        pawns: [(pawns.0 .0 as u8, pawns.0 .1 as u8), (pawns.1 .0 as u8, pawns.1 .1 as u8)],
        h_mask: 0, v_mask: 0, walls_left: [wl.0, wl.1], turn,
    };
    for (c, r) in h { g.h_mask |= 1u64 << (r * 8 + c); }
    for (c, r) in v { g.v_mask |= 1u64 << (r * 8 + c); }
    Ok(g)
}

pub fn parse_move(m: &Bound<'_, PyAny>) -> PyResult<Move> {
    let kind: String = m.get_item(0)?.extract()?;
    let c: i32 = m.get_item(1)?.extract()?;
    let r: i32 = m.get_item(2)?.extract()?;
    if kind == "step" { Ok(Move::Step { c, r }) }
    else { let o: String = m.get_item(3)?.extract()?; Ok(Move::Wall { c, r, orient: if o == "H" { 0 } else { 1 } }) }
}

impl<'py> IntoPyObject<'py> for Move {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;
    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        match self {
            Move::Step { c, r } => Ok(("step", c, r).into_pyobject(py)?.into_any()),
            Move::Wall { c, r, orient } => Ok(("wall", c, r, if orient == 0 { "H" } else { "V" }).into_pyobject(py)?.into_any()),
        }
    }
}

fn state_to_py(py: Python<'_>, g: &GameState) -> PyResult<Py<PyAny>> {
    let mut h: Vec<(i32, i32)> = Vec::new();
    let mut v: Vec<(i32, i32)> = Vec::new();
    for i in 0..64 {
        if (g.h_mask >> i) & 1 != 0 { h.push((i as i32 % 8, i as i32 / 8)); }
        if (g.v_mask >> i) & 1 != 0 { v.push((i as i32 % 8, i as i32 / 8)); }
    }
    h.sort();
    v.sort();
    let pawns = ((g.pawns[0].0 as i32, g.pawns[0].1 as i32), (g.pawns[1].0 as i32, g.pawns[1].1 as i32));
    let wl = (g.walls_left[0], g.walls_left[1]);
    Ok((pawns, h, v, wl, g.turn).into_pyobject(py)?.into_any().unbind())
}

#[pyfunction]
#[pyo3(name = "legal_moves")]
fn legal_moves_py(state: &Bound<'_, PyAny>) -> PyResult<Vec<Move>> { Ok(legal_moves(&parse_state(state)?)) }

#[pyfunction]
#[pyo3(name = "shortest_path_len")]
fn shortest_path_len_py(state: &Bound<'_, PyAny>, player: usize) -> PyResult<Option<u32>> { Ok(bfs_dist(&parse_state(state)?, player)) }

#[pyfunction]
#[pyo3(name = "is_blocked")]
fn is_blocked_py(state: &Bound<'_, PyAny>, a: (i32, i32), b: (i32, i32)) -> PyResult<bool> { Ok(is_blocked(&parse_state(state)?, a, b)) }

#[pyfunction]
#[pyo3(name = "apply_move")]
fn apply_move_py(py: Python<'_>, state: &Bound<'_, PyAny>, mv: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let g = apply_move(&parse_state(state)?, &parse_move(mv)?);
    state_to_py(py, &g)
}

#[pyfunction]
#[pyo3(name = "winner")]
fn winner_py(state: &Bound<'_, PyAny>) -> PyResult<Option<usize>> { Ok(winner(&parse_state(state)?)) }

#[pyfunction]
#[pyo3(name = "is_terminal")]
fn is_terminal_py(state: &Bound<'_, PyAny>) -> PyResult<bool> { Ok(is_terminal(&parse_state(state)?)) }

#[pyfunction]
#[pyo3(name = "encode_planes")]
fn encode_planes_py<'py>(py: Python<'py>, state: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyArray3<f32>>> {
    let g = parse_state(state)?;
    let mut buf = vec![0f32; 6 * 81];
    encode_planes(&g, &mut buf);
    let arr = numpy::ndarray::Array3::from_shape_vec((6, 9, 9), buf).expect("shape 6x9x9");
    Ok(arr.into_pyarray(py))
}

#[pyfunction]
#[pyo3(name = "move_to_action")]
fn move_to_action_py(mv: &Bound<'_, PyAny>, state: &Bound<'_, PyAny>) -> PyResult<usize> {
    let st = parse_state(state)?;
    let m = parse_move(mv)?;
    if !legal_moves(&st).contains(&m) {
        return Err(PyValueError::new_err(
            "move_to_action: move is not legal for the given state",
        ));
    }
    Ok(move_to_action(&m, &st))
}

#[pyfunction]
#[pyo3(name = "action_to_move")]
fn action_to_move_py(idx: usize, state: &Bound<'_, PyAny>) -> PyResult<Move> {
    if idx >= 140 {
        return Err(PyValueError::new_err(format!(
            "action_to_move: idx {} out of range [0,140)", idx
        )));
    }
    Ok(action_to_move(idx, &parse_state(state)?))
}

#[pyclass]
pub struct Tree {
    inner: CoreTree,
}

#[pymethods]
impl Tree {
    #[new]
    fn new(state: &Bound<'_, PyAny>, c_puct: f64, seed: u64) -> PyResult<Tree> {
        Ok(Tree { inner: CoreTree::new(parse_state(state)?, c_puct, seed) })
    }

    fn prepare_leaf<'py>(&mut self, py: Python<'py>) -> Option<Bound<'py, PyArray3<f32>>> {
        let mut buf = vec![0f32; 6 * 81];
        match self.inner.prepare_leaf(&mut buf) {
            Leaf::Parked => {
                let arr = numpy::ndarray::Array3::from_shape_vec((6, 9, 9), buf).unwrap();
                Some(arr.into_pyarray(py))
            }
            Leaf::Terminal => None,
        }
    }

    fn receive(&mut self, policy: PyReadonlyArray1<f32>, value: f64) -> PyResult<()> {
        let pol = policy.as_slice()?;
        if pol.len() != 140 {
            return Err(PyValueError::new_err(format!(
                "receive: expected policy of length 140, got {}", pol.len()
            )));
        }
        self.inner.receive(pol, value);
        Ok(())
    }

    fn run_heuristic(&mut self, sims: u32) -> PyResult<Move> {
        self.inner.run_heuristic(sims).ok_or_else(|| PyValueError::new_err("run_heuristic: root has no legal moves (terminal state?)"))
    }

    #[pyo3(signature = (alpha, eps=0.25))]
    fn apply_root_noise(&mut self, alpha: f64, eps: f64) {
        self.inner.apply_root_noise(alpha, eps);
    }

    fn best_move(&mut self, temp: f64) -> PyResult<(Move, Vec<f32>)> {
        let (mv, pi) = self.inner.best_move(temp).ok_or_else(|| PyValueError::new_err("best_move: root has no children (terminal or unsearched?)"))?;
        Ok((mv, pi.to_vec()))
    }

    fn advance(&mut self, mv: &Bound<'_, PyAny>) -> PyResult<()> {
        self.inner.advance(parse_move(mv)?);
        Ok(())
    }

    fn root_visits(&self) -> u32 {
        self.inner.root_visits()
    }
}

#[pyclass]
pub struct SelfPlayPool {
    inner: CorePool,
}

#[pymethods]
impl SelfPlayPool {
    #[new]
    #[pyo3(signature = (n_games, total_games, sims, c_puct=1.5, seed=0,
                        dirichlet_alpha=0.5, dirichlet_eps=0.25,
                        temp_moves=10, max_plies=200, carryover=true))]
    fn new(n_games: u32, total_games: u32, sims: u32, c_puct: f64, seed: u64,
           dirichlet_alpha: f64, dirichlet_eps: f64, temp_moves: u32, max_plies: u32,
           carryover: bool) -> SelfPlayPool {
        let cfg = Config { sims, c_puct, dirichlet_alpha, dirichlet_eps,
                           temp_moves, max_plies, carryover };
        SelfPlayPool { inner: CorePool::new(n_games, total_games, cfg, seed) }
    }

    fn step<'py>(&mut self, py: Python<'py>) -> Option<Bound<'py, PyArray4<f32>>> {
        let (buf, m) = py.detach(|| self.inner.step());
        if m == 0 {
            return None;
        }
        let arr = numpy::ndarray::Array4::from_shape_vec((m, 6, 9, 9), buf).unwrap();
        Some(arr.into_pyarray(py))
    }

    fn feed(&mut self, policy: PyReadonlyArray2<f32>, value: PyReadonlyArray1<f32>) -> PyResult<()> {
        let pol = policy.as_slice()?;
        let val = value.as_slice()?;
        let m = self.inner.pending_len();
        if pol.len() != m * 140 || val.len() != m {
            return Err(PyValueError::new_err(format!(
                "feed: expected policy ({m}, 140) and value ({m},) matching the last step()'s batch; got policy {} elems, value {} elems",
                pol.len(),
                val.len()
            )));
        }
        self.inner.feed(pol, val);
        Ok(())
    }

    fn drain<'py>(&mut self, py: Python<'py>) -> PyResult<Vec<Py<PyAny>>> {
        let mut out = Vec::new();
        for ex in self.inner.drain() {
            let planes = numpy::ndarray::Array3::from_shape_vec((6, 9, 9), ex.planes).unwrap().into_pyarray(py);
            let pi = numpy::ndarray::Array1::from_vec(ex.pi).into_pyarray(py);
            let feats = numpy::ndarray::Array1::from_vec(ex.feats.to_vec()).into_pyarray(py);
            out.push((planes, pi, ex.z, feats).into_pyobject(py)?.into_any().unbind());
        }
        Ok(out)
    }

    fn games_remaining(&self) -> u32 {
        self.inner.games_remaining()
    }

    fn active(&self) -> usize {
        self.inner.active()
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(legal_moves_py, m)?)?;
    m.add_function(wrap_pyfunction!(shortest_path_len_py, m)?)?;
    m.add_function(wrap_pyfunction!(is_blocked_py, m)?)?;
    m.add_function(wrap_pyfunction!(apply_move_py, m)?)?;
    m.add_function(wrap_pyfunction!(winner_py, m)?)?;
    m.add_function(wrap_pyfunction!(is_terminal_py, m)?)?;
    m.add_function(wrap_pyfunction!(encode_planes_py, m)?)?;
    m.add_function(wrap_pyfunction!(move_to_action_py, m)?)?;
    m.add_function(wrap_pyfunction!(action_to_move_py, m)?)?;
    m.add_class::<Tree>()?;
    m.add_class::<SelfPlayPool>()?;
    Ok(())
}
