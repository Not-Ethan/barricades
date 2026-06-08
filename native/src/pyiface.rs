use pyo3::prelude::*;
use pyo3::types::PyModule;

use crate::bitboard::bfs_dist;
use crate::movegen::{is_blocked, legal_moves};
use crate::state::{apply_move, is_terminal, winner, GameState, Move};

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

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(legal_moves_py, m)?)?;
    m.add_function(wrap_pyfunction!(shortest_path_len_py, m)?)?;
    m.add_function(wrap_pyfunction!(is_blocked_py, m)?)?;
    m.add_function(wrap_pyfunction!(apply_move_py, m)?)?;
    m.add_function(wrap_pyfunction!(winner_py, m)?)?;
    m.add_function(wrap_pyfunction!(is_terminal_py, m)?)?;
    Ok(())
}
