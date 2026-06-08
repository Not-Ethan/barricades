use pyo3::prelude::*;

mod bitboard;
mod coords;
mod encoding;
mod movegen;
mod pyiface;
mod state;

#[pymodule]
fn barricades_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    pyiface::register(m)
}
