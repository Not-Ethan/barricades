//! Profiling CLI for the exact solver.
//!
//! Usage: `solve <W> <H> <WALLS>` — builds the `W x H` board with `WALLS` walls
//! per player, solves the initial position to its game-theoretic value, and
//! prints one line of search statistics (value, nodes visited, transposition
//! table size, an estimate of the TT memory footprint, and wall-clock time).
//!
//! The figures drive the RunPod feasibility estimate, so the emphasis is on the
//! search cost — `nodes` counts every internal node entered by the main
//! alpha-beta search *and* the wall-less race endgame; `tt_entries` is the live
//! transposition-table size; `tt_bytes` is a rough lower-bound estimate of the
//! TT footprint (entry count times the per-entry key+value size), not true RSS.

use std::process::ExitCode;
use std::time::Instant;

use quoridor_solver::board::Board;
use quoridor_solver::solver::Solver;

fn parse_u8(s: &str) -> Option<u8> {
    s.parse::<u8>().ok()
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 4 {
        eprintln!("usage: {} <W> <H> <WALLS>", args.first().map_or("solve", |s| s.as_str()));
        return ExitCode::FAILURE;
    }
    let (w, h, walls) = match (parse_u8(&args[1]), parse_u8(&args[2]), parse_u8(&args[3])) {
        (Some(w), Some(h), Some(walls)) => (w, h, walls),
        _ => {
            eprintln!("usage: {} <W> <H> <WALLS>  (all three must be u8)", args[0]);
            return ExitCode::FAILURE;
        }
    };

    let board = Board::new(w, h, walls);
    let start = board.initial();

    let mut solver = Solver::new(&board);
    let t0 = Instant::now();
    let value = solver.solve(&start);
    let elapsed = t0.elapsed();

    // `tt_bytes` is a rough lower-bound estimate of the dominant TT memory
    // (entries times per-entry key+value size); it ignores HashMap overhead, so
    // it under-counts true RSS — a TT-footprint estimate, not real resident size.
    let tt_entries = solver.tt_len();
    let tt_bytes = solver.tt_bytes();

    println!(
        "W×H={}×{} walls={}  value={:?}  nodes={}  tt_entries={}  tt_bytes≈{}  time={:.3}s",
        w,
        h,
        walls,
        value,
        solver.nodes,
        tt_entries,
        tt_bytes,
        elapsed.as_secs_f64(),
    );

    ExitCode::SUCCESS
}
