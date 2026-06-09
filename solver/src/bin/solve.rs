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
    // Staged-measurement toggles (default ON). `QS_ORDERING=0` disables
    // killer/history ordering; `QS_SYMMETRY=0` disables mirror TT
    // canonicalization. Neither changes the returned value.
    let off = |k: &str| std::env::var(k).map(|v| v == "0").unwrap_or(false);
    if off("QS_ORDERING") {
        solver.set_use_ordering(false);
    }
    if off("QS_SYMMETRY") {
        solver.set_use_symmetry(false);
    }
    let t0 = Instant::now();
    let value = solver.solve(&start);
    let elapsed = t0.elapsed();

    // The main TT is now a DENSE, fixed-capacity packed-key array. `tt_entries`
    // is the live occupied-slot count; `tt_capacity` the fixed slot count;
    // `tt_bytes` the exact heap footprint of the flat array (capacity *
    // entry_size, fully resident); `entry_size` the per-slot byte size. Capacity
    // is set by `QS_TT_MB` (megabytes; default 2048).
    let tt_entries = solver.tt_len();
    let tt_capacity = solver.tt_capacity();
    let tt_bytes = solver.tt_bytes();
    let fill_pct = if tt_capacity > 0 {
        100.0 * tt_entries as f64 / tt_capacity as f64
    } else {
        0.0
    };

    println!(
        "W×H={}×{} walls={}  value={:?}  nodes={}  tt_entries={}  tt_capacity={}  tt_fill={:.1}%  tt_bytes={}  entry_size={}  time={:.3}s",
        w,
        h,
        walls,
        value,
        solver.nodes,
        tt_entries,
        tt_capacity,
        fill_pct,
        tt_bytes,
        Solver::tt_entry_size(),
        elapsed.as_secs_f64(),
    );

    ExitCode::SUCCESS
}
