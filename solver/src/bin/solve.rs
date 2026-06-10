//! Profiling CLI for the exact solver.
//!
//! Usage: `solve <W> <H> <WALLS>` — builds the `W x H` board with `WALLS` walls
//! per player, solves the initial position to its game-theoretic value, and
//! prints one line of search statistics (value, threads used, nodes visited,
//! transposition-table fill, TT memory footprint, race-memo fill, and
//! wall-clock time).
//!
//! Environment knobs:
//!   * `QS_ENGINE`   — `dfpn` selects the df-pn engine (Stage 1); default is
//!     the verified alpha-beta solver. Both compute the same exact value.
//!   * `QS_THREADS`  — worker threads for the lazy-SMP search (default
//!     num_cpus). `QS_THREADS=1` reproduces the single-thread value/behaviour.
//!   * `QS_TT_MB`    — main transposition-table budget in MiB (default 2048).
//!   * `QS_RACE_MB`  — bounded race-memo budget in MiB (default 1024); the race
//!     memo is config-granular LRU and exact, so the cap is value-neutral.
//!   * `QS_ORDERING=0` / `QS_SYMMETRY=0` — disable ordering / mirror TT
//!     canonicalization for staged measurement (neither changes the value).
//!   * `QS_T4=0` — disable the Theorem-4 one-sided frozen-race bounds
//!     (depth-infinity TT synthesis; value-neutral A/B knob, default ON). The
//!     `t4_fires` / `t4_cutoffs` stats count bound evaluations and the
//!     whole-subtree cutoffs they produced.
//!   * `QS_FOOTPRINT=0` — disable the Theorem-1 Win-direction wall-relevance
//!     footprint (mustplay) pruning (value-neutral A/B knob, default ON). The
//!     `fp_attempts` / `fp_extracted` / `fp_prunes` stats count extraction
//!     attempts, verified certificates compiled to masks, and wall moves
//!     skipped with zero search; `fp_avg_bits` is the mean footprint size in
//!     anchor bits over successful extractions.
//!   * df-pn only: `QS_DFPN_MB` (least-work TT MiB, default 1024), `QS_EPS`
//!     (1+ε trick, default 0.25), `QS_DFPN_H=0` (disable df-pn+ leaf init),
//!     `QS_DFPN_LOOP_CAP`, `QS_DFPN_SIM_BUDGET`, `QS_DFPN_FALLBACK_MB`
//!     (embedded AB fallback TT), and FDFPN dynamic widening (Stage 2):
//!     `QS_DFPN_WIDEN=0` disables, `QS_DFPN_WIDEN_BASE` / `QS_DFPN_WIDEN_FRAC`
//!     set the window (defaults 4 / 0.25).
//!
//! The parallel value is provably identical to the single-thread value (parallel
//! alpha-beta over a shared TT is exact). `nodes` counts every internal node
//! entered by the main alpha-beta search *and* the wall-less race endgame,
//! SUMMED across all worker threads (so it varies run-to-run with the thread
//! count; only the VALUE is deterministic). `tt_bytes` is the EXACT resident
//! heap footprint of the dense TT array (capacity * entry size).

use std::process::ExitCode;
use std::time::Instant;

use quoridor_solver::board::Board;
use quoridor_solver::dfpn::DfpnSolver;
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

    // Engine selection: `QS_ENGINE=dfpn` runs the df-pn engine (Stage 1);
    // anything else (default) runs the verified alpha-beta solver, which
    // remains the differential oracle.
    let engine = std::env::var("QS_ENGINE").unwrap_or_default();
    if engine.eq_ignore_ascii_case("dfpn") {
        let mut solver = DfpnSolver::new(&board);
        let t0 = Instant::now();
        let value = solver.solve(&start);
        let elapsed = t0.elapsed();
        let st = solver.stats;
        let fill_pct = if solver.tt_capacity() > 0 {
            100.0 * solver.tt_len() as f64 / solver.tt_capacity() as f64
        } else {
            0.0
        };
        let widen = match solver.widening() {
            Some((base, frac)) => format!("base={base},frac={frac}"),
            None => "off".to_string(),
        };
        println!(
            "W×H={}×{} walls={}  engine=dfpn  widen={}  value={:?}  nodes={}  mid_nodes={}  race_nodes={}  \
             tt_entries={}  tt_capacity={}  tt_fill={:.1}%  tt_bytes={}  rep_hits={}  twins={}  \
             sims={}  sim_nodes={}  sim_verified={}  fallbacks={} (child={} node={} twin={} root={})  \
             race_entries={}  time={:.3}s",
            w,
            h,
            walls,
            widen,
            value,
            st.total_nodes(),
            st.nodes,
            st.race_nodes,
            solver.tt_len(),
            solver.tt_capacity(),
            fill_pct,
            solver.tt_bytes(),
            st.rep_hits,
            st.twin_stores,
            st.sim_calls,
            st.sim_nodes,
            st.sim_verified,
            st.fallbacks(),
            st.fallback_child,
            st.fallback_node,
            st.fallback_twin,
            st.fallback_root,
            solver.race_tt_len(),
            elapsed.as_secs_f64(),
        );
        return ExitCode::SUCCESS;
    }

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

    let fp_avg_bits = if solver.fp_extractions > 0 {
        solver.fp_mask_bits as f64 / solver.fp_extractions as f64
    } else {
        0.0
    };
    println!(
        "W×H={}×{} walls={}  value={:?}  threads={}  nodes={}  t4_fires={}  t4_cutoffs={}  fp_attempts={}  fp_extracted={}  fp_prunes={}  fp_avg_bits={:.1}  tt_entries={}  tt_capacity={}  tt_fill={:.1}%  tt_bytes={}  entry_size={}  race_entries={}  race_configs={}  time={:.3}s",
        w,
        h,
        walls,
        value,
        solver.threads(),
        solver.nodes,
        solver.t4_fires,
        solver.t4_cutoffs,
        solver.fp_attempts,
        solver.fp_extractions,
        solver.fp_prunes,
        fp_avg_bits,
        tt_entries,
        tt_capacity,
        fill_pct,
        tt_bytes,
        Solver::tt_entry_size(),
        solver.race_tt_len(),
        solver.race_config_count(),
        elapsed.as_secs_f64(),
    );

    ExitCode::SUCCESS
}
