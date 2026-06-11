#!/usr/bin/env bash
# RunPod solver setup — run ON THE POD (Ubuntu CPU pod). Idempotent.
# Usage: bash runpod_solver_setup.sh   (repo already rsynced/cloned to ~/barricades)
set -euo pipefail

echo "=== [1/4] toolchain ==="
command -v cc >/dev/null || { apt-get update && apt-get install -y build-essential curl git; }
command -v cargo >/dev/null || {
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
  . "$HOME/.cargo/env"
}
. "$HOME/.cargo/env" 2>/dev/null || true
rustc --version && cargo --version

echo "=== [2/4] build ==="
cd "$HOME/barricades/solver"
cargo build --release --bin solve
ls -la target/release/solve

echo "=== [3/4] exactness gates (targeted; ~10 min) ==="
# Skip the multi-CPU-hour legacy audits (zz_sign_audit / zzz_audit_*) — their
# coverage is subsumed by these suites.
cargo test --release \
  --lib \
  --test dsu_walls --test wall_legality --test writeup_values \
  --test race_exactness --test race_cap_neutral --test race_memo_persistence \
  --test parallel_exact --test symmetry --test tt_depthfold \
  --test theorem4 --test footprint --test diff_vs_smallboard \
  2>&1 | grep -E 'test result|error|FAILED'
echo "ALL GATES MUST READ 'ok'. A single failure = STOP, do not run the ladder."

echo "=== [4/4] pin the binary ==="
cp target/release/solve /root/qs_solve_pinned
echo "Pinned at /root/qs_solve_pinned — the ladder runs from the PIN, never the"
echo "build path (rebuilds under a running binary cause SIGABRT on some systems)."
echo "Setup complete. Run the ladder with: bash ~/barricades/scripts/runpod_solver_ladder.sh"
