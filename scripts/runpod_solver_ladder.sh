#!/usr/bin/env bash
# 6x5 wall-count ladder runner — run ON THE POD inside tmux.
# Usage: bash runpod_solver_ladder.sh [start_W] [end_W]
# Sizes caches from the pod's RAM, runs each rung from the pinned binary with
# heartbeats, logs everything, and stops the ladder on the first failure.
set -uo pipefail

START_W=${1:-5}
END_W=${2:-10}
BIN=/root/qs_solve_pinned
OUT=$HOME/ladder_logs && mkdir -p "$OUT"

# --- size to the machine ---
NCPU=$(nproc)
MEM_GB=$(awk '/MemTotal/{printf "%d", $2/1048576}' /proc/meminfo)
# leave ~20% headroom; split ~2:1 between main TT and race cache
TT_MB=$(( MEM_GB * 1024 * 8 / 10 * 2 / 3 ))
RACE_MB=$(( MEM_GB * 1024 * 8 / 10 * 1 / 3 ))
echo "pod: ${NCPU} vCPU, ${MEM_GB} GB -> QS_TT_MB=${TT_MB} QS_RACE_MB=${RACE_MB} QS_THREADS=${NCPU}"

for W in $(seq "$START_W" "$END_W"); do
  LOG="$OUT/6x5_w${W}.log"
  echo "=== 6x5 W${W} -> ${LOG} ($(date -u)) ==="
  QS_THREADS=$NCPU QS_TT_MB=$TT_MB QS_RACE_MB=$RACE_MB \
  QS_RACE_SHARDS=$(( NCPU * 4 )) QS_PROGRESS_SECS=300 \
    /usr/bin/time -v "$BIN" 6 5 "$W" 2>&1 | tee "$LOG"
  RC=${PIPESTATUS[0]}
  if [ "$RC" -ne 0 ]; then
    echo "!!! W${W} exited rc=${RC} — ladder STOPPED. Check ${LOG} (and dmesg for OOM)."
    exit "$RC"
  fi
  grep -E 'value=' "$LOG" | tail -1
done
echo "=== ladder complete W${START_W}..W${END_W} ($(date -u)) ==="
echo "rsync $OUT back to the dev machine and append results to docs/superpowers/solver-phase1-measurements.md"
