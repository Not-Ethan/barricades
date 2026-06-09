#!/bin/bash
# RSS effect of the bounded race cache on 6x5 W3.
#
# 6x5 W3 does not finish quickly, so we time-box each run (timeout) and let
# /usr/bin/time -l report PEAK resident set size (max RSS) observed up to the
# kill point. The dense main TT is held SMALL and FIXED (QS_TT_MB) so the only
# variable driving RSS is the race cache size. Under a small race cap the RSS
# plateaus; under a huge cap it grows unbounded toward the historical multi-GB.
#
# Usage: bash measure_rss.sh [seconds] [tt_mb]
set -u
SECS="${1:-90}"
TTMB="${2:-512}"
BIN=./target/release/solve

run() {
  local cap="$1"
  echo "=== 6x5 W3  QS_RACE_MB=${cap}  QS_TT_MB=${TTMB}  threads=8  window=${SECS}s ==="
  # /usr/bin/time WRAPS timeout (not the reverse): when timeout kills the solve,
  # `time` still reports the peak RSS observed up to that point. (If time wrapped
  # the solve and timeout killed time, the report would be lost.)
  QS_THREADS=8 QS_TT_MB="$TTMB" QS_RACE_MB="$cap" \
    /usr/bin/time -l timeout "${SECS}" "$BIN" 6 5 3 2>&1 \
    | grep -iE "maximum resident set size|value=|race_entries"
  echo
}

run 256
run 8000
