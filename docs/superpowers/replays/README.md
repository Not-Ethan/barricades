# Canonical optimal lines — 6×5

Replays extracted with `solve <W> <H> <WALLS> --pv`: every move is
value-preserving (verified by warm-TT requery, value invariant asserted per
ply; the terminal must deliver the root's promised winner or extraction
panics). Race-phase plies are exactly DTM-optimal (fastest win / slowest
loss); wall-phase plies are value-optimal with a natural-play heuristic —
NOT guaranteed fastest/slowest (true wall-phase DTM needs a non-folding
distance search; see pv.rs docs). W0–W4 extracted locally; W10 from the pod.

Reproducibility: the W4 line was independently re-extracted on a 16-thread
x86 pod (vs ARM locally) and the 23-ply move sequence is identical — PV
selection is deterministic across architectures and thread counts because
values are exact and the candidate ordering is fixed.

Line lengths (W0 6 plies, W1 8, W2 18, W3 28, W4 23, W10 35) are properties of the
SELECTION HEURISTIC, not game-theoretic lengths — the loser's resistance is
not provably maximal nor the winner provably fastest (these replays predate
the race-DTM upgrade; full-game DTM needs a non-folding distance search).
Do not read the length progression as a finding. The VALUES at every ply are
exact. Qualitative content is real: e.g. the W4 transition game — P0 races
4 plies, then converts tempo into a 4-wall cage around P1's corridor and
walks the top of the board to (1,4).

The W10 line (35 plies) shows budget saturation inside perfect play: with
ten walls per side available, the optimal game places only 5 (P0) + 2 (P1) —
the same geometry exhaustion that flattens the solving-cost curve (see the
W15 experiment in 6x5-solved-results.md) caps how many walls perfect play
can usefully spend.
