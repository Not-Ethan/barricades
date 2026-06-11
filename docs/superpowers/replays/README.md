# Canonical optimal lines — 6×5

Replays extracted with `solve <W> <H> <WALLS> --pv`: every move is
value-preserving (verified by warm-TT requery, value invariant asserted per
ply; the terminal must deliver the root's promised winner or extraction
panics). Race-phase plies are exactly DTM-optimal (fastest win / slowest
loss); wall-phase plies are value-optimal with a natural-play heuristic —
NOT guaranteed fastest/slowest (true wall-phase DTM needs a non-folding
distance search; see pv.rs docs). W0–W4 extracted locally; W10 from the pod.

Canonical game lengths: W0 6 plies (P1), W1 8 (P1), W2 18 (P1), W3 28 (P1),
W4 23 (P0 — the transition game: P0 races 4 plies, then builds a 4-wall cage
around P1's approach and walks the top of the board to (1,4)).
