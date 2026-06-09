"""Generate the cross-language differential fixture for the Rust solver's
move generator.

Walks seeded random `smallboard` games at 3x3 (W=1) and 5x5 (W=3). For each
visited non-terminal position it emits one JSON record describing the position
and the lexicographically-sorted set of legal-move keys, where:

  step to cell (c, r)  -> "S {c} {r}"
  horizontal wall      -> "H {wc} {wr}"
  vertical wall        -> "V {wc} {wr}"

The Rust test (`solver/tests/diff_vs_smallboard.rs`) reconstructs each position,
computes its own legal moves, encodes them with the SAME keys, and asserts the
sorted lists match. Dumps >=800 records to
`solver/tests/fixtures/smallboard_diff.json`.

Run:  source .venv/bin/activate && python scripts/gen_solver_fixture.py
"""

import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from smallboard.engine import Engine, Step, Wall  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_PATH = os.path.join(
    REPO_ROOT, "solver", "tests", "fixtures", "smallboard_diff.json"
)

# (N, W) board configs to walk.
CONFIGS = [(3, 1), (5, 3)]
# Plenty of games to comfortably clear the >=800-record floor.
GAMES_PER_CONFIG = 400
MAX_PLIES = 40
MIN_RECORDS = 800
# Cap the committed fixture size while keeping it well above the floor and
# diverse across both configs.
MAX_RECORDS = 2000


def move_key(m):
    """Encode a smallboard move into the shared key string."""
    if isinstance(m, Step):
        c, r = m.to_cell
        return f"S {c} {r}"
    # Wall: orient is "H" / "V", (c, r) are the anchors == our (wc, wr).
    return f"{m.orient} {m.c} {m.r}"


def record_for(eng, s):
    """Build the JSON record for non-terminal position `s`."""
    keys = sorted(move_key(m) for m in eng.legal_moves(s))
    return {
        "w": eng.N,
        "h": eng.N,
        "walls": eng.W,
        "pawns": [list(p) for p in s.pawns],
        "h_walls": sorted([c, r] for (c, r) in s.h_walls),
        "v_walls": sorted([c, r] for (c, r) in s.v_walls),
        "walls_left": list(s.walls_left),
        "turn": s.turn,
        "moves": keys,
    }


def main():
    rng = random.Random(0xC0FFEE)
    records = []
    seen = set()  # de-dup identical positions to keep the fixture diverse

    for (n, w) in CONFIGS:
        eng = Engine(n, w)
        for g in range(GAMES_PER_CONFIG):
            game_rng = random.Random((n * 1_000_003) ^ (w * 7919) ^ g)
            s = eng.initial_state()
            for _ in range(MAX_PLIES):
                if eng.is_terminal(s):
                    break
                moves = eng.legal_moves(s)
                if not moves:
                    break
                # Fingerprint the position; emit a record only once per state.
                fp = (
                    n,
                    w,
                    s.pawns,
                    frozenset(s.h_walls),
                    frozenset(s.v_walls),
                    s.walls_left,
                    s.turn,
                )
                if fp not in seen:
                    seen.add(fp)
                    records.append(record_for(eng, s))
                s = eng.apply_move(s, game_rng.choice(moves))

    # Keep a deterministic order, then a stable shuffle for variety, and cap
    # the size while preserving config balance.
    records.sort(key=lambda rec: json.dumps(rec, sort_keys=True))
    rng.shuffle(records)
    if len(records) > MAX_RECORDS:
        # Interleave configs so the cap keeps both 3x3 and 5x5 positions.
        small = [r for r in records if r["w"] == 3]
        big = [r for r in records if r["w"] == 5]
        keep_small = small[: MAX_RECORDS // 2]
        keep_big = big[: MAX_RECORDS - len(keep_small)]
        records = keep_small + keep_big
        rng.shuffle(records)

    assert len(records) >= MIN_RECORDS, (
        f"only {len(records)} records, need >= {MIN_RECORDS}"
    )

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        # One compact JSON record per line: small file, easy to diff.
        f.write("[\n")
        for i, rec in enumerate(records):
            sep = "," if i + 1 < len(records) else ""
            f.write(json.dumps(rec, separators=(",", ":")) + sep + "\n")
        f.write("]\n")

    by_cfg = {}
    for rec in records:
        by_cfg[(rec["w"], rec["walls"])] = by_cfg.get((rec["w"], rec["walls"]), 0) + 1
    print(f"wrote {len(records)} records to {OUT_PATH}")
    print(f"  by config (w,walls): {dict(sorted(by_cfg.items()))}")


if __name__ == "__main__":
    main()
