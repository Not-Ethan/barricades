"""Baseline strength report: gauntlet over real agents at fast budgets.

Usage
-----
    python scripts/strength_report.py [--games N] [--include-az]

Options
-------
--games N       Number of games per ordered pair (default 6).
--include-az    Also test the AZ agent (slow; near-random without a trained model).
"""
import argparse
import os
import sys

# Make the repo root importable when run as `python scripts/strength_report.py`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.registry import make_agent
from agents.strength import gauntlet


def build_factories(include_az: bool) -> dict:
    """Return the factory dict for the default gauntlet."""
    factories = {
        "random":   lambda seed: make_agent("random", seed=seed),
        "greedy":   lambda seed: make_agent("greedy", seed=seed),
        "minimax":  lambda seed: make_agent("minimax", seed=seed, time_budget=0.05),
        "mcts":     lambda seed: make_agent("mcts",    seed=seed, time_budget=0.05),
    }
    if include_az:
        factories["az"] = lambda seed: make_agent("az", seed=seed)
    return factories


def _fmt_rate(v) -> str:
    if v is None:
        return "  N/A "
    return f"{v:6.1%}"


def print_report(result: dict, games: int) -> None:
    names = result["names"]
    n = len(names)

    header_width = max(len(nm) for nm in names)
    col_w = max(6, header_width + 2)

    print()
    print("=" * 70)
    print("  BARRICADES ENGINE STRENGTH REPORT")
    print(f"  {games} games per ordered pair  |  round-robin")
    print("=" * 70)

    # ---- Win matrix --------------------------------------------------------
    print()
    print("WIN MATRIX  (row = player-0 / column = player-1)")
    print("  Each cell: wins by the ROW agent as player-0 against the COLUMN agent.")
    print()

    # Header row
    pad = " " * (header_width + 2)
    header = pad + "".join(f"{nm:>{col_w}}" for nm in names)
    print(header)
    print(pad + "-" * (col_w * n))

    for a in names:
        row = f"{a:<{header_width}}  "
        for b in names:
            if a == b:
                row += f"{'---':>{col_w}}"
            else:
                row += f"{result['wins'][a][b]:>{col_w}}"
        print(row)

    # ---- Per-agent summary -------------------------------------------------
    print()
    print("PER-AGENT SUMMARY")
    print()
    col_labels = ["Agent", "Win-rate", "Wasted-wall%", "Avg plies"]
    col_widths = [header_width + 2, 10, 14, 12]
    header_line = "".join(f"{lbl:<{w}}" for lbl, w in zip(col_labels, col_widths))
    print(header_line)
    print("-" * sum(col_widths))

    for a in names:
        wr = result["winrate"][a]
        wwr = result["wasted_wall_rate"][a]
        avg_p = result["avg_plies"]   # global; per-agent would need more plumbing
        wwr_str = _fmt_rate(wwr)
        print(
            f"{a:<{col_widths[0]}}"
            f"{wr:>9.1%}  "
            f"{wwr_str:>12}  "
            f"{avg_p:>10.1f}"
        )

    print()
    print(f"Global avg plies per game: {result['avg_plies']:.1f}")
    print()

    # ---- Ranking -----------------------------------------------------------
    ranked = sorted(names, key=lambda nm: result["winrate"][nm], reverse=True)
    print("RANKING (by overall win-rate)")
    for rank, nm in enumerate(ranked, 1):
        wwr = result["wasted_wall_rate"][nm]
        wwr_str = _fmt_rate(wwr)
        wr = result["winrate"][nm]
        print(f"  #{rank}  {nm:<{header_width}}  winrate={wr:.1%}  "
              f"wasted_wall={wwr_str}")

    print()
    print("=" * 70)


def main():
    ap = argparse.ArgumentParser(
        description="Print a round-robin strength report for the barricades engines."
    )
    ap.add_argument("--games", type=int, default=6,
                    help="Games per ordered pair (default: 6).")
    ap.add_argument("--include-az", action="store_true",
                    help="Include the AZ agent (slow; near-random without a model).")
    args = ap.parse_args()

    factories = build_factories(include_az=args.include_az)
    n = len(factories)
    ordered_pairs = n * (n - 1)
    total_games = ordered_pairs * args.games
    print(f"Running gauntlet: {n} agents, {ordered_pairs} ordered pairs, "
          f"{args.games} games each = {total_games} total games …")
    sys.stdout.flush()

    result = gauntlet(factories, games=args.games, max_plies=400)
    print_report(result, games=args.games)


if __name__ == "__main__":
    main()
