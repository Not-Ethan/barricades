"""Strength-measurement harness for Quoridor barricades engines.

Public API
----------
play_recorded_game(agent0, agent1, max_plies=400) -> dict
wasted_wall_rate(records, player=None) -> float | None
agent_stats(results, player) -> dict
gauntlet(factories, games=10, max_plies=400) -> dict
"""

from __future__ import annotations

from typing import Any

from core import initial_state, apply_move, is_terminal, winner, shortest_path_len, Wall

# Sentinel used to represent "infinity" when a path length is None.
_INF = 10_000_000


# ---------------------------------------------------------------------------
# play_recorded_game
# ---------------------------------------------------------------------------

def play_recorded_game(agent0, agent1, max_plies: int = 400) -> dict:
    """Play one game and return a dict with detailed per-move records.

    Returns
    -------
    {
        "winner": 0 | 1 | None,
        "plies": int,
        "records": [
            {
                "player": int,       # who moved (0 or 1)
                "is_wall": bool,     # True if the move was a Wall placement
                "opp_dist_before": int | None,  # opponent's shortest-path BEFORE move
                "opp_dist_after":  int | None,  # opponent's shortest-path AFTER move
            },
            ...
        ]
    }
    """
    agents = (agent0, agent1)
    state = initial_state()
    records: list[dict[str, Any]] = []

    for _ in range(max_plies):
        if is_terminal(state):
            break

        mover = state.turn
        opponent = 1 - mover

        # Measure the opponent's path length BEFORE the move.
        opp_dist_before = shortest_path_len(state, opponent)

        # Select and apply the move.
        move = agents[mover].select_move(state)
        is_wall_move = isinstance(move, Wall)
        state = apply_move(state, move)

        # Measure the opponent's path length AFTER the move.
        opp_dist_after = shortest_path_len(state, opponent)

        records.append({
            "player": mover,
            "is_wall": is_wall_move,
            "opp_dist_before": opp_dist_before,
            "opp_dist_after": opp_dist_after,
        })

    game_winner = winner(state) if is_terminal(state) else None
    return {
        "winner": game_winner,
        "plies": len(records),
        "records": records,
    }


# ---------------------------------------------------------------------------
# wasted_wall_rate
# ---------------------------------------------------------------------------

def wasted_wall_rate(records: list[dict], player: int | None = None) -> float | None:
    """Fraction of wall moves that did NOT increase the opponent's shortest path.

    A wall is "wasted" if ``opp_dist_after <= opp_dist_before``.
    ``None`` distances are treated as very large (infinity-like): specifically,
    - ``None`` treated as ``_INF`` for comparison purposes.

    Parameters
    ----------
    records:
        The ``records`` list from a ``play_recorded_game`` result (or a
        hand-built list with the same schema).
    player:
        If given, only consider walls placed by this player.
        If ``None``, consider all walls.

    Returns
    -------
    float in [0.0, 1.0] or None if no walls were placed (by the filtered player).
    """
    wall_records = [
        r for r in records
        if r["is_wall"] and (player is None or r["player"] == player)
    ]
    if not wall_records:
        return None

    def _dist(v) -> int:
        return _INF if v is None else v

    wasted = sum(
        1 for r in wall_records
        if _dist(r["opp_dist_after"]) <= _dist(r["opp_dist_before"])
    )
    return wasted / len(wall_records)


# ---------------------------------------------------------------------------
# agent_stats
# ---------------------------------------------------------------------------

def agent_stats(results: list[dict], player: int) -> dict:
    """Aggregate statistics for one fixed-seat player across multiple games.

    Parameters
    ----------
    results:
        A list of dicts as returned by ``play_recorded_game``.
    player:
        The seat (0 or 1) that was occupied by the agent of interest.

    Returns
    -------
    {
        "walls_placed": int,
        "wasted_wall_rate": float | None,
        "avg_plies": float,
    }
    """
    all_records: list[dict] = []
    total_plies = 0

    for result in results:
        all_records.extend(result["records"])
        total_plies += result["plies"]

    walls_placed = sum(
        1 for r in all_records if r["is_wall"] and r["player"] == player
    )
    wwr = wasted_wall_rate(all_records, player=player)
    avg_plies = total_plies / len(results) if results else 0.0

    return {
        "walls_placed": walls_placed,
        "wasted_wall_rate": wwr,
        "avg_plies": avg_plies,
    }


# ---------------------------------------------------------------------------
# gauntlet
# ---------------------------------------------------------------------------

def gauntlet(
    factories: dict[str, Any],
    games: int = 10,
    max_plies: int = 400,
) -> dict:
    """Round-robin gauntlet.

    Parameters
    ----------
    factories:
        Mapping ``name -> callable(seed: int) -> agent``.
    games:
        Number of games per ORDERED pair (a as player 0, b as player 1).
    max_plies:
        Per-game move cap passed to ``play_recorded_game``.

    Returns
    -------
    {
        "names": [str, ...],                          # sorted list of agent names
        "wins":  {a: {b: int}},                       # wins[a][b] = games won by a as p0 vs b as p1
        "losses": {a: {b: int}},
        "draws":  {a: {b: int}},
        "winrate": {a: float},                        # wins / total games played (all seats)
        "wasted_wall_rate": {a: float | None},        # aggregated over both seats
        "avg_plies": float,                            # global average across all games
    }
    """
    names = sorted(factories.keys())
    n = len(names)

    # Initialise accumulators.
    wins:   dict[str, dict[str, int]] = {a: {b: 0 for b in names if b != a} for a in names}
    losses: dict[str, dict[str, int]] = {a: {b: 0 for b in names if b != a} for a in names}
    draws:  dict[str, dict[str, int]] = {a: {b: 0 for b in names if b != a} for a in names}

    # Per-agent accumulator for wall metrics (all records across all games).
    all_records_for: dict[str, list] = {a: [] for a in names}
    total_wins: dict[str, int] = {a: 0 for a in names}
    total_losses: dict[str, int] = {a: 0 for a in names}
    total_draws: dict[str, int] = {a: 0 for a in names}
    total_plies = 0
    total_games = 0

    for a_name in names:
        for b_name in names:
            if a_name == b_name:
                continue
            factory_a = factories[a_name]
            factory_b = factories[b_name]

            for g in range(games):
                seed_a = g
                seed_b = 1000 + g
                agent0 = factory_a(seed_a)
                agent1 = factory_b(seed_b)

                result = play_recorded_game(agent0, agent1, max_plies=max_plies)
                w = result["winner"]
                total_plies += result["plies"]
                total_games += 1

                # a_name is player 0, b_name is player 1.
                if w == 0:
                    wins[a_name][b_name] += 1
                    losses[b_name][a_name] += 1
                    total_wins[a_name] += 1
                    total_losses[b_name] += 1
                elif w == 1:
                    losses[a_name][b_name] += 1
                    wins[b_name][a_name] += 1
                    total_losses[a_name] += 1
                    total_wins[b_name] += 1
                else:
                    draws[a_name][b_name] += 1
                    draws[b_name][a_name] += 1
                    total_draws[a_name] += 1
                    total_draws[b_name] += 1

                # Accumulate records for each agent in their seat.
                records = result["records"]
                for rec in records:
                    if rec["player"] == 0:
                        all_records_for[a_name].append(rec)
                    else:
                        all_records_for[b_name].append(rec)

    # Compute per-agent winrates.
    # Each agent plays (n-1)*games games as p0 and (n-1)*games games as p1.
    games_per_agent = (n - 1) * games * 2  # both seat directions
    winrate: dict[str, float] = {}
    for a in names:
        total_a = total_wins[a] + total_losses[a] + total_draws[a]
        winrate[a] = total_wins[a] / total_a if total_a > 0 else 0.0

    # Compute per-agent wasted wall rate (aggregated across all records, all seats).
    wasted: dict[str, float | None] = {}
    for a in names:
        wasted[a] = wasted_wall_rate(all_records_for[a])

    avg_plies = total_plies / total_games if total_games > 0 else 0.0

    return {
        "names": names,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "winrate": winrate,
        "wasted_wall_rate": wasted,
        "avg_plies": avg_plies,
    }
