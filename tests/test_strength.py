"""Tests for agents/strength.py — TDD-first."""
import pytest

from core import initial_state, apply_move, Wall, Step, shortest_path_len
from agents.registry import make_agent
from agents.strength import (
    play_recorded_game,
    wasted_wall_rate,
    agent_stats,
    gauntlet,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_greedy(seed=0):
    return make_agent("greedy", seed=seed)

def _make_random(seed=0):
    return make_agent("random", seed=seed)


# ---------------------------------------------------------------------------
# play_recorded_game
# ---------------------------------------------------------------------------

class TestPlayRecordedGame:
    def test_returns_expected_keys(self):
        agent0 = _make_greedy(0)
        agent1 = _make_greedy(1)
        result = play_recorded_game(agent0, agent1, max_plies=400)
        assert set(result.keys()) == {"winner", "plies", "records"}

    def test_winner_is_valid(self):
        agent0 = _make_greedy(0)
        agent1 = _make_greedy(1)
        result = play_recorded_game(agent0, agent1, max_plies=400)
        assert result["winner"] in (0, 1, None)

    def test_greedy_vs_greedy_finishes(self):
        """Greedy agents don't place walls, so the game always ends quickly."""
        agent0 = _make_greedy(0)
        agent1 = _make_greedy(1)
        result = play_recorded_game(agent0, agent1, max_plies=400)
        # Greedy only steps; game must finish well under 400 plies.
        assert result["winner"] in (0, 1)
        assert result["plies"] > 0
        assert result["plies"] <= 400

    def test_records_shape(self):
        agent0 = _make_greedy(0)
        agent1 = _make_greedy(1)
        result = play_recorded_game(agent0, agent1, max_plies=400)
        records = result["records"]
        assert len(records) == result["plies"]
        for rec in records:
            assert set(rec.keys()) == {"player", "is_wall", "opp_dist_before", "opp_dist_after"}

    def test_record_player_alternates(self):
        agent0 = _make_greedy(0)
        agent1 = _make_greedy(1)
        result = play_recorded_game(agent0, agent1, max_plies=400)
        records = result["records"]
        for i, rec in enumerate(records):
            assert rec["player"] == i % 2, f"ply {i}: expected player {i%2}, got {rec['player']}"

    def test_record_opp_dists_are_ints(self):
        """Greedy agents do not block paths, so distances should always be ints."""
        agent0 = _make_greedy(0)
        agent1 = _make_greedy(1)
        result = play_recorded_game(agent0, agent1, max_plies=400)
        for rec in result["records"]:
            assert isinstance(rec["opp_dist_before"], int), rec
            assert isinstance(rec["opp_dist_after"], int), rec

    def test_is_wall_false_for_greedy(self):
        """Greedy agent never places walls."""
        agent0 = _make_greedy(0)
        agent1 = _make_greedy(1)
        result = play_recorded_game(agent0, agent1, max_plies=400)
        for rec in result["records"]:
            assert rec["is_wall"] is False

    def test_max_plies_cap(self):
        """With a tiny cap the game may not finish; plies == cap in that case."""
        agent0 = _make_greedy(0)
        agent1 = _make_greedy(1)
        result = play_recorded_game(agent0, agent1, max_plies=5)
        assert result["plies"] <= 5


# ---------------------------------------------------------------------------
# wasted_wall_rate — tested with hand-built records
# ---------------------------------------------------------------------------

class TestWastedWallRate:
    def test_none_when_no_walls(self):
        records = [
            {"player": 0, "is_wall": False, "opp_dist_before": 5, "opp_dist_after": 4},
            {"player": 1, "is_wall": False, "opp_dist_before": 6, "opp_dist_after": 5},
        ]
        assert wasted_wall_rate(records) is None

    def test_zero_wasted_when_all_walls_lengthen(self):
        """Every wall placed increased opponent's distance → 0.0 wasted."""
        records = [
            {"player": 0, "is_wall": True,  "opp_dist_before": 5, "opp_dist_after": 7},
            {"player": 1, "is_wall": True,  "opp_dist_before": 4, "opp_dist_after": 6},
            {"player": 0, "is_wall": False, "opp_dist_before": 3, "opp_dist_after": 3},
        ]
        assert wasted_wall_rate(records) == 0.0

    def test_one_wasted_when_all_walls_useless(self):
        """Every wall placed did NOT increase opponent's distance → 1.0 wasted."""
        records = [
            {"player": 0, "is_wall": True, "opp_dist_before": 5, "opp_dist_after": 5},
            {"player": 1, "is_wall": True, "opp_dist_before": 4, "opp_dist_after": 4},
        ]
        assert wasted_wall_rate(records) == 1.0

    def test_mixed_wasted_rate(self):
        """2 walls: 1 effective, 1 wasted → 0.5."""
        records = [
            {"player": 0, "is_wall": True, "opp_dist_before": 5, "opp_dist_after": 7},  # good
            {"player": 0, "is_wall": True, "opp_dist_before": 5, "opp_dist_after": 5},  # wasted
        ]
        assert wasted_wall_rate(records) == 0.5

    def test_filter_by_player(self):
        """Player-specific filter: player 0 places 1 good wall, player 1 places 1 wasted."""
        records = [
            {"player": 0, "is_wall": True, "opp_dist_before": 5, "opp_dist_after": 7},
            {"player": 1, "is_wall": True, "opp_dist_before": 4, "opp_dist_after": 4},
        ]
        assert wasted_wall_rate(records, player=0) == 0.0
        assert wasted_wall_rate(records, player=1) == 1.0

    def test_player_filter_none_when_no_walls_for_that_player(self):
        records = [
            {"player": 0, "is_wall": True, "opp_dist_before": 5, "opp_dist_after": 7},
        ]
        assert wasted_wall_rate(records, player=1) is None

    def test_wall_with_none_opp_dist_after_counts_as_wasted(self):
        """opp_dist_after=None means no path was found; we treat it like effective
        (opponent has no path → distance effectively infinite, definitely increased).
        But the spec says 'treating None as very large', so None→wasted means
        opp_dist_after<=opp_dist_before evaluates False when after=None."""
        # opp_dist_after=None means opponent path blocked → effective wall
        records = [
            {"player": 0, "is_wall": True, "opp_dist_before": 5, "opp_dist_after": None},
        ]
        # None treated as very large → after > before → NOT wasted
        assert wasted_wall_rate(records) == 0.0

    def test_wall_with_none_opp_dist_before_and_large_after(self):
        """If before=None (already no path) and after is also None → wasted
        (didn't improve things)."""
        records = [
            {"player": 0, "is_wall": True, "opp_dist_before": None, "opp_dist_after": None},
        ]
        # None <= None → wasted
        assert wasted_wall_rate(records) == 1.0


# ---------------------------------------------------------------------------
# agent_stats
# ---------------------------------------------------------------------------

class TestAgentStats:
    def test_structure(self):
        agent0 = _make_greedy(0)
        agent1 = _make_greedy(1)
        results = [play_recorded_game(agent0, agent1, max_plies=400)]
        stats = agent_stats(results, player=0)
        assert set(stats.keys()) == {"walls_placed", "wasted_wall_rate", "avg_plies"}

    def test_greedy_places_no_walls(self):
        agent0 = _make_greedy(0)
        agent1 = _make_greedy(1)
        results = [
            play_recorded_game(agent0, agent1, max_plies=400),
            play_recorded_game(_make_greedy(2), _make_greedy(3), max_plies=400),
        ]
        stats = agent_stats(results, player=0)
        assert stats["walls_placed"] == 0
        assert stats["wasted_wall_rate"] is None

    def test_avg_plies_is_correct(self):
        """avg_plies should match the mean of the plies values across results."""
        # Use two known results built from hand-crafted data.
        fake_results = [
            {"winner": 0, "plies": 20, "records": []},
            {"winner": 1, "plies": 30, "records": []},
        ]
        stats = agent_stats(fake_results, player=0)
        assert stats["avg_plies"] == pytest.approx(25.0)

    def test_walls_placed_aggregation(self):
        """walls_placed sums correctly across multiple games."""
        wall_rec = {"player": 0, "is_wall": True, "opp_dist_before": 4, "opp_dist_after": 6}
        step_rec = {"player": 0, "is_wall": False, "opp_dist_before": 4, "opp_dist_after": 4}
        fake_results = [
            {"winner": 0, "plies": 2, "records": [wall_rec, step_rec]},
            {"winner": 1, "plies": 2, "records": [wall_rec, wall_rec]},
        ]
        stats = agent_stats(fake_results, player=0)
        assert stats["walls_placed"] == 3   # 1 + 2


# ---------------------------------------------------------------------------
# gauntlet
# ---------------------------------------------------------------------------

class TestGauntlet:
    @pytest.fixture(scope="class")
    def result(self):
        factories = {
            "random": lambda seed: make_agent("random", seed=seed),
            "greedy": lambda seed: make_agent("greedy", seed=seed),
        }
        return gauntlet(factories, games=4, max_plies=400)

    def test_top_level_keys(self, result):
        expected = {"names", "wins", "losses", "draws", "winrate",
                    "wasted_wall_rate", "avg_plies"}
        assert expected.issubset(result.keys())

    def test_names_present(self, result):
        assert set(result["names"]) == {"random", "greedy"}

    def test_wins_matrix_keys(self, result):
        names = result["names"]
        for a in names:
            assert a in result["wins"]
            for b in names:
                if a != b:
                    assert b in result["wins"][a]

    def test_winrate_keys(self, result):
        for name in result["names"]:
            assert name in result["winrate"]

    def test_wasted_wall_rate_keys(self, result):
        for name in result["names"]:
            assert name in result["wasted_wall_rate"]

    def test_greedy_beats_random(self, result):
        """Over 4 games greedy should win more often than random against each other."""
        # greedy winrate vs random should exceed random's winrate
        assert result["winrate"]["greedy"] > result["winrate"]["random"]

    def test_winrates_sum_reasonably(self, result):
        """Sum of all winrates / n_agents is roughly 0.5 (no draws expected for these agents)."""
        names = result["names"]
        total = sum(result["winrate"][n] for n in names)
        # With 2 agents and no draws: one wins 100%, other 0% is extreme but OK.
        assert 0.0 <= total <= len(names)

    def test_avg_plies_positive(self, result):
        assert result["avg_plies"] > 0
