"""Tests for POST /analyze and dict_to_state (TDD: written before implementation)."""
import pytest
from fastapi.testclient import TestClient

from core.state import GameState, initial_state
from server.serialize import state_to_dict, dict_to_state


# ---------------------------------------------------------------------------
# dict_to_state round-trip tests
# ---------------------------------------------------------------------------

def _make_client():
    from server.app import create_app
    return TestClient(create_app())


def test_dict_to_state_roundtrip_initial():
    s = initial_state()
    d = state_to_dict(s, "g", ["human", "human"])
    s2 = dict_to_state(d)
    assert s2 == s


def test_dict_to_state_roundtrip_with_walls():
    s = GameState(
        pawns=((3, 2), (5, 6)),
        h_walls=frozenset({(2, 3), (4, 1)}),
        v_walls=frozenset({(1, 5)}),
        walls_left=(7, 8),
        turn=1,
    )
    d = state_to_dict(s, "g", ["human", "human"])
    s2 = dict_to_state(d)
    assert s2 == s


def test_dict_to_state_roundtrip_many_walls():
    s = GameState(
        pawns=((0, 0), (8, 8)),
        h_walls=frozenset({(0, 0), (1, 2), (3, 4)}),
        v_walls=frozenset({(5, 6), (7, 7)}),
        walls_left=(5, 3),
        turn=0,
    )
    d = state_to_dict(s, "g", ["greedy", "minimax"])
    s2 = dict_to_state(d)
    assert s2 == s


def test_dict_to_state_raw_dict():
    """dict_to_state also works on a raw dict (not produced by state_to_dict)."""
    d = {
        "pawns": [[4, 0], [4, 8]],
        "h_walls": [],
        "v_walls": [],
        "walls_left": [10, 10],
        "turn": 0,
    }
    s = dict_to_state(d)
    assert s == initial_state()


# ---------------------------------------------------------------------------
# POST /analyze – valid position (initial state, two engines)
# ---------------------------------------------------------------------------

def _initial_pos():
    s = initial_state()
    return {
        "pawns": [list(s.pawns[0]), list(s.pawns[1])],
        "h_walls": [],
        "v_walls": [],
        "walls_left": [10, 10],
        "turn": 0,
    }


def test_analyze_valid_position_initial_state():
    c = _make_client()
    body = {
        "position": _initial_pos(),
        "engines": [
            {"name": "greedy"},
            {"name": "minimax", "params": {"time_budget": 0.05}},
        ],
    }
    r = c.post("/analyze", json=body)
    assert r.status_code == 200, r.text
    data = r.json()

    assert data["valid"] is True
    assert data["winner"] is None
    assert isinstance(data["static_eval"], (int, float))
    assert data["turn"] == 0

    # Legal moves present
    assert len(data["legal"]["steps"]) > 0
    assert len(data["legal"]["walls"]) > 0

    # Both engines reported
    assert len(data["results"]) == 2
    names = {r["engine"] for r in data["results"]}
    assert names == {"greedy", "minimax"}

    for result in data["results"]:
        assert "best_move" in result
        assert isinstance(result["value"], (int, float))
        assert isinstance(result["candidates"], list)
        assert isinstance(result["stats"], dict)


# ---------------------------------------------------------------------------
# POST /analyze – finished position (winner present)
# ---------------------------------------------------------------------------

def test_analyze_finished_position_reports_winner():
    c = _make_client()
    # Player 0's pawn at goal row (row 8 = N-1)
    body = {
        "position": {
            "pawns": [[4, 8], [4, 0]],   # player 0 at row 8 = goal for player 0
            "h_walls": [],
            "v_walls": [],
            "walls_left": [10, 10],
            "turn": 1,
        },
        "engines": [{"name": "greedy"}],
    }
    r = c.post("/analyze", json=body)
    assert r.status_code == 200, r.text
    data = r.json()

    assert data["valid"] is True
    assert data["winner"] == 0
    assert data["results"] == []
    assert data["legal"]["steps"] == []
    assert data["legal"]["walls"] == []


# ---------------------------------------------------------------------------
# POST /analyze – illegal position (pawn walled off)
# ---------------------------------------------------------------------------

def test_analyze_illegal_position_pawn_walled_off():
    """Construct a position where player 0 is completely walled off (no path to goal)."""
    c = _make_client()
    # Player 0 is at (4, 0). Place horizontal walls along entire row-0 boundary
    # to seal player 0 into the bottom row.
    # H-wall at (c, r) blocks the top edge of cell (c, r) and (c+1, r).
    # To block between row 0 and row 1 we place H-walls at r=0:
    # (0,0),(1,0),(2,0),(3,0) block columns 0-1,1-2,2-3,3-4 top edges
    # (4,0),(5,0),(6,0),(7,0) block columns 4-5,5-6,6-7,7-8 top edges
    # These 8 H-walls create a solid barrier, cutting off player 0.
    body = {
        "position": {
            "pawns": [[4, 0], [4, 8]],
            "h_walls": [[c, 0] for c in range(8)],   # 8 walls across row 0→1 boundary
            "v_walls": [],
            "walls_left": [10, 10],
            "turn": 0,
        },
        "engines": [{"name": "greedy"}],
    }
    r = c.post("/analyze", json=body)
    assert r.status_code == 200, r.text
    data = r.json()

    assert data["valid"] is False
    assert "reason" in data
    assert len(data["reason"]) > 0
    # Engines must NOT have been run
    assert "results" not in data


# ---------------------------------------------------------------------------
# POST /analyze – unknown engine name → 400
# ---------------------------------------------------------------------------

def test_analyze_unknown_engine_returns_400():
    c = _make_client()
    body = {
        "position": _initial_pos(),
        "engines": [{"name": "bogus_engine_xyz"}],
    }
    r = c.post("/analyze", json=body)
    assert r.status_code == 400


# ---------------------------------------------------------------------------
# POST /analyze – invalid position: out-of-range pawn
# ---------------------------------------------------------------------------

def test_analyze_invalid_pawn_out_of_range():
    c = _make_client()
    body = {
        "position": {
            "pawns": [[9, 0], [4, 8]],   # col 9 is off-board (0..8)
            "h_walls": [],
            "v_walls": [],
            "walls_left": [10, 10],
            "turn": 0,
        },
        "engines": [],
    }
    r = c.post("/analyze", json=body)
    assert r.status_code == 200
    data = r.json()
    assert data["valid"] is False
    assert "reason" in data


def test_analyze_invalid_pawns_same_cell():
    c = _make_client()
    body = {
        "position": {
            "pawns": [[4, 0], [4, 0]],  # same cell
            "h_walls": [],
            "v_walls": [],
            "walls_left": [10, 10],
            "turn": 0,
        },
        "engines": [],
    }
    r = c.post("/analyze", json=body)
    assert r.status_code == 200
    data = r.json()
    assert data["valid"] is False
    assert "reason" in data


def test_analyze_invalid_walls_left_out_of_range():
    c = _make_client()
    body = {
        "position": {
            "pawns": [[4, 0], [4, 8]],
            "h_walls": [],
            "v_walls": [],
            "walls_left": [11, 5],  # 11 > 10 is invalid
            "turn": 0,
        },
        "engines": [],
    }
    r = c.post("/analyze", json=body)
    assert r.status_code == 200
    data = r.json()
    assert data["valid"] is False
    assert "reason" in data


def test_analyze_invalid_wall_anchor_out_of_range():
    c = _make_client()
    body = {
        "position": {
            "pawns": [[4, 0], [4, 8]],
            "h_walls": [[8, 0]],  # col 8 is out of range (0..7)
            "v_walls": [],
            "walls_left": [9, 10],
            "turn": 0,
        },
        "engines": [],
    }
    r = c.post("/analyze", json=body)
    assert r.status_code == 200
    data = r.json()
    assert data["valid"] is False
    assert "reason" in data
