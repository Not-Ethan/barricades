from core.state import GameState, Step, Wall, initial_state
from server.serialize import state_to_dict, parse_move, move_to_dict


def test_state_to_dict_initial():
    d = state_to_dict(initial_state(), game_id="g1", controllers=["human", "greedy"])
    assert d["id"] == "g1"
    assert d["pawns"] == [[4, 0], [4, 8]]
    assert d["h_walls"] == [] and d["v_walls"] == []
    assert d["walls_left"] == [10, 10]
    assert d["turn"] == 0
    assert d["winner"] is None
    assert d["controllers"] == ["human", "greedy"]
    # legal moves present and shaped right
    assert [4, 1] in d["legal"]["steps"]
    assert any(w["orient"] in ("H", "V") for w in d["legal"]["walls"])


def test_state_to_dict_reports_winner():
    s = GameState(((4, 8), (4, 1)), frozenset(), frozenset(), (10, 10), 1)
    d = state_to_dict(s, game_id="g", controllers=["human", "human"])
    assert d["winner"] == 0
    assert d["legal"]["steps"] == [] and d["legal"]["walls"] == []  # game over


def test_parse_move_step_and_wall():
    assert parse_move({"type": "step", "to": [4, 1]}) == Step((4, 1))
    assert parse_move({"type": "wall", "c": 3, "r": 3, "orient": "H"}) == Wall(3, 3, "H")


def test_move_to_dict_roundtrip():
    for m in [Step((2, 5)), Wall(1, 2, "V")]:
        assert parse_move(move_to_dict(m)) == m


def test_parse_move_rejects_garbage():
    import pytest
    with pytest.raises(ValueError):
        parse_move({"type": "teleport"})


def test_parse_move_rejects_incomplete_step_and_wall():
    import pytest
    with pytest.raises(ValueError):
        parse_move({"type": "step"})
    with pytest.raises(ValueError):
        parse_move({"type": "wall", "c": 1, "r": 1})  # missing orient
