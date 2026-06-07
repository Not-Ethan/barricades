import pytest
from core.state import Step
from server.games import Game, GameStore


def test_game_applies_and_tracks_history():
    g = Game(game_id="g1", controllers=["human", "human"])
    assert g.state.turn == 0
    g.apply(Step((4, 1)))
    assert g.state.pawns[0] == (4, 1)
    assert g.state.turn == 1
    assert g.move_count == 1


def test_game_rejects_illegal_move():
    g = Game(game_id="g1", controllers=["human", "human"])
    with pytest.raises(ValueError):
        g.apply(Step((8, 8)))   # not adjacent


def test_undo_restores_previous_state():
    g = Game(game_id="g1", controllers=["human", "human"])
    g.apply(Step((4, 1)))
    g.undo()
    assert g.state.pawns[0] == (4, 0)
    assert g.move_count == 0


def test_undo_on_fresh_game_is_noop_or_error():
    g = Game(game_id="g1", controllers=["human", "human"])
    g.undo()  # should not crash; stays at initial
    assert g.move_count == 0


def test_store_creates_unique_ids():
    store = GameStore()
    a = store.create(["human", "human"])
    b = store.create(["human", "greedy"])
    assert a.id != b.id
    assert store.get(a.id) is a
    with pytest.raises(KeyError):
        store.get("nope")
