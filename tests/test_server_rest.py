from fastapi.testclient import TestClient
from server.app import create_app


def client():
    return TestClient(create_app())


def test_list_agents_includes_baselines():
    r = client().get("/agents")
    assert r.status_code == 200
    names = r.json()["agents"]
    assert "random" in names and "greedy" in names


def test_create_and_get_game():
    c = client()
    r = c.post("/games", json={"controllers": ["human", "greedy"]})
    assert r.status_code == 200
    gid = r.json()["id"]
    g = c.get(f"/games/{gid}")
    assert g.status_code == 200
    assert g.json()["controllers"] == ["human", "greedy"]
    assert g.json()["turn"] == 0


def test_legal_move_applied():
    c = client()
    gid = c.post("/games", json={"controllers": ["human", "human"]}).json()["id"]
    r = c.post(f"/games/{gid}/move", json={"type": "step", "to": [4, 1]})
    assert r.status_code == 200
    assert r.json()["pawns"][0] == [4, 1]
    assert r.json()["turn"] == 1


def test_illegal_move_rejected_with_400():
    c = client()
    gid = c.post("/games", json={"controllers": ["human", "human"]}).json()["id"]
    r = c.post(f"/games/{gid}/move", json={"type": "step", "to": [8, 8]})
    assert r.status_code == 400


def test_unknown_game_404():
    assert client().get("/games/nope").status_code == 404


def test_undo_endpoint():
    c = client()
    gid = c.post("/games", json={"controllers": ["human", "human"]}).json()["id"]
    c.post(f"/games/{gid}/move", json={"type": "step", "to": [4, 1]})
    r = c.post(f"/games/{gid}/undo")
    assert r.json()["pawns"][0] == [4, 0]


def test_engine_move_endpoint_advances_turn():
    c = client()
    # both human so we can drive: but engine_move uses controller of side to move.
    gid = c.post("/games", json={"controllers": ["greedy", "human"]}).json()["id"]
    r = c.post(f"/games/{gid}/engine_move")
    assert r.status_code == 200
    body = r.json()
    assert body["state"]["turn"] == 1            # greedy (player 0) moved
    assert body["analysis"] is not None          # analysis included


def test_engine_move_rejected_when_human_to_move():
    c = client()
    gid = c.post("/games", json={"controllers": ["human", "greedy"]}).json()["id"]
    r = c.post(f"/games/{gid}/engine_move")
    assert r.status_code == 400                   # side to move is human
