from fastapi.testclient import TestClient
from server.app import create_app


def test_ws_step_plays_one_engine_move():
    c = TestClient(create_app())
    gid = c.post("/games", json={"controllers": ["greedy", "greedy"]}).json()["id"]
    with c.websocket_connect(f"/games/{gid}/stream") as ws:
        ws.send_json({"action": "step"})
        msg = ws.receive_json()
        assert msg["state"]["turn"] == 1
        assert msg["state"]["move_count"] == 1
        assert "analysis" in msg


def test_ws_play_runs_to_terminal():
    c = TestClient(create_app())
    gid = c.post("/games", json={"controllers": ["greedy", "greedy"]}).json()["id"]
    with c.websocket_connect(f"/games/{gid}/stream") as ws:
        ws.send_json({"action": "play"})
        last = None
        # greedy vs greedy terminates quickly; read until winner set
        for _ in range(500):
            msg = ws.receive_json()
            last = msg
            if msg["state"]["winner"] is not None:
                break
        assert last["state"]["winner"] in (0, 1)
