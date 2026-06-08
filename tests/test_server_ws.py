from fastapi.testclient import TestClient
from server.app import create_app

# Engine-vs-engine streaming is client-driven: the client requests one move at a
# time via {"action": "step"}; the server applies and returns exactly that move,
# so its state never runs ahead of the displayed position.


def test_ws_step_plays_one_engine_move():
    c = TestClient(create_app())
    gid = c.post("/games", json={"controllers": ["greedy", "greedy"]}).json()["id"]
    with c.websocket_connect(f"/games/{gid}/stream") as ws:
        ws.send_json({"action": "step"})
        msg = ws.receive_json()
        assert msg["state"]["turn"] == 1
        assert msg["state"]["move_count"] == 1
        assert "analysis" in msg


def test_ws_repeated_steps_advance_in_strict_order():
    # Driving with repeated steps must yield move_count 1, 2, 3, ... with no gaps,
    # repeats, or reordering, ending in a win.
    c = TestClient(create_app())
    gid = c.post("/games", json={"controllers": ["greedy", "greedy"]}).json()["id"]
    counts = []
    with c.websocket_connect(f"/games/{gid}/stream") as ws:
        for _ in range(500):
            ws.send_json({"action": "step"})
            msg = ws.receive_json()
            if msg.get("done"):
                break
            counts.append(msg["state"]["move_count"])
            if msg["state"]["winner"] is not None:
                break
    assert counts == list(range(1, len(counts) + 1))
    assert counts  # at least one move happened


def test_ws_step_after_terminal_returns_done():
    c = TestClient(create_app())
    gid = c.post("/games", json={"controllers": ["greedy", "greedy"]}).json()["id"]
    with c.websocket_connect(f"/games/{gid}/stream") as ws:
        # play to completion
        for _ in range(500):
            ws.send_json({"action": "step"})
            msg = ws.receive_json()
            if msg["state"]["winner"] is not None:
                break
        assert msg["state"]["winner"] in (0, 1)
        # one more step on a finished game reports done, no further moves
        ws.send_json({"action": "step"})
        done = ws.receive_json()
        assert done["done"] is True


def test_ws_unknown_action_errors():
    c = TestClient(create_app())
    gid = c.post("/games", json={"controllers": ["greedy", "greedy"]}).json()["id"]
    with c.websocket_connect(f"/games/{gid}/stream") as ws:
        ws.send_json({"action": "bogus"})
        msg = ws.receive_json()
        assert "error" in msg
