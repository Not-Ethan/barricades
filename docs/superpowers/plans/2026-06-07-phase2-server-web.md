# Phase 2: Server + Web UI — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Make Quoridor playable in the browser: a FastAPI server exposing the `core` engine over REST + WebSocket, and a lightweight canvas UI supporting human-vs-human, human-vs-engine, engine-vs-engine, and an analysis/debug view.

**Architecture:** Thin server holding no rules of its own — it wraps `core` and `agents`. In-memory game sessions. REST for state-changing actions; WebSocket for streaming engine-vs-engine playouts and analysis. A no-build vanilla-JS (ES modules) frontend rendered on a single `<canvas>`, served as static files by FastAPI. The frontend never computes legality — it asks the server.

**Tech Stack:** Python 3.11+, FastAPI, uvicorn, pydantic (v2), pytest + httpx (FastAPI `TestClient`). Frontend: plain JS ES modules + Canvas 2D, no bundler. Browser verification via Playwright MCP tools.

---

## Conventions / facts this plan relies on
- `core` exports: `initial_state`, `legal_moves`, `legal_steps`, `legal_walls`, `apply_move`, `is_terminal`, `winner`, `GameState`, `Step`, `Wall`, `goal_row`. Cells `(col,row)` 0..8 (a..i / 1..9). Walls anchors `(c,r)` 0..7, orient "H"/"V".
- `agents` exports via `agents.registry`: `available_agents()`, `make_agent(name, **kwargs)`. Agents have `select_move(state)` and `analyze(state) -> Analysis(best_move, value, candidates, stats)`.
- **Registry is read dynamically** — do NOT hardcode agent names. Phase 3 adds `"minimax"` concurrently; reading `available_agents()` picks it up with no code change here.

## File Structure
```
server/
  __init__.py
  serialize.py    # GameState <-> dict, Move parsing, Analysis -> dict
  games.py        # Game session + in-memory GameStore
  schemas.py      # pydantic request/response models
  app.py          # FastAPI app: REST routes, WS route, static mount
web/
  index.html
  style.css
  api.js          # REST + WS client
  board.js        # canvas rendering + pointer interaction
  main.js         # mode wiring, controls, app state
tests/
  test_serialize.py
  test_server_rest.py
  test_server_ws.py
```

Add deps to `pyproject.toml`: `fastapi`, `uvicorn[standard]`, `httpx` (test). Install into `.venv`.

---

## JSON contract (single source of truth — implement exactly)

**State out** (`GET /games/{id}`, and inside other responses):
```json
{
  "id": "g1",
  "pawns": [[4,0],[4,8]],
  "h_walls": [[3,3]],
  "v_walls": [],
  "walls_left": [9,10],
  "turn": 0,
  "winner": null,
  "controllers": ["human","greedy"],
  "legal": {
    "steps": [[4,1],[3,0],[5,0]],
    "walls": [{"c":0,"r":0,"orient":"H"}, ...]
  },
  "move_count": 3
}
```

**New game in** (`POST /games`):
```json
{ "controllers": ["human","greedy"] }
```
`controllers[i]` is `"human"` or an agent name from `available_agents()`. Optional per-agent params may be sent as `{"name":"minimax","params":{"time_budget":1.0}}` in place of a string; support both the string form and the object form.

**Move in** (`POST /games/{id}/move`):
```json
{ "type": "step", "to": [4,1] }
// or
{ "type": "wall", "c": 3, "r": 3, "orient": "H" }
```

**Analysis out** (WS messages and `engine move` responses):
```json
{
  "best_move": {"type":"step","to":[4,1]},
  "value": 1.2,
  "candidates": [ {"move": {...}, "score": 1.2}, ... ],
  "stats": {"nodes": 1234, "depth": 2, "time_ms": 210}
}
```

---

## Task 1: Serialization (pure, fully TDD-able)

**Files:** Create `server/__init__.py` (empty), `server/serialize.py`, test `tests/test_serialize.py`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_serialize.py
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
```

- [ ] **Step 2: Run, verify fail.**

- [ ] **Step 3: Implement `server/serialize.py`**

```python
from core.state import Step, Wall
from core.rules import legal_steps, legal_walls, winner, is_terminal


def move_to_dict(move):
    if isinstance(move, Step):
        return {"type": "step", "to": list(move.to_cell)}
    if isinstance(move, Wall):
        return {"type": "wall", "c": move.c, "r": move.r, "orient": move.orient}
    raise ValueError(f"not a move: {move!r}")


def parse_move(d):
    t = d.get("type")
    if t == "step":
        return Step(tuple(d["to"]))
    if t == "wall":
        return Wall(int(d["c"]), int(d["r"]), d["orient"])
    raise ValueError(f"unknown move type: {t!r}")


def _legal_dict(state):
    if is_terminal(state):
        return {"steps": [], "walls": []}
    return {
        "steps": [list(c) for c in legal_steps(state)],
        "walls": [{"c": w.c, "r": w.r, "orient": w.orient} for w in legal_walls(state)],
    }


def state_to_dict(state, game_id, controllers, move_count=0):
    return {
        "id": game_id,
        "pawns": [list(state.pawns[0]), list(state.pawns[1])],
        "h_walls": sorted([list(a) for a in state.h_walls]),
        "v_walls": sorted([list(a) for a in state.v_walls]),
        "walls_left": list(state.walls_left),
        "turn": state.turn,
        "winner": winner(state),
        "controllers": list(controllers),
        "legal": _legal_dict(state),
        "move_count": move_count,
    }


def analysis_to_dict(analysis):
    return {
        "best_move": move_to_dict(analysis.best_move),
        "value": analysis.value,
        "candidates": [{"move": move_to_dict(m), "score": s}
                       for m, s in analysis.candidates],
        "stats": analysis.stats,
    }
```

- [ ] **Step 4: Run, verify pass. Step 5: Commit** `feat: server JSON serialization`.

---

## Task 2: Game session + in-memory store

**Files:** Create `server/games.py`, test `tests/test_games.py`.

A `Game` holds a history of `GameState` (for undo + move_count), the controllers, and an id. `GameStore` maps id → Game and creates ids.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_games.py
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
```

- [ ] **Step 2: Run, verify fail.**

- [ ] **Step 3: Implement `server/games.py`**

```python
from itertools import count

from core.state import initial_state
from core.rules import legal_moves, apply_move


class Game:
    def __init__(self, game_id, controllers):
        self.id = game_id
        self.controllers = list(controllers)   # ["human" | agent-name, ...]
        # Raw controller specs (str or {name,params}); the app may override this
        # with objects carrying agent params. Defaulted so it always exists.
        self._specs = list(controllers)
        self.history = [initial_state()]

    @property
    def state(self):
        return self.history[-1]

    @property
    def move_count(self):
        return len(self.history) - 1

    def apply(self, move):
        if move not in legal_moves(self.state):
            raise ValueError(f"illegal move: {move!r}")
        self.history.append(apply_move(self.state, move))

    def undo(self):
        if len(self.history) > 1:
            self.history.pop()


class GameStore:
    def __init__(self):
        self._games = {}
        self._ids = count(1)

    def create(self, controllers):
        gid = f"g{next(self._ids)}"
        g = Game(gid, controllers)
        self._games[gid] = g
        return g

    def get(self, gid):
        if gid not in self._games:
            raise KeyError(gid)
        return self._games[gid]
```

- [ ] **Step 4: Run, verify pass. Step 5: Commit** `feat: in-memory game sessions`.

---

## Task 3: REST API

**Files:** Create `server/schemas.py`, `server/app.py`, test `tests/test_server_rest.py`.

Endpoints: `GET /agents`, `POST /games`, `GET /games/{id}`, `POST /games/{id}/move`, `POST /games/{id}/undo`, `POST /games/{id}/engine_move` (compute+apply the move for the side to move when it is engine-controlled; returns new state + analysis). Static files mounted at `/` from `web/`.

Controller spec parsing: accept either `"greedy"` or `{"name": "...", "params": {...}}`. Build agents lazily via `make_agent`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_server_rest.py
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
```

- [ ] **Step 2: Run, verify fail.**

- [ ] **Step 3: Implement `server/schemas.py`** (pydantic v2 models for request bodies):

```python
from typing import Union
from pydantic import BaseModel


class ControllerObj(BaseModel):
    name: str
    params: dict = {}


class NewGame(BaseModel):
    controllers: list[Union[str, ControllerObj]]


class MoveIn(BaseModel):
    type: str
    to: list[int] | None = None
    c: int | None = None
    r: int | None = None
    orient: str | None = None
```

- [ ] **Step 4: Implement `server/app.py`** with a `create_app()` factory (fresh store per app, so tests are isolated):

```python
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from agents.registry import available_agents, make_agent
from server.games import GameStore
from server.schemas import NewGame, MoveIn
from server.serialize import state_to_dict, parse_move, analysis_to_dict

WEB_DIR = Path(__file__).resolve().parent.parent / "web"


def _controller_names(controllers):
    out = []
    for c in controllers:
        out.append(c if isinstance(c, str) else c.name)
    return out


def _make_controller_agent(spec):
    if isinstance(spec, str):
        return make_agent(spec)
    return make_agent(spec.name, **spec.params)


def create_app():
    app = FastAPI(title="Barricades")
    store = GameStore()

    def state_payload(game):
        return state_to_dict(game.state, game.id,
                             _controller_names(game.controllers), game.move_count)

    @app.get("/agents")
    def list_agents():
        return {"agents": available_agents()}

    @app.post("/games")
    def new_game(body: NewGame):
        names = _controller_names(body.controllers)
        game = store.create(names)
        # stash original specs for engine params
        game._specs = body.controllers
        return state_payload(game)

    @app.get("/games/{gid}")
    def get_game(gid: str):
        try:
            game = store.get(gid)
        except KeyError:
            raise HTTPException(404, "no such game")
        return state_payload(game)

    @app.post("/games/{gid}/move")
    def move(gid: str, body: MoveIn):
        try:
            game = store.get(gid)
        except KeyError:
            raise HTTPException(404, "no such game")
        try:
            game.apply(parse_move(body.model_dump()))
        except ValueError as e:
            raise HTTPException(400, str(e))
        return state_payload(game)

    @app.post("/games/{gid}/undo")
    def undo(gid: str):
        try:
            game = store.get(gid)
        except KeyError:
            raise HTTPException(404, "no such game")
        game.undo()
        return state_payload(game)

    @app.post("/games/{gid}/engine_move")
    def engine_move(gid: str):
        try:
            game = store.get(gid)
        except KeyError:
            raise HTTPException(404, "no such game")
        spec = game._specs[game.state.turn]
        if (isinstance(spec, str) and spec == "human"):
            raise HTTPException(400, "side to move is human")
        agent = _make_controller_agent(spec)
        analysis = agent.analyze(game.state)
        game.apply(analysis.best_move)
        return {"state": state_payload(game),
                "analysis": analysis_to_dict(analysis)}

    if WEB_DIR.exists():
        app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
    return app


app = create_app()
```

- [ ] **Step 5: Add deps + run.** Update `pyproject.toml` (add a `dependencies = ["fastapi", "uvicorn[standard]"]` under `[project]`, and pytest dep `httpx`). Install: `. .venv/bin/activate && pip install -q fastapi "uvicorn[standard]" httpx`. Run `pytest tests/test_server_rest.py -q` → PASS.
- [ ] **Step 6: Commit** `feat: REST API for games and engine moves`.

---

## Task 4: WebSocket engine-vs-engine stream

**Files:** Modify `server/app.py` (add WS route), test `tests/test_server_ws.py`.

WS `/games/{id}/stream`: client connects, sends `{"action":"play"}` / `{"action":"step"}` / `{"action":"pause"}`. On `step`, server computes one engine move for the side to move (must be engine-controlled), applies it, and sends `{"state":..., "analysis":...}`. On `play`, server auto-steps until terminal, sending one message per move (test uses `step` to stay deterministic). If the side to move is human, send `{"error":"..."}`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_server_ws.py
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
```

- [ ] **Step 2: Run, verify fail.**

- [ ] **Step 3: Implement the WS route** inside `create_app()` (before the static mount):

```python
    from fastapi import WebSocket, WebSocketDisconnect

    @app.websocket("/games/{gid}/stream")
    async def stream(ws: WebSocket, gid: str):
        await ws.accept()
        try:
            game = store.get(gid)
        except KeyError:
            await ws.send_json({"error": "no such game"})
            await ws.close()
            return

        def one_engine_move():
            from core.rules import is_terminal
            if is_terminal(game.state):
                return {"state": state_payload(game), "analysis": None,
                        "done": True}
            spec = game._specs[game.state.turn]
            if isinstance(spec, str) and spec == "human":
                return {"error": "side to move is human"}
            agent = _make_controller_agent(spec)
            analysis = agent.analyze(game.state)
            game.apply(analysis.best_move)
            return {"state": state_payload(game),
                    "analysis": analysis_to_dict(analysis)}

        try:
            while True:
                cmd = await ws.receive_json()
                action = cmd.get("action")
                if action == "step":
                    await ws.send_json(one_engine_move())
                elif action == "play":
                    from core.rules import is_terminal
                    while not is_terminal(game.state):
                        msg = one_engine_move()
                        await ws.send_json(msg)
                        if "error" in msg:
                            break
                elif action == "pause":
                    await ws.send_json({"paused": True})
                else:
                    await ws.send_json({"error": f"unknown action {action!r}"})
        except WebSocketDisconnect:
            return
```

- [ ] **Step 4: Run, verify pass. Step 5: Commit** `feat: websocket engine-vs-engine stream`.

---

## Task 5: Frontend — board rendering + API client

**Files:** Create `web/index.html`, `web/style.css`, `web/api.js`, `web/board.js`. No tests (verified via Playwright in Task 7); keep modules small and focused.

**`web/api.js`** — thin fetch/WS wrapper: `listAgents()`, `newGame(controllers)`, `getGame(id)`, `sendMove(id, move)`, `undo(id)`, `engineMove(id)`, `openStream(id, onMessage)`. Each returns parsed JSON / a WebSocket.

**`web/board.js`** — `class Board` that owns a `<canvas>`:
- Geometry: 9×9 cells. Constants: `CELL=64`, `GAP=10` (gap is where walls render), margin for labels. Map cell `(c,r)` → pixel; row 0 at the BOTTOM (rank 1), so `y = margin + (8 - r) * (CELL+GAP)`.
- `render(state, {legalSteps, mode, pathOverlay})`: draw cells, file/rank labels (a–i, 1–9), the two pawns (player 0 = red, player 1 = blue, matching the reference screenshot), placed walls (h_walls as horizontal bars in the gap above row r spanning cols c..c+1; v_walls as vertical bars right of col c spanning rows r..r+1), wall-anchor dots at cell corners, and—in move mode—highlight legal destination cells with a green outline.
- Hit-testing for input:
  - `cellAt(px,py)` → `(c,r)` or null (for step moves).
  - `wallSlotAt(px,py)` → `{c,r,orient}` or null: in wall mode, decide H vs V from whether the pointer is nearer a horizontal or vertical gridline gap, and snap to a valid 0..7 anchor.
- Hover preview in wall mode: `previewWall(slot, isLegal)` draws the candidate wall in green (legal) or red (illegal).
- The board emits callbacks: `onStep(cell)`, `onWall(slot)` set by `main.js`.

Keep `board.js` focused on drawing + hit-testing only; no network or game-flow logic.

- [ ] **Step 1:** Write `web/index.html` — a canvas, a sidebar with: new-game controls (two dropdowns for controllers populated from `/agents` plus "human"), turn indicator, walls-remaining for both players, a move/wall mode toggle, undo + new-game buttons, an engine-vs-engine control row (play/pause/step/speed), and a collapsible "Analysis" panel (eval number, candidate list, nodes/depth/time). Load `main.js` as `<script type="module">`.
- [ ] **Step 2:** Write `web/style.css` — clean, minimal, dark-on-light; layout = canvas left, sidebar right.
- [ ] **Step 3:** Write `web/api.js` and `web/board.js` per the spec above.
- [ ] **Step 4: Commit** `feat: web board rendering and API client`.

---

## Task 6: Frontend — app wiring (`web/main.js`)

**Files:** Create `web/main.js`.

Owns app state: current `gameId`, latest `state` JSON, current `mode` ("move"/"wall"), and whether each side is human/engine (from `state.controllers`).

Behavior:
- On load: fetch `/agents`, populate the two controller dropdowns (options: `human` + each agent name). Start a default human-vs-greedy game.
- **Human move:** in move mode, clicking a legal cell → `sendMove(step)`; in wall mode, clicking a valid slot → `sendMove(wall)`. After each human move, if the new side-to-move is an engine, call `engineMove()` and render the returned state + analysis (loop while side-to-move is engine and game not over, so engine-vs-human chains correctly).
- **Human vs human:** never auto-calls engine.
- **Engine vs engine:** "Start" opens the WS stream; play/pause/step buttons send `{action:...}`; each incoming message re-renders board + analysis; a speed slider throttles `play` rendering (client-side delay between applying messages — server streams; client paces display).
- **Analysis panel:** whenever a response includes `analysis`, show value, the candidate moves (formatted like `e1→e2` / `wall H c3`), and stats. Add a checkbox "show shortest paths" that requests/draws the BFS path overlay — compute the overlay client-side is NOT possible (walls logic lives server-side); instead derive a simple highlight from the engine's `best_move`. (Keep overlay minimal: highlight best_move target. Full BFS overlay is deferred.)
- Illegal-move feedback: the server returns 400; surface a brief inline message and re-render legal moves.

- [ ] **Step 1:** Implement `web/main.js` wiring all four modes against the REST + WS API. Keep functions small (render, onCellClick, onSlotClick, refreshFromState, runEngineTurnsIfNeeded, startEvE).
- [ ] **Step 2: Commit** `feat: frontend app wiring for all four modes`.

---

## Task 7: End-to-end browser verification (Playwright)

**Files:** none (verification only). Uses Playwright MCP browser tools.

- [ ] **Step 1:** Start the server: `. .venv/bin/activate && uvicorn server.app:app --port 8123` (run in background).
- [ ] **Step 2:** Navigate to `http://localhost:8123/`. Take a snapshot/screenshot; confirm the board renders with both pawns and labels.
- [ ] **Step 3:** Human-vs-engine: click a legal cell for player 0; confirm the pawn moves and the engine (greedy) replies (turn returns to 0). Screenshot.
- [ ] **Step 4:** Wall mode: toggle to wall mode, hover a slot (confirm green/red preview), place a legal wall; confirm it renders in the gap and walls-remaining decremented.
- [ ] **Step 5:** Engine-vs-engine: start a greedy-vs-greedy game, hit Step a few times and then Play; confirm the board updates and a winner is eventually declared. Screenshot.
- [ ] **Step 6:** Confirm the Analysis panel shows eval/candidates/stats when an engine moves.
- [ ] **Step 7:** Stop the server. Document findings; if any interaction is broken, fix the relevant `web/*.js` and re-verify (do not skip). **Commit** any fixes `fix: <issue> from e2e verification`.

---

## Done criteria
- `pytest -q` green (serialize, games, REST, WS tests).
- Server runs; the four modes all work in a real browser (Playwright-verified with screenshots).
- Frontend computes no legality itself — all from the server.
- Only new files + `pyproject.toml` touched; `agents/registry.py` NOT modified (clean seam vs Phase 3).
