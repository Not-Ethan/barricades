from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

from agents.heuristics import evaluate
from agents.registry import available_agents, make_agent
from core.coords import N, on_board
from core.rules import has_path_to_goal, winner as core_winner, _overlaps
from core.state import Wall
from server.games import GameStore
from server.schemas import AnalyzeRequest, NewGame, MoveIn
from server.serialize import (
    _legal_dict, analysis_to_dict, dict_to_state, parse_move, state_to_dict,
)

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
        valid = set(available_agents()) | {"human"}
        bad = [n for n in names if n not in valid]
        if bad:
            raise HTTPException(400, f"unknown controller(s): {bad}")
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
        if isinstance(spec, str) and spec == "human":
            raise HTTPException(400, "side to move is human")
        agent = _make_controller_agent(spec)
        analysis = agent.analyze(game.state)
        game.apply(analysis.best_move)
        return {"state": state_payload(game),
                "analysis": analysis_to_dict(analysis)}

    # ------------------------------------------------------------------
    # POST /analyze  — stateless position analysis (does not touch GameStore)
    # ------------------------------------------------------------------

    def _validate_position(state):
        """Validate a GameState; return (ok, reason). reason is None when ok."""
        # 1. Pawns on-board and distinct
        for i, p in enumerate(state.pawns):
            if not on_board(p):
                return False, f"pawn {i} is off-board: {p}"
        if state.pawns[0] == state.pawns[1]:
            return False, "both pawns occupy the same cell"

        # 2. walls_left each in 0..10
        for i, wl in enumerate(state.walls_left):
            if not (0 <= wl <= 10):
                return False, f"walls_left[{i}] out of range 0..10: {wl}"

        # 3. Wall anchors in range 0..N-2 and non-overlapping/non-crossing
        for orient, wall_set in (("H", state.h_walls), ("V", state.v_walls)):
            for (c, r) in wall_set:
                if not (0 <= c <= N - 2 and 0 <= r <= N - 2):
                    return False, (
                        f"{orient}-wall anchor ({c},{r}) out of range 0..{N-2}"
                    )

        # Check for overlaps/crosses by rebuilding incrementally
        from core.state import GameState as GS
        tmp = GS(state.pawns, frozenset(), frozenset(), state.walls_left, state.turn)
        for (c, r) in state.h_walls:
            w = Wall(c, r, "H")
            if _overlaps(tmp, w):
                return False, f"H-wall at ({c},{r}) overlaps or crosses another wall"
            tmp = GS(tmp.pawns, tmp.h_walls | {(c, r)}, tmp.v_walls,
                     tmp.walls_left, tmp.turn)
        for (c, r) in state.v_walls:
            w = Wall(c, r, "V")
            if _overlaps(tmp, w):
                return False, f"V-wall at ({c},{r}) overlaps or crosses another wall"
            tmp = GS(tmp.pawns, tmp.h_walls, tmp.v_walls | {(c, r)},
                     tmp.walls_left, tmp.turn)

        # 4. Both pawns must have a path to goal (unless the game is already won)
        if core_winner(state) is None:
            for player in (0, 1):
                if not has_path_to_goal(state, player):
                    return False, f"player {player} has no path to their goal"

        return True, None

    @app.post("/analyze")
    def analyze(body: AnalyzeRequest):
        # Build state
        pos = body.position.model_dump()
        state = dict_to_state(pos)

        # Validate
        ok, reason = _validate_position(state)
        if not ok:
            return {"valid": False, "reason": reason}

        # Check for finished game
        w = core_winner(state)
        from agents.heuristics import WIN_SCORE
        if w is not None:
            static_eval = WIN_SCORE if w == state.turn else -WIN_SCORE
            return {
                "valid": True,
                "winner": w,
                "static_eval": static_eval,
                "turn": state.turn,
                "legal": {"steps": [], "walls": []},
                "results": [],
            }

        # Compute static eval and legal moves
        static_eval = evaluate(state, state.turn)
        legal = _legal_dict(state)

        # Run each engine
        results = []
        for spec in body.engines:
            try:
                agent = make_agent(spec.name, **spec.params)
            except ValueError as exc:
                raise HTTPException(400, str(exc))
            analysis = agent.analyze(state)
            results.append({"engine": spec.name, **analysis_to_dict(analysis)})

        return {
            "valid": True,
            "winner": None,
            "static_eval": static_eval,
            "turn": state.turn,
            "legal": legal,
            "results": results,
        }

    import asyncio

    from fastapi import WebSocket, WebSocketDisconnect
    from core.rules import is_terminal

    @app.websocket("/games/{gid}/stream")
    async def stream(ws: WebSocket, gid: str):
        # Engine-vs-engine streaming is CLIENT-DRIVEN: the client requests one
        # move at a time via {"action": "step"}, so the server's game state never
        # races ahead of the display. Playback cadence and pause/step are handled
        # entirely on the client (it simply stops/starts requesting). The engine
        # search runs off the event loop so a move never blocks other I/O.
        await ws.accept()
        try:
            game = store.get(gid)
        except KeyError:
            await ws.send_json({"error": "no such game"})
            await ws.close()
            return

        async def next_move_msg():
            if is_terminal(game.state):
                return {"state": state_payload(game), "analysis": None,
                        "done": True}
            spec = game._specs[game.state.turn]
            if isinstance(spec, str) and spec == "human":
                return {"error": "side to move is human"}
            agent = _make_controller_agent(spec)
            analysis = await asyncio.to_thread(agent.analyze, game.state)
            game.apply(analysis.best_move)
            return {"state": state_payload(game),
                    "analysis": analysis_to_dict(analysis)}

        try:
            while True:
                cmd = await ws.receive_json()
                if cmd.get("action") == "step":
                    await ws.send_json(await next_move_msg())
                else:
                    await ws.send_json(
                        {"error": f"unknown action {cmd.get('action')!r}"})
        except WebSocketDisconnect:
            return

    if WEB_DIR.exists():
        app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
    return app


app = create_app()
