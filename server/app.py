from pathlib import Path

from fastapi import FastAPI, HTTPException
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
