from core.state import Step, Wall, GameState
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
        to = d.get("to")
        if not (isinstance(to, (list, tuple)) and len(to) == 2):
            raise ValueError("step move requires 'to' as [c, r]")
        return Step((int(to[0]), int(to[1])))
    if t == "wall":
        if d.get("c") is None or d.get("r") is None or d.get("orient") not in ("H", "V"):
            raise ValueError("wall move requires 'c', 'r', and 'orient' in {H, V}")
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


def dict_to_state(d) -> GameState:
    """Inverse of the core fields in state_to_dict.

    Accepts a dict with keys: pawns, h_walls, v_walls, walls_left, turn.
    Lists are converted to tuples; wall sets become frozensets.
    """
    pawns = tuple(tuple(p) for p in d["pawns"])
    h_walls = frozenset(tuple(w) for w in d["h_walls"])
    v_walls = frozenset(tuple(w) for w in d["v_walls"])
    walls_left = tuple(d["walls_left"])
    turn = int(d["turn"])
    return GameState(pawns=pawns, h_walls=h_walls, v_walls=v_walls,
                     walls_left=walls_left, turn=turn)


def analysis_to_dict(analysis):
    return {
        "best_move": move_to_dict(analysis.best_move),
        "value": analysis.value,
        "candidates": [{"move": move_to_dict(m), "score": s}
                       for m, s in analysis.candidates],
        "stats": analysis.stats,
    }
