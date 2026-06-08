# Analysis Board — Design Spec

**Date:** 2026-06-08
**Status:** Approved (design phase)
**Branch:** `engine-strength`

## Overview

A position-setup + engine-analysis tool in the existing web UI. The user can edit
an arbitrary board (place both pawns, add/remove walls, set walls-remaining per
side, choose whose turn it is), then run the engines on that exact position and
see their evaluations — including, for two engines side by side, each engine's
position eval, its best move (highlighted on the board), and its top candidate
moves with scores, plus the static heuristic eval.

This is a tool for *understanding and tuning* the engines (it's what we'll use to
test new evaluation features), so per-move scores and engine comparison are the
point.

## Approach (chosen)

Stateless **`POST /analyze`** endpoint (no game session) + a **Setup mode** added
to the existing UI. Reuses the canvas board renderer and all existing engines.
(Rejected: reusing the game-session model — conflates analysis with play;
reimplementing eval in JS — duplicates logic, can't run the Python engines.)

## Architecture

### Server (`server/`)
- **`serialize.py`**: add `dict_to_state(d) -> GameState`, the inverse of
  `state_to_dict` for the core fields (`pawns`, `h_walls`, `v_walls`,
  `walls_left`, `turn`). Reuse `analysis_to_dict`, `move_to_dict`.
- **`app.py`**: add `POST /analyze`.
  - **Request**:
    ```json
    {
      "position": {"pawns": [[c,r],[c,r]], "h_walls": [[c,r]...],
                   "v_walls": [[c,r]...], "walls_left": [n0,n1], "turn": 0},
      "engines": [{"name": "minimax", "params": {"time_budget": 0.3}},
                  {"name": "mcts",    "params": {"time_budget": 0.3}}]
    }
    ```
  - **Validation** (before running any engine): pawns on-board and distinct;
    `walls_left` each in 0..10; total walls placed + walls_left ≤ 10 per side is
    NOT required (positions may be hand-set, but `walls_left` is what the engines
    use); wall anchors in range and non-overlapping/non-crossing; **both pawns
    have a path to goal**; neither pawn already on its own goal row (that's a
    finished game — still allow, but mark `winner`). If invalid →
    `{"valid": false, "reason": "<human-readable>"}` and engines are NOT run.
  - **Response (valid)**:
    ```json
    {
      "valid": true,
      "winner": null,
      "static_eval": 1.0,                     // heuristics.evaluate(state, state.turn)
      "turn": 0,
      "legal": {"steps": [...], "walls": [...]},
      "results": [
        {"engine": "minimax", "best_move": {...}, "value": 1.2,
         "candidates": [{"move": {...}, "score": 1.2}, ...],
         "stats": {...}}
      ]
    }
    ```
  - `static_eval` is from the side-to-move's perspective (the eval's natural POV),
    labelled as such in the UI.
  - Engines built via the existing registry/`make_agent` with the request's
    params; modest default budget (0.3s). Each engine's `analyze(state)` is
    serialized via `analysis_to_dict`.
- **`schemas.py`**: add a pydantic `AnalyzeRequest` (position + engines list).

### Frontend (`web/`)
- **`board.js`**: add an **edit interaction mode** distinct from play
  (`move`/`wall`): in edit mode a click either (a) places the currently-selected
  piece (red pawn / blue pawn) on a cell, or (b) toggles a wall at a gap (add if
  empty, remove if a wall is there). Rendering gains: highlight an engine's
  best-move target; (optional) lightly mark candidate-move targets.
- **`index.html`**: a **Setup** section with: a piece selector (Red pawn / Blue
  pawn / Wall), walls-remaining spinners (×2), a turn radio (Red/Blue), Clear and
  Reset-to-start buttons; an **Analyze** panel with a budget control (Fast/Normal)
  and a two-column **engine comparison** (default minimax vs mcts) showing each
  engine's eval, best move, and top candidate moves with scores, plus the static
  eval and a legality indicator.
- **`api.js`**: add `analyzePosition(position, engines)` → `POST /analyze`.
- **`main.js`**: Setup-mode state (current position being edited, selected piece);
  wire editor clicks to mutate the local position and re-render with a live
  legality check; the Analyze button calls `analyzePosition` and renders the
  side-by-side results + board highlights. Setup mode is entered via a top-level
  toggle and is mutually exclusive with the play modes.

## Data flow
Editor builds a position object → (Analyze) `POST /analyze` → server
`dict_to_state` → validate → `evaluate` + each engine's `analyze` → JSON →
UI renders static eval + per-engine eval/best-move/candidates + board highlights.

## Error handling
- Illegal position → server `{valid:false, reason}`; UI shows the reason and does
  not display engine results. The editor also shows a live legality hint
  (recomputed client-side is not required; rely on the server response on Analyze,
  and a simple "looks illegal" hint is optional).
- Unknown engine name / bad params → 400 (reuse existing validation).

## Testing
- **Server** (`tests/test_analyze.py`): valid position → `valid:true` with
  `static_eval`, `legal`, and per-engine `results` containing `candidates`;
  illegal position (a pawn walled off) → `valid:false` with a reason and no
  engine run; `dict_to_state(state_to_dict(s)) == s` round-trip for several
  states; a position with a winner reports `winner`.
- **End-to-end** (Playwright, controller-run): enter Setup mode, place pawns +
  a wall, set turn, Analyze, confirm the static eval and both engines' evals +
  candidate lists render and the best move highlights on the board.

## Out of scope (YAGNI)
- Saving/loading named positions, position-string import/export, move-list
  navigation, or analysis of full game histories. (Can be added later.)
- More than two engines compared at once.
