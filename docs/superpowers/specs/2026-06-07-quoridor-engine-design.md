# Barricades (Quoridor) — Game + Engine Design

**Date:** 2026-06-07
**Status:** Approved (design phase)

## Overview

Build a local implementation of Quoridor ("barricade chess") plus a pluggable
engine to play it. The game is perfect-information and Markovian. The goal is a
**learning playground**: a clean, fast game core with a pluggable agent
interface, a web UI for playing and inspecting games, and room to grow from
simple bots up to MCTS and an AlphaZero-style self-play network.

**Stack:** Python for the game core, agents, and server; a lightweight browser
front-end (canvas + vanilla TypeScript/JS) for the UI.

## Rules (standard Quoridor)

- 9×9 board. Two pawns start on opposite edges (one per player).
- Each turn a player either **moves their pawn one cell orthogonally**
  ("rook-contiguous", no diagonals) **or places a wall**.
- **Walls are 2-segment**: each wall spans two cell-edges, sits on the slot grid
  between rows/columns, and may not overlap or cross an existing wall.
- **10 walls per player.**
- **Path constraint:** a wall placement is illegal if it would completely cut
  off *either* pawn from its goal edge. There must always be a path for both
  players.
- **Jumping:** when the pawns are adjacent, a player may jump straight over the
  opponent. If a wall or board edge is directly behind the opponent, the player
  may instead jump diagonally (to either side that is not wall-blocked).
- **Win:** reach the opposite edge (the goal row for your side).

## Architecture

```
barricades/
  core/        rules, state, BFS, coordinates       (Phase 1)
  agents/      Agent interface, bots, arena         (Phases 1, 3, 4)
  server/      FastAPI REST + WebSocket bridge      (Phase 2)
  web/         canvas UI, four play modes           (Phase 2)
  tests/       core rule tests, agent sanity        (ongoing)
```

Layering principle: all rules live in `core`. `agents`, `server`, and `web` sit
on top and never re-implement rules — the core is the single source of truth, so
the UI and engine can never disagree.

## Section 1 — Game core (`barricades/core`)

Headless, no I/O, written for correctness and clarity. The hot path is isolated
behind a frozen public API so it can later be re-implemented (numpy/bitboards)
without touching consumers.

### State

- `GameState`: pawn positions `(player0, player1)`, walls placed, walls-remaining
  per player, and whose turn it is.
- **Immutable with copy-on-move:** `apply_move` returns a *new* state. This keeps
  search (which explores many branches) correct and simple; the copy cost is
  accepted now and optimized later if needed.
- Board is 9×9 cells. Walls live on an 8×8 grid of **slots**, each slot holding a
  horizontal or vertical 2-segment wall, making overlap/cross checks simple
  lookups.

### Coordinates

- Cells as `(col, row)` with `a1`–`i9` string conversion (matches the UI labels).
- One canonical conversion module so UI and engine never disagree.

### Moves

A `Move` is one of:
- `Step(to_cell)` — pawn move, including straight-jump and diagonal-jump cases.
- `Wall(slot, orientation)` — place a 2-segment wall.

### The three rules that need real care (each gets dedicated, well-tested code)

1. `legal_steps(state)` — adjacency, blocked-by-wall, and full jump logic
   (straight jump over opponent; diagonal jump when a wall/edge is behind them).
2. `legal_walls(state)` — in-bounds, no overlap/cross with existing walls, player
   has walls left, **and** does not fully block either pawn from its goal row.
3. `has_path_to_goal(state, player)` — BFS from the pawn to its goal edge,
   respecting walls. Used for wall legality and as the engine's core heuristic.

### Public API (frozen seam)

```
initial_state() -> GameState
legal_moves(state) -> list[Move]
apply_move(state, move) -> GameState        # assumes the move is legal
is_terminal(state) -> bool
winner(state) -> int | None
shortest_path_len(state, player) -> int | None   # BFS distance, for heuristics
```

Design decisions:
- Immutable state, copy-on-move.
- `apply_move` trusts legality; legality is checked separately via `legal_moves`
  (faster in search loops).

## Section 2 — Agent interface (`barricades/agents`)

The pluggable seam. Every bot implements one small interface; server, UI, and the
head-to-head harness treat all agents identically.

### Interface

```
class Agent:
    name: str
    def select_move(self, state: GameState) -> Move: ...
```

Optional richer output for the analysis/debug view:

```
class Analysis:
    best_move: Move
    value: float                          # eval of position (current player POV)
    candidates: list[tuple[Move, float]]  # scored options
    stats: dict                           # nodes searched, depth, time, etc.

# optional on Agent; defaults to wrapping select_move
def analyze(self, state) -> Analysis
```

The debug UI calls `analyze` when available, otherwise just shows the move.

### Agents (built across phases)

- `RandomAgent` — legal random move. Baseline/sanity.
- `GreedyAgent` — minimize own shortest-path / maximize the path gap (pure BFS
  heuristic, no search). Validates the heuristic.
- `MinimaxAgent` — alpha-beta with shortest-path-difference eval, iterative
  deepening, time budget. The classic solid AI.
- `MCTSAgent` — UCT; later swappable to use NN priors/value.
- `AZAgent` — AlphaZero-lite: policy+value net guiding MCTS, trained by self-play.

### Registry

Lists and instantiates agents by name with params (e.g. `minimax(depth=4,
time=2s)`) for the UI dropdowns and the arena.

### Arena (`agents/arena.py`)

Play N games between two agents (alternating who starts), report win rates.
Pure-core, headless, scriptable — the evaluation backbone for comparing bots.

Design decision: agents are **stateless across moves** (they receive the full
`GameState` each turn; may build internal search trees but need not persist them).

## Section 3 — Server (`barricades/server`)

A thin FastAPI app bridging browser and Python core/agents. Holds no rules of its
own.

### Session model

- A `Game` session holds a `GameState` history (for undo + move list) plus which
  agent (if any) controls each side.
- **In-memory only** (single-user local tool); sessions in a dict keyed by id.
  Optional "export game as JSON" later.

### REST endpoints

```
POST /games                 # new game; body picks mode + agent(s) per side
GET  /games/{id}            # full state: pawns, walls, turn, walls-left,
                            #   legal moves, status
POST /games/{id}/move       # human submits Step/Wall; validated via legal_moves
POST /games/{id}/undo
GET  /agents                # registry list for UI dropdowns
```

### WebSocket `/games/{id}/stream`

- Streams state updates as **engine-vs-engine** plays out, with server-side
  pacing + play/pause/step.
- Pushes the **analysis** payload (eval, candidates, stats, BFS path) after each
  engine move for the live debug view.

Move legality is always enforced server-side via `core.legal_moves` — the browser
is never trusted, and the same function powers the UI's legal-move highlighting.

## Section 4 — Frontend (`barricades/web`)

Lightweight browser UI. Renders state and sends moves; all rules/AI live behind
the server.

**Tech:** plain HTML + a single `<canvas>` board + vanilla TS/JS, no framework to
start (rendering is isolated, so a framework can be added later if it grows).

### Rendering

- 9×9 cells with `a`–`i` / `1`–`9` labels, two pawns, placed walls on the slot
  grid, and corner wall-anchor dots.
- Legal-move affordances: highlight legal destination cells; in wall mode, preview
  the wall slot under the cursor (green = valid, red = illegal, including the
  "would block all paths" case the server reports).

### Interaction

- Toggle move mode / wall mode. Click a highlighted cell to step; hover+click a
  slot to place a wall.
- Sidebar: whose turn, walls remaining per side, move list, undo, new-game.

### Four modes, one board component

- **Human vs engine** — pick an agent for one side; moves post to REST, engine
  reply returns (over WS for the analysis payload).
- **Human vs human** — both sides interactive, no agent.
- **Engine vs engine** — pick two agents; WS stream with play / pause / step /
  speed.
- **Analysis / debug** — togglable panel: engine eval, scored candidate moves,
  search stats, and a BFS shortest-path overlay per pawn. Available in any mode
  with an engine present.

The frontend never computes legality itself — it asks the server (which asks
`core`).

## Section 5 — Build phasing

Each phase ends with something runnable.

**Phase 1 — Core + tests (foundation).**
Implement `core` (state, coords, move gen, jumps, wall legality, BFS /
`shortest_path_len`, terminal/winner). Heavy unit tests on the tricky rules:
jump/diagonal-jump, wall overlap/cross, path-must-exist. Add `RandomAgent`,
`GreedyAgent`, and the `arena` harness; prove correctness by playing thousands of
headless games.

**Phase 2 — Server + frontend, playable.**
FastAPI endpoints + WS, then the canvas UI. Deliverable: human-vs-human and
human-vs-greedy in the browser, with legal-move highlights and wall previews.
"The game, locally," done.

**Phase 3 — Classic engine.**
`MinimaxAgent` (alpha-beta + shortest-path-difference eval, iterative deepening,
time budget). Wire up the analysis/debug view. Use `arena` to confirm it beats
greedy.

**Phase 4 — Search/learning playground.**
`MCTSAgent` (UCT). If self-play is too slow, swap the core hot path to
numpy/bitboards behind the frozen API. Then `AZAgent`: policy+value net (PyTorch),
self-play training loop, evaluation via `arena`. Open-ended research phase.

## Testing strategy

- **Core:** unit tests per rule function, with explicit cases for every jump
  variant, wall overlap/cross combination, and path-blocking scenario. Property
  test: `apply_move` output always passes an invariant check (valid pawn
  positions, wall counts, both pawns still have a path).
- **Agents:** `RandomAgent` only ever returns legal moves (fuzz over many random
  states); `GreedyAgent` beats `RandomAgent` decisively in the arena.
- **Server:** legality enforced server-side; illegal submitted moves are rejected.
- **Regression:** interesting games exportable/replayable as JSON.

## Out of scope (YAGNI for now)

- Persistence / database (in-memory sessions only).
- Multi-user / networked play.
- Authentication.
- Mobile-specific UI.
- Opening books / endgame tablebases.
