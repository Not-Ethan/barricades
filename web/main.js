/**
 * main.js — App wiring for all four game modes:
 *   1. Human vs Human
 *   2. Human vs Engine
 *   3. Engine vs Human (engine is player 0)
 *   4. Engine vs Engine (WebSocket stream)
 *
 * Also wires the Setup/Analysis mode (position editor + engine comparison).
 *
 * Owns app state; delegates rendering to Board, networking to api.js.
 */

import { listAgents, newGame, getGame, sendMove, undo, engineMove, openStream, analyzePosition } from "./api.js";
import { Board } from "./board.js";

// ---------------------------------------------------------------------------
// DOM refs — Play mode
// ---------------------------------------------------------------------------
const canvas       = document.getElementById("board-canvas");
const turnEl       = document.getElementById("turn-indicator");
const errorEl      = document.getElementById("error-msg");
const walls0El     = document.getElementById("walls-left-0");
const walls1El     = document.getElementById("walls-left-1");
const moveCountEl  = document.getElementById("move-count");
const ctrl0El      = document.getElementById("ctrl-0");
const ctrl1El      = document.getElementById("ctrl-1");
const btnNewGame   = document.getElementById("btn-new-game");
const btnUndo      = document.getElementById("btn-undo");
const modeRadios   = document.querySelectorAll("input[name='mode']");
const btnEveStart  = document.getElementById("btn-eve-start");
const btnEveStep   = document.getElementById("btn-eve-step");
const btnEvePause  = document.getElementById("btn-eve-pause");
const speedSlider  = document.getElementById("speed-slider");
const speedLabel   = document.getElementById("speed-label");
const analysisValueEl = document.getElementById("analysis-value");
const analysisNodesEl = document.getElementById("analysis-nodes");
const analysisDepthEl = document.getElementById("analysis-depth");
const analysisTimeEl  = document.getElementById("analysis-time");
const candidatesList  = document.getElementById("candidates-list");
const btnToggleAnalysis = document.getElementById("btn-toggle-analysis");
const analysisBody  = document.getElementById("analysis-body");
const showBestMove  = document.getElementById("show-best-move");

// ---------------------------------------------------------------------------
// DOM refs — App mode switch
// ---------------------------------------------------------------------------
const appModeRadios   = document.querySelectorAll("input[name='app-mode']");
const playControls    = document.getElementById("play-controls");
const setupControls   = document.getElementById("setup-controls");

// ---------------------------------------------------------------------------
// DOM refs — Setup mode
// ---------------------------------------------------------------------------
const setupPieceRadios   = document.querySelectorAll("input[name='setup-piece']");
const setupWalls0El      = document.getElementById("setup-walls-0");
const setupWalls1El      = document.getElementById("setup-walls-1");
const setupTurnRadios    = document.querySelectorAll("input[name='setup-turn']");
const btnSetupReset      = document.getElementById("btn-setup-reset");
const btnSetupClearWalls = document.getElementById("btn-setup-clear-walls");
const btnAnalyze         = document.getElementById("btn-analyze");
const setupBudgetEl      = document.getElementById("setup-budget");
const setupEngine0El     = document.getElementById("setup-engine-0");
const setupEngine1El     = document.getElementById("setup-engine-1");
const setupLegalityEl    = document.getElementById("setup-legality");
const setupStaticEvalEl  = document.getElementById("setup-static-eval");
const setupEngineName0   = document.getElementById("setup-engine-name-0");
const setupEngineName1   = document.getElementById("setup-engine-name-1");
const setupEngineVal0    = document.getElementById("setup-engine-val-0");
const setupEngineVal1    = document.getElementById("setup-engine-val-1");
const setupEngineBest0   = document.getElementById("setup-engine-best-0");
const setupEngineBest1   = document.getElementById("setup-engine-best-1");
const setupCandidates0   = document.getElementById("setup-candidates-0");
const setupCandidates1   = document.getElementById("setup-candidates-1");

// ---------------------------------------------------------------------------
// App state
// ---------------------------------------------------------------------------
let gameId   = null;
let state    = null;   // latest server state dict
let mode     = "move"; // "move" | "wall"
let analysis = null;   // latest analysis dict or null
let eveWs    = null;   // active WebSocket for engine-vs-engine, or null
let evePaused = false;
let eveGen    = 0;     // bumped whenever the EvE stream is (re)started/closed;
                       // in-flight async callbacks bail when their captured gen
                       // no longer matches (prevents stale renders after New Game)
let eveWaiting = false; // true while a step request is in flight (one at a time)

// Setup mode state
let appMode = "play";  // "play" | "setup"

const DEFAULT_SETUP_POSITION = () => ({
  pawns: [[4, 0], [4, 8]],
  h_walls: [],
  v_walls: [],
  walls_left: [10, 10],
  turn: 0,
});

let setupPosition = DEFAULT_SETUP_POSITION();

/** Returns the currently selected piece in setup mode. */
function getSelectedPiece() {
  for (const r of setupPieceRadios) {
    if (r.checked) return r.value;
  }
  return "red";
}

const board = new Board(canvas);
board.onStep = onCellClick;
board.onWall = onSlotClick;
board.onEditCell = onEditCell;
board.onEditWall = onEditWall;

// ---------------------------------------------------------------------------
// Initialisation
// ---------------------------------------------------------------------------
async function init() {
  // Populate agent dropdowns (play mode + setup mode)
  try {
    const { agents } = await listAgents();
    for (const sel of [ctrl0El, ctrl1El]) {
      for (const name of agents) {
        const opt = document.createElement("option");
        opt.value = name;
        opt.textContent = name.charAt(0).toUpperCase() + name.slice(1);
        sel.appendChild(opt);
      }
    }
    // Default: human vs greedy
    ctrl1El.value = "greedy";

    // Populate setup engine dropdowns
    for (const sel of [setupEngine0El, setupEngine1El]) {
      // Clear existing options first
      sel.innerHTML = "";
      for (const name of agents) {
        const opt = document.createElement("option");
        opt.value = name;
        opt.textContent = name.charAt(0).toUpperCase() + name.slice(1);
        sel.appendChild(opt);
      }
    }
    // Defaults: engine 0 = minimax, engine 1 = mcts (if available)
    if (agents.includes("minimax")) setupEngine0El.value = "minimax";
    if (agents.includes("mcts"))    setupEngine1El.value = "mcts";
  } catch (e) {
    showError("Failed to fetch agents: " + e.message);
  }

  // Start default game
  await startNewGame();
}

// ---------------------------------------------------------------------------
// New game
// ---------------------------------------------------------------------------
async function startNewGame() {
  closeEveWs();
  clearError();
  const c0 = ctrl0El.value;
  const c1 = ctrl1El.value;

  try {
    state = await newGame([c0, c1]);
    gameId = state.id;
    analysis = null;
    refreshFromState(state, null);
    // If engine goes first, trigger it
    await runEngineTurnsIfNeeded();
  } catch (e) {
    showError("Failed to create game: " + e.message);
  }
}

btnNewGame.addEventListener("click", startNewGame);

// ---------------------------------------------------------------------------
// App mode switch (Play <-> Setup/Analysis)
// ---------------------------------------------------------------------------
for (const radio of appModeRadios) {
  radio.addEventListener("change", () => {
    appMode = radio.value;
    if (appMode === "setup") {
      enterSetupMode();
    } else {
      exitSetupMode();
    }
  });
}

function enterSetupMode() {
  playControls.classList.add("hidden");
  setupControls.classList.remove("hidden");

  // Stop any ongoing EvE game
  closeEveWs();

  // Put board in edit mode
  board.setEditMode(true, getSelectedPiece);

  // Render the current setup position
  renderSetupBoard();

  // Update the UI inputs from setupPosition
  syncSetupInputsFromPosition();

  // Clear any previous analysis results
  clearSetupAnalysis();
}

function exitSetupMode() {
  playControls.classList.remove("hidden");
  setupControls.classList.add("hidden");

  // Put board back in play mode
  board.setEditMode(false, null);

  // Re-render the play state
  if (state) {
    refreshFromState(state, analysis);
  }
}

// ---------------------------------------------------------------------------
// Setup mode — piece selector change updates board hover mode
// ---------------------------------------------------------------------------
for (const radio of setupPieceRadios) {
  radio.addEventListener("change", () => {
    if (appMode === "setup") {
      // Re-draw to update hover/preview behavior
      renderSetupBoard();
    }
  });
}

// ---------------------------------------------------------------------------
// Setup mode — walls-remaining inputs
// ---------------------------------------------------------------------------
setupWalls0El.addEventListener("change", () => {
  setupPosition.walls_left[0] = Math.max(0, Math.min(10, parseInt(setupWalls0El.value, 10) || 0));
  setupWalls0El.value = String(setupPosition.walls_left[0]);
});

setupWalls1El.addEventListener("change", () => {
  setupPosition.walls_left[1] = Math.max(0, Math.min(10, parseInt(setupWalls1El.value, 10) || 0));
  setupWalls1El.value = String(setupPosition.walls_left[1]);
});

// ---------------------------------------------------------------------------
// Setup mode — turn radio
// ---------------------------------------------------------------------------
for (const radio of setupTurnRadios) {
  radio.addEventListener("change", () => {
    setupPosition.turn = parseInt(radio.value, 10);
  });
}

// ---------------------------------------------------------------------------
// Setup mode — Reset / Clear buttons
// ---------------------------------------------------------------------------
btnSetupReset.addEventListener("click", () => {
  setupPosition = DEFAULT_SETUP_POSITION();
  syncSetupInputsFromPosition();
  clearSetupAnalysis();
  renderSetupBoard();
});

btnSetupClearWalls.addEventListener("click", () => {
  setupPosition.h_walls = [];
  setupPosition.v_walls = [];
  clearSetupAnalysis();
  renderSetupBoard();
});

// ---------------------------------------------------------------------------
// Setup mode — edit callbacks
// ---------------------------------------------------------------------------

/** Called when user clicks a cell in edit mode. */
function onEditCell(cell, piece) {
  const [col, row] = cell;
  const pawnIdx = piece === "red" ? 0 : 1;
  const otherIdx = 1 - pawnIdx;

  // Don't move onto the other pawn's cell
  const other = setupPosition.pawns[otherIdx];
  if (other[0] === col && other[1] === row) return;

  setupPosition.pawns[pawnIdx] = [col, row];
  clearSetupAnalysis();
  renderSetupBoard();
}

/** Called when user clicks a wall slot in edit mode. */
function onEditWall(slot) {
  const { c, r, orient } = slot;
  const key = `${c},${r}`;
  const list = orient === "H" ? "h_walls" : "v_walls";

  // Toggle: remove if present, add if absent
  const existing = setupPosition[list];
  const idx = existing.findIndex(([wc, wr]) => wc === c && wr === r);
  if (idx !== -1) {
    setupPosition[list] = existing.filter((_, i) => i !== idx);
  } else {
    setupPosition[list] = [...existing, [c, r]];
  }

  clearSetupAnalysis();
  renderSetupBoard();
}

// ---------------------------------------------------------------------------
// Setup mode — render
// ---------------------------------------------------------------------------

/** Build a state-like object for board.render from setupPosition. */
function setupPositionAsRenderState(bestMove) {
  return {
    pawns: setupPosition.pawns,
    h_walls: setupPosition.h_walls,
    v_walls: setupPosition.v_walls,
    walls_left: setupPosition.walls_left,
    turn: setupPosition.turn,
    winner: null,
    // No legal steps/walls — edit mode doesn't need them for highlighting
    legal: { steps: [], walls: [] },
    // move_count not strictly required for rendering, but provide a fallback
    move_count: 0,
    controllers: ["human", "human"],
  };
}

/** Re-render the board in setup mode, optionally with a best-move highlight. */
function renderSetupBoard(bestMove) {
  const renderState = setupPositionAsRenderState();
  board.render(renderState, { mode: "move", bestMove: bestMove || null });
  // Re-enable edit mode after render (render resets internal mode)
  board.setEditMode(true, getSelectedPiece);
}

/** Sync the HTML inputs to match setupPosition values. */
function syncSetupInputsFromPosition() {
  setupWalls0El.value = String(setupPosition.walls_left[0]);
  setupWalls1El.value = String(setupPosition.walls_left[1]);

  for (const radio of setupTurnRadios) {
    radio.checked = parseInt(radio.value, 10) === setupPosition.turn;
  }
}

// ---------------------------------------------------------------------------
// Setup mode — Analyze button
// ---------------------------------------------------------------------------
btnAnalyze.addEventListener("click", async () => {
  clearSetupAnalysis();
  setupLegalityEl.textContent = "Analyzing…";
  setupLegalityEl.style.color = "#555";

  const budget = parseFloat(setupBudgetEl.value);
  const eng0 = setupEngine0El.value;
  const eng1 = setupEngine1El.value;

  const engines = [
    { name: eng0, params: { time_budget: budget } },
    { name: eng1, params: { time_budget: budget } },
  ];

  // Sync walls_left and turn from inputs before sending
  setupPosition.walls_left[0] = Math.max(0, Math.min(10, parseInt(setupWalls0El.value, 10) || 0));
  setupPosition.walls_left[1] = Math.max(0, Math.min(10, parseInt(setupWalls1El.value, 10) || 0));
  setupPosition.turn = parseInt(
    Array.from(setupTurnRadios).find((r) => r.checked)?.value ?? "0", 10
  );

  try {
    const result = await analyzePosition(setupPosition, engines);

    if (!result.valid) {
      setupLegalityEl.textContent = "Invalid position: " + result.reason;
      setupLegalityEl.style.color = "#c0392b";
      return;
    }

    setupLegalityEl.textContent = result.winner !== null
      ? `Game over — winner: Player ${result.winner + 1}`
      : "Position is valid.";
    setupLegalityEl.style.color = result.winner !== null ? "#e67e22" : "#27ae60";

    // Static eval
    setupStaticEvalEl.textContent = typeof result.static_eval === "number"
      ? result.static_eval.toFixed(3)
      : "—";

    // Render engine results columns
    const resultsCols = [
      {
        nameEl: setupEngineName0,
        valEl:  setupEngineVal0,
        bestEl: setupEngineBest0,
        listEl: setupCandidates0,
      },
      {
        nameEl: setupEngineName1,
        valEl:  setupEngineVal1,
        bestEl: setupEngineBest1,
        listEl: setupCandidates1,
      },
    ];

    const results = result.results || [];
    for (let i = 0; i < 2; i++) {
      const col = resultsCols[i];
      const r = results[i];
      if (!r) {
        col.nameEl.textContent = engines[i].name;
        col.valEl.textContent = "—";
        col.bestEl.textContent = "—";
        col.listEl.innerHTML = "";
        continue;
      }
      col.nameEl.textContent = r.engine || engines[i].name;
      col.valEl.textContent = typeof r.value === "number" ? r.value.toFixed(3) : "—";
      col.bestEl.textContent = formatMove(r.best_move);

      col.listEl.innerHTML = "";
      const candidates = r.candidates || [];
      for (const { move, score } of candidates.slice(0, 8)) {
        const li = document.createElement("li");
        li.textContent = `${formatMove(move)} (${typeof score === "number" ? score.toFixed(3) : score})`;
        col.listEl.appendChild(li);
      }
    }

    // Highlight the first engine's best move on the board
    const firstBest = results[0] ? results[0].best_move : null;
    renderSetupBoard(firstBest);

  } catch (e) {
    setupLegalityEl.textContent = "Error: " + e.message;
    setupLegalityEl.style.color = "#c0392b";
  }
});

/** Clear the setup analysis result display. */
function clearSetupAnalysis() {
  setupLegalityEl.textContent = "";
  setupStaticEvalEl.textContent = "—";
  for (const el of [setupEngineVal0, setupEngineVal1]) el.textContent = "—";
  for (const el of [setupEngineBest0, setupEngineBest1]) el.textContent = "—";
  for (const el of [setupCandidates0, setupCandidates1]) el.innerHTML = "";
  setupEngineName0.textContent = "Engine 1";
  setupEngineName1.textContent = "Engine 2";
}

// ---------------------------------------------------------------------------
// Mode toggle (play mode: move / wall)
// ---------------------------------------------------------------------------
for (const radio of modeRadios) {
  radio.addEventListener("change", () => {
    mode = radio.value;
    board.setMode(mode);
  });
}

// ---------------------------------------------------------------------------
// Undo
// ---------------------------------------------------------------------------
btnUndo.addEventListener("click", async () => {
  if (!gameId) return;
  clearError();
  try {
    state = await undo(gameId);
    analysis = null;
    refreshFromState(state, null);
  } catch (e) {
    showError("Undo failed: " + e.message);
  }
});

// ---------------------------------------------------------------------------
// Human input handlers
// ---------------------------------------------------------------------------

/** Called when user clicks a cell in move mode. */
async function onCellClick(col, row) {
  if (!gameId || !state) return;
  if (state.winner !== null) return;
  const turn = state.turn;
  if (state.controllers[turn] !== "human") return;

  clearError();
  try {
    state = await sendMove(gameId, { type: "step", to: [col, row] });
    analysis = null;
    refreshFromState(state, null);
    await runEngineTurnsIfNeeded();
  } catch (e) {
    if (e.status === 400) {
      showError("Illegal move.");
    } else {
      showError("Move failed: " + e.message);
    }
    // Re-render so legal moves are still highlighted
    refreshFromState(state, analysis);
  }
}

/** Called when user clicks a wall slot in wall mode. */
async function onSlotClick({ c, r, orient }) {
  if (!gameId || !state) return;
  if (state.winner !== null) return;
  const turn = state.turn;
  if (state.controllers[turn] !== "human") return;

  clearError();
  try {
    state = await sendMove(gameId, { type: "wall", c, r, orient });
    analysis = null;
    refreshFromState(state, null);
    await runEngineTurnsIfNeeded();
  } catch (e) {
    if (e.status === 400) {
      showError("Illegal wall placement.");
    } else {
      showError("Move failed: " + e.message);
    }
    refreshFromState(state, analysis);
  }
}

// ---------------------------------------------------------------------------
// Engine automation
// ---------------------------------------------------------------------------

/**
 * After a human move (or game start), call engine_move in a loop while the
 * current side to move is an engine and the game is not over.
 * Stops for engine-vs-engine mode (handled by EVE WS instead).
 */
async function runEngineTurnsIfNeeded() {
  if (!state || state.winner !== null) return;

  // Don't auto-play if both sides are engines (EVE mode uses WS)
  const bothEngines = state.controllers.every((c) => c !== "human");
  if (bothEngines) return;

  while (state.winner === null) {
    const turn = state.turn;
    if (state.controllers[turn] === "human") break;

    try {
      const resp = await engineMove(gameId);
      state = resp.state;
      analysis = resp.analysis;
      refreshFromState(state, analysis);
    } catch (e) {
      showError("Engine error: " + e.message);
      break;
    }
  }
}

// ---------------------------------------------------------------------------
// Engine vs Engine — WebSocket stream
// ---------------------------------------------------------------------------

function isEveGame() {
  return state && state.controllers.every((c) => c !== "human");
}

btnEveStart.addEventListener("click", () => {
  if (!gameId || !isEveGame()) return;
  startEvE();
});

btnEveStep.addEventListener("click", () => {
  // Advance exactly one move (also works while paused). At most one request in
  // flight, so it can never run ahead of the display.
  if (eveWs && eveWs.readyState === WebSocket.OPEN && !eveWaiting &&
      !(state && state.winner !== null)) {
    eveWaiting = true;
    eveWs.send(JSON.stringify({ action: "step" }));
  }
});

btnEvePause.addEventListener("click", () => {
  // Pause/resume is purely client-side: stop or restart requesting moves. The
  // server is request-driven, so it's already exactly at the displayed move.
  if (!eveWs) return;
  evePaused = !evePaused;
  btnEvePause.textContent = evePaused ? "Resume" : "Pause";
  if (!evePaused) eveRequestNext();
});

speedSlider.addEventListener("input", () => {
  speedLabel.textContent = `${speedSlider.value}ms`;
});

/**
 * Request the next engine move — one at a time — unless paused, finished, or a
 * request is already in flight. Each response (handled in startEvE) paces by the
 * speed slider and then calls this again, forming the auto-play loop. Because
 * only one request is ever outstanding, the server never runs ahead of the
 * displayed position, so pause/step are exact and there is nothing to buffer.
 */
function eveRequestNext() {
  if (!eveWs || eveWs.readyState !== WebSocket.OPEN) return;
  if (evePaused || eveWaiting) return;
  if (state && state.winner !== null) return;
  eveWaiting = true;
  eveWs.send(JSON.stringify({ action: "step" }));
}

function startEvE() {
  closeEveWs();           // bumps eveGen and closes any prior stream
  const myGen = eveGen;   // this stream owns the current generation
  evePaused = false;
  eveWaiting = false;
  btnEvePause.textContent = "Pause";

  eveWs = openStream(gameId, async (msg) => {
    if (myGen !== eveGen) return;   // superseded (New Game / restart) — ignore
    eveWaiting = false;
    if (msg.error) {
      showError("Stream error: " + msg.error);
      return;
    }
    if (msg.state) {
      state = msg.state;
      analysis = msg.analysis || null;
      refreshFromState(state, analysis);
    }
    if (msg.done || (msg.state && msg.state.winner !== null)) {
      closeEveWs();                 // game over
      return;
    }
    // Pace, then request the next move (re-checking we're still the live stream
    // after the async gap — a New Game during the delay must not resume us).
    const delay = parseInt(speedSlider.value, 10);
    if (delay > 0) await sleep(delay);
    if (myGen !== eveGen) return;
    eveRequestNext();
  });

  eveWs.addEventListener("open", () => {
    if (myGen !== eveGen) return;
    eveRequestNext();               // kick off the first move
  });

  // An ABNORMAL close (server gone, network drop) leaves the loop with no reply.
  // Our own intentional closes bump eveGen first, so they bail this check.
  eveWs.addEventListener("close", () => {
    if (myGen !== eveGen) return;
    eveWaiting = false;
    if (state && state.winner === null) showError("Engine stream closed.");
  });
}

function closeEveWs() {
  eveGen++;            // invalidate any in-flight callbacks/pending timers
  eveWaiting = false;
  evePaused = false;                  // keep the Pause/Resume label honest
  btnEvePause.textContent = "Pause";
  if (eveWs) {
    eveWs.close();
    eveWs = null;
  }
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

/**
 * Refresh all UI from a state dict (and optional analysis).
 */
function refreshFromState(s, an) {
  // Render board
  const bestMove = (an && showBestMove.checked) ? an.best_move : null;
  board.render(s, { mode, bestMove });

  // Turn indicator
  const isOver = s.winner !== null;
  if (isOver) {
    const winner = s.winner + 1;
    turnEl.textContent = `Player ${winner} wins!`;
  } else {
    const p = s.turn + 1;
    const ctrl = s.controllers[s.turn];
    const who = ctrl === "human" ? "Human" : ctrl.charAt(0).toUpperCase() + ctrl.slice(1);
    turnEl.textContent = `Player ${p} to move (${who})`;
  }

  // Walls remaining
  walls0El.textContent = `${s.walls_left[0]} walls`;
  walls1El.textContent = `${s.walls_left[1]} walls`;

  // Move count
  moveCountEl.textContent = String(s.move_count);

  // Analysis panel
  renderAnalysis(an);
}

function renderAnalysis(an) {
  if (!an) {
    analysisValueEl.textContent = "—";
    analysisNodesEl.textContent = "—";
    analysisDepthEl.textContent = "—";
    analysisTimeEl.textContent = "—";
    candidatesList.innerHTML = "";
    return;
  }

  analysisValueEl.textContent = typeof an.value === "number"
    ? an.value.toFixed(3)
    : "—";

  const stats = an.stats || {};
  analysisNodesEl.textContent = stats.nodes != null ? String(stats.nodes) : "—";
  analysisDepthEl.textContent = stats.depth != null ? String(stats.depth) : "—";
  analysisTimeEl.textContent  = stats.time_ms != null ? `${stats.time_ms}ms` : "—";

  candidatesList.innerHTML = "";
  const candidates = an.candidates || [];
  for (const { move, score } of candidates.slice(0, 10)) {
    const li = document.createElement("li");
    li.textContent = `${formatMove(move)}  (${typeof score === "number" ? score.toFixed(3) : score})`;
    candidatesList.appendChild(li);
  }
}

/** Format a move dict as a short human-readable string. */
function formatMove(m) {
  if (!m) return "?";
  if (m.type === "step") {
    const [c, r] = m.to;
    return `${String.fromCharCode(97 + c)}${r + 1}`;
  }
  if (m.type === "wall") {
    return `wall ${m.orient} ${String.fromCharCode(97 + m.c)}${m.r + 1}`;
  }
  return JSON.stringify(m);
}

// ---------------------------------------------------------------------------
// Analysis panel toggle
// ---------------------------------------------------------------------------
btnToggleAnalysis.addEventListener("click", () => {
  const hidden = analysisBody.style.display === "none";
  analysisBody.style.display = hidden ? "" : "none";
  btnToggleAnalysis.textContent = hidden ? "Hide" : "Show";
});

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function showError(msg) {
  errorEl.textContent = msg;
  errorEl.classList.remove("hidden");
  setTimeout(() => clearError(), 4000);
}

function clearError() {
  errorEl.textContent = "";
  errorEl.classList.add("hidden");
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------
init().catch((e) => {
  console.error("Init error:", e);
  showError("Startup failed: " + e.message);
});
