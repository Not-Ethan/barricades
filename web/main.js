/**
 * main.js — App wiring for all four game modes:
 *   1. Human vs Human
 *   2. Human vs Engine
 *   3. Engine vs Human (engine is player 0)
 *   4. Engine vs Engine (WebSocket stream)
 *
 * Owns app state; delegates rendering to Board, networking to api.js.
 */

import { listAgents, newGame, getGame, sendMove, undo, engineMove, openStream } from "./api.js";
import { Board } from "./board.js";

// ---------------------------------------------------------------------------
// DOM refs
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
// App state
// ---------------------------------------------------------------------------
let gameId   = null;
let state    = null;   // latest server state dict
let mode     = "move"; // "move" | "wall"
let analysis = null;   // latest analysis dict or null
let eveWs    = null;   // active WebSocket for engine-vs-engine, or null
let evePaused = false;

const board = new Board(canvas);
board.onStep = onCellClick;
board.onWall = onSlotClick;

// ---------------------------------------------------------------------------
// Initialisation
// ---------------------------------------------------------------------------
async function init() {
  // Populate agent dropdowns
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
// Mode toggle
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
  if (eveWs && eveWs.readyState === WebSocket.OPEN) {
    eveWs.send(JSON.stringify({ action: "step" }));
  }
});

btnEvePause.addEventListener("click", () => {
  if (eveWs && eveWs.readyState === WebSocket.OPEN) {
    evePaused = !evePaused;
    if (!evePaused) {
      eveWs.send(JSON.stringify({ action: "play" }));
    } else {
      eveWs.send(JSON.stringify({ action: "pause" }));
    }
    btnEvePause.textContent = evePaused ? "Resume" : "Pause";
  }
});

speedSlider.addEventListener("input", () => {
  speedLabel.textContent = `${speedSlider.value}ms`;
});

function startEvE() {
  closeEveWs();
  evePaused = false;
  btnEvePause.textContent = "Pause";

  let lastMsgTime = 0;

  eveWs = openStream(gameId, async (msg) => {
    if (msg.error) {
      showError("Stream error: " + msg.error);
      return;
    }
    if (msg.state) {
      // Pace rendering by speed slider delay
      const delay = parseInt(speedSlider.value, 10);
      const now = Date.now();
      const elapsed = now - lastMsgTime;
      if (elapsed < delay) {
        await sleep(delay - elapsed);
      }
      lastMsgTime = Date.now();

      state = msg.state;
      analysis = msg.analysis || null;
      refreshFromState(state, analysis);
    }
    if (msg.done || (msg.state && msg.state.winner !== null)) {
      // Game over
      closeEveWs();
    }
  });

  // Send "play" to start auto-stepping
  eveWs.addEventListener("open", () => {
    eveWs.send(JSON.stringify({ action: "play" }));
  });
}

function closeEveWs() {
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
