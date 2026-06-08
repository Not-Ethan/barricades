/**
 * api.js — thin REST + WebSocket client for the Barricades server.
 * All functions return parsed JSON (or a WebSocket). No game logic here.
 */

const BASE = "";  // same-origin; adjust if running cross-origin

async function _json(resp) {
  if (!resp.ok) {
    const text = await resp.text();
    throw Object.assign(new Error(`HTTP ${resp.status}: ${text}`), { status: resp.status, body: text });
  }
  return resp.json();
}

/** GET /agents → { agents: ["greedy", "random", ...] } */
export async function listAgents() {
  return _json(await fetch(`${BASE}/agents`));
}

/**
 * POST /games → state dict.
 * @param {Array<string|{name:string,params:object}>} controllers
 */
export async function newGame(controllers) {
  return _json(await fetch(`${BASE}/games`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ controllers }),
  }));
}

/** GET /games/{id} → state dict */
export async function getGame(id) {
  return _json(await fetch(`${BASE}/games/${id}`));
}

/**
 * POST /games/{id}/move → state dict.
 * @param {string} id
 * @param {{ type:"step", to:[number,number] }|{ type:"wall", c:number, r:number, orient:string }} move
 */
export async function sendMove(id, move) {
  return _json(await fetch(`${BASE}/games/${id}/move`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(move),
  }));
}

/** POST /games/{id}/undo → state dict */
export async function undo(id) {
  return _json(await fetch(`${BASE}/games/${id}/undo`, { method: "POST" }));
}

/** POST /games/{id}/engine_move → { state, analysis } */
export async function engineMove(id) {
  return _json(await fetch(`${BASE}/games/${id}/engine_move`, { method: "POST" }));
}

/**
 * POST /analyze → analysis response.
 * @param {object} position - { pawns, h_walls, v_walls, walls_left, turn }
 * @param {Array<{name:string,params:object}>} engines
 */
export async function analyzePosition(position, engines) {
  return _json(await fetch(`${BASE}/analyze`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ position, engines }),
  }));
}

/**
 * Open a WebSocket to /games/{id}/stream.
 * @param {string} id
 * @param {(msg: object) => void} onMessage
 * @returns {WebSocket}
 */
export function openStream(id, onMessage) {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  const ws = new WebSocket(`${proto}//${location.host}/games/${id}/stream`);
  ws.addEventListener("message", (ev) => {
    try {
      onMessage(JSON.parse(ev.data));
    } catch (e) {
      console.error("WS parse error", e, ev.data);
    }
  });
  ws.addEventListener("error", (ev) => {
    console.error("WS error", ev);
  });
  return ws;
}
