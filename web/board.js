/**
 * board.js — Canvas rendering + hit-testing for the Quoridor board.
 * No network or game-flow logic here; only drawing and pointer math.
 *
 * Coordinate convention (matches the server):
 *   cells (col, row): col 0..8 (a..i), row 0..8 (rank 1..9).
 *   Row 0 is at the BOTTOM of the board (rank 1), row 8 at the top (rank 9).
 *   Player 0 starts at (4,0) — bottom centre — and marches toward row 8.
 *   Player 1 starts at (4,8) — top centre — and marches toward row 0.
 *
 * Wall anchors (c, r): 0..7. A horizontal wall at (c,r) fills the gap
 * ABOVE cells (c,r) and (c+1,r), i.e. between rows r and r+1.
 * A vertical wall at (c,r) fills the gap to the RIGHT of cells (c,r) and
 * (c,r+1), i.e. between cols c and c+1.
 */

const COLS = 9;
const ROWS = 9;
const CELL = 64;       // cell size in pixels
const GAP = 10;        // wall-slot gap width in pixels
const MARGIN = 32;     // space for rank/file labels

// Colours
const C_CELL       = "#f7f4ee";
const C_CELL_HOVER = "#e8f5e9";
const C_LEGAL      = "#a5d6a7";
const C_LEGAL_OUT  = "#2e7d32";
const C_WALL_H     = "#5d4037";
const C_WALL_V     = "#5d4037";
const C_WALL_PRE_OK  = "#43a047";
const C_WALL_PRE_BAD = "#e53935";
const C_P0         = "#c62828";  // player 0 — red
const C_P1         = "#1565c0";  // player 1 — blue
const C_BEST_MOVE  = "#ffb300";  // amber highlight for best-move target
const C_LABEL      = "#888";

/** Convert (col, row) to top-left pixel of the cell on the canvas. */
function cellToPixel(col, row) {
  const x = MARGIN + col * (CELL + GAP);
  const y = MARGIN + (ROWS - 1 - row) * (CELL + GAP);
  return { x, y };
}

/** Canvas total size. */
function canvasSize() {
  const w = MARGIN + COLS * CELL + (COLS - 1) * GAP + MARGIN;
  const h = MARGIN + ROWS * CELL + (ROWS - 1) * GAP + MARGIN;
  return { w, h };
}

export class Board {
  /**
   * @param {HTMLCanvasElement} canvas
   */
  constructor(canvas) {
    this._canvas = canvas;
    const { w, h } = canvasSize();
    canvas.width = w;
    canvas.height = h;
    this._ctx = canvas.getContext("2d");

    // Callback hooks set by main.js
    this.onStep = null;   // (col, row) => void
    this.onWall = null;   // ({c, r, orient}) => void
    this.onEditCell = null;  // (cell, piece) => void  — edit mode, pawn click
    this.onEditWall = null;  // (slot) => void         — edit mode, wall click

    // Internal state
    this._state = null;
    this._mode = "move";       // "move" | "wall"
    this._editMode = false;    // true when in Setup/Edit mode
    this._editGetPiece = null; // () => "red"|"blue"|"wall"
    this._legalSteps = [];
    this._previewWall = null;  // {c, r, orient, ok}
    this._hoverCell = null;    // [col, row] | null
    this._bestMove = null;     // move dict from analysis, or null

    canvas.addEventListener("mousemove",  (e) => this._onMouseMove(e));
    canvas.addEventListener("mouseleave", ()  => this._onMouseLeave());
    canvas.addEventListener("click",      (e) => this._onClick(e));
  }

  // ------------------------------------------------------------------
  // Public API
  // ------------------------------------------------------------------

  /**
   * Render the board from a state dict.
   * @param {object} state      - server state dict
   * @param {object} [opts]
   * @param {"move"|"wall"} [opts.mode]
   * @param {object|null} [opts.bestMove]   - analysis.best_move dict or null
   */
  render(state, { mode = "move", bestMove = null } = {}) {
    this._state = state;
    this._mode = mode;
    this._legalSteps = state.legal ? state.legal.steps : [];
    this._bestMove = bestMove;
    this._previewWall = null;
    this._draw();
  }

  /**
   * Draw (or clear) a wall preview in wall mode.
   * @param {{c:number,r:number,orient:string}|null} slot
   * @param {boolean} isLegal
   */
  previewWall(slot, isLegal) {
    this._previewWall = slot ? { ...slot, ok: isLegal } : null;
    this._draw();
  }

  setMode(mode) {
    this._mode = mode;
    this._previewWall = null;
    this._hoverCell = null;
    this._draw();
  }

  /**
   * Enable or disable edit mode.
   * @param {boolean} enabled
   * @param {(() => "red"|"blue"|"wall")|null} getSelectedPiece
   */
  setEditMode(enabled, getSelectedPiece = null) {
    this._editMode = enabled;
    this._editGetPiece = getSelectedPiece;
    this._previewWall = null;
    this._hoverCell = null;
    this._draw();
  }

  // ------------------------------------------------------------------
  // Hit-testing
  // ------------------------------------------------------------------

  /**
   * Convert a canvas pixel to a board (col, row) or null.
   * Returns a cell only if the pointer is inside a cell square (not in a gap).
   */
  cellAt(px, py) {
    const ox = px - MARGIN;
    const oy = py - MARGIN;
    if (ox < 0 || oy < 0) return null;

    const stride = CELL + GAP;
    const col = Math.floor(ox / stride);
    const row = ROWS - 1 - Math.floor(oy / stride);
    if (col < 0 || col >= COLS || row < 0 || row >= ROWS) return null;

    // Check pointer is inside the cell (not in the gap)
    const localX = ox - col * stride;
    const localY = oy - (ROWS - 1 - row) * stride;
    if (localX > CELL || localY > CELL) return null;
    return [col, row];
  }

  /**
   * In wall mode: snap pointer to the nearest wall anchor slot.
   * Returns {c, r, orient} or null if out of range.
   */
  wallSlotAt(px, py) {
    const ox = px - MARGIN;
    const oy = py - MARGIN;
    const stride = CELL + GAP;

    // Determine which cell column/row the pointer is near (including gaps)
    const colF = ox / stride;
    const rowF = (ROWS - 1) - oy / stride;

    const col = Math.floor(colF);
    const row = Math.floor(rowF);

    // Fraction within the stride
    const fracX = colF - col;  // 0..1 across col then gap
    const fracY = rowF - row;

    // Is pointer in a vertical gap (between cols)? fracX > CELL/stride
    const inVGap = fracX > CELL / stride && col >= 0 && col < COLS - 1;
    // Is pointer in a horizontal gap (between rows)? fracY > CELL/stride but we measure top-to-bottom
    // rowF increases upward, fracY is fractional part going upward, so gap is at high fracY (near next row boundary)
    const inHGap = fracY > CELL / stride && row >= 0 && row < ROWS - 1;

    if (!inVGap && !inHGap) {
      // Pointer is inside a cell. Only snap to a gap when the pointer is
      // reasonably close to one; a click in the dead-center of a cell places
      // no wall (predictable — you must move toward a gap line).
      const distToVGap = Math.abs(fracX - CELL / stride);
      const distToHGap = Math.abs(fracY - CELL / stride);
      const NEAR = 0.25; // must be within ~1/4 stride of a gap line to snap
      if (Math.min(distToVGap, distToHGap) > NEAR) {
        return null;
      }
      if (distToVGap < distToHGap && col >= 0 && col < COLS - 1 && row >= 0 && row <= ROWS - 2) {
        return this._snapWall(col, row, "V");
      }
      if (row >= 0 && row < ROWS - 1 && col >= 0 && col <= COLS - 2) {
        return this._snapWall(col, row, "H");
      }
      return null;
    }

    if (inVGap && inHGap) {
      // Corner: pick orient by which gap centre is closer
      const dv = Math.abs(fracX - (CELL + GAP / 2) / stride);
      const dh = Math.abs(fracY - (CELL + GAP / 2) / stride);
      const orient = dv < dh ? "V" : "H";
      if (orient === "V" && col >= 0 && col < COLS - 1 && row >= 0 && row <= ROWS - 2) {
        return this._snapWall(col, row, "V");
      }
      if (orient === "H" && row >= 0 && row < ROWS - 1 && col >= 0 && col <= COLS - 2) {
        return this._snapWall(col, row, "H");
      }
      return null;
    }

    if (inVGap) {
      if (col >= 0 && col < COLS - 1 && row >= 0 && row <= ROWS - 2) {
        return this._snapWall(col, row, "V");
      }
      return null;
    }

    // inHGap
    if (row >= 0 && row < ROWS - 1 && col >= 0 && col <= COLS - 2) {
      return this._snapWall(col, row, "H");
    }
    return null;
  }

  /** Clamp wall anchor to valid 0..7 range. */
  _snapWall(col, row, orient) {
    const c = Math.max(0, Math.min(7, col));
    const r = Math.max(0, Math.min(7, row));
    return { c, r, orient };
  }

  // ------------------------------------------------------------------
  // Drawing
  // ------------------------------------------------------------------

  _draw() {
    const ctx = this._ctx;
    const { w, h } = canvasSize();
    ctx.clearRect(0, 0, w, h);

    this._drawCells();
    this._drawLabels();
    if (this._state) {
      this._drawWalls(this._state.h_walls, "H");
      this._drawWalls(this._state.v_walls, "V");
      this._drawPawns(this._state.pawns);
    }
    if (this._previewWall) {
      this._drawWallPreview(this._previewWall);
    }
  }

  _drawCells() {
    const ctx = this._ctx;
    const legalSet = new Set(this._legalSteps.map(([c, r]) => `${c},${r}`));

    for (let col = 0; col < COLS; col++) {
      for (let row = 0; row < ROWS; row++) {
        const { x, y } = cellToPixel(col, row);
        const key = `${col},${row}`;
        const isLegal = this._mode === "move" && legalSet.has(key);
        const isHover = this._hoverCell &&
          this._hoverCell[0] === col && this._hoverCell[1] === row;

        // Best-move highlight
        const isBest = this._bestMove &&
          this._bestMove.type === "step" &&
          this._bestMove.to[0] === col && this._bestMove.to[1] === row;

        ctx.fillStyle = isBest ? C_BEST_MOVE
          : isLegal ? C_LEGAL
          : isHover ? C_CELL_HOVER
          : C_CELL;
        ctx.fillRect(x, y, CELL, CELL);

        if (isLegal) {
          ctx.strokeStyle = C_LEGAL_OUT;
          ctx.lineWidth = 2;
          ctx.strokeRect(x + 1, y + 1, CELL - 2, CELL - 2);
        }
      }
    }
  }

  _drawLabels() {
    const ctx = this._ctx;
    ctx.fillStyle = C_LABEL;
    ctx.font = "12px system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";

    for (let col = 0; col < COLS; col++) {
      const { x } = cellToPixel(col, 0);
      const label = String.fromCharCode(97 + col); // a..i
      ctx.fillText(label, x + CELL / 2, MARGIN / 2);
    }

    ctx.textAlign = "right";
    for (let row = 0; row < ROWS; row++) {
      const { y } = cellToPixel(0, row);
      ctx.fillText(String(row + 1), MARGIN - 6, y + CELL / 2);
    }
  }

  _drawWalls(walls, orient) {
    const ctx = this._ctx;
    ctx.fillStyle = orient === "H" ? C_WALL_H : C_WALL_V;

    for (const [c, r] of walls) {
      this._fillWallSlot(ctx, c, r, orient, ctx.fillStyle);
    }
  }

  /**
   * Fill a wall slot (anchor c,r, orient H or V) with a given colour.
   * H wall: horizontal bar in the gap ABOVE row r, spanning cols c..c+1.
   * V wall: vertical bar in the gap to the RIGHT of col c, spanning rows r..r+1.
   */
  _fillWallSlot(ctx, c, r, orient, color) {
    ctx.fillStyle = color;
    if (orient === "H") {
      // Gap above row r = between cells at row r and row r+1
      const x0 = cellToPixel(c, 0).x;
      const y0 = cellToPixel(0, r).y;
      // The gap starts at y0 + CELL (below the top of row r's cell … wait:
      // row increases upward, so "above row r" means lower y on canvas.
      // cellToPixel(0, r).y is the TOP of row r's cell.
      // The gap between row r and row r+1 is at y = cellToPixel(0,r).y - GAP.
      const gapY = cellToPixel(0, r).y - GAP;
      const spanW = 2 * CELL + GAP;   // spans cols c and c+1
      ctx.fillRect(x0, gapY, spanW, GAP);
    } else {
      // V wall: gap to the right of col c, spanning rows r and r+1
      const { x: cellX } = cellToPixel(c, 0);
      const gapX = cellX + CELL;
      const { y: y1 } = cellToPixel(0, r + 1);  // top of upper row
      const { y: y0 } = cellToPixel(0, r);       // top of lower row
      const spanH = 2 * CELL + GAP;
      ctx.fillRect(gapX, y1, GAP, spanH);
    }
  }

  _drawWallPreview({ c, r, orient, ok }) {
    const ctx = this._ctx;
    const color = ok ? C_WALL_PRE_OK : C_WALL_PRE_BAD;
    ctx.globalAlpha = 0.65;
    this._fillWallSlot(ctx, c, r, orient, color);
    ctx.globalAlpha = 1.0;
  }

  _drawPawns(pawns) {
    const ctx = this._ctx;
    const colors = [C_P0, C_P1];

    pawns.forEach(([col, row], i) => {
      const { x, y } = cellToPixel(col, row);
      const cx = x + CELL / 2;
      const cy = y + CELL / 2;
      const r = CELL * 0.3;

      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, 2 * Math.PI);
      ctx.fillStyle = colors[i];
      ctx.fill();
      ctx.strokeStyle = "#fff";
      ctx.lineWidth = 2;
      ctx.stroke();

      // Player number label
      ctx.fillStyle = "#fff";
      ctx.font = `bold ${Math.round(r)}px system-ui, sans-serif`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(String(i + 1), cx, cy);
    });
  }

  // ------------------------------------------------------------------
  // Event handlers
  // ------------------------------------------------------------------

  _pointerPos(e) {
    const rect = this._canvas.getBoundingClientRect();
    // Map displayed coordinates back to the canvas's internal resolution so
    // hit-testing stays correct when CSS scales the canvas down to fit.
    const sx = this._canvas.width / rect.width;
    const sy = this._canvas.height / rect.height;
    return {
      px: (e.clientX - rect.left) * sx,
      py: (e.clientY - rect.top) * sy,
    };
  }

  _onMouseMove(e) {
    if (!this._state) return;
    const { px, py } = this._pointerPos(e);

    if (this._editMode) {
      const piece = this._editGetPiece ? this._editGetPiece() : "red";
      if (piece === "wall") {
        const slot = this.wallSlotAt(px, py);
        if (slot) {
          // In edit mode, show preview as "ok" (green) — validity determined on click
          this.previewWall(slot, true);
        } else {
          this.previewWall(null, false);
        }
      } else {
        const cell = this.cellAt(px, py);
        this._hoverCell = cell;
        this._draw();
      }
      return;
    }

    if (this._mode === "wall") {
      const slot = this.wallSlotAt(px, py);
      if (slot) {
        const isLegal = this._isWallLegal(slot);
        this.previewWall(slot, isLegal);
      } else {
        this.previewWall(null, false);
      }
    } else {
      const cell = this.cellAt(px, py);
      this._hoverCell = cell;
      this._draw();
    }
  }

  _onMouseLeave() {
    this._hoverCell = null;
    this._previewWall = null;
    this._draw();
  }

  _onClick(e) {
    if (!this._state) return;
    const { px, py } = this._pointerPos(e);

    if (this._editMode) {
      // Edit mode: dispatch to edit callbacks based on selected piece type
      const piece = this._editGetPiece ? this._editGetPiece() : "red";
      if (piece === "wall") {
        const slot = this.wallSlotAt(px, py);
        if (slot && this.onEditWall) this.onEditWall(slot);
      } else {
        const cell = this.cellAt(px, py);
        if (cell && this.onEditCell) this.onEditCell(cell, piece);
      }
      return;
    }

    if (this._mode === "wall") {
      const slot = this.wallSlotAt(px, py);
      if (slot && this.onWall) this.onWall(slot);
    } else {
      const cell = this.cellAt(px, py);
      if (cell && this.onStep) this.onStep(cell[0], cell[1]);
    }
  }

  _isWallLegal(slot) {
    if (!this._state || !this._state.legal) return false;
    return this._state.legal.walls.some(
      (w) => w.c === slot.c && w.r === slot.r && w.orient === slot.orient
    );
  }
}
