// The Workspace timeline: the issue track above the curvature kymograph
// (head at the top, tail at the bottom, one column per frame; blue and
// orange are the two bending directions), with the current frame, the
// selection, a fix's preview range and Relabel's keyframes drawn over it.
// Developers can add one extra per-frame series below.
//
// Everything is in workspace rows; the page converts to frames. Click seeks,
// drag selects a range (for a fix without an issue), the wheel zooms about
// the cursor (Shift+wheel pans) and double-click shows the whole recording.
// In keyframe mode a click adds or removes a keyframe instead.

const ISSUE_H = 10, GAP = 3, KYMO_H = 96, AXIS_H = 16, SERIES_H = 34;
const ISSUE_COLORS = {unreviewed: "#e0b04d", reviewed: "#3f7a5a", fixed: "#6cb4ff"};
const DRAG_PX = 4;

export class Timeline {
  constructor(container, {onSeek, onSelect, onKeyframe}) {
    this.container = container;
    this.canvas = document.createElement("canvas");
    this.canvas.className = "ws-timeline-canvas";
    container.append(this.canvas);
    this.onSeek = onSeek; this.onSelect = onSelect; this.onKeyframe = onKeyframe;
    this.rows = 0; this.first = 0; this.step = 1;
    this.window = [0, 1];
    this.row = 0;
    this.kymograph = null; this.issues = []; this.selection = null; this.preview = null; this.keyframes = null;
    this.series = null; this.hover = null; this.drag = null; this.selected = null;
    this._frame = 0;
    this._bind();
    new ResizeObserver(() => this.draw()).observe(container);
  }

  get height() { return ISSUE_H + GAP + KYMO_H + AXIS_H + (this.series ? SERIES_H : 0); }

  setRange(rows, first, step) {
    const changed = rows !== this.rows;
    this.rows = rows; this.first = first; this.step = step;
    if (changed) this.window = [0, Math.max(1, rows)];
    this.draw();
  }

  frameOf(row) { return this.first + row * this.step; }

  set(key, value) { this[key] = value; this.draw(); }

  setRow(row) {
    this.row = row;
    // Keep the cursor in view when zoomed.
    const [a, b] = this.window, span = b - a;
    if (row < a || row >= b) this.window = this._clamp(row - span / 2, span);
    this.draw();
  }

  showRows(first, last) {
    const span = Math.max(last - first + 1, Math.min(this.rows, 40)) * 1.6;
    if (last - first + 1 > (this.window[1] - this.window[0])) this.window = this._clamp((first + last + 1) / 2 - span / 2, span);
    else if (first < this.window[0] || last >= this.window[1]) this.window = this._clamp((first + last + 1) / 2 - (this.window[1] - this.window[0]) / 2, this.window[1] - this.window[0]);
    this.draw();
  }

  _clamp(start, span) {
    span = Math.min(Math.max(span, Math.min(this.rows, 20)), this.rows);
    start = Math.max(0, Math.min(this.rows - span, start));
    return [start, start + span];
  }

  _x(row, width) { const [a, b] = this.window; return (row - a) / (b - a) * width; }
  _row(x, width) { const [a, b] = this.window; return Math.max(0, Math.min(this.rows - 1, Math.floor(a + x / width * (b - a)))); }

  draw() {
    if (this._frame) return;
    this._frame = requestAnimationFrame(() => { this._frame = 0; this._draw(); });
  }

  _draw() {
    const width = this.container.clientWidth, height = this.height, dpr = window.devicePixelRatio || 1;
    this.canvas.style.height = `${height}px`;
    this.canvas.width = Math.max(1, Math.round(width * dpr)); this.canvas.height = Math.round(height * dpr);
    const g = this.canvas.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, width, height);
    if (!this.rows) return;
    const style = getComputedStyle(document.documentElement);
    const muted = style.getPropertyValue("--muted").trim() || "#9aa6b0", border = style.getPropertyValue("--border").trim() || "#2a343c";
    const [a, b] = this.window, perRow = width / (b - a);
    const kymoTop = ISSUE_H + GAP;
    // Issue track.
    g.fillStyle = "#11171c";
    g.fillRect(0, 0, width, ISSUE_H);
    for (const issue of this.issues) {
      const [r0, r1] = issue.rows;
      if (r1 < a || r0 >= b) continue;
      const x0 = this._x(r0, width), x1 = Math.max(x0 + 2, this._x(r1 + 1, width));
      g.fillStyle = ISSUE_COLORS[issue.state] || muted;
      g.fillRect(x0, 1, x1 - x0, ISSUE_H - 2);
      if (issue.id === this.selected) { g.strokeStyle = "#ffffff"; g.lineWidth = 1.5; g.strokeRect(x0 + 0.5, 0.75, x1 - x0 - 1, ISSUE_H - 1.5); }
    }
    // Kymograph.
    g.fillStyle = "#0b0f12";
    g.fillRect(0, kymoTop, width, KYMO_H);
    if (this.kymograph) {
      g.imageSmoothingEnabled = perRow < 1;
      g.drawImage(this.kymograph, a, 0, b - a, this.kymograph.height, 0, kymoTop, width, KYMO_H);
    }
    g.fillStyle = muted; g.font = "10px system-ui, sans-serif"; g.textBaseline = "top";
    g.fillText("head", 4, kymoTop + 2);
    g.textBaseline = "bottom";
    g.fillText("tail", 4, kymoTop + KYMO_H - 2);
    // Extra series (developer).
    let bottom = kymoTop + KYMO_H;
    if (this.series) {
      const top = bottom + 2, h = SERIES_H - 4, {values, label} = this.series;
      const finite = values.slice(Math.floor(a), Math.ceil(b)).filter((v) => Number.isFinite(v));
      const lo = Math.min(...finite), hi = Math.max(...finite), span = hi - lo || 1;
      g.strokeStyle = "#6cb4ff"; g.lineWidth = 1; g.beginPath();
      let pen = false;
      const stride = Math.max(1, Math.floor((b - a) / width / 2));
      for (let r = Math.floor(a); r < Math.min(this.rows, Math.ceil(b)); r += stride) {
        const v = values[r];
        if (!Number.isFinite(v)) { pen = false; continue; }
        const x = this._x(r + 0.5, width), y = top + h - (v - lo) / span * h;
        if (pen) g.lineTo(x, y); else g.moveTo(x, y);
        pen = true;
      }
      g.stroke();
      g.fillStyle = muted; g.textBaseline = "top";
      g.fillText(`${label}  ${Number.isFinite(lo) ? `${+lo.toFixed(3)}–${+hi.toFixed(3)}` : ""}`, 4, top);
      bottom += SERIES_H;
    }
    // Axis.
    g.strokeStyle = border; g.beginPath(); g.moveTo(0, bottom + 0.5); g.lineTo(width, bottom + 0.5); g.stroke();
    const frames = (b - a) * this.step, target = frames / Math.max(1, width / 90);
    const tick = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000].find((t) => t >= target) || 100000;
    g.fillStyle = muted; g.textBaseline = "top"; g.textAlign = "center";
    for (let f = Math.ceil(this.frameOf(a) / tick) * tick; f <= this.frameOf(b); f += tick) {
      const x = this._x((f - this.first) / this.step, width);
      g.fillRect(x, bottom, 1, 3);
      if (x > 14 && x < width - 14) g.fillText(String(f), x, bottom + 3);
    }
    g.textAlign = "start";
    const full = this.height - AXIS_H;
    // Preview range, selection, keyframes, cursor.
    const band = (range, fill, stroke) => {
      if (!range) return;
      const x0 = this._x(range[0], width), x1 = Math.max(x0 + 2, this._x(range[1] + 1, width));
      g.fillStyle = fill; g.fillRect(x0, 0, x1 - x0, full);
      g.strokeStyle = stroke; g.lineWidth = 1; g.strokeRect(x0 + 0.5, 0.5, x1 - x0 - 1, full - 1);
    };
    band(this.preview, "rgba(255, 122, 217, 0.10)", "rgba(255, 122, 217, 0.7)");
    const selection = this.drag?.moved ? [Math.min(this.drag.start, this.drag.end), Math.max(this.drag.start, this.drag.end)] : this.selection;
    band(selection, "rgba(108, 180, 255, 0.14)", "rgba(108, 180, 255, 0.85)");
    if (this.keyframes) {
      g.fillStyle = "#ff7ad9";
      for (const r of this.keyframes) {
        const x = this._x(r + 0.5, width);
        g.fillRect(x - 1, 0, 2, full);
        g.beginPath(); g.moveTo(x - 4, 0); g.lineTo(x + 4, 0); g.lineTo(x, 6); g.fill();
      }
    }
    // The current frame: its column outlined when frames are wide, else a line.
    const x0 = this._x(this.row, width);
    if (perRow > 4) {
      g.fillStyle = "rgba(255,255,255,0.16)"; g.fillRect(x0, 0, perRow, full);
      g.strokeStyle = "#ffffff"; g.lineWidth = 1; g.strokeRect(Math.round(x0) + 0.5, 0.5, Math.max(1, Math.round(perRow) - 1), full - 1);
    } else {
      g.fillStyle = "#ffffff"; g.fillRect(Math.round(x0 + perRow / 2) - 1, 0, 2, full);
    }
    if (this.hover !== null && !this.drag) {
      const hx = this._x(this.hover + 0.5, width), text = String(this.frameOf(this.hover));
      g.fillStyle = "rgba(255,255,255,0.35)"; g.fillRect(Math.round(hx), ISSUE_H, 1, full - ISSUE_H);
      g.font = "11px system-ui, sans-serif";
      const w = g.measureText(text).width + 8, left = Math.min(width - w, Math.max(0, hx + 6));
      g.fillStyle = "rgba(15,20,24,0.9)"; g.fillRect(left, kymoTop + 2, w, 16);
      g.fillStyle = "#e6ebef"; g.textBaseline = "top"; g.fillText(text, left + 4, kymoTop + 4);
    }
  }

  _bind() {
    const canvas = this.canvas, width = () => this.container.clientWidth;
    canvas.addEventListener("pointerdown", (event) => {
      if (event.button !== 0 || !this.rows) return;
      canvas.setPointerCapture(event.pointerId);
      const row = this._row(event.offsetX, width());
      this.drag = {x: event.offsetX, start: row, end: row, moved: false};
      if (!this.keyframes) this.onSeek(row);
    });
    canvas.addEventListener("pointermove", (event) => {
      if (!this.rows) return;
      this.hover = this._row(event.offsetX, width());
      if (this.drag) {
        this.drag.end = this.hover;
        if (Math.abs(event.offsetX - this.drag.x) > DRAG_PX && !this.keyframes) this.drag.moved = true;
      }
      this.draw();
    });
    canvas.addEventListener("pointerleave", () => { this.hover = null; this.draw(); });
    const end = () => {
      const drag = this.drag;
      this.drag = null;
      if (!drag) return;
      if (drag.moved) this.onSelect({first: Math.min(drag.start, drag.end), last: Math.max(drag.start, drag.end)});
      else if (this.keyframes) this.onKeyframe(drag.end);
      this.draw();
    };
    canvas.addEventListener("pointerup", end);
    canvas.addEventListener("pointercancel", () => { this.drag = null; this.draw(); });
    canvas.addEventListener("dblclick", () => { this.window = [0, this.rows]; this.draw(); });
    canvas.addEventListener("wheel", (event) => {
      if (!this.rows) return;
      event.preventDefault();
      const [a, b] = this.window, span = b - a;
      if (event.shiftKey || Math.abs(event.deltaX) > Math.abs(event.deltaY)) {
        const delta = (event.shiftKey ? event.deltaY : event.deltaX) / width() * span;
        this.window = this._clamp(a + delta, span);
      } else {
        const anchor = a + event.offsetX / width() * span, next = span * Math.exp(event.deltaY * 0.002);
        this.window = this._clamp(anchor - (anchor - a) * next / span, next);
      }
      this.draw();
    }, {passive: false});
  }
}
