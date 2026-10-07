// Edit mask, the Workspace's inline brush on one frame: Worm / Background,
// a size, Undo, and Network proposal (the segmenter's own mask for the
// frame, replacing the draft). Save writes the frame's mask override
// (POST /api/workspaces/<ws>/mask, the labels as a PNG: 0 background,
// 255 worm, 128 ignore kept as it was) and the page then refits the
// affected stretch with a job that keeps its result itself. The Ignore
// brush, refinements and the other proposals live in Labeling only.

import {api, post, query} from "./api.js";
import {decodeGray, encodeGray} from "./workspace_draw.js";

const WORM = 255, BACKGROUND = 0;
const DRAFT = [255, 79, 163];
const HISTORY = 30;

export class MaskEditor {
  constructor(workspace) {
    this.workspace = workspace;
    this.reset();
  }

  reset() {
    this.frame = null; this.width = 0; this.height = 0;
    this.pixels = null; this.saved = null; this.base = null; this.revision = null;
    this.history = []; this.brush = WORM; this.diameter = this.diameter || 12;
    this.painting = false; this.last = null; this.cursor = null;
    this.overlay = null; this.image = null;
  }

  get active() { return this.pixels !== null; }
  get dirty() { return !!this.pixels && this.pixels.some((v, i) => v !== this.saved[i]); }
  get hasProposal() { return !!this.base; }

  async start(frame) {
    const payload = await api(`/api/workspaces/${encodeURIComponent(this.workspace)}/mask${query({frame})}`);
    const [mask, base] = await Promise.all([decodeGray(payload.mask), decodeGray(payload.base_mask)]);
    this.reset();
    this.frame = frame; this.width = payload.width; this.height = payload.height; this.revision = payload.revision;
    this.pixels = Uint8Array.from(mask.data, (v) => (v >= 192 ? WORM : v > 64 ? 128 : BACKGROUND));
    this.saved = this.pixels.slice();
    this.base = base ? Uint8Array.from(base.data, (v) => (v >= 192 ? WORM : BACKGROUND)) : null;
    this.overlay = document.createElement("canvas");
    this.overlay.width = this.width; this.overlay.height = this.height;
    this.image = this.overlay.getContext("2d").createImageData(this.width, this.height);
    this._paint(0, 0, this.width, this.height);
  }

  stop() { this.reset(); }

  _paint(x0, y0, w, h) {
    const data = this.image.data;
    for (let y = y0; y < y0 + h; y++) {
      for (let x = x0; x < x0 + w; x++) {
        const i = y * this.width + x, o = i * 4, v = this.pixels[i];
        data[o] = DRAFT[0]; data[o + 1] = v === 128 ? 205 : DRAFT[1]; data[o + 2] = v === 128 ? 40 : DRAFT[2];
        data[o + 3] = v === BACKGROUND ? 0 : 120;
      }
    }
    this.overlay.getContext("2d").putImageData(this.image, 0, 0, x0, y0, w, h);
  }

  _dab(cx, cy) {
    const r = this.diameter / 2;
    const x0 = Math.max(0, Math.floor(cx - r)), x1 = Math.min(this.width, Math.ceil(cx + r));
    const y0 = Math.max(0, Math.floor(cy - r)), y1 = Math.min(this.height, Math.ceil(cy + r));
    for (let y = y0; y < y1; y++) {
      for (let x = x0; x < x1; x++) if ((x + 0.5 - cx) ** 2 + (y + 0.5 - cy) ** 2 <= r * r) this.pixels[y * this.width + x] = this.brush;
    }
    if (x1 > x0 && y1 > y0) this._paint(x0, y0, x1 - x0, y1 - y0);
  }

  _stroke(a, b) {
    const steps = Math.max(1, Math.ceil(Math.hypot(b.x - a.x, b.y - a.y) / Math.max(1, this.diameter / 4)));
    for (let i = 0; i <= steps; i++) this._dab(a.x + (b.x - a.x) * i / steps, a.y + (b.y - a.y) * i / steps);
  }

  _remember() {
    this.history.push(this.pixels.slice());
    if (this.history.length > HISTORY) this.history.shift();
  }

  // FrameCanvas pointer events (image coordinates); returns whether the view must redraw.
  pointer({type, x, y}) {
    if (!this.active) return false;
    this.cursor = {x, y};
    if (type === "down") {
      this._remember();
      this.painting = true;
      this.last = {x, y};
      this._stroke(this.last, this.last);
    } else if (type === "move" && this.painting) {
      this._stroke(this.last, {x, y});
      this.last = {x, y};
    } else if (type === "up") {
      this.painting = false;
      this.last = null;
    }
    return true;
  }

  setBrush(value) { this.brush = value === "background" ? BACKGROUND : WORM; }
  get brushName() { return this.brush === WORM ? "worm" : "background"; }
  resize(delta) { this.diameter = Math.max(2, Math.min(120, this.diameter + delta)); }

  undo() {
    if (!this.history.length) return false;
    this.pixels = this.history.pop();
    this._paint(0, 0, this.width, this.height);
    return true;
  }

  useProposal() {
    if (!this.base) return false;
    this._remember();
    // The network's mask replaces the draft; pixels a person excluded stay excluded.
    this.pixels = Uint8Array.from(this.base, (v, i) => (this.pixels[i] === 128 && v === BACKGROUND ? 128 : v));
    this._paint(0, 0, this.width, this.height);
    return true;
  }

  async save() {
    const mask = encodeGray(this.width, this.height, this.pixels);
    return post(`/api/workspaces/${encodeURIComponent(this.workspace)}/mask`, {frame: this.frame, mask, revision: this.revision});
  }

  draw(g, view) {
    if (!this.active) return;
    g.drawImage(this.overlay, 0, 0);
    if (this.cursor) {
      g.beginPath();
      g.arc(this.cursor.x, this.cursor.y, this.diameter / 2, 0, Math.PI * 2);
      g.strokeStyle = this.brush === WORM ? "#ff4fa3" : "#ffffff";
      g.lineWidth = 1.5 / view.scale;
      g.stroke();
    }
  }
}
