// One video frame on a canvas with overlay layers, pan and zoom; shared by
// the Workspace and Labeling pages.
//
//   const view = new FrameCanvas(container);
//   view.setImage(bitmap);                     // ImageBitmap, <img>, <canvas> or ImageData
//   view.setLayer("mask", (g, v) => {...});    // drawn in image coordinates, in insertion order
//   view.setLayerVisible("mask", false);
//   view.onPointer = (event) => {...};         // {type: "down"|"move"|"up"|"rightclick", x, y, event}
//   view.fit();
//
// Left drag goes to onPointer (painting, tracing, clicking); right, middle or
// Shift drag pans, and a right click that did not pan goes to onPointer as
// "rightclick"; the wheel zooms about the cursor. Only the page fits the
// frame to the view (its Fit button and the 0 key call fit()).
// Layer draw functions get the 2D context already transformed to image
// pixels and the view, so a line of width 1 / v.scale is one screen pixel.

export class FrameCanvas {
  constructor(container) {
    this.container = container;
    container.classList.add("frame-canvas");
    this.canvas = document.createElement("canvas");
    this.canvas.tabIndex = 0;
    container.append(this.canvas);
    this.image = null;
    this.width = 0;
    this.height = 0;
    this.layers = new Map();
    this.view = {scale: 1, tx: 0, ty: 0};
    this.onPointer = null;
    this.onViewChange = null;
    this._pan = null;
    this._fitted = false;
    this._frame = 0;
    this._bind();
    new ResizeObserver(() => this._resize()).observe(container);
  }

  setImage(image) {
    const width = image.width, height = image.height;
    if (image instanceof ImageData) {
      const canvas = document.createElement("canvas");
      canvas.width = width; canvas.height = height;
      canvas.getContext("2d").putImageData(image, 0, 0);
      image = canvas;
    }
    const sizeChanged = width !== this.width || height !== this.height;
    this.image = image; this.width = width; this.height = height;
    if (sizeChanged || !this._fitted) this.fit();
    else this.redraw();
  }

  setLayer(id, draw, visible = true) {
    const layer = this.layers.get(id);
    this.layers.set(id, {draw, visible: layer ? layer.visible : visible});
    this.redraw();
  }

  removeLayer(id) { this.layers.delete(id); this.redraw(); }

  setLayerVisible(id, visible) {
    const layer = this.layers.get(id);
    if (layer) { layer.visible = visible; this.redraw(); }
  }

  isLayerVisible(id) { return !!this.layers.get(id)?.visible; }

  fit() {
    const rect = this.canvas.getBoundingClientRect();
    if (!this.width || !rect.width || !rect.height) return;
    const scale = Math.min(rect.width / this.width, rect.height / this.height) * 0.98;
    this.view = {scale, tx: (rect.width - this.width * scale) / 2, ty: (rect.height - this.height * scale) / 2};
    this._fitted = true;
    this.redraw();
    this.onViewChange?.(this.view);
  }

  // Screen (client) coordinates → image pixel coordinates.
  toImage(clientX, clientY) {
    const rect = this.canvas.getBoundingClientRect();
    return {x: (clientX - rect.left - this.view.tx) / this.view.scale, y: (clientY - rect.top - this.view.ty) / this.view.scale};
  }

  redraw() {
    if (this._frame) return;
    this._frame = requestAnimationFrame(() => { this._frame = 0; this._draw(); });
  }

  _draw() {
    const g = this.canvas.getContext("2d"), dpr = window.devicePixelRatio || 1;
    g.setTransform(1, 0, 0, 1, 0, 0);
    g.clearRect(0, 0, this.canvas.width, this.canvas.height);
    if (!this.image) return;
    const {scale, tx, ty} = this.view;
    g.setTransform(dpr * scale, 0, 0, dpr * scale, dpr * tx, dpr * ty);
    g.imageSmoothingEnabled = scale < 1;
    g.drawImage(this.image, 0, 0);
    for (const layer of this.layers.values()) {
      if (!layer.visible) continue;
      g.save();
      layer.draw(g, this.view);
      g.restore();
    }
  }

  _resize() {
    const rect = this.container.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
    this.canvas.width = Math.max(1, Math.round(rect.width * dpr));
    this.canvas.height = Math.max(1, Math.round(rect.height * dpr));
    if (!this._fitted) this.fit();
    this.redraw();
  }

  _bind() {
    const canvas = this.canvas;
    canvas.addEventListener("contextmenu", (event) => event.preventDefault());
    canvas.addEventListener("pointerdown", (event) => {
      canvas.focus();
      canvas.setPointerCapture(event.pointerId);
      if (event.button === 1 || event.button === 2 || event.shiftKey) {
        this._pan = {x: event.clientX, y: event.clientY, tx: this.view.tx, ty: this.view.ty, button: event.button, shift: event.shiftKey};
        return;
      }
      if (event.button === 0) this._emit("down", event);
    });
    canvas.addEventListener("pointermove", (event) => {
      if (this._pan) {
        this.view.tx = this._pan.tx + event.clientX - this._pan.x;
        this.view.ty = this._pan.ty + event.clientY - this._pan.y;
        this.redraw();
        return;
      }
      this._emit("move", event);
    });
    const end = (event) => {
      if (this._pan) {
        const pan = this._pan, click = Math.hypot(event.clientX - pan.x, event.clientY - pan.y) < 4;
        this._pan = null;
        if (click && pan.button === 2 && !pan.shift && event.type === "pointerup") this._emit("rightclick", event);
        else this.onViewChange?.(this.view);
        return;
      }
      this._emit("up", event);
    };
    canvas.addEventListener("pointerup", end);
    canvas.addEventListener("pointercancel", end);
    canvas.addEventListener("wheel", (event) => {
      event.preventDefault();
      const rect = canvas.getBoundingClientRect();
      const cx = event.clientX - rect.left, cy = event.clientY - rect.top;
      const factor = Math.exp(-event.deltaY * 0.0015);
      const scale = Math.min(64, Math.max(0.05, this.view.scale * factor));
      const ratio = scale / this.view.scale;
      this.view = {scale, tx: cx - (cx - this.view.tx) * ratio, ty: cy - (cy - this.view.ty) * ratio};
      this.redraw();
      this.onViewChange?.(this.view);
    }, {passive: false});
  }

  _emit(type, event) {
    if (!this.onPointer || !this.image) return;
    const {x, y} = this.toImage(event.clientX, event.clientY);
    this.onPointer({type, x, y, event});
  }
}

// Draw a single-channel mask (Uint8Array, 0/1 or 0/255, width × height) as a
// colored overlay; returns a canvas to reuse while the mask is unchanged.
export function maskCanvas(mask, width, height, [r, g, b], alpha = 0.45) {
  const canvas = document.createElement("canvas");
  canvas.width = width; canvas.height = height;
  const image = new ImageData(width, height), data = image.data, a = Math.round(alpha * 255);
  for (let i = 0; i < mask.length; i++) {
    if (!mask[i]) continue;
    const o = i * 4;
    data[o] = r; data[o + 1] = g; data[o + 2] = b; data[o + 3] = a;
  }
  canvas.getContext("2d").putImageData(image, 0, 0);
  return canvas;
}

// A head-first polyline (array of [x, y]) with head and tail markers.
export function drawMidline(g, view, points, {color = "#57d68d", head = "#57d68d", tail = "#ff5d5d", width = 2} = {}) {
  if (!points?.length) return;
  const px = 1 / view.scale;
  g.lineWidth = width * px;
  g.strokeStyle = color;
  g.lineJoin = g.lineCap = "round";
  g.beginPath();
  points.forEach(([x, y], i) => (i ? g.lineTo(x, y) : g.moveTo(x, y)));
  g.stroke();
  const marker = ([x, y], fill, square) => {
    const r = 5 * px;
    g.fillStyle = fill;
    g.strokeStyle = "#000";
    g.lineWidth = px;
    g.beginPath();
    if (square) g.rect(x - r, y - r, 2 * r, 2 * r); else g.arc(x, y, r, 0, Math.PI * 2);
    g.fill(); g.stroke();
  };
  marker(points[0], head, true);
  marker(points[points.length - 1], tail, false);
}
