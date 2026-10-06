"use strict";

// Label rasters shared by the workspace mask editor (masks.js) and Paint
// (paint.js). Labels are PNG values: 0 background, 127 ignore, 255 worm.
const maskTools = (() => {
  const normalize = labels => labels?.map(v => v === 128 ? 127 : v);
  const brushColor = value => value === 255 ? "#ff4fa3" : value === 127 ? "#ffd23c" : "#57d68d";
  function render(canvas, width, height, labels, proposal = false) {
    canvas.width = width; canvas.height = height;
    const c = canvas.getContext("2d"), image = c.createImageData(width, height);
    for (let i = 0; i < labels.length; i++) {
      const v = labels[i], j = i * 4;
      image.data[j] = proposal ? 40 : 255; image.data[j + 1] = proposal ? 220 : v === 127 ? 205 : 65; image.data[j + 2] = proposal ? 255 : v === 127 ? 40 : 155;
      image.data[j + 3] = v === 0 ? 0 : 255;
    }
    c.putImageData(image, 0, 0);
    return canvas;
  }
  function encode(width, height, labels) {
    const out = document.createElement("canvas"); out.width = width; out.height = height;
    const c = out.getContext("2d"), image = c.createImageData(width, height);
    for (let i = 0; i < labels.length; i++) { const j = i * 4; image.data[j] = image.data[j + 1] = image.data[j + 2] = labels[i]; image.data[j + 3] = 255; }
    c.putImageData(image, 0, 0); return out.toDataURL("image/png");
  }
  function dab(labels, width, height, x, y, diameter, value) {
    const r = diameter / 2;
    for (let yy = Math.max(0, Math.floor(y - r)); yy < Math.min(height, Math.ceil(y + r)); yy++)
      for (let xx = Math.max(0, Math.floor(x - r)); xx < Math.min(width, Math.ceil(x + r)); xx++)
        if ((xx + .5 - x) ** 2 + (yy + .5 - y) ** 2 <= r * r) labels[yy * width + xx] = value;
  }
  function stroke(labels, width, height, a, b, diameter, value) {
    const n = Math.max(1, Math.ceil(Math.hypot(b.x - a.x, b.y - a.y) / Math.max(1, diameter / 4)));
    for (let i = 0; i <= n; i++) dab(labels, width, height, a.x + (b.x - a.x) * i / n, a.y + (b.y - a.y) * i / n, diameter, value);
  }
  // Combine a proposal into the draft; ignore pixels outside the worm survive every mode but replace.
  function combine(pixels, proposal, mode) {
    return pixels.map((value, i) => {
      const p = proposal[i] === 255, w = value === 255, other = value === 127 ? 127 : 0;
      return mode === "union" ? (p || w ? 255 : other) : mode === "intersect" ? (p && w ? 255 : other) : mode === "subtract" ? (w && !p ? 255 : other) : proposal[i];
    });
  }
  const combineMode = (event, fallback) => event?.shiftKey ? "union" : event?.altKey ? "intersect" : event?.ctrlKey || event?.metaKey ? "subtract" : fallback;
  return {normalize, brushColor, render, encode, dab, stroke, combine, combineMode};
})();
