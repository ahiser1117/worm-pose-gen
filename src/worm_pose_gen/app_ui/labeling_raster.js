// Rasters of the Labeling page: masks as label arrays, grey images, and the
// overlays drawn from them.
//
// A mask is a Uint8Array of store labels, one per pixel: 0 background,
// 1 worm, 255 excluded from the loss (only migrated labels have those; no
// brush makes them). The server sends and takes masks as PNG data URLs with
// 0 background, 255 worm and 128 excluded (app/images.py, mask_to_png_values).

export const WORM = 1, BACKGROUND = 0, IGNORE = 255;

export function loadImage(url) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("could not decode an image from the server"));
    image.src = url;
  });
}

// The grey values (red channel) of a PNG data URL as a Uint8Array.
export async function grayOf(url) {
  const image = await loadImage(url);
  const canvas = document.createElement("canvas");
  canvas.width = image.width; canvas.height = image.height;
  const g = canvas.getContext("2d");
  g.drawImage(image, 0, 0);
  const rgba = g.getImageData(0, 0, image.width, image.height).data, out = new Uint8Array(image.width * image.height);
  for (let i = 0; i < out.length; i++) out[i] = rgba[i * 4];
  return out;
}

export async function decodeMask(url) {
  const values = await grayOf(url);
  for (let i = 0; i < values.length; i++) values[i] = values[i] >= 192 ? WORM : values[i] >= 64 ? IGNORE : BACKGROUND;
  return values;
}

export function encodeMask(mask, width, height) {
  const canvas = document.createElement("canvas");
  canvas.width = width; canvas.height = height;
  const g = canvas.getContext("2d"), image = g.createImageData(width, height), data = image.data;
  for (let i = 0; i < mask.length; i++) {
    const v = mask[i] === WORM ? 255 : mask[i] === IGNORE ? 128 : 0, o = i * 4;
    data[o] = data[o + 1] = data[o + 2] = v; data[o + 3] = 255;
  }
  g.putImageData(image, 0, 0);
  return canvas.toDataURL("image/png");
}

// A grey array as a canvas the frame view can draw.
export function grayCanvas(values, width, height) {
  const canvas = document.createElement("canvas");
  canvas.width = width; canvas.height = height;
  const g = canvas.getContext("2d"), image = g.createImageData(width, height), data = image.data;
  for (let i = 0; i < values.length; i++) { const o = i * 4; data[o] = data[o + 1] = data[o + 2] = values[i]; data[o + 3] = 255; }
  g.putImageData(image, 0, 0);
  return canvas;
}

// The mask as a coloured overlay: worm pink, excluded pixels yellow; a proposal preview cyan.
export function maskOverlay(mask, width, height, preview = false) {
  const canvas = document.createElement("canvas");
  canvas.width = width; canvas.height = height;
  const g = canvas.getContext("2d"), image = g.createImageData(width, height), data = image.data;
  for (let i = 0; i < mask.length; i++) {
    const v = mask[i];
    if (!v) continue;
    const o = i * 4;
    if (preview) { data[o] = 40; data[o + 1] = 210; data[o + 2] = 255; }
    else if (v === IGNORE) { data[o] = 255; data[o + 1] = 205; data[o + 2] = 40; }
    else { data[o] = 255; data[o + 1] = 70; data[o + 2] = 160; }
    data[o + 3] = 255;
  }
  g.putImageData(image, 0, 0);
  return canvas;
}

// A round brush dab of `value` with diameter `size` at (x, y); excluded pixels stay as they are only under no brush.
function dab(mask, width, height, x, y, size, value) {
  const r = size / 2;
  const y0 = Math.max(0, Math.floor(y - r)), y1 = Math.min(height, Math.ceil(y + r));
  const x0 = Math.max(0, Math.floor(x - r)), x1 = Math.min(width, Math.ceil(x + r));
  for (let yy = y0; yy < y1; yy++) {
    for (let xx = x0; xx < x1; xx++) {
      if ((xx + 0.5 - x) ** 2 + (yy + 0.5 - y) ** 2 <= r * r) mask[yy * width + xx] = value;
    }
  }
}

export function stroke(mask, width, height, a, b, size, value) {
  const steps = Math.max(1, Math.ceil(Math.hypot(b.x - a.x, b.y - a.y) / Math.max(1, size / 4)));
  for (let i = 0; i <= steps; i++) dab(mask, width, height, a.x + (b.x - a.x) * i / steps, a.y + (b.y - a.y) * i / steps, size, value);
}

// A proposal from a 0..255 map: worm where map >= threshold (0..1). The
// network's map is its worm probability; Threshold's is the frame's darkness
// (the worm is dark on a bright background), so one slider serves both.
export function thresholded(map, threshold, invert = false) {
  const cut = Math.round(threshold * 255), out = new Uint8Array(map.length);
  for (let i = 0; i < map.length; i++) out[i] = (invert ? 255 - map[i] : map[i]) >= cut ? WORM : BACKGROUND;
  return out;
}

// A proposal replaces the mask, except that pixels excluded from the loss stay excluded where the proposal sees no worm.
export function applyProposal(mask, proposal) {
  const out = new Uint8Array(mask.length);
  for (let i = 0; i < mask.length; i++) out[i] = proposal[i] === WORM ? WORM : mask[i] === IGNORE ? IGNORE : BACKGROUND;
  return out;
}

export function sameMask(a, b) {
  if (!a || !b || a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

// The A-P field (0 undefined, else 1 + 254 * ap) coloured head (orange) to tail (blue).
export function apOverlay(values, width, height) {
  const canvas = document.createElement("canvas");
  canvas.width = width; canvas.height = height;
  const g = canvas.getContext("2d"), image = g.createImageData(width, height), data = image.data;
  const head = [255, 150, 40], tail = [60, 120, 255];
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    if (!v) continue;
    const t = (v - 1) / 254, o = i * 4;
    data[o] = head[0] + (tail[0] - head[0]) * t; data[o + 1] = head[1] + (tail[1] - head[1]) * t; data[o + 2] = head[2] + (tail[2] - head[2]) * t;
    data[o + 3] = 255;
  }
  g.putImageData(image, 0, 0);
  return canvas;
}

// The A-P values of the same body traced the other way round.
export function reversedAp(values) {
  const out = new Uint8Array(values.length);
  for (let i = 0; i < values.length; i++) out[i] = values[i] ? 256 - values[i] : 0;
  return out;
}

// frame[t + lag] - frame[t - lag] around mid-grey, scaled so the 99th percentile of the change reaches the ends.
export function differenceCanvas(later, earlier, width, height) {
  const n = later.length, sample = [];
  for (let i = 0; i < n; i += 97) sample.push(Math.abs(later[i] - earlier[i]));
  sample.sort((a, b) => a - b);
  const scale = 127 / Math.max(4, sample[Math.floor(sample.length * 0.99)] || 0);
  const out = new Uint8Array(n);
  for (let i = 0; i < n; i++) out[i] = Math.max(0, Math.min(255, 128 + scale * (later[i] - earlier[i])));
  return grayCanvas(out, width, height);
}
