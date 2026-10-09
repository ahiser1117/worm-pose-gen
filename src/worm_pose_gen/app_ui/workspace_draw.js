// Drawing helpers of the Workspace page: decoding the server's PNG layers,
// the colour maps of the kymograph, the A-P field and the body model's raw
// outputs, and the body outline
// drawn from a pose (centerline + width profile) rather than a raster, so it
// stays sharp at any zoom and is there even while frames play.

export function loadImage(url) {
  return new Promise((resolve, reject) => {
    if (!url) { resolve(null); return; }
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("could not decode an image from the server"));
    image.src = url;
  });
}

// A gray PNG data URL as {width, height, data: Uint8Array} (its red channel).
export async function decodeGray(url) {
  const image = await loadImage(url);
  if (!image) return null;
  const canvas = document.createElement("canvas");
  canvas.width = image.width; canvas.height = image.height;
  const g = canvas.getContext("2d", {willReadFrequently: true});
  g.drawImage(image, 0, 0);
  const rgba = g.getImageData(0, 0, image.width, image.height).data;
  const data = new Uint8Array(image.width * image.height);
  for (let i = 0; i < data.length; i++) data[i] = rgba[i * 4];
  return {width: image.width, height: image.height, data};
}

// Encode labels (0 background, 255 worm, 128 ignore) as the PNG data URL the mask endpoints take.
export function encodeGray(width, height, values) {
  const canvas = document.createElement("canvas");
  canvas.width = width; canvas.height = height;
  const g = canvas.getContext("2d"), image = g.createImageData(width, height);
  for (let i = 0; i < values.length; i++) {
    const o = i * 4;
    image.data[o] = image.data[o + 1] = image.data[o + 2] = values[i];
    image.data[o + 3] = 255;
  }
  g.putImageData(image, 0, 0);
  return canvas.toDataURL("image/png");
}

const mix = (a, b, t) => a.map((v, i) => Math.round(v + (b[i] - v) * t));

// Diverging map for signed curvature: blue (bends one way), dark (straight), orange (the other).
function divergingLut() {
  const negative = [74, 163, 255], zero = [30, 37, 43], positive = [255, 140, 66];
  const lut = new Uint8ClampedArray(256 * 4);
  for (let v = 1; v < 256; v++) {
    const t = (v - 128) / 127;
    const color = t < 0 ? mix(zero, negative, Math.sqrt(-t)) : mix(zero, positive, Math.sqrt(t));
    lut.set([...color, 255], v * 4);
  }
  return lut; // value 0 (no pose) stays transparent
}

// Sequential map for the A-P field, head (0) to tail (1).
function apLut() {
  const stops = [[87, 214, 141], [108, 180, 255], [184, 120, 255], [255, 93, 93]];
  const lut = new Uint8ClampedArray(256 * 4);
  for (let v = 1; v < 256; v++) {
    const t = (v - 1) / 254 * (stops.length - 1), k = Math.min(stops.length - 2, Math.floor(t));
    lut.set([...mix(stops[k], stops[k + 1], t - k), 170], v * 4);
  }
  return lut;
}

// A raw model output through the A-P colours: every value drawn, 0 (head) included.
function rawApLut() {
  const lut = apLut();
  lut.set(lut.subarray(4, 8), 0);
  return lut;
}

// A raw probability channel as one colour whose opacity is the value.
export function probabilityLut([r, g, b]) {
  const lut = new Uint8ClampedArray(256 * 4);
  for (let v = 0; v < 256; v++) lut.set([r, g, b, Math.round(v * 0.9)], v * 4);
  return lut;
}

export const KYMOGRAPH_LUT = divergingLut();
export const AP_LUT = apLut();
export const RAW_AP_LUT = rawApLut();

// A canvas of a decoded gray image through a lookup table of 256 RGBA entries.
export function colorize({width, height, data}, lut) {
  const canvas = document.createElement("canvas");
  canvas.width = width; canvas.height = height;
  const g = canvas.getContext("2d"), image = g.createImageData(width, height);
  for (let i = 0; i < data.length; i++) {
    const o = data[i] * 4;
    image.data.set(lut.subarray(o, o + 4), i * 4);
  }
  g.putImageData(image, 0, 0);
  return canvas;
}

// The body's outline from its centerline and full width at each point.
export function drawOutline(g, view, centerline, widths, color = "#ffd166") {
  if (!centerline?.length || !widths?.length || centerline.length !== widths.length) return;
  const left = [], right = [], n = centerline.length;
  for (let i = 0; i < n; i++) {
    const a = centerline[Math.max(0, i - 1)], b = centerline[Math.min(n - 1, i + 1)];
    let dx = b[0] - a[0], dy = b[1] - a[1];
    const length = Math.hypot(dx, dy) || 1;
    dx /= length; dy /= length;
    const r = widths[i] / 2;
    left.push([centerline[i][0] - dy * r, centerline[i][1] + dx * r]);
    right.push([centerline[i][0] + dy * r, centerline[i][1] - dx * r]);
  }
  g.beginPath();
  [...left, ...right.reverse()].forEach(([x, y], i) => (i ? g.lineTo(x, y) : g.moveTo(x, y)));
  g.closePath();
  g.lineWidth = 1.5 / view.scale;
  g.strokeStyle = color;
  g.lineJoin = "round";
  g.stroke();
}

// A polyline in image coordinates (a before/after pose), optionally dashed.
export function drawCurve(g, view, points, {color, width = 2, dash = null}) {
  if (!points?.length) return;
  g.lineWidth = width / view.scale;
  g.strokeStyle = color;
  g.lineJoin = g.lineCap = "round";
  if (dash) g.setLineDash(dash.map((d) => d / view.scale));
  g.beginPath();
  points.forEach(([x, y], i) => (i ? g.lineTo(x, y) : g.moveTo(x, y)));
  g.stroke();
  g.setLineDash([]);
  const [hx, hy] = points[0], r = 4 / view.scale;
  g.fillStyle = color;
  g.fillRect(hx - r, hy - r, 2 * r, 2 * r);
}
