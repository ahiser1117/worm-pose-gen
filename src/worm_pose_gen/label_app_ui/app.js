"use strict";

// Label values match the store: 0 background, 1 worm, 255 ignore.
const WORM = 1, BACKGROUND = 0, IGNORE = 255;
const $ = (selector) => document.querySelector(selector);

const state = {
  info: null,
  frame: null,           // payload from /api/frame
  width: 0, height: 0,
  image: null, imageRaw: null,        // HTMLImageElement
  showRaw: false,
  label: null,           // Uint8Array label values
  proposals: {},         // name -> Uint8Array
  probability: null,     // Uint8Array 0..255 network probability
  undo: [],
  brush: { mode: "worm", size: 12 },
  overlay: 0.45,
  view: { scale: 1, tx: 0, ty: 0 },
  painting: false, panning: false, last: null,
  dirty: false,
  history: [], historyPos: -1,
};

const canvas = $("#canvas");
const ctx = canvas.getContext("2d");
const overlayCanvas = document.createElement("canvas");
const overlayCtx = overlayCanvas.getContext("2d");
let overlayData = null;

function setStatus(text, kind) {
  const node = $("#status");
  node.textContent = text;
  node.className = "status" + (kind ? " " + kind : "");
}

function setLoading(on) { $("#loading").hidden = !on; }

async function api(path, options) {
  const response = await fetch(path, options);
  const payload = await response.json();
  if (!response.ok || payload.error) throw new Error(payload.error || response.statusText);
  return payload;
}

function post(path, body) {
  return api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
}

function loadImage(url) {
  return new Promise((resolve, reject) => {
    if (!url) return resolve(null);
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("image decode failed"));
    image.src = url;
  });
}

// Decode a mask PNG data URL into label values through an offscreen canvas.
async function decodeMask(url) {
  const image = await loadImage(url);
  if (!image) return null;
  const off = document.createElement("canvas");
  off.width = image.width; off.height = image.height;
  const c = off.getContext("2d");
  c.drawImage(image, 0, 0);
  const data = c.getImageData(0, 0, off.width, off.height).data;
  const out = new Uint8Array(off.width * off.height);
  for (let i = 0; i < out.length; i++) {
    const v = data[i * 4];
    out[i] = v >= 192 ? WORM : (v > 64 ? IGNORE : BACKGROUND);
  }
  return out;
}

async function decodeGray(url) {
  const image = await loadImage(url);
  if (!image) return null;
  const off = document.createElement("canvas");
  off.width = image.width; off.height = image.height;
  const c = off.getContext("2d");
  c.drawImage(image, 0, 0);
  const data = c.getImageData(0, 0, off.width, off.height).data;
  const out = new Uint8Array(off.width * off.height);
  for (let i = 0; i < out.length; i++) out[i] = data[i * 4];
  return out;
}

function encodeMask(label) {
  const off = document.createElement("canvas");
  off.width = state.width; off.height = state.height;
  const c = off.getContext("2d");
  const data = c.createImageData(state.width, state.height);
  for (let i = 0; i < label.length; i++) {
    const v = label[i] === WORM ? 255 : (label[i] === IGNORE ? 128 : 0);
    data.data[i * 4] = v; data.data[i * 4 + 1] = v; data.data[i * 4 + 2] = v; data.data[i * 4 + 3] = 255;
  }
  c.putImageData(data, 0, 0);
  return off.toDataURL("image/png");
}

// ---------- rendering ----------

function renderOverlay() {
  if (!state.label) return;
  const data = overlayData.data;
  const alpha = Math.round(state.overlay * 255);
  for (let i = 0; i < state.label.length; i++) {
    const v = state.label[i];
    const o = i * 4;
    if (v === WORM) { data[o] = 255; data[o + 1] = 79; data[o + 2] = 163; data[o + 3] = alpha; }
    else if (v === IGNORE) { data[o] = 255; data[o + 1] = 210; data[o + 2] = 60; data[o + 3] = alpha; }
    else { data[o + 3] = 0; }
  }
  overlayCtx.putImageData(overlayData, 0, 0);
}

function draw() {
  const rect = canvas.getBoundingClientRect();
  if (canvas.width !== Math.round(rect.width) || canvas.height !== Math.round(rect.height)) {
    canvas.width = Math.round(rect.width); canvas.height = Math.round(rect.height);
  }
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.fillStyle = "#05080a";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  if (!state.image) return;
  ctx.setTransform(state.view.scale, 0, 0, state.view.scale, state.view.tx, state.view.ty);
  ctx.imageSmoothingEnabled = state.view.scale < 1;
  ctx.drawImage(state.showRaw && state.imageRaw ? state.imageRaw : state.image, 0, 0);
  if (state.overlay > 0) ctx.drawImage(overlayCanvas, 0, 0);
  if (state.last && !state.panning) {
    ctx.beginPath();
    ctx.arc(state.last.x, state.last.y, state.brush.size / 2, 0, Math.PI * 2);
    ctx.strokeStyle = state.brush.mode === "worm" ? "#ff4fa3" : (state.brush.mode === "ignore" ? "#ffd23c" : "#57d68d");
    ctx.lineWidth = 1 / state.view.scale;
    ctx.stroke();
  }
}

function fitView() {
  const rect = canvas.getBoundingClientRect();
  const scale = Math.min(rect.width / state.width, rect.height / state.height) * 0.98;
  state.view = { scale, tx: (rect.width - state.width * scale) / 2, ty: (rect.height - state.height * scale) / 2 };
  draw();
}

function toImage(event) {
  const rect = canvas.getBoundingClientRect();
  const x = (event.clientX - rect.left - state.view.tx) / state.view.scale;
  const y = (event.clientY - rect.top - state.view.ty) / state.view.scale;
  return { x, y };
}

// ---------- editing ----------

function pushUndo() {
  state.undo.push(state.label.slice());
  if (state.undo.length > 40) state.undo.shift();
}

function undo() {
  if (!state.undo.length) return;
  state.label = state.undo.pop();
  state.dirty = true;
  renderOverlay(); draw();
}

function stamp(x, y) {
  const value = state.brush.mode === "worm" ? WORM : (state.brush.mode === "ignore" ? IGNORE : BACKGROUND);
  const r = state.brush.size / 2;
  const x0 = Math.max(0, Math.floor(x - r)), x1 = Math.min(state.width - 1, Math.ceil(x + r));
  const y0 = Math.max(0, Math.floor(y - r)), y1 = Math.min(state.height - 1, Math.ceil(y + r));
  const r2 = r * r;
  for (let yy = y0; yy <= y1; yy++) {
    for (let xx = x0; xx <= x1; xx++) {
      const dx = xx - x, dy = yy - y;
      if (dx * dx + dy * dy <= r2) state.label[yy * state.width + xx] = value;
    }
  }
}

function paintLine(a, b) {
  const distance = Math.hypot(b.x - a.x, b.y - a.y);
  const steps = Math.max(1, Math.ceil(distance / Math.max(1, state.brush.size / 4)));
  for (let i = 0; i <= steps; i++) {
    const t = i / steps;
    stamp(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t);
  }
}

function combine(source, mode) {
  if (!source) { setStatus("that proposal is not available for this frame", "error"); return; }
  pushUndo();
  const label = state.label;
  for (let i = 0; i < label.length; i++) {
    const s = source[i] === WORM, l = label[i] === WORM;
    let v;
    if (mode === "union") v = (s || l) ? WORM : (label[i] === IGNORE ? IGNORE : BACKGROUND);
    else if (mode === "intersect") v = (s && l) ? WORM : (label[i] === IGNORE ? IGNORE : BACKGROUND);
    else if (mode === "subtract") v = (l && !s) ? WORM : (label[i] === IGNORE ? IGNORE : BACKGROUND);
    else v = source[i];
    label[i] = v;
  }
  state.dirty = true;
  renderOverlay(); draw();
}

function thresholdedNetwork() {
  if (!state.probability) return null;
  const cutoff = Number($("#threshold").value) * 255;
  const out = new Uint8Array(state.probability.length);
  for (let i = 0; i < out.length; i++) out[i] = state.probability[i] >= cutoff ? WORM : BACKGROUND;
  return out;
}

function proposalByName(name) {
  if (name === "network") return thresholdedNetwork();
  return state.proposals[name] || null;
}

function modeFromEvent(event) {
  if (event.shiftKey) return "union";
  if (event.altKey) return "intersect";
  if (event.ctrlKey || event.metaKey) return "subtract";
  return "replace";
}

// ---------- frame loading ----------

async function showFrame(recording, index, { record = true } = {}) {
  setLoading(true);
  try {
    const payload = await api(`/api/frame?recording=${encodeURIComponent(recording)}&index=${index}`);
    state.frame = payload;
    state.width = payload.width; state.height = payload.height;
    state.image = await loadImage(payload.image);
    state.imageRaw = await loadImage(payload.image_raw);
    state.proposals = {
      classical: await decodeMask(payload.proposals.classical),
      raw_threshold: await decodeMask(payload.proposals.raw_threshold),
      existing: await decodeMask(payload.existing_mask),
    };
    state.probability = await decodeGray(payload.proposals.network_probability);
    overlayCanvas.width = state.width; overlayCanvas.height = state.height;
    overlayData = overlayCtx.createImageData(state.width, state.height);
    const initial = state.proposals.existing || thresholdedNetwork() || state.proposals.classical;
    state.label = initial ? initial.slice() : new Uint8Array(state.width * state.height);
    state.undo = [];
    state.dirty = false;
    if (record) {
      state.history = state.history.slice(0, state.historyPos + 1);
      state.history.push({ recording, index });
      state.historyPos = state.history.length - 1;
    }
    $("#recording").value = recording;
    $("#frame-index").value = index;
    const existing = payload.existing_record;
    $("#frame-info").textContent =
      `${recording} frame ${index} of ${payload.frame_count}` +
      (existing ? `\nsaved: split ${existing.split}, rev ${existing.revision}, from ${existing.label_source}` : "\nunlabeled") +
      (payload.network_uncertain_fraction != null ? `\nnetwork uncertain fraction ${(payload.network_uncertain_fraction * 100).toFixed(2)}%` : "\nno network loaded") +
      `\nproposals in ${payload.proposal_seconds.toFixed(2)} s`;
    $("#proposal-info").textContent = existing
      ? "started from the saved label"
      : (state.probability ? "started from the network proposal" : "started from the classical proposal");
    $("#refine-info").textContent = "";
    renderOverlay(); fitView();
    setStatus("ready", "ok");
  } catch (error) {
    setStatus(error.message, "error");
  } finally {
    setLoading(false);
  }
}

async function nextFrame() {
  const mode = $("#next-mode").value;
  const payload = await post("/api/next", {
    mode,
    recording: mode === "sequential" ? $("#recording").value : null,
    current_index: state.frame ? state.frame.frame_index : null,
    stride: Number($("#stride").value),
    candidates: Number($("#candidates").value),
  });
  await showFrame(payload.recording, payload.frame_index);
}

async function saveLabel() {
  if (!state.label) return false;
  setLoading(true);
  try {
    const payload = await post("/api/save", {
      recording: state.frame.recording,
      frame_index: state.frame.frame_index,
      mask: encodeMask(state.label),
      label_source: state.probability ? "network+manual" : "classical+manual",
    });
    state.dirty = false;
    updateDataset(payload.counts);
    setStatus(`saved ${payload.record.sample_id} → ${payload.record.split} (rev ${payload.record.revision})`, "ok");
    return true;
  } catch (error) {
    setStatus(error.message, "error");
    return false;
  } finally {
    setLoading(false);
  }
}

async function refine(method) {
  if (!state.label) return;
  setLoading(true);
  try {
    const payload = await post("/api/refine", {
      recording: state.frame.recording, frame_index: state.frame.frame_index,
      mask: encodeMask(state.label), method,
    });
    const refined = await decodeMask(payload.mask);
    pushUndo();
    state.label = refined;
    state.dirty = true;
    renderOverlay(); draw();
    $("#refine-info").textContent = `${method}: ` + Object.entries(payload.info).map(([k, v]) => `${k}=${typeof v === "number" ? v.toFixed(3) : v}`).join(", ");
    setStatus(`refined with ${method}`, "ok");
  } catch (error) {
    setStatus(error.message, "error");
  } finally {
    setLoading(false);
  }
}

function updateDataset(counts) {
  const total = counts.train + counts.val + counts.test;
  $("#dataset-info").textContent = `${total} labeled: train ${counts.train}, val ${counts.val}, test ${counts.test}\n${state.info.dataset_root}\nnetwork: ${state.info.checkpoint || "none"} on ${state.info.device}`;
}

async function stepHistory(delta) {
  const position = state.historyPos + delta;
  if (position < 0 || position >= state.history.length) return;
  state.historyPos = position;
  const entry = state.history[position];
  await showFrame(entry.recording, entry.index, { record: false });
}

// ---------- events ----------

canvas.addEventListener("contextmenu", (event) => event.preventDefault());
canvas.addEventListener("pointerdown", (event) => {
  if (!state.label) return;
  canvas.setPointerCapture(event.pointerId);
  const point = toImage(event);
  if (event.button === 1 || event.button === 2 || event.shiftKey) {
    state.panning = true; state.last = { x: event.clientX, y: event.clientY };
    return;
  }
  state.painting = true;
  pushUndo();
  stamp(point.x, point.y);
  state.last = point;
  state.dirty = true;
  renderOverlay(); draw();
});
canvas.addEventListener("pointermove", (event) => {
  if (state.panning) {
    state.view.tx += event.clientX - state.last.x;
    state.view.ty += event.clientY - state.last.y;
    state.last = { x: event.clientX, y: event.clientY };
    draw();
    return;
  }
  const point = toImage(event);
  if (state.painting) {
    paintLine(state.last, point);
    renderOverlay();
  }
  state.last = point;
  draw();
});
const endStroke = () => { state.painting = false; state.panning = false; draw(); };
canvas.addEventListener("pointerup", endStroke);
canvas.addEventListener("pointercancel", endStroke);
canvas.addEventListener("pointerleave", () => { state.last = null; draw(); });
canvas.addEventListener("wheel", (event) => {
  event.preventDefault();
  if (!state.image) return;
  const factor = Math.exp(-event.deltaY * 0.0015);
  const rect = canvas.getBoundingClientRect();
  const cx = event.clientX - rect.left, cy = event.clientY - rect.top;
  const scale = Math.min(20, Math.max(0.1, state.view.scale * factor));
  state.view.tx = cx - (cx - state.view.tx) * (scale / state.view.scale);
  state.view.ty = cy - (cy - state.view.ty) * (scale / state.view.scale);
  state.view.scale = scale;
  draw();
}, { passive: false });
window.addEventListener("resize", draw);

function setBrush(mode) {
  state.brush.mode = mode;
  document.querySelectorAll("[data-brush]").forEach((b) => b.classList.toggle("active", b.dataset.brush === mode));
  draw();
}
function setBrushSize(size) {
  state.brush.size = Math.min(80, Math.max(1, size));
  $("#brush-size").value = state.brush.size;
  $("#brush-size-value").textContent = `${state.brush.size} px`;
  draw();
}

document.querySelectorAll("[data-brush]").forEach((b) => b.addEventListener("click", () => setBrush(b.dataset.brush)));
document.querySelectorAll("[data-proposal]").forEach((b) => b.addEventListener("click", (event) => combine(proposalByName(b.dataset.proposal), modeFromEvent(event))));
document.querySelectorAll("[data-refine]").forEach((b) => b.addEventListener("click", () => refine(b.dataset.refine)));
$("#brush-size").addEventListener("input", (event) => setBrushSize(Number(event.target.value)));
$("#threshold").addEventListener("input", (event) => { $("#threshold-value").textContent = Number(event.target.value).toFixed(2); });
$("#undo").addEventListener("click", undo);
$("#clear").addEventListener("click", () => { pushUndo(); state.label.fill(BACKGROUND); state.dirty = true; renderOverlay(); draw(); });
$("#toggle-raw").addEventListener("click", () => { state.showRaw = !state.showRaw; draw(); });
$("#toggle-overlay").addEventListener("click", () => { state.overlay = state.overlay > 0.4 ? 0.2 : (state.overlay > 0 ? 0 : 0.45); renderOverlay(); draw(); });
$("#fit-view").addEventListener("click", fitView);
$("#go").addEventListener("click", () => showFrame($("#recording").value, Number($("#frame-index").value)));
$("#next").addEventListener("click", () => nextFrame().catch((e) => setStatus(e.message, "error")));
$("#prev").addEventListener("click", () => stepHistory(-1));
$("#save-next").addEventListener("click", async () => { if (await saveLabel()) await nextFrame().catch((e) => setStatus(e.message, "error")); });
$("#delete").addEventListener("click", async () => {
  if (!state.frame || !state.frame.existing_record) { setStatus("this frame has no saved label", "error"); return; }
  const payload = await post("/api/delete", { sample_id: state.frame.existing_record.sample_id });
  updateDataset(payload.counts);
  setStatus("deleted saved label", "ok");
  await showFrame(state.frame.recording, state.frame.frame_index, { record: false });
});

window.addEventListener("keydown", async (event) => {
  if (event.target.tagName === "INPUT" || event.target.tagName === "SELECT") {
    if (event.key === "Enter" && event.target.id === "frame-index") $("#go").click();
    return;
  }
  const key = event.key;
  const proposalKeys = { "1": "network", "2": "classical", "4": "raw_threshold", "5": "existing" };
  if (proposalKeys[key]) { combine(proposalByName(proposalKeys[key]), modeFromEvent(event)); event.preventDefault(); return; }
  switch (key.toLowerCase()) {
    case "3": refine("mask_fit"); break;
    case "b": setBrush("worm"); break;
    case "e": setBrush("background"); break;
    case "i": setBrush("ignore"); break;
    case "[": setBrushSize(state.brush.size - (state.brush.size > 10 ? 4 : 1)); break;
    case "]": setBrushSize(state.brush.size + (state.brush.size >= 10 ? 4 : 1)); break;
    case "z": undo(); break;
    case "h": refine("fill_holes"); break;
    case "l": refine("largest_component"); break;
    case "d": refine("dilate"); break;
    case "s": refine("erode"); break;
    case "f": state.showRaw = !state.showRaw; draw(); break;
    case "o": $("#toggle-overlay").click(); break;
    case "0": fitView(); break;
    case "n": nextFrame().catch((e) => setStatus(e.message, "error")); break;
    case "p": stepHistory(-1); break;
    case " ": case "enter": event.preventDefault(); if (await saveLabel()) await nextFrame().catch((e) => setStatus(e.message, "error")); break;
    default: return;
  }
  event.preventDefault();
});

window.addEventListener("beforeunload", (event) => { if (state.dirty) { event.preventDefault(); event.returnValue = ""; } });

async function boot() {
  try {
    state.info = await api("/api/state");
    const select = $("#recording");
    select.innerHTML = "";
    for (const recording of state.info.recordings) {
      const option = document.createElement("option");
      option.value = recording.name;
      option.textContent = `${recording.name} (${recording.frame_count} frames, ${recording.labeled} labeled)`;
      select.appendChild(option);
    }
    updateDataset(state.info.counts);
    if (!state.info.checkpoint) $("#next-mode").value = "random";
    await nextFrame();
  } catch (error) {
    setStatus(error.message, "error");
  }
}

boot();
