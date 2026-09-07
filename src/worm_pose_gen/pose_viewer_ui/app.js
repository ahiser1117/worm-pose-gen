"use strict";

// Pose run viewer: scrub a stored run, composite the pipeline's layers on the
// frame, and read every statistic the run tracks with synced timelines.

const $ = (selector) => document.querySelector(selector);

// Pixel layers are composited server-side PNGs; vector layers are drawn from
// the centerlines.  Order here is draw order; the first nine get number keys.
const LAYERS = [
  { id: "image", name: "Image (flat-fielded)", kind: "base", on: true, alpha: 1.0 },
  { id: "probability", name: "Probability heat", kind: "pixel", on: false, alpha: 0.6, color: [255, 200, 0] },
  { id: "mask_raw", name: "Raw mask (threshold)", kind: "pixel", on: true, alpha: 0.35, color: [80, 190, 255] },
  { id: "fill_adds", name: "Hole fill adds", kind: "pixel", on: true, alpha: 0.85, color: [255, 150, 30] },
  { id: "largest_drops", name: "Largest-component drops", kind: "pixel", on: true, alpha: 0.85, color: [255, 70, 255] },
  { id: "residual", name: "Residual (blue missed / red extra)", kind: "pixel", on: false, alpha: 0.6, color: [140, 140, 255] },
  { id: "tube", name: "Tube outline", kind: "pixel", on: true, alpha: 1.0, color: [90, 220, 140] },
  { id: "centerline", name: "Centerline + head □ / tail ○", kind: "vector", on: true, alpha: 1.0, color: [255, 80, 165] },
  { id: "independent", name: "Independent fit (before propagation)", kind: "both", on: true, alpha: 0.9, color: [200, 200, 200] },
  { id: "final_outline", name: "Final mask outline", kind: "pixel", on: false, alpha: 0.9, color: [255, 255, 255] },
  { id: "tube_fill", name: "Tube fill", kind: "pixel", on: false, alpha: 0.3, color: [90, 220, 140] },
  { id: "width_ticks", name: "Width ticks along the body", kind: "vector", on: false, alpha: 0.9, color: [255, 80, 165] },
  { id: "compare", name: "Compare run pose", kind: "vector", on: true, alpha: 1.0, color: [255, 170, 60] },
  { id: "starts", name: "Fitter starts (press Starts)", kind: "vector", on: true, alpha: 0.9, color: [120, 255, 255] },
  { id: "crop", name: "Crop window", kind: "vector", on: false, alpha: 0.8, color: [150, 150, 150] },
];

const NOTE_TAGS = ["coil", "edge", "fragment", "orientation", "length", "jump", "segmentation", "propagation", "good example", "other"];
const SOURCE_NAMES = { 0: "independent", 1: "forward", 2: "backward" };
const KIND_COLORS = { clean: "#57d68d", watch: "#e0b04d", ambiguous: "#ff5d5d", unfitted: "#444" };
const GROUP_COLORS = { failure: "#ff5d5d", coil: "#e0b04d", edge: "#80c8ff", mask: "#ff70ff" };
const EXTRA_SERIES = [
  ["pose_jump_px", "Pose jump px"], ["self_contact_px", "Self-contact px"], ["width_px", "Width px"],
  ["energy", "Soft-Dice energy"], ["total_energy", "Total energy"], ["points_in_fov", "Points in view"],
  ["pixels_filled", "Hole-fill px"], ["pixels_outside_largest", "Outside largest px"], ["components", "Components"],
  ["taper_asymmetry", "Taper asymmetry"], ["orientation_gap", "Orientation gap"], ["length_deviation", "Length deviation (log)"],
  ["n_starts", "Starts tried"],
];

const state = {
  info: null,
  runs: [],
  run: null,             // /api/run payload of the selected run
  runName: null,
  compare: null,         // /api/run payload of the compare run
  compareName: null,
  compareRows: null,     // frame_index -> row in the compare run
  row: 0,
  frame: null,           // decoded frame payload
  frameCache: new Map(), // cacheKey -> payload (with decoded images attached)
  decoded: null,         // Uint8Array layers of the current frame
  image: null, imageRaw: null,
  showRaw: false,
  starts: null,
  comparePose: null,
  view: { scale: 1, tx: 0, ty: 0 },
  panning: false, last: null,
  playing: null,
  loading: 0,
  fitPending: false,     // fit the view when the next frame arrives
  userView: false,       // the user zoomed or panned; keep their view on resize
  timeline: { start: 0, end: 0, dragging: false },
  charts: [],
  notes: [],
  noteTags: new Set(),
  extra: "pose_jump_px",
  requestId: 0,
};

const canvas = $("#canvas");
const ctx = canvas.getContext("2d");
const overlayCanvas = document.createElement("canvas");
const overlayCtx = overlayCanvas.getContext("2d");

function setStatus(text, kind) {
  const node = $("#status");
  node.textContent = text;
  node.className = "status" + (kind ? " " + kind : "");
}

function setLoading(delta) {
  state.loading = Math.max(0, state.loading + delta);
  updateLoadingIndicator();
}

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

// Decode a gray PNG data URL to its red channel.
async function decodeGray(url) {
  const image = await loadImage(url);
  if (!image) return null;
  const off = document.createElement("canvas");
  off.width = image.width; off.height = image.height;
  const c = off.getContext("2d", { willReadFrequently: true });
  c.drawImage(image, 0, 0);
  const data = c.getImageData(0, 0, off.width, off.height).data;
  const out = new Uint8Array(off.width * off.height);
  for (let i = 0; i < out.length; i++) out[i] = data[i * 4];
  return out;
}

// Numbers: integers print as such unless a digit count is asked for.
const fmt = (v, digits) => {
  if (v === null || v === undefined || Number.isNaN(v)) return "–";
  if (typeof v !== "number") return String(v);
  if (digits === undefined) return Number.isInteger(v) ? String(v) : v.toFixed(3);
  return v.toFixed(digits);
};

// ---------------------------------------------------------------- runs

function runOptionText(entry) {
  const iou = entry.iou_median === null || entry.iou_median === undefined ? "" : ` · IoU ${entry.iou_median.toFixed(3)}`;
  return `${entry.name}${iou}`;
}

function renderRunList() {
  const filter = $("#run-filter").value.trim().toLowerCase();
  const select = $("#run");
  select.innerHTML = "";
  for (const entry of state.runs) {
    if (filter && !(entry.name + " " + entry.recording).toLowerCase().includes(filter)) continue;
    const option = document.createElement("option");
    option.value = entry.name;
    option.textContent = runOptionText(entry);
    option.title = `${entry.recording} frames ${entry.frames[0]}–${entry.frames[1]} · ${entry.mask_cleanup} · checkpoint ${entry.checkpoint_sha}`;
    if (entry.name === state.runName) option.selected = true;
    select.appendChild(option);
  }
}

function describeRun(entry, run) {
  const lines = [
    `${entry.recording}  frames ${entry.frames[0]}–${entry.frames[1]} (${entry.frame_count}, step ${entry.step})`,
    `mask: ${entry.mask_cleanup} · threshold ${run ? run.threshold : "?"} · segmenter ${entry.checkpoint_sha || "?"}`,
    `IoU median ${fmt(entry.iou_median)} · min ${fmt(entry.iou_min)} · below 0.9: ${fmt(entry["frames_below_0.9"])} · ${entry.propagated ? "propagated" : "independent only"} · git ${entry.git_commit}`,
  ];
  if (run && !run.recording_readable) lines.push(`recording not readable: ${run.recording_error}`);
  return lines.join("\n");
}

function runSummaryText(run) {
  const s = run.summary_iou || {};
  const l = run.summary_length || {};
  const p = run.propagation;
  const a = run.ambiguity || {};
  const prior = run.prior;
  const lines = [];
  lines.push(`IoU median ${fmt(s.median)} · p10 ${fmt(s.p10)} · min ${fmt(s.min)} · ≥0.9: ${fmt(s["fraction_at_least_0.9"])}`);
  lines.push(`length median ${fmt(l.median, 0)} px (p10 ${fmt(l.p10, 0)}, p90 ${fmt(l.p90, 0)}) · beyond prior 2σ: ${fmt(l.beyond_2_sigma_of_prior)}`);
  if (prior) lines.push(`prior: length ${prior.length_px.toFixed(0)} ± ${(100 * prior.log_length_sigma).toFixed(0)}% · width ${prior.width_px.toFixed(1)} px · from ${prior.frames_used}/${prior.frames_candidates} frames`);
  if (a.flag_counts) lines.push("flags: " + Object.entries(a.flag_counts).filter(([, v]) => v).map(([k, v]) => `${k} ${v}`).join(", "));
  if (a.frames_with_score_at_least_2 !== undefined) lines.push(`score ≥ 2: ${a.frames_with_score_at_least_2} frames · score ≥ 1: ${a.frames_with_score_at_least_1}`);
  if (p) lines.push(`propagation: ${p.stretches.length} stretches, ${p.frames_in_stretches} frames, ${p.frames_replaced} replaced (fwd ${p.replaced_by_source.forward}, bwd ${p.replaced_by_source.backward}); stretch IoU ${fmt(p.stretch_iou_median_before)} → ${fmt(p.stretch_iou_median_after)}`);
  lines.push(`cleanup: fill holes ${run.cleanup.fill_holes} (r ${run.cleanup.hole_radius}) · largest only ${run.cleanup.largest_only}`);
  if (!run.has_independent_pose) lines.push("independent pose not stored in this run (older fitter); only its IoU/score are shown");
  return lines.join("\n");
}

async function selectRun(name, frameIndex) {
  if (!name) return;
  setLoading(1);
  try {
    const run = await api(`/api/run?name=${encodeURIComponent(name)}`);
    abortLoad("light"); abortLoad("full"); abortLoad("compare"); cancelPendingFull();
    loads.lightWanted = null;
    state.run = run;
    state.runName = name;
    state.frameCache.clear();
    state.frame = null; state.decoded = null;
    state.starts = null;
    state.compare = null; state.compareName = null; state.comparePose = null;
    const entry = run.entry;
    $("#run-info").textContent = describeRun(entry, run);
    $("#run-summary").textContent = runSummaryText(run);
    renderRunList();
    const compare = $("#compare");
    compare.innerHTML = '<option value="">none</option>';
    for (const other of run.compatible_runs) {
      const option = document.createElement("option");
      option.value = other.name;
      option.textContent = `${other.overlap ? "" : "(no overlap) "}${other.name} · ${other.frames[0]}–${other.frames[1]} · IoU ${fmt(other.iou_median)} · ${other.mask_cleanup}`;
      compare.appendChild(option);
    }
    populateWorst();
    state.timeline = { start: 0, end: run.series.frame_index.length, dragging: false };
    buildCharts();
    renderNotes();
    let row = 0;
    if (frameIndex !== undefined && frameIndex !== null) {
      const found = run.series.frame_index.indexOf(frameIndex);
      row = found >= 0 ? found : 0;
    }
    await showRow(row, { fit: true });
  } catch (error) {
    setStatus(error.message, "error");
  } finally {
    setLoading(-1);
  }
}

async function selectCompare(name) {
  state.compareName = name || null;
  state.compare = null; state.compareRows = null; state.comparePose = null;
  if (name) {
    try {
      const run = await api(`/api/run?name=${encodeURIComponent(name)}`);
      state.compare = run;
      state.compareRows = new Map(run.series.frame_index.map((f, i) => [f, i]));
    } catch (error) {
      setStatus(error.message, "error");
    }
  }
  drawCharts();
  fetchCompare(state.row);
  renderDetails();
}

function populateWorst() {
  const series = state.run.series;
  const rows = [];
  for (let r = 0; r < series.iou.length; r++) if (series.fitted[r] && series.iou[r] !== null) rows.push(r);
  rows.sort((a, b) => series.iou[a] - series.iou[b]);
  const select = $("#worst");
  select.innerHTML = '<option value="">lowest IoU frames…</option>';
  for (const r of rows.slice(0, 40)) {
    const option = document.createElement("option");
    option.value = r;
    option.textContent = `frame ${series.frame_index[r]}  IoU ${series.iou[r].toFixed(3)}  score ${series.ambiguity_score ? series.ambiguity_score[r] : "?"}`;
    select.appendChild(option);
  }
}

// ---------------------------------------------------------------- frames
//
// Two tiers keep scrubbing smooth.  A light request (image, tube, pose and
// statistics; no segmenter) follows the cursor at once and aborts whatever
// it superseded; the full mask layers are asked for only once the cursor
// has rested.  Payloads and their decoded images are cached by row.

const loads = { light: null, lightRow: null, lightWanted: null, full: null, compare: null, fullTimer: null, fullResolve: null, prefetching: false, seq: 0, applied: 0 };
const FULL_DELAY_MS = 150;
const CACHE_ENTRIES = 24;

function thresholdParam() { return $("#threshold-on").checked ? $("#threshold").value : ""; }

function cacheKey(row, detail, raw) { return `${row}|${thresholdParam()}|${detail}|${raw ? 1 : 0}`; }

function frameUrl(row, detail, raw) {
  const frame = state.run.series.frame_index[row];
  const threshold = thresholdParam() ? `&threshold=${thresholdParam()}` : "";
  return `/api/frame?run=${encodeURIComponent(state.runName)}&frame=${frame}&detail=${detail}${threshold}${raw ? "&raw=1" : ""}`;
}

async function fetchJson(url, controller) {
  const response = await fetch(url, controller ? { signal: controller.signal } : undefined);
  const payload = await response.json();
  if (!response.ok || payload.error) throw new Error(payload.error || response.statusText);
  return payload;
}

const isAbort = (error) => error && error.name === "AbortError";

function abortLoad(kind) {
  if (loads[kind]) { loads[kind].abort(); loads[kind] = null; }
}


function trimCache() {
  while (state.frameCache.size > CACHE_ENTRIES) state.frameCache.delete(state.frameCache.keys().next().value);
}

function cachedFrame(row, raw) {
  return state.frameCache.get(cacheKey(row, "full", raw)) || null;
}

async function loadTier(row, detail, controller) {
  const raw = state.showRaw;
  let payload = state.frameCache.get(cacheKey(row, "full", raw));
  if (!payload && detail === "light") payload = state.frameCache.get(cacheKey(row, "light", raw));
  if (payload) return payload;
  payload = await fetchJson(frameUrl(row, detail, raw), controller);
  state.frameCache.set(cacheKey(row, detail, raw), payload);
  trimCache();
  return payload;
}

async function decodeFrame(payload) {
  const layers = payload.layers || {};
  const names = ["probability", "mask_raw", "mask_filled", "mask_largest", "mask_final", "tube", "tube_independent"];
  const decoded = {};
  await Promise.all(names.map(async (name) => { decoded[name] = layers[name] ? await decodeGray(layers[name]) : null; }));
  decoded.width = payload.width; decoded.height = payload.height;
  return decoded;
}

// Decode once per payload, then show it if the cursor is still on its row.
// While scrubbing, a light payload a step behind the cursor is still shown
// (``stale``) so the view keeps moving; anything older than what is on
// screen is dropped.
async function applyPayload(payload, row, stale = false) {
  if (!payload._decoded) {
    const [decoded, image, imageRaw] = await Promise.all([decodeFrame(payload), loadImage(payload.layers && payload.layers.image), loadImage(payload.image_raw)]);
    payload._decoded = decoded; payload._image = image; payload._imageRaw = imageRaw;
  }
  if (row !== state.row && !stale) return false;
  if (stale && row !== state.row && payload._seq !== undefined && payload._seq < loads.applied) return false;
  if (payload._seq !== undefined) loads.applied = Math.max(loads.applied, payload._seq);
  state.frame = payload;
  state.decoded = payload._decoded;
  state.image = payload._image;
  state.imageRaw = payload._imageRaw;
  if (state.fitPending) { fitView(); state.fitPending = false; }
  buildOverlay();
  draw();
  renderDetails();
  return true;
}

function frameStatus(row, payload) {
  const n = state.run.series.frame_index.length;
  const frame = state.run.series.frame_index[row];
  if (payload.errors && payload.errors.length) { setStatus(payload.errors.join("; "), "error"); return; }
  setStatus(`frame ${frame} · row ${row + 1}/${n}${payload.detail === "light" ? " · mask layers…" : ""}`, payload.detail === "light" ? "" : "ok");
}

function updateLoadingIndicator() {
  const busy = loads.full !== null || loads.fullTimer !== null || state.loading > 0;
  $("#loading").hidden = !busy;
  $("#loading").textContent = state.loading > 0 ? "Working…" : "mask layers…";
}

function fetchCompare(row) {
  abortLoad("compare");
  state.comparePose = null;
  if (!state.compareName) return;
  const controller = new AbortController();
  loads.compare = controller;
  const frame = state.run.series.frame_index[row];
  fetchJson(`/api/pose?run=${encodeURIComponent(state.compareName)}&frame=${frame}`, controller)
    .then((payload) => {
      if (row !== state.row) return;
      state.comparePose = payload.present ? payload : null;
      draw(); renderDetails();
    })
    .catch((error) => { if (!isAbort(error)) setStatus(error.message, "error"); })
    .finally(() => { if (loads.compare === controller) loads.compare = null; });
}

// One background loop fills the cache with the full layers of the rows just
// ahead of the cursor, one request at a time, re-aiming after each; it runs
// while the cursor rests and during playback, and idles when nothing is left.
const PREFETCH_AHEAD = 4;

async function prefetchLoop() {
  if (loads.prefetching || !state.run) return;
  loads.prefetching = true;
  const runName = state.runName;
  try {
    for (;;) {
      if (state.runName !== runName) break;
      const n = state.run.series.frame_index.length;
      let target = null;
      for (let k = 1; k <= PREFETCH_AHEAD; k++) {
        const r = state.row + k;
        if (r < n && !cachedFrame(r, state.showRaw)) { target = r; break; }
      }
      if (target === null) break;
      const payload = await loadTier(target, "full", null);
      if (state.runName !== runName) break;
      await applyPayload(payload, -1);  // decode ahead of time; never shown from here
    }
  } catch (error) {
    console.warn("prefetch", error);
  } finally {
    loads.prefetching = false;
  }
}

// Light requests: at most one in flight; when it lands it is shown even if
// the cursor moved on, and the cursor's current row is requested next.
function requestLight(row) {
  loads.lightWanted = row;
  if (loads.light) return;
  const controller = new AbortController();
  loads.light = controller;
  loads.lightRow = row;
  const seq = ++loads.seq;
  loadTier(row, "light", controller)
    .then((payload) => { payload._seq = Math.max(payload._seq || 0, seq); return applyPayload(payload, row, true).then((ok) => { if (ok && row === state.row) frameStatus(row, payload); }); })
    .catch((error) => { if (!isAbort(error)) setStatus(error.message, "error"); })
    .finally(() => {
      loads.light = null;
      if (loads.lightWanted !== null && loads.lightWanted !== row && !cachedFrame(loads.lightWanted, state.showRaw)) requestLight(loads.lightWanted);
    });
}

// Move the cursor.  Resolves when the full layers are shown for this row, or
// false when the cursor moved on first.
function showRow(row, options = {}) {
  if (!state.run) return Promise.resolve(false);
  const n = state.run.series.frame_index.length;
  row = Math.max(0, Math.min(n - 1, row));
  state.row = row;
  $("#frame-index").value = state.run.series.frame_index[row];
  if (options.fit) state.fitPending = true;
  if (!options.keepStarts) state.starts = null;
  drawCharts();
  updateHash();
  abortLoad("full");
  cancelPendingFull();
  fetchCompare(row);
  renderNotes();
  const full = cachedFrame(row, state.showRaw);
  if (full) {
    loads.lightWanted = null;
    full._seq = ++loads.seq;
    updateLoadingIndicator();
    return applyPayload(full, row).then((ok) => { if (ok) { frameStatus(row, full); prefetchLoop(); } return ok; });
  }
  requestLight(row);
  const delay = options.immediate || state.playing ? 0 : FULL_DELAY_MS;
  return new Promise((resolve) => {
    loads.fullResolve = resolve;
    loads.fullTimer = setTimeout(async () => {
      loads.fullTimer = null; loads.fullResolve = null;
      if (row !== state.row) return resolve(false);
      const controller = new AbortController();
      loads.full = controller;
      updateLoadingIndicator();
      try {
        const payload = await loadTier(row, "full", controller);
        payload._seq = ++loads.seq;
        const ok = await applyPayload(payload, row);
        if (ok) { frameStatus(row, payload); prefetchLoop(); }
        resolve(ok);
      } catch (error) {
        if (!isAbort(error)) setStatus(error.message, "error");
        resolve(false);
      } finally {
        if (loads.full === controller) loads.full = null;
        updateLoadingIndicator();
      }
    }, delay);
    updateLoadingIndicator();
  });
}

// A superseded full request that has not been sent yet still owes its caller an answer.
function cancelPendingFull() {
  if (loads.fullTimer) { clearTimeout(loads.fullTimer); loads.fullTimer = null; }
  if (loads.fullResolve) { const resolve = loads.fullResolve; loads.fullResolve = null; resolve(false); }
}

function step(delta) { showRow(state.row + delta, { keepView: true }); }

// ---------------------------------------------------------------- compositing

function layer(id) { return LAYERS.find((l) => l.id === id); }

function buildOverlay() {
  const d = state.decoded;
  if (!d || !d.width) return;
  const W = d.width, H = d.height, N = W * H;
  overlayCanvas.width = W; overlayCanvas.height = H;
  const out = overlayCtx.createImageData(W, H);
  const px = out.data;
  const on = (id) => { const l = layer(id); return l.on && l.alpha > 0 ? l : null; };
  const prob = on("probability"), raw = on("mask_raw"), adds = on("fill_adds"), drops = on("largest_drops");
  const residual = on("residual"), tube = on("tube"), tubeFill = on("tube_fill"), outline = on("final_outline"), indep = on("independent");
  const P = d.probability, MR = d.mask_raw, MF = d.mask_filled, ML = d.mask_largest, MX = d.mask_final, T = d.tube, TI = d.tube_independent;
  const blend = (i, color, alpha) => {
    const o = i * 4, a = px[o + 3] / 255;
    const na = alpha + a * (1 - alpha);
    px[o] = (color[0] * alpha + px[o] * a * (1 - alpha)) / na;
    px[o + 1] = (color[1] * alpha + px[o + 1] * a * (1 - alpha)) / na;
    px[o + 2] = (color[2] * alpha + px[o + 2] * a * (1 - alpha)) / na;
    px[o + 3] = na * 255;
  };
  const edge = (M, i, x, y) => M[i] && (x === 0 || y === 0 || x === W - 1 || y === H - 1 || !M[i - 1] || !M[i + 1] || !M[i - W] || !M[i + W]);
  const heat = (v) => {
    // black -> blue -> yellow -> white, so 0.5 sits at a clear hue change.
    const t = v / 255;
    return t < 0.5 ? [0, 60 * t * 2, 255 * t * 2] : [255 * (t - 0.5) * 2, 60 + 195 * (t - 0.5) * 2, 255 - 255 * (t - 0.5) * 2];
  };
  for (let i = 0; i < N; i++) {
    const x = i % W, y = (i - x) / W;
    if (prob && P && P[i] > 8) blend(i, heat(P[i]), prob.alpha * Math.min(1, P[i] / 128));
    if (raw && MR && MR[i]) blend(i, raw.color, raw.alpha);
    if (adds && MF && MR && MF[i] && !MR[i]) blend(i, adds.color, adds.alpha);
    if (drops && ML && MF && MF[i] && !ML[i]) blend(i, drops.color, drops.alpha);
    if (residual && MX && T) {
      if (MX[i] && !T[i]) blend(i, [40, 80, 255], residual.alpha);
      else if (T[i] && !MX[i]) blend(i, [255, 50, 50], residual.alpha);
    }
    if (tubeFill && T && T[i]) blend(i, tubeFill.color, tubeFill.alpha);
    if (outline && MX && edge(MX, i, x, y)) blend(i, outline.color, outline.alpha);
    if (indep && TI && edge(TI, i, x, y)) blend(i, indep.color, indep.alpha * 0.8);
    if (tube && T && edge(T, i, x, y)) blend(i, tube.color, tube.alpha);
  }
  overlayCtx.putImageData(out, 0, 0);
  renderLegend();
}

function renderLegend() {
  const parts = [];
  for (const l of LAYERS) {
    if (!l.on || l.kind === "base") continue;
    if (l.id === "independent" && !(state.decoded && state.decoded.tube_independent) && !(state.frame && state.frame.pose && state.frame.pose.independent)) continue;
    if (l.id === "compare" && !state.comparePose) continue;
    if (l.id === "starts" && !state.starts) continue;
    if (l.id === "residual") { parts.push(`<span><span class="swatch" style="background:rgb(40,80,255)"></span>mask missed</span><span><span class="swatch" style="background:rgb(255,50,50)"></span>tube extra</span>`); continue; }
    parts.push(`<span><span class="swatch" style="background:rgb(${l.color.join(",")})"></span>${l.name.replace(/ \(.*\)$/, "")}</span>`);
  }
  $("#legend").innerHTML = parts.join("");
  $("#legend").hidden = parts.length === 0;
}

// ---------------------------------------------------------------- stage drawing

function resizeCanvas() {
  const rect = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  const width = Math.max(1, Math.round(rect.width * ratio));
  const height = Math.max(1, Math.round(rect.height * ratio));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  if (!state.userView && state.decoded) fitView();
  else draw();
}

function fitView() {
  const d = state.decoded;
  if (!d || !d.width) return;
  const rect = canvas.getBoundingClientRect();
  const scale = Math.min(rect.width / d.width, rect.height / d.height) * 0.98;
  state.view = { scale, tx: (rect.width - d.width * scale) / 2, ty: (rect.height - d.height * scale) / 2 };
  state.userView = false;
  draw();
}

function toImage(clientX, clientY) {
  const rect = canvas.getBoundingClientRect();
  const v = state.view;
  return { x: (clientX - rect.left - v.tx) / v.scale, y: (clientY - rect.top - v.ty) / v.scale };
}

function drawCurve(points, color, width, dash, alpha) {
  if (!points || points.length < 2) return;
  ctx.save();
  ctx.globalAlpha = alpha === undefined ? 1 : alpha;
  ctx.strokeStyle = color;
  ctx.lineWidth = width / state.view.scale;
  ctx.setLineDash(dash ? dash.map((v) => v / state.view.scale) : []);
  ctx.beginPath();
  ctx.moveTo(points[0][0], points[0][1]);
  for (let i = 1; i < points.length; i++) ctx.lineTo(points[i][0], points[i][1]);
  ctx.stroke();
  ctx.restore();
}

function drawEnds(points, color) {
  const s = state.view.scale;
  ctx.save();
  ctx.strokeStyle = color; ctx.lineWidth = 2 / s;
  const [hx, hy] = points[0];
  ctx.strokeStyle = "rgb(255,210,60)";
  ctx.strokeRect(hx - 5 / s, hy - 5 / s, 10 / s, 10 / s);
  const [tx, ty] = points[points.length - 1];
  ctx.strokeStyle = "rgb(80,200,255)";
  ctx.beginPath(); ctx.arc(tx, ty, 6 / s, 0, Math.PI * 2); ctx.stroke();
  ctx.restore();
}

function drawWidthTicks(points, profile, color, every = 4) {
  ctx.save();
  ctx.strokeStyle = color; ctx.lineWidth = 1 / state.view.scale; ctx.globalAlpha = 0.9;
  ctx.beginPath();
  for (let i = 0; i < points.length; i += every) {
    const a = points[Math.max(0, i - 1)], b = points[Math.min(points.length - 1, i + 1)];
    let nx = -(b[1] - a[1]), ny = b[0] - a[0];
    const norm = Math.hypot(nx, ny) || 1;
    nx /= norm; ny /= norm;
    const half = profile[i] / 2;
    ctx.moveTo(points[i][0] - nx * half, points[i][1] - ny * half);
    ctx.lineTo(points[i][0] + nx * half, points[i][1] + ny * half);
  }
  ctx.stroke();
  ctx.restore();
}

function draw() {
  const ratio = window.devicePixelRatio || 1;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const d = state.decoded;
  if (!d || !d.width) return;
  const v = state.view;
  ctx.setTransform(ratio * v.scale, 0, 0, ratio * v.scale, ratio * v.tx, ratio * v.ty);
  ctx.imageSmoothingEnabled = v.scale < 1;
  const base = layer("image");
  const image = state.showRaw && state.imageRaw ? state.imageRaw : state.image;
  if (base.on && image) { ctx.globalAlpha = base.alpha; ctx.drawImage(image, 0, 0); ctx.globalAlpha = 1; }
  else { ctx.fillStyle = "#000"; ctx.fillRect(0, 0, d.width, d.height); }
  ctx.drawImage(overlayCanvas, 0, 0);
  const pose = state.frame && state.frame.pose;
  if (pose) {
    if (layer("crop").on && pose.crop) {
      const [x0, x1, y0, y1] = pose.crop;
      ctx.save(); ctx.strokeStyle = "rgba(160,160,160,0.8)"; ctx.setLineDash([6 / v.scale, 4 / v.scale]); ctx.lineWidth = 1 / v.scale;
      ctx.strokeRect(x0, y0, x1 - x0, y1 - y0); ctx.restore();
    }
    const indep = layer("independent");
    if (indep.on && pose.independent) {
      drawCurve(pose.independent.centerline_xy, `rgb(${indep.color.join(",")})`, 2, [8, 5], indep.alpha);
    }
    const cmp = layer("compare");
    if (cmp.on && state.comparePose && state.comparePose.pose) {
      drawCurve(state.comparePose.pose.centerline_xy, `rgb(${cmp.color.join(",")})`, 2, [3, 4], cmp.alpha);
      drawEnds(state.comparePose.pose.centerline_xy, `rgb(${cmp.color.join(",")})`);
    }
    if (layer("width_ticks").on) drawWidthTicks(pose.centerline_xy, pose.width_profile, "rgba(255,80,165,0.9)");
    const line = layer("centerline");
    if (line.on) {
      drawCurve(pose.centerline_xy, `rgb(${line.color.join(",")})`, 2, null, line.alpha);
      drawEnds(pose.centerline_xy, `rgb(${line.color.join(",")})`);
    }
  }
  const starts = layer("starts");
  if (starts.on && state.starts) {
    const palette = ["rgb(120,255,255)", "rgb(255,255,120)", "rgb(200,120,255)", "rgb(120,255,160)", "rgb(255,160,120)"];
    state.starts.forEach((s, i) => {
      drawCurve(s.centerline_xy, palette[i % palette.length], 1.5, [2, 3], starts.alpha);
      const [x, y] = s.centerline_xy[0];
      ctx.save(); ctx.fillStyle = palette[i % palette.length]; ctx.font = `${11 / v.scale}px system-ui`;
      ctx.fillText(s.name, x + 6 / v.scale, y - 6 / v.scale); ctx.restore();
    });
  }
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  renderCaption();
}

function renderCaption() {
  const f = state.frame;
  if (!f) { $("#caption").textContent = ""; return; }
  const s = f.stats;
  const parts = [`${state.run.entry.recording} · frame ${f.frame_index}`];
  if (s.fitted) {
    parts.push(`IoU ${fmt(s.iou)}`);
    if (s.iou_independent !== undefined && s.source) parts.push(`(indep ${fmt(s.iou_independent)})`);
    parts.push(`len ${fmt(s.body_length_px, 0)} px`, `width ${fmt(s.width_px, 1)} px`, `in view ${fmt(s.in_view_fraction, 2)}`);
    if (s.source_name && s.source_name !== "independent") parts.push(s.source_name);
    if (s.ambiguity_score) parts.push(`score ${s.ambiguity_score}`);
  } else parts.push("no fit");
  if (f.threshold !== state.run.threshold) parts.push(`threshold ${f.threshold} (override)`);
  $("#caption").textContent = parts.join("  ");
}

// ---------------------------------------------------------------- details panel

function row(label, value, cls) {
  return `<tr><td>${label}</td><td class="num ${cls || ""}">${value}</td></tr>`;
}
function section(label) { return `<tr class="section"><td colspan="2">${label}</td></tr>`; }

function renderDetails() {
  const f = state.frame;
  if (!f) return;
  const s = f.stats;
  const c = s.classification;
  const badge = $("#classification");
  badge.textContent = c.label;
  badge.className = `badge ${c.kind}`;
  $("#tags").innerHTML = c.tags.map((t) => `<span>${t}</span>`).join("");
  const cmp = state.comparePose && state.comparePose.stats;
  const both = (key, digits) => cmp ? `${fmt(s[key], digits)} <span style="color:#ffaa3c">/ ${fmt(cmp[key], digits)}</span>` : fmt(s[key], digits);
  const rows = [];
  rows.push(section("fit" + (cmp ? " (this / compare)" : "")));
  rows.push(row("fitted", s.fitted ? "yes" : "no"));
  rows.push(row("IoU", both("iou")));
  if (s.iou_independent !== undefined) rows.push(row("IoU independent", fmt(s.iou_independent)));
  rows.push(row("source", s.source_name || "independent"), row("best start", s.best_start || "–"), row("starts tried", fmt(s.n_starts)));
  rows.push(row("soft-Dice energy", fmt(s.energy, 4)), row("total energy", fmt(s.total_energy, 4)));
  if (s.stretch) rows.push(row("stretch", `#${s.stretch.index + 1} frames ${s.stretch.frames[0]}–${s.stretch.frames[1]} (${s.stretch.length})`));
  rows.push(section("body"));
  rows.push(row("length px", both("body_length_px", 1)));
  if (s.length_vs_prior_sigmas !== undefined) rows.push(row("length vs prior", `${fmt(s.length_vs_prior_sigmas, 2)} σ`, Math.abs(s.length_vs_prior_sigmas) > 2 ? "diff" : ""));
  rows.push(row("width px", both("width_px", 2)));
  if (s.width_vs_prior_sigmas !== undefined) rows.push(row("width vs prior", `${fmt(s.width_vs_prior_sigmas, 2)} σ`));
  rows.push(row("in view", `${fmt(s.points_in_fov)}/${state.run.n_points} (${fmt(s.in_view_fraction, 2)})`));
  rows.push(row("taper asymmetry", fmt(s.taper_asymmetry, 3)), row("orientation gap", fmt(s.orientation_gap, 4)), row("reversed start won", s.reversed ? "yes" : "no"));
  rows.push(row("tube area px", fmt(s.tube_area_px, 0)));
  if (s.crop) rows.push(row("crop x0 x1 y0 y1", s.crop.join(" ")));
  rows.push(section("mask (stored)"));
  rows.push(row("worm px", fmt(s.worm_pixels)), row("raw px", fmt(s.raw_worm_pixels)), row("hole fill px", fmt(s.pixels_filled)));
  rows.push(row("components", fmt(s.components)), row("outside largest px", fmt(s.pixels_outside_largest)), row("mask on border", s.mask_on_border ? "yes" : "no"));
  rows.push(section("ambiguity signals"));
  rows.push(row("area ratio mask/tube", fmt(s.area_ratio, 3)), row("self-contact px", fmt(s.self_contact_px, 1)), row("pose jump px", fmt(s.pose_jump_px, 1)));
  rows.push(row("length deviation (log)", fmt(s.length_deviation, 4)));
  rows.push(row("score", fmt(s.ambiguity_score)));
  if (s.score_independent !== undefined) rows.push(row("score independent", fmt(s.score_independent)));
  $("#stats").innerHTML = rows.join("");

  const flags = s.flags.map((fl) => {
    const test = fl.threshold === null || fl.threshold === undefined ? fmt(fl.value) : `${fmt(fl.value)} ${fl.test} ${fmt(fl.threshold)}`;
    return `<tr class="${fl.fired ? "fired" : ""} ${fl.group}" title="${fl.description}"><td><span class="dot"></span>${fl.name}</td><td class="num">${test}</td><td>${fl.group}</td></tr>`;
  });
  $("#flags").innerHTML = flags.join("");

  const m = f.mask_stats, st = f.mask_stats_stored || {};
  const mrows = [];
  if (m) {
    const pair = (label, key) => {
      const differs = st[key] !== undefined && st[key] !== m[key];
      mrows.push(`<tr><td>${label}</td><td class="num">${fmt(m[key])}</td><td class="num ${differs ? "diff" : ""}">${st[key] === undefined ? "" : "stored " + fmt(st[key])}</td></tr>`);
    };
    mrows.push(`<tr><td>threshold</td><td class="num">${f.threshold}</td><td class="num">${f.threshold !== state.run.threshold ? "run " + state.run.threshold : ""}</td></tr>`);
    pair("raw mask px", "raw_worm_pixels"); pair("hole fill adds px", "pixels_filled"); pair("components", "components");
    pair("outside largest px", "pixels_outside_largest"); pair("final mask px", "worm_pixels");
    if (f.checkpoint && state.run.entry.checkpoint_path && !f.checkpoint.endsWith(state.run.entry.checkpoint_path.split("/").slice(-2).join("/")) && f.checkpoint !== state.run.entry.checkpoint_path) {
      mrows.push(`<tr><td colspan="3" class="diff">segmenter differs from the run's: ${f.checkpoint}</td></tr>`);
    }
  } else if (f.detail === "light") mrows.push('<tr><td colspan="3">mask layers loading…</td></tr>');
  else mrows.push(`<tr><td colspan="3">no mask layers (${(f.errors || []).join("; ") || "recording or checkpoint unavailable"})</td></tr>`);
  $("#mask-stats").innerHTML = mrows.join("");

  drawWidthChart();
  drawCurvatureChart();
  renderLayerAvailability();
}

// ---------------------------------------------------------------- small charts

// A chart canvas is laid out at its CSS size and backed at device pixels;
// drawing happens in CSS pixels through the transform.
function chartBox(canvasNode) {
  const ratio = window.devicePixelRatio || 1;
  // Setting canvas.height rewrites the height attribute, so the intended CSS
  // height lives in data-height and is never read back from the element.
  if (!canvasNode.dataset.height) canvasNode.dataset.height = canvasNode.getAttribute("height");
  const cssHeight = Number(canvasNode.dataset.height);
  if (canvasNode.style.height !== `${cssHeight}px`) canvasNode.style.height = `${cssHeight}px`;
  const rect = canvasNode.getBoundingClientRect();
  const width = Math.max(1, Math.round(rect.width * ratio));
  const height = Math.max(1, Math.round(cssHeight * ratio));
  if (canvasNode.width !== width) canvasNode.width = width;
  if (canvasNode.height !== height) canvasNode.height = height;
  const c = canvasNode.getContext("2d");
  c.setTransform(ratio, 0, 0, ratio, 0, 0);
  c.clearRect(0, 0, rect.width, cssHeight);
  return { c, w: rect.width, h: cssHeight };
}

function polyline(c, xs, ys, color, dash, width = 1.5) {
  c.save(); c.strokeStyle = color; c.lineWidth = width; c.setLineDash(dash || []);
  c.beginPath();
  let started = false;
  for (let i = 0; i < xs.length; i++) {
    if (ys[i] === null || ys[i] === undefined || Number.isNaN(ys[i])) { started = false; continue; }
    if (!started) { c.moveTo(xs[i], ys[i]); started = true; } else c.lineTo(xs[i], ys[i]);
  }
  c.stroke(); c.restore();
}

function drawWidthChart() {
  const node = $("#width-chart");
  const { c, w, h } = chartBox(node);
  const pose = state.frame && state.frame.pose;
  if (!pose) return;
  const series = [
    { y: pose.width_profile, color: "#57d68d", dash: null, width: 2 },
    { y: pose.width_template_profile, color: "#9aa6b0", dash: [2, 3] },
    { y: pose.width_prior_profile, color: "#9aa6b0", dash: [6, 4] },
    { y: pose.independent && pose.independent.width_profile, color: "#cccccc", dash: [8, 5] },
    { y: state.comparePose && state.comparePose.pose && state.comparePose.pose.width_profile, color: "#ffaa3c", dash: [3, 4] },
  ].filter((s) => s.y);
  const max = Math.max(...series.flatMap((s) => s.y)) * 1.1 || 1;
  const pad = { l: 30, r: 6, t: 6, b: 16 };
  const n = pose.width_profile.length;
  const X = (i) => pad.l + (i / (n - 1)) * (w - pad.l - pad.r);
  const Y = (v) => pad.t + (1 - v / max) * (h - pad.t - pad.b);
  c.fillStyle = "#9aa6b0"; c.font = "10px system-ui";
  for (const tick of [0, max / 2, max]) { c.fillText(tick.toFixed(0), 2, Y(tick) + 3); c.strokeStyle = "#1f2a33"; c.beginPath(); c.moveTo(pad.l, Y(tick)); c.lineTo(w - pad.r, Y(tick)); c.stroke(); }
  c.fillText("head", pad.l, h - 4); c.fillText("tail", w - pad.r - 18, h - 4);
  const xs = Array.from({ length: n }, (_, i) => X(i));
  for (const s of series) polyline(c, xs, s.y.map(Y), s.color, s.dash, s.width || 1.2);
  // Shade the part of the body outside the camera, when known.
  const shape = state.run.image_shape;
  if (shape) {
    c.fillStyle = "rgba(128,200,255,0.15)";
    pose.centerline_xy.forEach(([x, y], i) => {
      if (x < 0 || y < 0 || x >= shape[1] || y >= shape[0]) c.fillRect(X(i) - 0.5, pad.t, Math.max(1, (w - pad.l - pad.r) / n), h - pad.t - pad.b);
    });
  }
}

function drawCurvatureChart() {
  const node = $("#curvature-chart");
  const { c, w, h } = chartBox(node);
  const pose = state.frame && state.frame.pose;
  if (!pose) return;
  const k = pose.curvature;
  const limit = 1 / Math.max(pose.width_px, 1);
  const max = Math.max(limit * 1.5, ...k.map(Math.abs)) * 1.05;
  const pad = { l: 40, r: 6, t: 4, b: 4 };
  const n = k.length;
  const X = (i) => pad.l + (i / (n - 1)) * (w - pad.l - pad.r);
  const Y = (v) => pad.t + (0.5 - v / (2 * max)) * (h - pad.t - pad.b);
  c.fillStyle = "rgba(255,93,93,0.12)";
  c.fillRect(pad.l, pad.t, w - pad.l - pad.r, Y(limit) - pad.t);
  c.fillRect(pad.l, Y(-limit), w - pad.l - pad.r, h - pad.b - Y(-limit));
  c.strokeStyle = "#2a343c"; c.beginPath(); c.moveTo(pad.l, Y(0)); c.lineTo(w - pad.r, Y(0)); c.stroke();
  c.fillStyle = "#9aa6b0"; c.font = "10px system-ui";
  c.fillText(`+${max.toFixed(3)}`, 2, pad.t + 9); c.fillText(`−${max.toFixed(3)}`, 2, h - pad.b - 2); c.fillText("1/width", 2, Y(limit) + 3);
  const xs = Array.from({ length: n }, (_, i) => X(i));
  polyline(c, xs, k.map(Y), "#57d68d", null, 1.5);
  if (state.comparePose && state.comparePose.pose) polyline(c, xs, state.comparePose.pose.curvature.map(Y), "#ffaa3c", [3, 4], 1);
}

// ---------------------------------------------------------------- timeline

function chartSpecs() {
  const run = state.run;
  const prior = run.prior;
  const specs = [
    { id: "iou", label: "IoU", height: 56, lines: [{ key: "iou", color: "#57d68d" }, { key: "iou_independent", color: "#9aa6b0", dash: [3, 3], opt: "independent" }], compare: "iou", hline: () => parseFloat($("#jump-iou").value), ymin: 0, ymax: 1 },
    { id: "length", label: "Body length px", height: 56, lines: [{ key: "body_length_px", color: "#57d68d" }], compare: "body_length_px", band: prior ? [prior.length_px * Math.exp(-2 * prior.log_length_sigma), prior.length_px * Math.exp(2 * prior.log_length_sigma)] : null },
    { id: "area", label: "Area px (mask, tube in view, raw)", height: 56, lines: [{ key: "worm_pixels", color: "#50beff" }, { key: "tube_area_visible_px", color: "#57d68d" }, { key: "raw_worm_pixels", color: "#9aa6b0", dash: [2, 3] }] },
    { id: "score", label: "Ambiguity score", height: 44, bars: "ambiguity_score", dots: { key: "score_independent", color: "#9aa6b0", opt: "independent" }, ymin: 0, ymax: 9 },
    { id: "extra", label: "", height: 50, lines: [{ key: null, color: "#e0b04d" }], compare: null },
    { id: "flags", label: "Flags", height: 9 * 10 + 2, raster: true },
    { id: "strip", label: "Class · source", height: 20, strip: true },
  ];
  return specs;
}

function buildCharts() {
  const container = $("#charts");
  container.innerHTML = "";
  state.charts = [];
  const extra = $("#extra-series");
  if (!extra.options.length) {
    for (const [key, label] of EXTRA_SERIES) { const o = document.createElement("option"); o.value = key; o.textContent = label; extra.appendChild(o); }
    extra.value = state.extra;
  }
  for (const spec of chartSpecs()) {
    const label = document.createElement("div");
    label.className = "label";
    label.textContent = spec.label;
    const node = document.createElement("canvas");
    node.dataset.height = spec.height;
    node.style.height = `${spec.height}px`;
    container.appendChild(label); container.appendChild(node);
    node.addEventListener("pointerdown", (event) => { state.timeline.dragging = true; node.setPointerCapture(event.pointerId); seekFromEvent(node, event); });
    node.addEventListener("pointermove", (event) => { if (state.timeline.dragging) seekFromEvent(node, event); });
    node.addEventListener("pointerup", () => { state.timeline.dragging = false; });
    node.addEventListener("wheel", (event) => { event.preventDefault(); zoomTimeline(node, event); }, { passive: false });
    node.addEventListener("dblclick", () => { state.timeline.start = 0; state.timeline.end = state.run.series.frame_index.length; drawCharts(); });
    state.charts.push({ spec, node, label });
  }
  drawCharts();
}

function timelineX(node, rowIndex) {
  const { start, end } = state.timeline;
  const w = node.getBoundingClientRect().width;
  return ((rowIndex + 0.5 - start) / (end - start)) * w;
}

function rowFromX(node, clientX) {
  const rect = node.getBoundingClientRect();
  const { start, end } = state.timeline;
  return Math.floor(start + ((clientX - rect.left) / rect.width) * (end - start));
}

function seekFromEvent(node, event) {
  const r = rowFromX(node, event.clientX);
  if (r !== state.row) showRow(r, { keepView: true });
}

function zoomTimeline(node, event) {
  const n = state.run.series.frame_index.length;
  const t = state.timeline;
  const pivot = rowFromX(node, event.clientX);
  const factor = event.deltaY > 0 ? 1.25 : 0.8;
  let span = Math.max(20, Math.min(n, Math.round((t.end - t.start) * factor)));
  let start = Math.round(pivot - (pivot - t.start) * (span / (t.end - t.start)));
  start = Math.max(0, Math.min(n - span, start));
  state.timeline.start = start; state.timeline.end = start + span;
  drawCharts();
}

function compareValues(key) {
  if (!state.compare || !$("#show-compare").checked || !state.compare.series[key]) return null;
  const series = state.run.series;
  const values = state.compare.series[key];
  return series.frame_index.map((f) => { const r = state.compareRows.get(f); return r === undefined ? null : values[r]; });
}

let chartsFrame = null;
function drawCharts() {
  if (chartsFrame) return;
  chartsFrame = requestAnimationFrame(() => { chartsFrame = null; drawChartsNow(); });
}

function drawChartsNow() {
  if (!state.run) return;
  const series = state.run.series;
  const n = series.frame_index.length;
  const { start, end } = state.timeline;
  $("#timeline-range").textContent = `frames ${series.frame_index[start]}–${series.frame_index[Math.max(start, end - 1)]} (${end - start} of ${n} rows)`;
  const showIndependent = $("#show-independent").checked;
  for (const { spec, node, label } of state.charts) {
    const { c, w, h } = chartBox(node);
    const X = (i) => ((i + 0.5 - start) / (end - start)) * w;
    const colw = Math.max(1, w / (end - start));
    // Propagation stretches shaded on every chart.
    for (const [a, b] of state.run.stretches) {
      if (b < start || a >= end) continue;
      c.fillStyle = "rgba(255,255,255,0.06)";
      c.fillRect(X(a) - colw / 2, 0, X(b) - X(a) + colw, h);
    }
    if (spec.raster) {
      const names = Object.keys(series.flags);
      const rowh = (h - 2) / Math.max(1, names.length);
      names.forEach((name, k) => {
        const group = Object.entries(state.info.flag_groups).find(([, list]) => list.includes(name));
        c.fillStyle = GROUP_COLORS[group ? group[0] : "failure"];
        const values = series.flags[name];
        for (let i = start; i < end; i++) if (values[i]) c.fillRect(X(i) - colw / 2, 1 + k * rowh, Math.max(1, colw), rowh - 1);
      });
      label.innerHTML = names.map((nm) => `<div style="height:${rowh}px;line-height:${rowh}px;font-size:9px;overflow:hidden">${nm}</div>`).join("");
    } else if (spec.strip) {
      const half = (h - 2) / 2;
      for (let i = start; i < end; i++) {
        c.fillStyle = KIND_COLORS[series.classification[i]] || "#444";
        c.fillRect(X(i) - colw / 2, 1, Math.max(1, colw), half - 1);
        if (series.source) {
          const s = series.source[i];
          c.fillStyle = s === 1 ? "#ffaa3c" : s === 2 ? "#c080ff" : series.fitted[i] ? "#2a343c" : "#111";
          c.fillRect(X(i) - colw / 2, 1 + half, Math.max(1, colw), half - 1);
        }
      }
    } else {
      const lines = spec.lines ? spec.lines.map((l) => ({ ...l, key: l.key === null ? state.extra : l.key })).filter((l) => series[l.key]) : [];
      if (spec.id === "extra") label.textContent = (EXTRA_SERIES.find(([k]) => k === state.extra) || [state.extra, state.extra])[1];
      const values = [];
      for (const l of lines) if (!(l.opt === "independent" && !showIndependent)) for (let i = start; i < end; i++) { const v = series[l.key][i]; if (v !== null && Number.isFinite(v)) values.push(v); }
      const compare = spec.compare ? compareValues(spec.id === "extra" ? state.extra : spec.compare) : (spec.id === "extra" ? compareValues(state.extra) : null);
      if (compare) for (let i = start; i < end; i++) if (compare[i] !== null && Number.isFinite(compare[i])) values.push(compare[i]);
      if (spec.bars) for (let i = start; i < end; i++) values.push(series[spec.bars][i]);
      if (spec.band) values.push(...spec.band);
      let ymin = spec.ymin !== undefined ? spec.ymin : Math.min(...values);
      let ymax = spec.ymax !== undefined ? spec.ymax : Math.max(...values);
      if (!Number.isFinite(ymin) || !Number.isFinite(ymax)) { ymin = 0; ymax = 1; }
      if (ymax - ymin < 1e-9) { ymax = ymin + 1; }
      if (spec.ymin === undefined) { const m = (ymax - ymin) * 0.08; ymin -= m; ymax += m; }
      const Y = (v) => 3 + (1 - (v - ymin) / (ymax - ymin)) * (h - 6);
      c.fillStyle = "#9aa6b0"; c.font = "9px system-ui";
      c.fillText(fmt(ymax, 2), 2, 10); c.fillText(fmt(ymin, 2), 2, h - 3);
      if (spec.band) {
        c.fillStyle = "rgba(87,214,141,0.12)";
        c.fillRect(0, Y(spec.band[1]), w, Y(spec.band[0]) - Y(spec.band[1]));
      }
      if (spec.hline) {
        const yv = spec.hline();
        c.strokeStyle = "rgba(255,93,93,0.6)"; c.setLineDash([3, 3]); c.beginPath(); c.moveTo(0, Y(yv)); c.lineTo(w, Y(yv)); c.stroke(); c.setLineDash([]);
      }
      if (spec.bars) {
        const vals = series[spec.bars];
        for (let i = start; i < end; i++) {
          if (!vals[i]) continue;
          c.fillStyle = vals[i] >= 2 ? "#ff5d5d" : "#e0b04d";
          c.fillRect(X(i) - colw / 2, Y(vals[i]), Math.max(1, colw), Y(0) - Y(vals[i]));
        }
        if (spec.dots && series[spec.dots.key] && !(spec.dots.opt === "independent" && !showIndependent)) {
          c.fillStyle = spec.dots.color;
          for (let i = start; i < end; i++) { const v = series[spec.dots.key][i]; if (v) c.fillRect(X(i) - 1, Y(v) - 1, 2, 2); }
        }
      }
      const xs = [];
      for (let i = start; i < end; i++) xs.push(X(i));
      for (const l of lines) {
        if (l.opt === "independent" && !showIndependent) continue;
        polyline(c, xs, series[l.key].slice(start, end).map((v) => (v === null ? null : Y(v))), l.color, l.dash, 1.2);
      }
      if (compare) polyline(c, xs, compare.slice(start, end).map((v) => (v === null ? null : Y(v))), "#ffaa3c", [3, 3], 1);
    }
    // Cursor.
    if (state.row >= start && state.row < end) {
      c.strokeStyle = "rgba(255,255,255,0.85)"; c.lineWidth = 1;
      c.beginPath(); c.moveTo(X(state.row), 0); c.lineTo(X(state.row), h); c.stroke();
    }
  }
}

// ---------------------------------------------------------------- jumps and playback

function jump(kind, direction) {
  const series = state.run.series;
  const n = series.frame_index.length;
  const minScore = parseInt($("#jump-score").value, 10) || 2;
  const maxIou = parseFloat($("#jump-iou").value);
  const test = {
    flag: (r) => series.fitted[r] && series.ambiguity_score && series.ambiguity_score[r] >= minScore,
    iou: (r) => series.fitted[r] && series.iou[r] !== null && series.iou[r] < maxIou,
    jump: (r) => series.flags.pose_jump && series.flags.pose_jump[r],
    edge: (r) => series.fitted[r] && ((series.mask_on_border && series.mask_on_border[r]) || series.points_in_fov[r] < state.run.n_points || (series.flags.edge_inside && series.flags.edge_inside[r])),
  }[kind];
  if (kind === "stretch") {
    const stretches = state.run.stretches;
    if (!stretches.length) { setStatus("no propagation stretches in this run", "error"); return; }
    let target = null;
    if (direction > 0) target = stretches.find(([a]) => a > state.row);
    else { const before = stretches.filter(([a]) => a < state.row && !(a <= state.row && state.row <= stretches.find(([x]) => x === a)[1])); target = before[before.length - 1]; }
    if (!target) { setStatus("no more stretches in that direction", "error"); return; }
    showRow(target[0], { keepView: true });
    return;
  }
  // Leave the current run of matching frames first, so repeated presses walk events, not frames.
  let r = state.row;
  while (r >= 0 && r < n && test(r)) r += direction;
  while (r >= 0 && r < n && !test(r)) r += direction;
  if (r < 0 || r >= n) { setStatus("no more matching frames in that direction", "error"); return; }
  showRow(r, { keepView: true });
}

function togglePlay() {
  if (state.playing) { clearInterval(state.playing); state.playing = null; $("#play").textContent = "Play"; return; }
  const period = 1000 / Math.max(1, Math.min(30, parseInt($("#fps").value, 10) || 10));
  // The cursor advances at the requested rate; frames whose full layers are
  // already cached show them, the others show the light tier and the mask
  // layers catch up when playback stops.
  state.playing = setInterval(() => {
    const n = state.run.series.frame_index.length;
    let next = state.row + 1;
    const stretch = state.frame && state.frame.stats && state.frame.stats.stretch;
    if ($("#loop-stretch").checked && stretch && next > stretch.rows[1]) next = stretch.rows[0];
    if (next >= n) { togglePlay(); return; }
    showRow(next, { keepView: true, immediate: true });
  }, period);
  $("#play").textContent = "Pause";
}

// ---------------------------------------------------------------- layers UI

function renderLayers() {
  const box = $("#layers");
  box.innerHTML = "";
  LAYERS.forEach((l, index) => {
    const node = document.createElement("div");
    node.className = "layer";
    node.dataset.layer = l.id;
    const key = index < 9 ? `${index + 1}` : "";
    node.innerHTML = `<input type="checkbox" ${l.on ? "checked" : ""} title="${key}"><span><span class="swatch" style="background:${l.color ? `rgb(${l.color.join(",")})` : "#888"}"></span>${l.name} <span class="key">${key}</span></span><input type="range" min="0" max="1" step="0.05" value="${l.alpha}" title="opacity">`;
    node.querySelector("input[type=checkbox]").addEventListener("change", (e) => { l.on = e.target.checked; relayer(l); });
    node.querySelector("input[type=range]").addEventListener("input", (e) => { l.alpha = parseFloat(e.target.value); relayer(l); });
    box.appendChild(node);
  });
}

function relayer(l) {
  if (l.kind === "pixel" || l.kind === "both") buildOverlay();
  renderLegend();
  draw();
}

function toggleLayer(index) {
  const l = LAYERS[index];
  if (!l) return;
  l.on = !l.on;
  const node = $(`.layer[data-layer="${l.id}"] input[type=checkbox]`);
  if (node) node.checked = l.on;
  relayer(l);
}

function renderLayerAvailability() {
  const d = state.decoded || {};
  const pose = state.frame && state.frame.pose;
  if (state.frame && state.frame.detail === "light") {
    for (const node of document.querySelectorAll(".layer")) node.classList.toggle("unavailable", ["independent", "compare", "starts"].includes(node.dataset.layer) && !{ independent: pose && pose.independent, compare: state.comparePose, starts: state.starts }[node.dataset.layer]);
    return;
  }
  const available = {
    probability: !!d.probability, mask_raw: !!d.mask_raw, fill_adds: !!d.mask_filled, largest_drops: !!d.mask_largest,
    residual: !!(d.mask_final && d.tube), tube: !!d.tube, tube_fill: !!d.tube, final_outline: !!d.mask_final,
    centerline: !!pose, width_ticks: !!pose, crop: !!pose,
    independent: !!(pose && pose.independent), compare: !!state.comparePose, starts: !!state.starts, image: true,
  };
  for (const node of document.querySelectorAll(".layer")) node.classList.toggle("unavailable", available[node.dataset.layer] === false);
}

// ---------------------------------------------------------------- notes

function renderNoteTags() {
  const box = $("#note-tags");
  box.innerHTML = "";
  for (const tag of NOTE_TAGS) {
    const button = document.createElement("button");
    button.type = "button"; button.className = "tag" + (state.noteTags.has(tag) ? " active" : ""); button.textContent = tag;
    button.addEventListener("click", () => { if (state.noteTags.has(tag)) state.noteTags.delete(tag); else state.noteTags.add(tag); renderNoteTags(); });
    box.appendChild(button);
  }
}

async function loadNotes() {
  try {
    const payload = await api("/api/notes");
    state.notes = payload.notes;
    $("#notes-path").textContent = `notes file: ${payload.path}`;
    renderNotes();
  } catch (error) { setStatus(error.message, "error"); }
}

function renderNotes() {
  const list = $("#note-list");
  const mine = state.notes.map((note, index) => ({ note, index })).filter(({ note }) => !state.run || note.recording === state.run.entry.recording);
  $("#note-count").textContent = `${mine.length} for this recording · ${state.notes.length} total`;
  list.innerHTML = "";
  if (!mine.length) { list.innerHTML = '<div class="empty">No notes yet. Mark a frame with tags and a comment.</div>'; return; }
  const currentFrame = state.run ? state.run.series.frame_index[state.row] : null;
  for (const { note, index } of mine.slice().reverse()) {
    const item = document.createElement("div");
    item.className = "item" + (note.frame_index === currentFrame ? " current" : "");
    item.innerHTML = `<span><b>${note.frame_index}</b> ${note.tags.join(", ")}${note.comment ? " — " + note.comment : ""}<div class="meta">${note.run}</div></span><span><button type="button" title="delete">×</button></span>`;
    item.addEventListener("click", async (event) => {
      if (event.target.tagName === "BUTTON") {
        event.stopPropagation();
        const payload = await post("/api/note/delete", { index });
        state.notes = payload.notes; renderNotes();
        return;
      }
      if (note.run !== state.runName && state.runs.some((r) => r.name === note.run)) await selectRun(note.run, note.frame_index);
      else { const r = state.run.series.frame_index.indexOf(note.frame_index); if (r >= 0) showRow(r, { keepView: true }); }
    });
    list.appendChild(item);
  }
}

async function saveNote() {
  if (!state.run) return;
  const comment = $("#note-comment").value.trim();
  const tags = [...state.noteTags];
  if (!tags.length && !comment) { setStatus("pick a tag or write a comment first", "error"); return; }
  try {
    const payload = await post("/api/note", { run: state.runName, frame_index: state.run.series.frame_index[state.row], tags, comment });
    state.notes = payload.notes;
    $("#note-comment").value = "";
    renderNotes();
    setStatus(`noted frame ${state.run.series.frame_index[state.row]}`, "ok");
  } catch (error) { setStatus(error.message, "error"); }
}

// ---------------------------------------------------------------- misc

let hashTimer = null;
function updateHash() {
  if (hashTimer) return;
  hashTimer = setTimeout(() => {
    hashTimer = null;
    if (!state.run) return;
    const frame = state.run.series.frame_index[state.row];
    history.replaceState(null, "", `#run=${encodeURIComponent(state.runName)}&frame=${frame}`);
  }, 250);
}

function parseHash() {
  const params = new URLSearchParams(location.hash.replace(/^#/, ""));
  return { run: params.get("run"), frame: params.has("frame") ? parseInt(params.get("frame"), 10) : null };
}

async function computeStarts() {
  if (!state.run) return;
  setLoading(1);
  try {
    const threshold = $("#threshold-on").checked ? `&threshold=${$("#threshold").value}` : "";
    const payload = await api(`/api/starts?run=${encodeURIComponent(state.runName)}&frame=${state.run.series.frame_index[state.row]}${threshold}`);
    state.starts = payload.starts;
    setStatus(`${payload.starts.length} starts: ${payload.starts.map((s) => s.name).join(", ")}`, "ok");
    renderLegend(); renderLayerAvailability(); draw();
  } catch (error) { setStatus(error.message, "error"); } finally { setLoading(-1); }
}

// ---------------------------------------------------------------- panels
//
// The three panels around the stage are sized by CSS variables; the
// splitters drag them, their buttons collapse and restore them, and the
// layout persists in localStorage.

const LAYOUT_DEFAULT = { left: 290, right: 360, bottom: 330 };
const LAYOUT_MIN = { left: 160, right: 220, bottom: 60 };
const layout = { sizes: { ...LAYOUT_DEFAULT }, collapsed: {}, restore: {} };

function loadLayout() {
  try {
    const saved = JSON.parse(localStorage.getItem("poseViewer.layout") || "null");
    if (saved && saved.sizes) { Object.assign(layout.sizes, saved.sizes); Object.assign(layout.collapsed, saved.collapsed || {}); }
  } catch (error) { /* first visit or blocked storage */ }
}

function saveLayout() {
  try { localStorage.setItem("poseViewer.layout", JSON.stringify({ sizes: layout.sizes, collapsed: layout.collapsed })); } catch (error) { /* ignore */ }
}

function applyLayout() {
  const app = $("#app");
  for (const side of ["left", "right", "bottom"]) {
    const size = layout.collapsed[side] ? 0 : layout.sizes[side];
    app.style.setProperty(`--${side}`, `${size}px`);
    const splitter = $(`#split-${side}`);
    splitter.classList.toggle("collapsed", !!layout.collapsed[side]);
    const button = splitter.querySelector("button");
    button.textContent = { left: layout.collapsed.left ? "▸" : "◂", right: layout.collapsed.right ? "◂" : "▸", bottom: layout.collapsed.bottom ? "▴" : "▾" }[side];
    button.title = layout.collapsed[side] ? "expand panel" : "collapse panel";
  }
  $("#timeline").classList.toggle("collapsed", !!layout.collapsed.bottom);
}

function togglePanel(side) {
  layout.collapsed[side] = !layout.collapsed[side];
  applyLayout();
  saveLayout();
}

function initSplitters() {
  loadLayout();
  applyLayout();
  for (const side of ["left", "right", "bottom"]) {
    const splitter = $(`#split-${side}`);
    splitter.querySelector("button").addEventListener("click", (event) => { event.stopPropagation(); togglePanel(side); });
    splitter.addEventListener("dblclick", () => { layout.sizes[side] = LAYOUT_DEFAULT[side]; layout.collapsed[side] = false; applyLayout(); saveLayout(); });
    splitter.addEventListener("pointerdown", (event) => {
      if (event.target.tagName === "BUTTON") return;
      event.preventDefault();
      splitter.setPointerCapture(event.pointerId);
      const rect = $("#app").getBoundingClientRect();
      const move = (e) => {
        let size = side === "left" ? e.clientX - rect.left : side === "right" ? rect.right - e.clientX : rect.bottom - e.clientY;
        size = Math.round(Math.max(0, Math.min(size, side === "bottom" ? rect.height * 0.8 : rect.width * 0.5)));
        if (size < LAYOUT_MIN[side] / 2) { layout.collapsed[side] = true; }
        else { layout.collapsed[side] = false; layout.sizes[side] = Math.max(LAYOUT_MIN[side], size); }
        applyLayout();
      };
      const up = () => { splitter.removeEventListener("pointermove", move); splitter.removeEventListener("pointerup", up); saveLayout(); };
      splitter.addEventListener("pointermove", move);
      splitter.addEventListener("pointerup", up);
    });
  }
}

function bindEvents() {
  $("#run").addEventListener("change", (e) => selectRun(e.target.value));
  $("#run-filter").addEventListener("input", renderRunList);
  $("#compare").addEventListener("change", (e) => selectCompare(e.target.value));
  $("#go").addEventListener("click", () => { const r = state.run.series.frame_index.indexOf(parseInt($("#frame-index").value, 10)); if (r >= 0) showRow(r, { keepView: true }); else setStatus("frame not in this run", "error"); });
  $("#frame-index").addEventListener("keydown", (e) => { if (e.key === "Enter") $("#go").click(); });
  $("#prev").addEventListener("click", () => step(-1));
  $("#next").addEventListener("click", () => step(1));
  $("#play").addEventListener("click", togglePlay);
  $("#worst").addEventListener("change", (e) => { if (e.target.value !== "") showRow(parseInt(e.target.value, 10), { keepView: true }); });
  for (const button of document.querySelectorAll("[data-jump]")) {
    const [kind, direction] = button.dataset.jump.split(":");
    button.addEventListener("click", () => jump(kind, parseInt(direction, 10)));
  }
  $("#toggle-raw").addEventListener("click", () => { state.showRaw = !state.showRaw; $("#toggle-raw").classList.toggle("active", state.showRaw); showRow(state.row, { keepView: true, keepStarts: true, immediate: true }); });
  $("#fit-view").addEventListener("click", fitView);
  $("#starts").addEventListener("click", computeStarts);
  $("#threshold").addEventListener("input", (e) => { $("#threshold-value").textContent = e.target.value; if ($("#threshold-on").checked) showRow(state.row, { keepView: true, keepStarts: true }); });
  $("#threshold-on").addEventListener("change", (e) => { $("#threshold-value").textContent = e.target.checked ? $("#threshold").value : "run"; showRow(state.row, { keepView: true, keepStarts: true, immediate: true }); });
  $("#note-save").addEventListener("click", saveNote);
  $("#note-comment").addEventListener("keydown", (e) => { if (e.key === "Enter") saveNote(); e.stopPropagation(); });
  $("#extra-series").addEventListener("change", (e) => { state.extra = e.target.value; drawCharts(); });
  $("#show-independent").addEventListener("change", drawCharts);
  $("#show-compare").addEventListener("change", drawCharts);
  $("#jump-iou").addEventListener("change", drawCharts);
  $("#timeline-toggle").addEventListener("click", () => togglePanel("bottom"));

  canvas.addEventListener("wheel", (event) => {
    event.preventDefault();
    const before = toImage(event.clientX, event.clientY);
    const factor = event.deltaY > 0 ? 0.9 : 1.1;
    state.view.scale = Math.max(0.05, Math.min(40, state.view.scale * factor));
    const rect = canvas.getBoundingClientRect();
    state.view.tx = event.clientX - rect.left - before.x * state.view.scale;
    state.view.ty = event.clientY - rect.top - before.y * state.view.scale;
    state.userView = true;
    draw();
  }, { passive: false });
  canvas.addEventListener("pointerdown", (event) => { state.panning = true; state.last = { x: event.clientX, y: event.clientY }; canvas.setPointerCapture(event.pointerId); canvas.style.cursor = "grabbing"; });
  canvas.addEventListener("pointermove", (event) => {
    if (state.panning && state.last) {
      state.view.tx += event.clientX - state.last.x; state.view.ty += event.clientY - state.last.y;
      state.last = { x: event.clientX, y: event.clientY }; state.userView = true; draw();
    } else if (state.decoded) {
      const p = toImage(event.clientX, event.clientY);
      const x = Math.floor(p.x), y = Math.floor(p.y);
      const d = state.decoded;
      if (x >= 0 && y >= 0 && x < d.width && y < d.height) {
        const i = y * d.width + x;
        const bits = [`x ${x} y ${y}`];
        if (d.probability) bits.push(`p ${(d.probability[i] / 255).toFixed(2)}`);
        if (d.mask_final) bits.push(d.mask_final[i] ? "mask" : "");
        if (d.tube) bits.push(d.tube[i] ? "tube" : "");
        canvas.title = bits.filter(Boolean).join(" · ");
      }
    }
  });
  canvas.addEventListener("pointerup", () => { state.panning = false; state.last = null; canvas.style.cursor = "grab"; });
  // Panels resize the stage and the charts without a window resize.
  let pending = null;
  const relayout = () => {
    if (pending) return;
    pending = requestAnimationFrame(() => { pending = null; resizeCanvas(); drawCharts(); if (state.frame) { drawWidthChart(); drawCurvatureChart(); } });
  };
  new ResizeObserver(relayout).observe($("#stage"));
  new ResizeObserver(relayout).observe($("#charts"));
  new ResizeObserver(relayout).observe($("#details"));
  window.addEventListener("resize", relayout);
  initSplitters();

  window.addEventListener("keydown", (event) => {
    const tag = document.activeElement && document.activeElement.tagName;
    if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA") { if (event.key === "Escape") document.activeElement.blur(); return; }
    const big = event.ctrlKey ? 100 : event.shiftKey ? 10 : 1;
    switch (event.key) {
      case "ArrowLeft": step(-big); event.preventDefault(); break;
      case "ArrowRight": step(big); event.preventDefault(); break;
      case " ": togglePlay(); event.preventDefault(); break;
      case "[": jump("flag", -1); break;
      case "]": jump("flag", 1); break;
      case ",": jump("iou", -1); break;
      case ".": jump("iou", 1); break;
      case "f": case "F": $("#toggle-raw").click(); break;
      case "0": fitView(); break;
      case "n": case "N": $("#note-comment").focus(); event.preventDefault(); break;
      case "s": case "S": computeStarts(); break;
      default:
        if (/^[1-9]$/.test(event.key)) toggleLayer(parseInt(event.key, 10) - 1);
    }
  });
}

async function boot() {
  renderLayers();
  renderNoteTags();
  bindEvents();
  resizeCanvas();
  try {
    state.info = await api("/api/state");
    state.runs = state.info.runs;
    renderRunList();
    await loadNotes();
    const errors = Object.entries(state.info.errors || {});
    if (errors.length) setStatus(`${errors.length} run directories skipped: ${errors.map(([p, e]) => `${p.split("/").pop()} (${e})`).join("; ")}`, "error");
    const wanted = parseHash();
    const initial = wanted.run && state.runs.some((r) => r.name === wanted.run) ? wanted.run : (state.runs[0] && state.runs[0].name);
    if (initial) await selectRun(initial, wanted.frame);
    else setStatus("no runs found; start the viewer with --run or --runs-root", "error");
  } catch (error) {
    setStatus(error.message, "error");
  }
}

boot();
