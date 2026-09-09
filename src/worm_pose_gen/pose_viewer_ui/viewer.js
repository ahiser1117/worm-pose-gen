"use strict";

// Source selection (workspaces and read-only runs), the two-tier frame
// loading that keeps scrubbing smooth, the compare source, review notes,
// the URL hash and the fitter starts.

// ---------------------------------------------------------------- source list

// A workspace whose files could not be read is listed with its error
// (rows shaped {name, path, kind, error}, no frames or recording).
function sourceOptionText(kind, entry) {
  if (entry.error) return `${entry.name} — unreadable`;
  if (kind === "workspace") {
    const summary = entry.summary || {};
    const iou = summary.iou && summary.iou.median !== undefined && summary.iou.median !== null ? ` · IoU ${fmt(summary.iou.median)}` : "";
    const fitted = summary.fitted !== undefined ? ` · ${summary.fitted}/${entry.frame_count} fit` : "";
    return `${entry.name}${iou}${fitted}`;
  }
  const iou = entry.iou_median === null || entry.iou_median === undefined ? "" : ` · IoU ${entry.iou_median.toFixed(3)}`;
  return `${entry.name}${iou}`;
}

function sourceOptionTitle(kind, entry) {
  if (entry.error) return `${entry.path || entry.name}\n${entry.error}`;
  const frames = entry.frames || [null, null];
  if (kind === "workspace") return `${entry.recording} frames ${frames[0]}–${frames[1]} step ${entry.step} · created ${fmtTime(entry.created_at)}${entry.imported_runs && entry.imported_runs.length ? ` · imported ${entry.imported_runs.join(", ")}` : ""}`;
  return `${entry.recording} frames ${frames[0]}–${frames[1]} · ${entry.mask_cleanup} · checkpoint ${entry.checkpoint_sha}`;
}

function renderRunList() {
  const filter = $("#run-filter").value.trim().toLowerCase();
  const select = $("#run");
  select.innerHTML = "";
  const groups = [["workspace", "Workspaces", state.workspaces], ["run", "Runs (read-only)", state.runs]];
  const current = currentSourceKey();
  for (const [kind, label, entries] of groups) {
    if (!entries.length && kind === "run") continue;
    const group = document.createElement("optgroup");
    group.label = entries.length ? `${label} (${entries.length})` : `${label} — none yet`;
    for (const entry of entries) {
      const haystack = `${entry.name} ${entry.recording || ""}`.toLowerCase();
      if (filter && !haystack.includes(filter)) continue;
      const option = document.createElement("option");
      option.value = sourceKey(kind, entry.name);
      option.textContent = sourceOptionText(kind, entry);
      option.title = sourceOptionTitle(kind, entry);
      option.disabled = !!entry.error;
      if (option.value === current) option.selected = true;
      group.appendChild(option);
    }
    select.appendChild(group);
  }
  $("#import-run").hidden = state.sourceKind !== "run";
}

function describeSource(run) {
  const entry = run.entry;
  const lines = [`${entry.recording}  frames ${entry.frames[0]}–${entry.frames[1]} (${entry.frame_count}, step ${entry.step})`];
  if (run.kind === "workspace") {
    const summary = run.workspace_summary || {};
    const prov = summary.provenance || {};
    lines.push(`workspace · ${fmt(summary.fitted)} fitted · masks ${summary.has_masks ? fmt(summary.mask_rows) + " rows" : "none"} · hypotheses ${summary.has_hypotheses ? "yes" : "no"} · prior ${summary.has_prior ? "yes" : "no"}`);
    // Live counts from the provenance block when the payload has one, else the summary's.
    let counts = "";
    if (run.provenance && run.provenance.algorithms.length) {
      const tally = new Map();
      for (const k of run.provenance.index) if (k >= 0) tally.set(k, (tally.get(k) || 0) + 1);
      counts = run.provenance.algorithms.map((id, k) => `${id} ${tally.get(k) || 0}`).join(", ");
      const edited = run.provenance.edited.reduce((a, v) => a + (v ? 1 : 0), 0);
      if (edited) counts += ` · ${edited} rows manually edited`;
    } else counts = Object.entries(prov).filter(([, v]) => v).map(([k, v]) => `${k} ${v}`).join(", ");
    if (counts) lines.push(`provenance: ${counts}`);
    const edits = run.edits_count !== undefined ? run.edits_count : summary.edits;
    if (edits !== undefined) lines.push(`edits ${edits} · snapshots ${(summary.snapshots || []).length}${(entry.imported_runs || []).length ? ` · imported ${entry.imported_runs.join(", ")}` : ""}`);
  } else {
    lines.push(`mask: ${entry.mask_cleanup} · threshold ${run.threshold !== undefined ? run.threshold : "?"} · segmenter ${entry.checkpoint_sha || "?"}`);
    lines.push(`IoU median ${fmt(entry.iou_median)} · min ${fmt(entry.iou_min)} · below 0.9: ${fmt(entry["frames_below_0.9"])} · ${entry.propagated ? "propagated" : "independent only"} · git ${entry.git_commit}`);
  }
  if (run.recording_readable === false) lines.push(`recording not readable: ${run.recording_error}`);
  return lines.join("\n");
}

function runSummaryText(run) {
  const s = run.summary_iou || {};
  const l = run.summary_length || {};
  const p = run.propagation;
  const a = run.ambiguity || {};
  const prior = run.prior;
  const cleanup = run.cleanup || {};
  const lines = [];
  if (!(run.series && run.series.fitted && run.series.fitted.some(Boolean))) lines.push("no fitted frames yet — run the segment, prior and fit stages from the Pipeline tab");
  lines.push(`IoU median ${fmt(s.median)} · p10 ${fmt(s.p10)} · min ${fmt(s.min)} · ≥0.9: ${fmt(s["fraction_at_least_0.9"])}`);
  lines.push(`length median ${fmt(l.median, 0)} px (p10 ${fmt(l.p10, 0)}, p90 ${fmt(l.p90, 0)}) · beyond prior 2σ: ${fmt(l.beyond_2_sigma_of_prior)}`);
  if (prior) lines.push(`prior: length ${fmt(prior.length_px, 0)} ± ${fmt(100 * prior.log_length_sigma, 0)}% · width ${fmt(prior.width_px, 1)} px · from ${prior.frames_used}/${prior.frames_candidates} frames`);
  if (a.flag_counts) lines.push("flags: " + Object.entries(a.flag_counts).filter(([, v]) => v).map(([k, v]) => `${k} ${v}`).join(", "));
  if (a.frames_with_score_at_least_2 !== undefined) lines.push(`score ≥ 2: ${a.frames_with_score_at_least_2} frames · score ≥ 1: ${a.frames_with_score_at_least_1}`);
  if (p && p.stretches) lines.push(`propagation: ${p.stretches.length} stretches, ${p.frames_in_stretches} frames, ${p.frames_replaced} replaced (fwd ${p.replaced_by_source && p.replaced_by_source.forward}, bwd ${p.replaced_by_source && p.replaced_by_source.backward}); stretch IoU ${fmt(p.stretch_iou_median_before)} → ${fmt(p.stretch_iou_median_after)}`);
  if (p && p.prediction_damping !== undefined) lines.push(`6a: damping ${p.prediction_damping}, temporal prior ${p.temporal_prior_weight} at ${p.temporal_prior_sigma_widths} widths; predicted starts won ${p.predicted_starts_won} of ${p.predicted_starts_offered}`);
  if (p && p.path) lines.push(`6c: refit ${p.refit_preset}, beam ${p.beam}, path ${p.path.enabled ? `T ${p.path.temperature}, distance ${p.path.distance_weight}, in-view ${p.path.inview_weight}` : "off"}; overrode lowest energy on ${p.path.frames_overriding_lowest_energy} frames, mirrored ${p.path.frames_mirrored}; replaced by ${JSON.stringify(p.replaced_by_source)}`);
  const t = run.track_length;
  if (t) lines.push(`6b: track length p50 ${fmt(t.track_length_p10_p50_p90 && t.track_length_p10_p50_p90[1], 0)} px (window ±${t.window}); ${t.frames_refit} frames refit (${t.frames_clipped} clipped, ${t.frames_deviating} off track) in ${fmt(t.seconds, 0)} s${p && p.jump_seeds !== undefined && p.jump_seeds !== null ? `; jump seeds ${p.jump_seeds} (${p.jump_seeds_below_score} below the score threshold)` : ""}`);
  const c = run.continuity;
  if (c) lines.push(`continuity: length jumps > ${Math.round(100 * c.length_jump_fraction)}%: ${c.length_jumps_over_fraction} · pose jumps > width: ${c.pose_jumps_over_width} · in-view changes: ${c.in_view_changes}` + (c.prediction_distance_px_p50_p90_max ? ` · distance to prediction p50/p90 ${fmt(c.prediction_distance_px_p50_p90_max[0], 1)} / ${fmt(c.prediction_distance_px_p50_p90_max[1], 1)} px` : ""));
  if (cleanup.fill_holes !== undefined) lines.push(`cleanup: fill holes ${cleanup.fill_holes} (r ${cleanup.hole_radius}) · largest only ${cleanup.largest_only}`);
  if (run.has_independent_pose === false) lines.push("independent pose not stored in this run (older fitter); only its IoU/score are shown");
  return lines.join("\n");
}

// A workspace payload carries the run shape plus workspace fields; fill in
// whatever an older or thinner server leaves out so the viewer code can rely
// on entry, series, stretches and cleanup being present.
function normaliseSourcePayload(payload, kind, name) {
  payload.kind = kind;
  const info = payload.info || payload.workspace || workspaceEntry(name) || {};
  const summary = payload.summary || (info && info.summary) || (workspaceEntry(name) || {}).summary || {};
  if (kind === "workspace") {
    payload.workspace_info = info;
    payload.workspace_summary = payload.workspace_summary || summary;
    if (!payload.provenance && payload.workspace_summary.provenance === undefined && summary.provenance) payload.workspace_summary.provenance = summary.provenance;
  }
  if (!payload.entry) {
    const frames = info.frames || [null, null];
    const iou = payload.summary_iou || summary.iou || {};
    payload.entry = {
      name, path: info.path || "", recording: recordingStem(info.recording || payload.recording), recording_path: info.recording || payload.recording || "",
      frames, step: info.step || 1, frame_count: info.frame_count || (payload.series && payload.series.frame_index ? payload.series.frame_index.length : 0),
      started_at: info.created_at || null, preset: null, iou_median: iou.median === undefined ? null : iou.median, iou_min: iou.min === undefined ? null : iou.min,
      "frames_below_0.9": iou["frames_below_0.9"] === undefined ? null : iou["frames_below_0.9"],
      mask_cleanup: payload.cleanup ? (payload.cleanup.fill_holes ? "fill" : "no fill") + " + " + (payload.cleanup.largest_only ? "largest" : "all components") : "",
      propagated: !!payload.propagation, checkpoint_sha: "", checkpoint_path: null, git_commit: "",
    };
  }
  if (!payload.series || !payload.series.frame_index) {
    // A fresh workspace with no state yet: rows are the frame range, nothing fitted.
    const [first, last] = payload.entry.frames;
    const step = payload.entry.step || 1;
    const index = [];
    if (first !== null && last !== null) for (let f = first; f <= last; f += step) index.push(f);
    payload.series = { frame_index: index, fitted: index.map(() => 0), flags: {} };
  }
  if (!payload.series.fitted) payload.series.fitted = payload.series.frame_index.map(() => 0);
  if (!payload.series.flags) payload.series.flags = {};
  payload.stretches = payload.stretches || [];
  payload.cleanup = payload.cleanup || {};
  if (payload.threshold === undefined) payload.threshold = null;
  normaliseProvenance(payload);
  return payload;
}

// Sources of the same recording (other runs and workspaces) that can be
// compared; the payload's compatible_runs list when the server computed it.
function compareCandidates(run) {
  const out = [];
  const seen = new Set();
  for (const other of run.compatible_runs || []) {
    const key = sourceKey(other.kind === "workspace" ? "workspace" : "run", other.name);
    seen.add(key);
    out.push({ key, text: `${other.overlap ? "" : "(no overlap) "}${other.name} · ${other.frames[0]}–${other.frames[1]} · IoU ${fmt(other.iou_median)}${other.mask_cleanup ? " · " + other.mask_cleanup : ""}` });
  }
  const stem = run.entry.recording;
  const current = currentSourceKey();
  for (const w of state.workspaces) {
    const key = sourceKey("workspace", w.name);
    if (w.error || !w.frames || key === current || seen.has(key) || recordingStem(w.recording) !== stem) continue;
    out.push({ key, text: `${w.name} · ${w.frames[0]}–${w.frames[1]} (workspace)` });
  }
  for (const r of state.runs) {
    const key = sourceKey("run", r.name);
    if (!r.frames || key === current || seen.has(key) || r.recording !== stem) continue;
    out.push({ key, text: `${r.name} · ${r.frames[0]}–${r.frames[1]} · IoU ${fmt(r.iou_median)} · ${r.mask_cleanup}` });
  }
  return out;
}

// ``options.keepView`` (a reload of the source shown) keeps the timeline range
// and the view when the frame count is unchanged.  A selection that another
// selection overtook while its payload was in flight is dropped, so a slow
// reload never brings back the source the user just left.
async function selectSource(kind, name, frameIndex, options = {}) {
  if (!name) return;
  const selection = ++loads.selection;
  setLoading(1);
  try {
    const run = normaliseSourcePayload(await api(sourcePayloadUrl(kind, name)), kind, name);
    if (selection !== loads.selection) return;
    const sameShape = !!options.keepView && state.run !== null && state.sourceKind === kind && state.runName === name
      && state.run.series.frame_index.length === run.series.frame_index.length;
    const timeline = sameShape ? { ...state.timeline, dragging: false } : null;
    abortLoad("light"); abortLoad("full"); abortLoad("compare"); abortLoad("prefetch"); cancelPendingFull();
    loads.lightWanted = null;
    loads.generation++;  // frame payloads still in flight belong to the previous selection
    state.run = run;
    state.runName = name;
    state.sourceKind = kind;
    state.frameCache.clear();
    state.frame = null; state.decoded = null;
    state.starts = null;
    state.compare = null; state.compareName = null; state.compareKind = null; state.comparePose = null;
    state.segments.clear();
    state.edits = []; state.editsError = null;
    $("#run-info").textContent = describeSource(run);
    $("#run-summary").textContent = runSummaryText(run);
    renderRunList();
    const compare = $("#compare");
    compare.innerHTML = '<option value="">none</option>';
    for (const candidate of compareCandidates(run)) {
      const option = document.createElement("option");
      option.value = candidate.key;
      option.textContent = candidate.text;
      compare.appendChild(option);
    }
    populateWorst();
    state.timeline = timeline || { start: 0, end: run.series.frame_index.length, dragging: false };
    buildCharts();
    renderNotes();
    renderEditControls();
    loadEdits();
    if (typeof onSourceChanged === "function") onSourceChanged();
    let row = 0;
    if (frameIndex !== undefined && frameIndex !== null) {
      const found = run.series.frame_index.indexOf(frameIndex);
      row = found >= 0 ? found : 0;
    }
    if (!run.series.frame_index.length) { setStatus("this source has no frames", "error"); draw(); return; }
    await showRow(row, { fit: !sameShape, keepView: sameShape });
  } catch (error) {
    setStatus(error.message, "error");
  } finally {
    setLoading(-1);
  }
}

// Backwards-compatible entry point (run directories).
function selectRun(name, frameIndex) { return selectSource("run", name, frameIndex); }

// Re-read the current source after a job wrote to it, keeping the cursor,
// the view and the compare source.
async function reloadSource() {
  if (!state.runName) return;
  const kind = state.sourceKind, name = state.runName;
  const frame = state.run && state.run.series.frame_index[state.row];
  const compare = state.compareName ? sourceKey(state.compareKind, state.compareName) : "";
  await selectSource(kind, name, frame, { keepView: true });
  if (state.sourceKind !== kind || state.runName !== name) return;  // the user moved on meanwhile
  if (compare) { $("#compare").value = compare; selectCompare(compare); }
}

async function selectCompare(key) {
  const source = parseSourceKey(key);
  state.compareName = source ? source.name : null;
  state.compareKind = source ? source.kind : null;
  state.compare = null; state.compareRows = null; state.comparePose = null;
  if (source) {
    try {
      const run = normaliseSourcePayload(await api(sourcePayloadUrl(source.kind, source.name)), source.kind, source.name);
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
  if (series.iou) for (let r = 0; r < series.iou.length; r++) if (series.fitted[r] && series.iou[r] !== null) rows.push(r);
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

const loads = {
  light: null, lightRow: null, lightWanted: null, full: null, compare: null, prefetch: null, fullTimer: null, fullResolve: null,
  prefetching: false, seq: 0, applied: 0,
  selection: 0,   // bumped by every selectSource call; a payload for an older selection is dropped
  generation: 0,  // bumped when the shown source changes; frame payloads of an older generation are never cached
};
const FULL_DELAY_MS = 150;
const CACHE_ENTRIES = 24;

function setLoading(delta) {
  state.loading = Math.max(0, state.loading + delta);
  updateLoadingIndicator();
}

function thresholdParam() { return $("#threshold-on").checked ? $("#threshold").value : ""; }

// The key names the source and its generation, so a payload fetched for one
// selection can never be served for another with the same row number.
function cacheKey(row, detail, raw) { return `${state.sourceKind}|${state.runName}|${loads.generation}|${row}|${thresholdParam()}|${detail}|${raw ? 1 : 0}`; }

function frameUrl(row, detail, raw) {
  const frame = state.run.series.frame_index[row];
  const threshold = thresholdParam() ? `&threshold=${thresholdParam()}` : "";
  return sourceUrl(state.sourceKind, state.runName, "frame", `frame=${frame}&detail=${detail}${threshold}${raw ? "&raw=1" : ""}`);
}

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
  const generation = loads.generation;
  const key = cacheKey(row, detail, raw);
  payload = await fetchJson(frameUrl(row, detail, raw), controller);
  // The source changed (or was reloaded) while the request was out: the payload describes the old one.
  if (generation !== loads.generation) throw new DOMException("source changed while loading", "AbortError");
  state.frameCache.set(key, payload);
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
  if (row === state.row) fetchSegment(row);
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
  fetchJson(sourceUrl(state.compareKind, state.compareName, "pose", `frame=${frame}`), controller)
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
      const controller = new AbortController();
      loads.prefetch = controller;
      let payload;
      try {
        payload = await loadTier(target, "full", controller);
      } finally {
        if (loads.prefetch === controller) loads.prefetch = null;
      }
      if (state.runName !== runName) break;
      await applyPayload(payload, -1);  // decode ahead of time; never shown from here
    }
  } catch (error) {
    if (!isAbort(error)) console.warn("prefetch", error);
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
  if (!state.run || !state.run.series.frame_index.length) return Promise.resolve(false);
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
    state.notes = payload.notes || [];
    $("#notes-path").textContent = `notes file: ${payload.path}`;
    renderNotes();
  } catch (error) { setStatus(error.message, "error"); }
}

// A note names the source it was taken on; open it as a workspace or a run,
// whichever the catalog knows.
function sourceOfNote(note) {
  if (note.workspace && workspaceEntry(note.workspace)) return { kind: "workspace", name: note.workspace };
  if (workspaceEntry(note.run)) return { kind: "workspace", name: note.run };
  if (runEntry(note.run)) return { kind: "run", name: note.run };
  return null;
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
    item.innerHTML = `<span><b>${note.frame_index}</b> ${escapeHtml((note.tags || []).join(", "))}${note.comment ? " — " + escapeHtml(note.comment) : ""}<div class="meta">${escapeHtml(note.run)}</div></span><span><button type="button" title="delete">×</button></span>`;
    item.addEventListener("click", async (event) => {
      if (event.target.tagName === "BUTTON") {
        event.stopPropagation();
        try {
          const payload = await post("/api/note/delete", { index });
          state.notes = payload.notes; renderNotes();
        } catch (error) { setStatus(error.message, "error"); }
        return;
      }
      const source = sourceOfNote(note);
      if (source && !(source.kind === state.sourceKind && source.name === state.runName)) await selectSource(source.kind, source.name, note.frame_index);
      else if (state.run) { const r = state.run.series.frame_index.indexOf(note.frame_index); if (r >= 0) showRow(r, { keepView: true }); }
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
    const frame = state.run.series.frame_index[state.row];
    const body = { run: state.runName, frame_index: frame, tags, comment, recording: state.run.entry.recording };
    if (isWorkspace()) body.workspace = state.runName;
    const payload = await post("/api/note", body);
    state.notes = payload.notes;
    $("#note-comment").value = "";
    renderNotes();
    setStatus(`noted frame ${frame}`, "ok");
  } catch (error) { setStatus(error.message, "error"); }
}

// ---------------------------------------------------------------- hash and starts

let hashTimer = null;
function updateHash() {
  if (hashTimer) return;
  hashTimer = setTimeout(() => {
    hashTimer = null;
    if (!state.run) return;
    const frame = state.run.series.frame_index[state.row];
    const key = isWorkspace() ? "ws" : "run";
    history.replaceState(null, "", `#${key}=${encodeURIComponent(state.runName)}&frame=${frame}`);
  }, 250);
}

function parseHash() {
  const params = new URLSearchParams(location.hash.replace(/^#/, ""));
  return { run: params.get("run"), ws: params.get("ws"), frame: params.has("frame") ? parseInt(params.get("frame"), 10) : null };
}

async function computeStarts() {
  if (!state.run) return;
  setLoading(1);
  try {
    const threshold = $("#threshold-on").checked ? `&threshold=${$("#threshold").value}` : "";
    const payload = await api(sourceUrl(state.sourceKind, state.runName, "starts", `frame=${state.run.series.frame_index[state.row]}${threshold}`));
    state.starts = payload.starts;
    setStatus(`${payload.starts.length} starts: ${payload.starts.map((s) => s.name).join(", ")}`, "ok");
    renderLegend(); renderLayerAvailability(); draw();
  } catch (error) { setStatus(error.message, "error"); } finally { setLoading(-1); }
}
