"use strict";

// Phase 2 interventions: picking a stored hypothesis, flipping the
// orientation of a frame or a segment, undoing through the edit log, and
// the provenance display (which algorithm produced each row, which rows a
// manual edit touched).  Everything goes through /api/workspaces/<name>/...;
// a read-only run or the stdlib viewer shows the controls disabled.

const PROVENANCE_COLORS = {
  independent_fit: "#3d8f6b",
  chain_forward: "#ffaa3c",
  chain_backward: "#c080ff",
  track_length_refit: "#50beff",
  manual: "#ff5da8",
};
const PROVENANCE_PALETTE = ["#e0b04d", "#57d68d", "#80c8ff", "#ff70ff", "#c8e070", "#70e0d0", "#e08870", "#a0a0ff"];
const ALGORITHM_SHORT = {
  independent_fit: "independent fit",
  chain_forward: "forward chain",
  chain_backward: "backward chain",
  track_length_refit: "track refit",
  "manual:pick": "manual pick",
  "manual:flip": "manual flip",
};
const EDIT_KIND_NAMES = { pick_hypothesis: "pick", flip_orientation: "flip", accept_path: "accept path", set_pose: "set pose", undo: "undo" };

function isManualAlgorithm(id) { return typeof id === "string" && id.startsWith("manual"); }

// Manual edits share one colour; every other algorithm gets a fixed colour
// when known, else one from the palette by its position in the run's list.
function provenanceColor(id, position) {
  if (!id) return "#1a242c";
  if (isManualAlgorithm(id)) return PROVENANCE_COLORS.manual;
  if (PROVENANCE_COLORS[id]) return PROVENANCE_COLORS[id];
  return PROVENANCE_PALETTE[Math.max(0, position || 0) % PROVENANCE_PALETTE.length];
}

function shortAlgorithm(id) {
  if (!id) return "";
  if (ALGORITHM_SHORT[id]) return ALGORITHM_SHORT[id];
  const bare = String(id).replace(/^[a-z_]+:/, "").replace(/_/g, " ");
  return bare.length > 22 ? bare.slice(0, 21) + "…" : bare;
}

// The run payload's provenance in one shape: {algorithms, index, edited,
// algorithm (per row id)}.  Older servers send only the per-row arrays.
function normaliseProvenance(payload) {
  const n = payload.series && payload.series.frame_index ? payload.series.frame_index.length : 0;
  let prov = payload.provenance;
  if (!prov || typeof prov !== "object" || Array.isArray(prov)) prov = {};
  let algorithms = Array.isArray(prov.algorithms) ? prov.algorithms.slice() : null;
  let index = Array.isArray(prov.index) ? prov.index.slice() : null;
  let perRow = Array.isArray(prov.algorithm) ? prov.algorithm : null;
  if (!index && perRow) {
    algorithms = algorithms || [];
    index = perRow.map((id) => {
      if (!id) return -1;
      let k = algorithms.indexOf(id);
      if (k < 0) { algorithms.push(id); k = algorithms.length - 1; }
      return k;
    });
  }
  if (!perRow && index && algorithms) perRow = index.map((k) => (k >= 0 && k < algorithms.length ? algorithms[k] : ""));
  const edited = Array.isArray(prov.edited) ? prov.edited.slice() : Array.from({ length: n }, () => 0);
  if (!index) { payload.provenance = null; return payload; }
  payload.provenance = { ...prov, algorithms: algorithms || [], index, algorithm: perRow, edited };
  return payload;
}

function editsSupported() { return isWorkspace() && serverIsApp(); }

function editsUrl(endpoint, query) {
  return `/api/workspaces/${encodeURIComponent(state.runName)}/${endpoint}${query ? "?" + query : ""}`;
}

// ---------------------------------------------------------------- edit log

async function loadEdits() {
  if (!editsSupported()) { state.edits = []; renderEdits(); return; }
  const name = state.runName;
  try {
    const payload = await api(editsUrl("edits"));
    if (state.runName !== name) return;
    const list = Array.isArray(payload) ? payload : payload.edits || [];
    // A raw log (older server) is oldest first without the flags; shape it.
    const shaped = list.length && list[0].undoable !== undefined;
    state.edits = shaped ? list : list.map(rawEditEntry).reverse();
    state.editsError = null;
  } catch (error) {
    state.edits = [];
    state.editsError = error.message;
  }
  renderEdits();
}

function rawEditEntry(e) {
  const payload = e.payload || {};
  const frames = payload.frames || [];
  return { id: e.id, kind: e.kind, time: e.time, rows: (payload.rows || []).length, frames: frames.length ? [Math.min(...frames), Math.max(...frames)] : null, note: payload.note || "", undone: false, undoable: false, summary: payload.summary || {} };
}

function newestUndoable() { return (state.edits || []).find((e) => e.undoable) || null; }

function editLabel(e) {
  const kind = EDIT_KIND_NAMES[e.kind] || e.kind;
  if (e.kind === "undo" && e.undoes) return `undo ${e.undoes}`;
  return kind;
}

function editFrames(e) {
  if (!e.frames) return "";
  return e.frames[0] === e.frames[1] ? `frame ${e.frames[0]}` : `frames ${e.frames[0]}–${e.frames[1]}${e.rows > 1 ? ` (${e.rows})` : ""}`;
}

function editSummaryText(e) {
  const changes = (e.summary && e.summary.changes) || [];
  const c = changes[0];
  if (!c) return "";
  const parts = [];
  if (c.before && c.after && c.before.iou !== c.after.iou) parts.push(`IoU ${fmt(c.before.iou)} → ${fmt(c.after.iou)}`);
  if (c.before && c.after && c.before.reversed !== c.after.reversed) parts.push("orientation reversed");
  if (c.before && c.after && c.before.algorithm !== c.after.algorithm) parts.push(`${c.before.algorithm || "–"} → ${c.after.algorithm || "–"}`);
  return parts.join(" · ") + (changes.length > 1 ? ` · +${changes.length - 1} rows` : "");
}

function renderEdits() {
  const list = $("#edits-list");
  const count = $("#edits-count");
  if (!list) return;
  const edits = state.edits || [];
  const live = edits.filter((e) => e.kind !== "undo" && !e.undone).length;
  count.textContent = editsSupported() ? `${live} live · ${edits.length} entries` : "";
  const undoButton = $("#edit-undo");
  const target = newestUndoable();
  undoButton.disabled = !editsSupported() || !target;
  undoButton.title = target ? `undo ${editLabel(target)} ${editFrames(target)} (Ctrl+Z)` : editsSupported() ? "nothing to undo" : "edits need a workspace on the app server";
  list.innerHTML = "";
  if (!editsSupported()) { list.innerHTML = `<div class="empty">${state.sourceKind === "run" ? "Read-only run: import it as a workspace to edit." : "Edits need a workspace on the app server."}</div>`; return; }
  if (state.editsError) { list.innerHTML = `<div class="empty">${escapeHtml(state.editsError)}</div>`; return; }
  if (!edits.length) { list.innerHTML = '<div class="empty">No edits yet. Pick a hypothesis or flip a frame.</div>'; return; }
  for (const e of edits) {
    const item = document.createElement("div");
    item.className = `item edit ${e.kind}${e.undone ? " undone" : ""}`;
    item.dataset.edit = e.id;
    const summary = editSummaryText(e);
    item.innerHTML = `<span><b>${escapeHtml(editLabel(e))}</b> ${escapeHtml(editFrames(e))}${e.note ? " — " + escapeHtml(e.note) : ""}<div class="meta">${escapeHtml(e.id)} · ${escapeHtml(fmtTime(e.time))}${e.algorithm && !isManualAlgorithm(e.algorithm) ? " · " + escapeHtml(e.algorithm) : ""}${summary ? " · " + escapeHtml(summary) : ""}${e.undone ? " · undone" : ""}</div></span><span>${target && e.id === target.id ? '<button type="button" title="undo this edit (Ctrl+Z)">Undo</button>' : ""}</span>`;
    item.addEventListener("click", (event) => {
      if (event.target.tagName === "BUTTON") { event.stopPropagation(); undoEdit(e.id); return; }
      if (e.frames && state.run) {
        const current = state.run.series.frame_index[state.row];
        const wanted = current >= e.frames[0] && current <= e.frames[1] ? current : e.frames[0];
        const r = state.run.series.frame_index.indexOf(wanted);
        if (r >= 0) showRow(r, { keepView: true });
      }
    });
    list.appendChild(item);
  }
}

// ---------------------------------------------------------------- segments

// The propagation stretch (or fitted run outside every stretch) around a
// row, from /segment; cached per row until an edit or a reload.  A row is
// requested once (the light and the full frame tier both land here: the
// in-flight request sits in the cache as {pending: true}) and only once
// the row has rested for a moment, so scrubbing does not queue a request
// per frame passed.
const SEGMENT_REST_MS = 150;
let segmentTimer = null;

function fetchSegment(row) {
  if (!editsSupported() || !state.run) { renderSegmentInfo(); return; }
  if (state.segments.has(row)) { renderSegmentInfo(); return; }
  renderSegmentInfo();
  clearTimeout(segmentTimer);
  segmentTimer = setTimeout(() => {
    if (row !== state.row || !state.run || state.segments.has(row)) return;
    const name = state.runName;
    const frame = state.run.series.frame_index[row];
    state.segments.set(row, { pending: true });
    api(editsUrl("segment", `frame=${frame}`))
      .then((payload) => { if (state.runName !== name) return; state.segments.set(row, payload); if (row === state.row) renderSegmentInfo(); })
      .catch((error) => { if (state.runName !== name) return; state.segments.set(row, { error: error.message }); if (row === state.row) renderSegmentInfo(); });
  }, SEGMENT_REST_MS);
}

function currentSegment() {
  const seg = state.segments.get(state.row);
  return !seg || seg.pending ? null : seg;
}

function renderSegmentInfo() {
  const note = $("#segment-info");
  const flipFrame = $("#flip-frame"), flipSegment = $("#flip-segment");
  if (!note) return;
  const fitted = state.frame && state.frame.stats && state.frame.stats.fitted;
  const enabled = editsSupported() && !!fitted;
  flipFrame.disabled = !enabled;
  flipSegment.disabled = !enabled;
  const why = !editsSupported() ? (state.sourceKind === "run" ? "read-only run: import it as a workspace to edit" : "edits need a workspace on the app server") : !fitted ? "no fit on this frame" : "";
  flipFrame.title = why || "reverse head and tail of this frame's pose";
  const seg = currentSegment();
  if (!editsSupported()) { note.textContent = why; flipSegment.title = why; return; }
  if (!seg) { note.textContent = "segment: …"; flipSegment.title = why || "reverse head and tail over the segment"; return; }
  if (seg.error) { note.textContent = `segment: ${seg.error}`; flipSegment.title = seg.error; return; }
  const n = seg.rows[1] - seg.rows[0] + 1;
  const text = `segment frames ${seg.frames[0]}–${seg.frames[1]} (${n} row${n === 1 ? "" : "s"}) · ${seg.in_stretch ? "propagation stretch" : "outside stretches"}`;
  note.textContent = text;
  flipSegment.title = why || `reverse head and tail on ${text}`;
}

// ---------------------------------------------------------------- hypotheses table

function renderHypothesesTable() {
  const table = $("#hypotheses");
  const head = $("#hypotheses-head");
  const section = $("#hypotheses-section");
  const f = state.frame;
  const pose = f && f.pose;
  if (!pose || !pose.hypotheses || !pose.hypotheses.length) { section.hidden = true; table.innerHTML = ""; return; }
  section.hidden = false;
  const path = pose.path || {};
  head.textContent = `${pose.hypotheses.length} candidates${path.mirrored ? " · chosen mirrored" : ""}${path.override ? ` · path overrode lowest energy by ${fmt(path.energy_gap, 4)}` : ""}`;
  const canEdit = editsSupported();
  const ranked = pose.hypotheses.slice().sort((a, b) => a.energy - b.energy);
  const rows = ['<tr><th></th><th>candidate</th><th class="num">energy</th><th class="num">IoU</th><th>start</th><th></th></tr>'];
  for (const h of ranked) {
    const color = `rgb(${(HYP_COLORS[h.source] || [200, 200, 200]).join(",")})`;
    const name = `${h.source}${h.source === "independent" ? "" : " #" + (h.beam + 1)}`;
    const start = (h.start || "").replace(/_(forward|backward)$/, "").replace("independent_refit", "refit");
    const asIs = h.chosen && !path.mirrored, mirrored = h.chosen && path.mirrored;
    const use = asIs ? '<span class="current">current</span>' : `<button type="button" data-pick="${h.index}" data-mirrored="0" title="make this candidate the frame's pose" ${canEdit ? "" : "disabled"}>use</button>`;
    const useMirrored = mirrored ? '<span class="current">current ↔</span>' : `<button type="button" data-pick="${h.index}" data-mirrored="1" title="make this candidate the frame's pose with head and tail swapped" ${canEdit ? "" : "disabled"}>use ↔</button>`;
    rows.push(`<tr class="${h.chosen ? "chosen" : ""}"><td><span class="swatch" style="background:${color}"></span></td><td>${h.chosen ? "✓ " : ""}${escapeHtml(name)}</td><td class="num">${fmt(h.energy, 4)}</td><td class="num">${fmt(h.iou)}</td><td class="dim">${escapeHtml(start)}</td><td class="actions">${use}${useMirrored}</td></tr>`);
  }
  table.innerHTML = rows.join("");
  for (const button of table.querySelectorAll("button[data-pick]")) {
    button.addEventListener("click", () => pickHypothesis(parseInt(button.dataset.pick, 10), button.dataset.mirrored === "1"));
  }
  $("#hypotheses-note").textContent = canEdit ? "" : state.sourceKind === "run" ? "read-only run: import it as a workspace to pick" : "picking needs a workspace on the app server";
}

// ---------------------------------------------------------------- submitting edits

// One edit at a time, whatever submits it (a pick, a flip, an undo here; a
// candidate set accept in regions.js): the server applies edits in order, but
// a second click while the first is in flight would be a second edit.
let editInFlight = false;

function claimEdit() {
  if (editInFlight) { setStatus("an edit is still being applied", "error"); return false; }
  editInFlight = true;
  return true;
}

function releaseEdit() { editInFlight = false; }

function editPending() { return editInFlight; }

async function submitEdit(body, describe) {
  if (!editsSupported()) { setStatus("edits need a workspace on the app server", "error"); return null; }
  if (!claimEdit()) return null;
  const name = state.runName;
  setLoading(1);
  try {
    const payload = await post(editsUrl("edits"), body);
    if (state.runName !== name) return payload;
    await applyEditResponse(payload);
    const edit = payload.edit || {};
    const rows = edit.rows ? edit.rows.length : 0;
    setStatus(`${describe}${rows > 1 ? ` (${rows} rows)` : ""} · ${edit.edit_id || ""}`, "ok");
    return payload;
  } catch (error) {
    setStatus(`${describe} failed: ${error.message}`, "error");
    return null;
  } finally {
    releaseEdit();
    setLoading(-1);
  }
}

function currentFrameIndex() { return state.run ? state.run.series.frame_index[state.row] : null; }

function pickHypothesis(index, mirrored) {
  const frame = currentFrameIndex();
  if (frame === null) return;
  return submitEdit({ kind: "pick_hypothesis", frame, index, mirrored: !!mirrored }, `picked candidate ${index}${mirrored ? " mirrored" : ""} on frame ${frame}`);
}

function flipFrame() {
  const frame = currentFrameIndex();
  if (frame === null) return;
  return submitEdit({ kind: "flip", frame, scope: "frame" }, `flipped frame ${frame}`);
}

function flipSegment() {
  const frame = currentFrameIndex();
  if (frame === null) return;
  const seg = currentSegment();
  const where = seg && seg.frames ? `frames ${seg.frames[0]}–${seg.frames[1]}` : `the segment of frame ${frame}`;
  return submitEdit({ kind: "flip", frame, scope: "segment" }, `flipped ${where}`);
}

function undoEdit(editId) {
  if (!editsSupported()) return;
  const target = editId ? (state.edits || []).find((e) => e.id === editId) : newestUndoable();
  if (!target) { setStatus("nothing to undo", "error"); return; }
  const body = { kind: "undo" };
  if (editId) body.edit = editId;
  return submitEdit(body, `undid ${editLabel(target)} ${editFrames(target)}`);
}

// Apply what an edit changed: the series patch (iou, source, ambiguity
// score, classification, provenance for the rows around the edit), then
// drop the cached frames, show the refreshed frame and reload the log.
async function applyEditResponse(payload) {
  const run = state.run;
  const series = run.series;
  const n = series.frame_index.length;
  const patch = payload.series_patch || {};
  for (const [key, values] of Object.entries(patch)) {
    if (!values || typeof values !== "object") continue;
    if (key === "provenance" || key === "edited") { patchProvenance(key, values); continue; }
    if (key === "provenance_index") continue;  // the algorithm ids under "provenance" already say it
    if (!series[key]) series[key] = Array.from({ length: n }, () => null);
    for (const [row, value] of Object.entries(values)) {
      const r = parseInt(row, 10);
      if (r >= 0 && r < n) series[key][r] = value;
    }
  }
  for (const [name, values] of Object.entries(payload.flags_patch || {})) {
    if (!series.flags[name]) series.flags[name] = Array.from({ length: n }, () => 0);
    for (const [row, value] of Object.entries(values)) { const r = parseInt(row, 10); if (r >= 0 && r < n) series.flags[name][r] = value; }
  }
  // The whole provenance block, when the server sends it, beats the patch.
  if (payload.provenance && typeof payload.provenance === "object") run.provenance = normaliseProvenance({ series, provenance: payload.provenance }).provenance;
  if (payload.edits_count !== undefined) run.edits_count = payload.edits_count;
  const edit = payload.edit || {};
  if (run.provenance && edit.rows && !patch.provenance && !payload.provenance) {
    // No provenance patch from the server: take the algorithms from the edit's summary.
    for (const change of (edit.summary && edit.summary.changes) || []) {
      if (change.after && change.after.algorithm !== undefined) setRowProvenance(change.row, change.after.algorithm);
    }
  }
  state.frameCache.clear();
  state.segments.clear();
  const row = state.row;
  const frame = payload.frame;
  if (frame && frame.frame_index === series.frame_index[row] && !thresholdParam() && !state.showRaw) {
    state.frameCache.set(cacheKey(row, "light", false), frame);
  }
  populateWorst();
  let edits;
  if (Array.isArray(payload.edits)) { state.edits = payload.edits; state.editsError = null; renderEdits(); edits = Promise.resolve(); }
  else edits = loadEdits();
  await Promise.all([edits, showRow(row, { keepView: true, keepStarts: true, immediate: true })]);
  if (run.provenance && !patch.edited && !payload.provenance) markEditedRowsFromLog();
  renderProvenanceLegend();
  drawCharts();
  if (typeof onSourceChanged === "function") onSourceChanged();
  $("#run-info").textContent = describeSource(run);
}

// Without an "edited" patch from the server, the rows a live edit touched
// are the frame ranges of the live log entries (exact for picks and flips,
// which edit consecutive rows).
function markEditedRowsFromLog() {
  const run = state.run;
  const series = run.series;
  const step = run.entry.step || 1;
  const edited = series.frame_index.map(() => 0);
  for (const e of state.edits || []) {
    if (e.kind === "undo" || e.undone || !e.frames) continue;
    for (let f = e.frames[0]; f <= e.frames[1]; f += step) { const r = series.frame_index.indexOf(f); if (r >= 0) edited[r] = 1; }
  }
  run.provenance.edited = edited;
}

function setRowProvenance(row, algorithm) {
  const prov = state.run.provenance;
  if (!prov) return;
  let k = prov.algorithms.indexOf(algorithm);
  if (k < 0 && algorithm) { prov.algorithms.push(algorithm); k = prov.algorithms.length - 1; }
  prov.index[row] = algorithm ? k : -1;
  prov.algorithm[row] = algorithm || "";
}

// A provenance patch value may be an algorithm id, an index into the
// payload's algorithms, or {algorithm, job, edited}.
function patchProvenance(key, values) {
  const prov = state.run.provenance;
  if (!prov) return;
  for (const [row, value] of Object.entries(values)) {
    const r = parseInt(row, 10);
    if (r < 0 || r >= prov.index.length) continue;
    if (key === "edited") { prov.edited[r] = value ? 1 : 0; continue; }
    if (typeof value === "number") { prov.index[r] = value; prov.algorithm[r] = value >= 0 ? prov.algorithms[value] || "" : ""; }
    else if (typeof value === "string") setRowProvenance(r, value);
    else if (value && typeof value === "object") {
      if (value.algorithm !== undefined) setRowProvenance(r, value.algorithm);
      if (value.edited !== undefined) prov.edited[r] = value.edited ? 1 : 0;
    } else if (value === null) setRowProvenance(r, "");
  }
}

// ---------------------------------------------------------------- provenance legend and jump

function renderProvenanceLegend() {
  const legend = $("#provenance-legend");
  const select = $("#jump-provenance");
  const prov = state.run && state.run.provenance;
  if (!prov || !prov.algorithms.length) {
    legend.hidden = true; legend.innerHTML = "";
    select.innerHTML = '<option value="manual">manual edits</option>';
    return;
  }
  const counts = new Map();
  for (const k of prov.index) if (k >= 0) counts.set(k, (counts.get(k) || 0) + 1);
  const edited = prov.edited.reduce((a, v) => a + (v ? 1 : 0), 0);
  const parts = [`<span class="title">provenance</span>`];
  prov.algorithms.forEach((id, k) => {
    parts.push(`<span title="${escapeHtml(id)} · ${counts.get(k) || 0} rows"><span class="swatch" style="background:${provenanceColor(id, k)}"></span>${escapeHtml(shortAlgorithm(id))} <small>${counts.get(k) || 0}</small></span>`);
  });
  parts.push(`<span title="rows a live manual edit touched"><span class="swatch tick"></span>edited <small>${edited}</small></span>`);
  legend.innerHTML = parts.join("");
  legend.hidden = false;
  const previous = select.value;
  select.innerHTML = '<option value="manual">manual edits</option>' + prov.algorithms.map((id, k) => `<option value="${k}">${escapeHtml(shortAlgorithm(id))}</option>`).join("");
  if ([...select.options].some((o) => o.value === previous)) select.value = previous;
}

// The jump test for the Provenance select: rows whose algorithm is the
// selected one, or rows a manual edit touched.
function provenanceJumpTest() {
  const prov = state.run && state.run.provenance;
  if (!prov) return null;
  const wanted = $("#jump-provenance").value;
  if (wanted === "manual") return (r) => !!prov.edited[r] || isManualAlgorithm(prov.algorithm[r]);
  const k = parseInt(wanted, 10);
  return (r) => prov.index[r] === k;
}

// Enable or disable every edit control for the current source.
function renderEditControls() {
  renderSegmentInfo();
  renderHypothesesTable();
  renderEdits();
  renderProvenanceLegend();
}
