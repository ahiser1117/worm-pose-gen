"use strict";

// Phase 3: region reruns and comparison.  A region is a row range of the
// current workspace with an anchor on either side (rows outside the region
// whose stored pose the chains start from); an algorithm from the registry
// runs on it as a job and leaves a candidate set (per-row candidates and the
// path chosen among them) that the user compares with the current state and
// with a second set before accepting it.  Everything goes through
// /api/algorithms, /api/workspaces/<name>/{region,candidates} and
// /api/outcomes; a read-only run or the stdlib viewer shows the section
// disabled and an older app server without these endpoints gets a clear
// message instead of a broken panel.
//
// Rows and frames: the server speaks frames on the endpoints the UI calls;
// the UI keeps the region and the candidate rows in ROWS of the current
// source (state.run.series.frame_index maps between them).

const CANDIDATE_SLOTS = ["A", "B"];
const CANDIDATE_LAYER_IDS = ["cand_a", "cand_b"];
// [key, label, digits, better] of the compact before -> after table.
const REGION_METRICS = [
  ["median_iou", "IoU", 3, "up", "median IoU"],
  ["frames_below_0_9", "<0.9", 0, "down", "frames below IoU 0.9"],
  ["pose_jumps_over_width", "pose", 0, "down", "pose jumps over a body width"],
  ["length_jumps_over_3pct", "len", 0, "down", "length jumps over 3%"],
  ["orientation_flips", "flips", 0, "down", "orientation flips"],
];
const COMPARE_METRICS = [
  ["median_iou", "median IoU", 3, "up"],
  ["p10_iou", "p10 IoU", 3, "up"],
  ["min_iou", "min IoU", 3, "up"],
  ["frames_below_0_9", "frames < 0.9", 0, "down"],
  ["pose_jumps_over_width", "pose jumps > width", 0, "down"],
  ["length_jumps_over_3pct", "length jumps > 3%", 0, "down"],
  ["orientation_flips", "orientation flips", 0, "down"],
  ["frames_fitted", "frames fitted", 0, null],
  ["seconds", "seconds", 1, null],
];
const REGION_AROUND_ROWS = 10;

const regions = { sourceKey: null, loadingDetail: new Set(), unsupported: null };

function regionsSupported() { return isWorkspace() && serverIsApp(); }

function regionUrl(endpoint, query) { return editsUrl(endpoint, query); }

function regionsWhyNot() {
  if (!state.run) return "select a workspace first";
  if (state.sourceKind === "run") return "read-only run: import it as a workspace to run algorithms on it";
  if (!serverIsApp()) return "region runs need a workspace on the app server";
  return "";
}

// ---------------------------------------------------------------- rows and frames

function frameOfRow(row) { return state.run ? state.run.series.frame_index[row] : null; }

function rowOfFrame(frame) { return state.run ? state.run.series.frame_index.indexOf(frame) : -1; }

function rowCount() { return state.run ? state.run.series.frame_index.length : 0; }

// A server value naming a row: a row number, {row, frame}, or a frame when
// ``frames`` says so.  -1 when it names nothing in this source.
function rowFromServer(value, frames = false) {
  if (value === null || value === undefined || value === "") return null;
  if (typeof value === "object") {
    if (value.row !== undefined && value.row !== null) return value.row;
    if (value.frame !== undefined && value.frame !== null) return rowOfFrame(value.frame);
    return null;
  }
  return frames ? rowOfFrame(value) : value;
}

// ---------------------------------------------------------------- the region form

function regionInputs() {
  return { first: $("#region-first"), last: $("#region-last"), before: $("#region-anchor-before"), after: $("#region-anchor-after") };
}

function writeRegionForm() {
  const inputs = regionInputs();
  const region = state.region;
  const put = (input, row) => { input.value = row === null || row === undefined || row < 0 ? "" : String(frameOfRow(row)); };
  if (!region) { for (const input of Object.values(inputs)) input.value = ""; }
  else { put(inputs.first, region.first); put(inputs.last, region.last); put(inputs.before, region.anchor_before); put(inputs.after, region.anchor_after); }
  renderRegionInfo();
  drawCharts();
}

// Parse the form (frames) into state.region (rows).  Anchors left blank are
// none; an anchor inside the region or a frame outside the source is an error.
function readRegionForm() {
  const inputs = regionInputs();
  // Typing the region by hand fills one field before the other: no region yet, not an error.
  const filled = [inputs.first, inputs.last].filter((input) => input.value.trim() !== "").length;
  if (filled < 2) {
    if (filled === 0 && !inputs.before.value.trim() && !inputs.after.value.trim()) { state.region = null; renderRegionInfo(); drawCharts(); return false; }
    renderRegionInfo("give both the first and the last frame of the region", { neutral: true });
    return false;
  }
  const parse = (input, label, optional) => {
    const text = input.value.trim();
    if (!text) { if (optional) return null; throw new Error(`${label}: give a frame`); }
    const frame = Number(text);
    if (!Number.isInteger(frame)) throw new Error(`${label}: expected a frame number`);
    const row = rowOfFrame(frame);
    if (row < 0) throw new Error(`${label}: frame ${frame} is not in this workspace`);
    return row;
  };
  try {
    const first = parse(inputs.first, "first frame", false), last = parse(inputs.last, "last frame", false);
    if (last < first) throw new Error("last frame must not be before the first");
    const before = parse(inputs.before, "anchor before", true), after = parse(inputs.after, "anchor after", true);
    if (before !== null && before >= first) throw new Error("the anchor before must lie before the region");
    if (after !== null && after <= last) throw new Error("the anchor after must lie after the region");
    const previous = state.region || {};
    state.region = { first, last, anchor_before: before, anchor_after: after, reason: previous.first === first && previous.last === last ? previous.reason : "set by hand" };
    renderRegionInfo();
    drawCharts();
    return true;
  } catch (error) {
    setStatus(error.message, "error");
    renderRegionInfo(error.message);
    return false;
  }
}

function setRegionRows(first, last, reason) {
  const n = rowCount();
  first = Math.max(0, Math.min(n - 1, Math.min(first, last)));
  last = Math.max(0, Math.min(n - 1, Math.max(first, last)));
  const previous = state.region || {};
  // Anchors survive a new range when they still lie outside it.
  const before = previous.anchor_before !== null && previous.anchor_before !== undefined && previous.anchor_before < first ? previous.anchor_before : null;
  const after = previous.anchor_after !== null && previous.anchor_after !== undefined && previous.anchor_after > last ? previous.anchor_after : null;
  state.region = { first, last, anchor_before: before, anchor_after: after, reason: reason || "selected on the timeline" };
  writeRegionForm();
}

function clearRegion() {
  state.region = null;
  writeRegionForm();
  setStatus("region cleared", "ok");
}

function regionAroundFrame() {
  if (!state.run) return;
  setRegionRows(state.row - REGION_AROUND_ROWS, state.row + REGION_AROUND_ROWS, `${REGION_AROUND_ROWS} frames either side of frame ${frameOfRow(state.row)}`);
}

// /region for the current frame: the stretch containing it padded, with
// proposed anchors (the nearest clean rows outside it).
async function proposeRegion() {
  const why = regionsWhyNot();
  if (why) { setStatus(why, "error"); return; }
  const name = state.runName;
  const frame = frameOfRow(state.row);
  setLoading(1);
  try {
    const payload = await api(regionUrl("region", `frame=${frame}`));
    if (state.runName !== name) return;
    const rows = Array.isArray(payload.rows) && payload.rows.length === 2 ? payload.rows : null;
    const frames = Array.isArray(payload.frames) && payload.frames.length === 2 ? payload.frames : null;
    const first = rows ? rows[0] : frames ? rowOfFrame(frames[0]) : rowFromServer(payload.first);
    const last = rows ? rows[1] : frames ? rowOfFrame(frames[1]) : rowFromServer(payload.last);
    if (first === null || last === null || first < 0 || last < 0) throw new Error("the server's region names no rows of this workspace");
    const before = rowFromServer(payload.anchor_before), after = rowFromServer(payload.anchor_after);
    state.region = { first, last, anchor_before: before !== null && before >= 0 ? before : null, anchor_after: after !== null && after >= 0 ? after : null, reason: payload.reason || "proposed by the server" };
    writeRegionForm();
    setStatus(`region frames ${frameOfRow(first)}–${frameOfRow(last)}: ${payload.reason || "proposed"}`, "ok");
  } catch (error) {
    setStatus(`region proposal failed: ${error.message}`, "error");
  } finally { setLoading(-1); }
}

// /region?first=&last= : anchors for the frames in the form (the nearest
// clean rows outside them), keeping the range as typed.
async function proposeAnchors() {
  const why = regionsWhyNot();
  if (why) { setStatus(why, "error"); return; }
  if (!readRegionForm()) return;
  const name = state.runName;
  const region = state.region;
  setLoading(1);
  try {
    const payload = await api(regionUrl("region", `first=${frameOfRow(region.first)}&last=${frameOfRow(region.last)}`));
    if (state.runName !== name || !state.region) return;
    const before = rowFromServer(payload.anchor_before), after = rowFromServer(payload.anchor_after);
    state.region = { ...state.region, anchor_before: before !== null && before >= 0 ? before : null, anchor_after: after !== null && after >= 0 ? after : null };
    writeRegionForm();
    const missing = [before === null ? "before" : "", after === null ? "after" : ""].filter(Boolean);
    setStatus(missing.length ? `no clean anchor ${missing.join(" or ")} the region within reach; give one by hand or run without` : `anchors frames ${frameOfRow(before)} / ${frameOfRow(after)}`, missing.length ? "error" : "ok");
  } catch (error) {
    setStatus(`anchor proposal failed: ${error.message}`, "error");
  } finally { setLoading(-1); }
}

function renderRegionInfo(problem, options) {
  const info = $("#region-info");
  const why = regionsWhyNot();
  $("#region-target").textContent = isWorkspace() ? `on ${state.runName}` : "select a workspace";
  const run = $("#region-run");
  const neutral = !!(options && options.neutral);
  if (problem) { info.textContent = problem; info.classList.toggle("error", !neutral); run.disabled = true; return; }
  info.classList.remove("error");
  if (why) { info.textContent = why; run.disabled = true; return; }
  const region = state.region;
  if (!region) { info.textContent = "No region: use the current stretch, Shift+drag on a timeline chart, or type frames."; run.disabled = true; return; }
  const n = region.last - region.first + 1;
  const anchor = (row) => (row === null || row === undefined ? "none" : `frame ${frameOfRow(row)}`);
  const parts = [`frames ${frameOfRow(region.first)}–${frameOfRow(region.last)} (${n} row${n === 1 ? "" : "s"})`, `anchors ${anchor(region.anchor_before)} / ${anchor(region.anchor_after)}`];
  if (region.reason) parts.push(region.reason);
  info.textContent = parts.join(" · ");
  run.disabled = !state.algorithm;
  $("#region-run-note").textContent = state.algorithm ? "" : "pick an algorithm";
}

// ---------------------------------------------------------------- Shift+drag selection on the timeline

function beginRangeSelection(row) {
  if (!state.run) return;
  state.rangeSelect = { anchor: row, current: row };
  drawCharts();
}

function updateRangeSelection(row) {
  if (!state.rangeSelect) return;
  state.rangeSelect.current = row;
  drawCharts();
}

function endRangeSelection(row) {
  const sel = state.rangeSelect;
  state.rangeSelect = null;
  if (!sel || !state.run) { drawCharts(); return; }
  const n = rowCount();
  const clamp = (r) => Math.max(0, Math.min(n - 1, r));
  const a = clamp(Math.min(sel.anchor, row)), b = clamp(Math.max(sel.anchor, row));
  setRegionRows(a, b, "selected on the timeline");
  const why = regionsWhyNot();
  setStatus(`selected frames ${frameOfRow(a)}–${frameOfRow(b)} (${b - a + 1} rows)${why ? " · " + why : ""}`, why ? "error" : "ok");
  if (!why) showTab("pipeline");
}

// The region (shaded, edges) and its anchors (dashed lines) on a timeline
// chart; a selection in progress as a lighter band.  Called from drawChartsNow.
function drawRegionOnChart(c, X, colw, h, start, end) {
  const band = (a, b, fill) => {
    if (b < start || a >= end) return;
    c.fillStyle = fill;
    c.fillRect(X(a) - colw / 2, 0, X(b) - X(a) + colw, h);
  };
  const region = state.region;
  if (region) {
    band(region.first, region.last, "rgba(0,220,255,0.10)");
    c.strokeStyle = "rgba(0,220,255,0.7)"; c.lineWidth = 1; c.setLineDash([]);
    for (const r of [region.first, region.last]) {
      if (r < start || r >= end) continue;
      const x = r === region.first ? X(r) - colw / 2 : X(r) + colw / 2;
      c.beginPath(); c.moveTo(x, 0); c.lineTo(x, h); c.stroke();
    }
    c.strokeStyle = "rgba(255,255,255,0.8)"; c.setLineDash([2, 3]);
    for (const r of [region.anchor_before, region.anchor_after]) {
      if (r === null || r === undefined || r < start || r >= end) continue;
      c.beginPath(); c.moveTo(X(r), 0); c.lineTo(X(r), h); c.stroke();
    }
    c.setLineDash([]);
  }
  const sel = state.rangeSelect;
  if (sel) band(Math.min(sel.anchor, sel.current), Math.max(sel.anchor, sel.current), "rgba(255,255,255,0.16)");
}

// ---------------------------------------------------------------- algorithms and their forms

async function loadAlgorithms() {
  try {
    const payload = await api("/api/algorithms");
    const list = Array.isArray(payload) ? payload : payload.algorithms || [];
    state.algorithms = list.map((a) => ({ ...a, parameters: a.parameters || a.params || [] }));
    state.algorithmsError = null;
    regions.unsupported = null;
  } catch (error) {
    state.algorithms = null;
    state.algorithmsError = error.message;
    regions.unsupported = /not served by this server/.test(error.message) ? "this server has no region endpoints (update worm-pose-app and restart it)" : null;
  }
  state.regionParams = readStorage("poseViewer.regionParams", {});
  const saved = readStorage("poseViewer.regionAlgorithm", null);
  if (state.algorithms && state.algorithms.length && !state.algorithms.some((a) => a.id === state.algorithm)) {
    state.algorithm = state.algorithms.some((a) => a.id === saved) ? saved : (state.algorithms.find((a) => a.scope === "region" || !a.scope) || state.algorithms[0]).id;
  }
  renderAlgorithmSelect();
}

function algorithmInfo(id) { return (state.algorithms || []).find((a) => a.id === id) || null; }

function renderAlgorithmSelect() {
  const select = $("#region-algorithm");
  select.innerHTML = "";
  if (!state.algorithms) {
    const option = document.createElement("option");
    option.value = ""; option.textContent = state.algorithmsError ? "unavailable" : "loading…";
    select.appendChild(option);
    $("#region-algorithm-help").textContent = state.algorithmsError ? `algorithms unavailable: ${regions.unsupported || state.algorithmsError}` : "";
    $("#region-params").innerHTML = "";
    renderRegionInfo();
    return;
  }
  for (const a of state.algorithms) {
    const option = document.createElement("option");
    option.value = a.id;
    option.textContent = `${a.label || a.id}${a.scope && a.scope !== "region" ? ` (${a.scope})` : ""}`;
    option.title = a.description || "";
    select.appendChild(option);
  }
  select.value = state.algorithm || "";
  renderAlgorithmForm();
}

function renderAlgorithmForm() {
  const box = $("#region-params");
  const help = $("#region-algorithm-help");
  box.innerHTML = "";
  const algorithm = algorithmInfo(state.algorithm);
  if (!algorithm) { help.textContent = ""; renderRegionInfo(); return; }
  help.textContent = algorithm.description || "";
  if (!algorithm.parameters.length) box.innerHTML = '<div class="empty">no parameters</div>';
  for (const param of algorithm.parameters) box.appendChild(renderRegionParamField(algorithm.id, param));
  if (algorithm.parameters.length) {
    const reset = document.createElement("button");
    reset.type = "button"; reset.className = "stage-reset"; reset.textContent = "defaults";
    reset.addEventListener("click", () => { delete state.regionParams[algorithm.id]; writeStorage("poseViewer.regionParams", state.regionParams); renderAlgorithmForm(); });
    box.appendChild(reset);
  }
  renderRegionInfo();
}

function regionParamValue(algorithmId, param) {
  const values = state.regionParams[algorithmId] || {};
  return param.name in values ? values[param.name] : undefined;
}

// A field of the algorithm's parameter form, like the stage forms plus the
// registry's choice type and bounds; edited values (only) are kept per
// algorithm in localStorage and sent with the job.
function renderRegionParamField(algorithmId, param) {
  const type = String(param.type || "str").toLowerCase();
  const kind = type === "choice" ? "choice" : paramInputType(param);
  const edited = regionParamValue(algorithmId, param);
  const value = edited === undefined ? param.default : edited;
  const field = document.createElement("label");
  field.className = "param" + (edited === undefined ? "" : " edited");
  const bounds = [param.minimum !== null && param.minimum !== undefined ? `min ${param.minimum}` : "", param.maximum !== null && param.maximum !== undefined ? `max ${param.maximum}` : ""].filter(Boolean).join(", ");
  field.title = `${param.help || ""}\ndefault: ${JSON.stringify(param.default)}${bounds ? " · " + bounds : ""}`.trim();
  const name = document.createElement("span");
  name.className = "param-name";
  name.textContent = param.name;
  field.appendChild(name);
  let input;
  if (kind === "choice") {
    input = document.createElement("select");
    for (const choice of param.choices || []) { const o = document.createElement("option"); o.value = String(choice); o.textContent = String(choice); input.appendChild(o); }
    input.value = String(value);
  } else if (kind === "bool") {
    input = document.createElement("input"); input.type = "checkbox"; input.checked = !!value;
  } else if (kind === "json") {
    input = document.createElement("input"); input.type = "text"; input.value = value === null || value === undefined ? "" : JSON.stringify(value); input.placeholder = "JSON";
  } else {
    input = document.createElement("input"); input.type = kind === "str" ? "text" : "number";
    if (kind === "float") input.step = "any";
    if (kind === "int") input.step = "1";
    if (param.minimum !== null && param.minimum !== undefined) input.min = String(param.minimum);
    if (param.maximum !== null && param.maximum !== undefined) input.max = String(param.maximum);
    input.value = value === null || value === undefined ? "" : String(value);
    input.placeholder = param.default === null || param.default === undefined ? "none" : String(param.default);
  }
  input.addEventListener("change", () => {
    const parsed = kind === "choice" ? { value: coerceChoice(param, input.value) } : parseParamInput(kind, input);
    if (!parsed.error && parsed.value !== null && (kind === "int" || kind === "float")) {
      if (param.minimum !== null && param.minimum !== undefined && parsed.value < param.minimum) parsed.error = `must be at least ${param.minimum}`;
      if (param.maximum !== null && param.maximum !== undefined && parsed.value > param.maximum) parsed.error = `must be at most ${param.maximum}`;
    }
    if (parsed.error) { setStatus(`${algorithmId}.${param.name}: ${parsed.error}`, "error"); return; }
    const values = state.regionParams[algorithmId] || (state.regionParams[algorithmId] = {});
    if (JSON.stringify(parsed.value) === JSON.stringify(param.default)) delete values[param.name]; else values[param.name] = parsed.value;
    writeStorage("poseViewer.regionParams", state.regionParams);
    field.classList.toggle("edited", param.name in values);
  });
  field.appendChild(input);
  return field;
}

// A choice value in the type of its default (numbers stay numbers).
function coerceChoice(param, text) {
  const match = (param.choices || []).find((choice) => String(choice) === text);
  return match === undefined ? text : match;
}

function collectRegionParams(algorithmId) { return { ...(state.regionParams[algorithmId] || {}) }; }

// ---------------------------------------------------------------- running a region

async function runRegion() {
  const why = regionsWhyNot();
  if (why) { setStatus(why, "error"); return; }
  if (!readRegionForm()) return;
  const region = state.region;
  const algorithm = algorithmInfo(state.algorithm);
  if (!algorithm) { setStatus("pick an algorithm", "error"); return; }
  const frame = (row) => (row === null || row === undefined ? null : frameOfRow(row));
  const body = {
    kind: "region", workspace: state.runName, algorithm: algorithm.id,
    first: frame(region.first), last: frame(region.last), anchor_before: frame(region.anchor_before), anchor_after: frame(region.anchor_after),
    params: collectRegionParams(algorithm.id), label: `region ${algorithm.id} frames ${frame(region.first)}–${frame(region.last)} on ${state.runName}`,
  };
  const button = $("#region-run");
  button.disabled = true;
  dismissCandidateJobs();
  try {
    const record = await post("/api/jobs", body);
    setStatus(`queued ${algorithm.id} on frames ${body.first}–${body.last} as ${record.id}`, "ok");
    await loadJobs();
  } catch (error) {
    setStatus(`region run failed: ${error.message}`, "error");
  } finally {
    button.disabled = false;
    renderRegionInfo();
  }
}

// Region jobs of this workspace that are still queued or running, above the
// sets, and the region jobs that failed or were cancelled since the last
// run was submitted or the list refreshed (a run that vanishes without a
// word looks like a lost job).
function renderCandidateJobs() {
  const box = $("#candidates-jobs");
  if (!box) return;
  const ofWorkspace = (j) => isWorkspace() && (j.spec || {}).kind === "region" && j.spec.workspace === state.runName;
  const broken = (j) => (j.state === "failed" || j.state === "cancelled") && (!regions.dismissedBefore || (j.finished_at || j.created_at || "") > regions.dismissedBefore);
  const jobs = state.jobs.filter((j) => ofWorkspace(j) && (jobActive(j) || broken(j)));
  box.innerHTML = "";
  for (const job of jobs) {
    const params = (job.spec || {}).params || {};
    const frames = job.spec.frames ? `frames ${job.spec.frames[0]}–${job.spec.frames[1]}` : "";
    const node = document.createElement("div");
    node.className = `cand-job${jobActive(job) ? "" : " " + job.state}`;
    const why = job.error || job.message;
    node.innerHTML = `${jobActive(job) ? '<span class="spin"></span>' : '<span class="dot"></span>'}<b>${escapeHtml(params.algorithm || job.spec.algorithm || "region")}</b> ${escapeHtml(frames)} · ${job.state}${job.state === "running" ? ` ${Math.round(100 * (job.progress || 0))}%` : ""} · ${escapeHtml(job.id)}${why ? `<div class="meta">${escapeHtml(why)}</div>` : ""}`;
    box.appendChild(node);
  }
}

// Failed region jobs older than now leave the Candidate sets section.
function dismissCandidateJobs() {
  regions.dismissedBefore = new Date().toISOString();
  renderCandidateJobs();
}

// ---------------------------------------------------------------- candidate sets

// A list entry in one shape: {id, algorithm, params, rows: [a, b], frames:
// [f0, f1], anchor_before, anchor_after (rows), metrics, metrics_before,
// created_at, accepted, job, candidates, path_rows}.
function normaliseSetEntry(entry) {
  const out = { ...entry };
  out.id = String(entry.id !== undefined ? entry.id : entry.candidate_set || "");
  const rows = Array.isArray(entry.rows) && entry.rows.length === 2 && typeof entry.rows[0] === "number" ? entry.rows : null;
  const frames = Array.isArray(entry.frames) && entry.frames.length === 2 ? entry.frames : null;
  out.rows = rows || (frames ? [rowOfFrame(frames[0]), rowOfFrame(frames[1])] : [entry.first, entry.last].map((v) => (v === undefined ? -1 : v)));
  out.frames = frames || (out.rows[0] >= 0 && out.rows[1] >= 0 ? [frameOfRow(out.rows[0]), frameOfRow(out.rows[1])] : null);
  const anchors = entry.anchors || {};
  out.anchor_before = rowFromServer(entry.anchor_before !== undefined ? entry.anchor_before : anchors.before);
  out.anchor_after = rowFromServer(entry.anchor_after !== undefined ? entry.anchor_after : anchors.after);
  out.metrics = entry.metrics || entry.metrics_after || {};
  out.metrics_before = entry.metrics_before || null;
  out.accepted = !!entry.accepted;
  // Rows accepted so far (a partial accept); the set stays open for the rest.
  out.accepted_rows = Array.isArray(entry.accepted_rows) ? entry.accepted_rows.map((r) => rowFromServer(r)).filter((r) => r !== null && r >= 0) : [];
  return out;
}

async function loadCandidateSets() {
  if (!regionsSupported()) { state.candidateSets = []; state.candidateSetsError = null; renderCandidateSets(); return; }
  const name = state.runName;
  try {
    const payload = await api(regionUrl("candidates"));
    if (state.runName !== name) return;
    const list = Array.isArray(payload) ? payload : payload.candidates || payload.sets || [];
    state.candidateSets = list.map(normaliseSetEntry);
    state.candidateSetsError = null;
  } catch (error) {
    if (state.runName !== name) return;
    state.candidateSets = [];
    state.candidateSetsError = error.message;
  }
  // Sets that vanished (discarded elsewhere) leave the overlay.
  state.shownSets = state.shownSets.map((id) => (id && state.candidateSets.some((s) => s.id === id) ? id : null));
  renderCandidateSets();
  renderComparison();
  renderFrameCandidateSets();
  relayerCandidates();
}

function candidateSetEntry(id) { return state.candidateSets.find((s) => s.id === id) || null; }

function candidateSlot(id) { return state.shownSets.indexOf(id); }

function slotColor(slot) { return layer(CANDIDATE_LAYER_IDS[slot]).color; }

function slotCss(slot) { return `rgb(${slotColor(slot).join(",")})`; }

function metricText(value, digits) {
  if (value === null || value === undefined || (typeof value === "number" && !Number.isFinite(value))) return "–";
  return typeof value === "number" ? value.toFixed(digits) : String(value);
}

// "before → after" with the after coloured by whether it improved.
function metricDelta(before, after, digits, better) {
  const b = before === undefined ? null : before, a = after === undefined ? null : after;
  if (b === null && a === null) return "–";
  if (b === null) return metricText(a, digits);
  let cls = "";
  if (a !== null && better && Math.abs(a - b) > (digits ? Math.pow(10, -digits) / 2 : 0)) cls = (better === "up") === (a > b) ? "better" : "worse";
  return `<span class="dim">${metricText(b, digits)}</span> → <span class="${cls}">${metricText(a, digits)}</span>`;
}

function paramsSummary(params) {
  const entries = Object.entries(params || {});
  if (!entries.length) return "defaults";
  return entries.map(([k, v]) => `${k}=${typeof v === "number" && !Number.isInteger(v) ? v.toPrecision(3) : JSON.stringify(v)}`).join(" ");
}

function renderCandidateSets() {
  const box = $("#candidates");
  const count = $("#candidates-count");
  if (!box) return;
  const sets = state.candidateSets || [];
  const live = sets.filter((s) => !s.accepted).length;
  count.textContent = regionsSupported() && sets.length ? `${live} open · ${sets.length} total` : "";
  box.innerHTML = "";
  if (!regionsSupported()) { box.innerHTML = `<div class="empty">${escapeHtml(regionsWhyNot())}</div>`; return; }
  if (state.candidateSetsError) { box.innerHTML = `<div class="empty">candidate sets unavailable: ${escapeHtml(regions.unsupported || state.candidateSetsError)}</div>`; return; }
  if (!sets.length) { box.innerHTML = '<div class="empty">No candidate sets yet. Set a region and run an algorithm on it.</div>'; return; }
  for (const s of sets) {
    const slot = candidateSlot(s.id);
    const node = document.createElement("div");
    node.className = `cand-set${slot >= 0 ? " shown slot-" + CANDIDATE_SLOTS[slot].toLowerCase() : ""}${s.accepted ? " accepted" : ""}`;
    node.dataset.set = s.id;
    if (slot >= 0) node.style.borderColor = slotCss(slot);
    const frames = s.frames ? `frames ${s.frames[0]}–${s.frames[1]} (${s.rows[1] - s.rows[0] + 1})` : "";
    const anchor = (row) => (row === null || row === undefined || row < 0 ? "none" : String(frameOfRow(row)));
    const cells = REGION_METRICS.map(([key, , digits, better, help]) => `<td class="num" title="${help} before → after">${metricDelta(s.metrics_before ? s.metrics_before[key] : null, s.metrics[key], digits, better)}</td>`).join("");
    const heads = REGION_METRICS.map(([, label, , , help]) => `<th class="num" title="${help}">${label}</th>`).join("");
    const partial = !s.accepted && s.accepted_rows && s.accepted_rows.length ? `${s.accepted_rows.length}/${s.path_rows || "?"} rows accepted` : "";
    const stateText = s.accepted ? `accepted${s.accepted_edit ? " · " + escapeHtml(s.accepted_edit) : ""}` : [partial, slot >= 0 ? `shown as ${CANDIDATE_SLOTS[slot]}` : ""].filter(Boolean).join(" · ");
    const acceptDisabled = s.accepted || acceptPending;
    const acceptTitle = s.accepted ? "already accepted" : acceptPending ? "an accept is being applied" : partial ? "accept the rest of the path (an undoable edit)" : "install the path's candidates as the poses of these frames (an undoable edit)";
    node.innerHTML =
      `<div class="cand-head">${slot >= 0 ? `<span class="slot" style="background:${slotCss(slot)}">${CANDIDATE_SLOTS[slot]}</span>` : ""}<b>${escapeHtml(shortAlgorithm(s.algorithm))}</b><span class="cand-frames">${escapeHtml(frames)}</span><span class="cand-state">${stateText}</span></div>` +
      `<div class="meta">${escapeHtml(s.id)} · ${escapeHtml(fmtTime(s.created_at))} · anchors ${anchor(s.anchor_before)} / ${anchor(s.anchor_after)} · ${fmt(s.candidates)} candidates${s.metrics.seconds ? ` · ${fmtDuration(s.metrics.seconds)}` : ""}<div class="params" title="${escapeHtml(JSON.stringify(s.params || {}))}">${escapeHtml(paramsSummary(s.params))}</div></div>` +
      `<table class="cand-metrics"><thead><tr>${heads}</tr></thead><tbody><tr>${cells}</tr></tbody></table>` +
      `<div class="row cand-actions"><button type="button" data-show>${slot >= 0 ? "Hide" : "Show"}</button><button type="button" data-goto title="go to the first frame of the region">go to</button><button type="button" data-accept class="primary" ${acceptDisabled ? "disabled" : ""} title="${acceptTitle}">Accept</button><button type="button" data-discard title="delete this candidate set">Discard</button></div>`;
    node.querySelector("[data-show]").addEventListener("click", () => toggleCandidateSet(s.id));
    node.querySelector("[data-goto]").addEventListener("click", () => { if (s.rows[0] >= 0) showRow(s.rows[0], { keepView: true }); });
    node.querySelector("[data-accept]").addEventListener("click", () => acceptCandidateSet(s.id));
    node.querySelector("[data-discard]").addEventListener("click", () => discardCandidateSet(s.id));
    box.appendChild(node);
  }
}

// ---- detail (per-row candidates and the path)

function normaliseCandidate(item, position) {
  return {
    index: item.index !== undefined ? item.index : position,
    centerline_xy: item.centerline_xy || null,
    energy: item.energy === undefined ? null : item.energy,
    iou: item.iou === undefined ? null : item.iou,
    source: item.source || "", start: item.start || "",
    chosen: !!item.chosen, mirrored: !!item.mirrored,
  };
}

// The detail payload in one shape: the list entry fields plus byRow (Map row
// -> {row, frame, candidates, chosen: position | null, mirrored}) and
// current_metrics.  Candidates may come per row ([{row | frame, candidates:
// [...], chosen?, mirrored?}] or {row: [...]}) or as one flat list with a row
// on each; the path as [[row, index, mirrored]], [{row, index, mirrored}] or
// {row: [index, mirrored]}; without a path the candidates' chosen flags decide.
function normaliseCandidateDetail(payload, entry) {
  const detail = normaliseSetEntry({ ...(entry || {}), ...payload, rows: Array.isArray(payload.rows) && payload.rows.length === 2 && typeof payload.rows[0] === "number" ? payload.rows : entry ? entry.rows : undefined });
  const byRow = new Map();
  const rowOf = (item) => (item.row !== undefined && item.row !== null ? item.row : item.frame !== undefined ? rowOfFrame(item.frame) : -1);
  const ensure = (r) => { if (!byRow.has(r)) byRow.set(r, { row: r, frame: frameOfRow(r), candidates: [], chosen: null, mirrored: false }); return byRow.get(r); };
  // The list entry's "candidates" is a count; the per-row lists live under per_row (or candidates as lists).
  let source = Array.isArray(payload.candidates) || (payload.candidates && typeof payload.candidates === "object") ? payload.candidates : undefined;
  if (source === undefined && Array.isArray(payload.per_row)) source = payload.per_row;
  if (source === undefined && Array.isArray(payload.rows) && payload.rows.length && typeof payload.rows[0] === "object") source = payload.rows;
  if (source === undefined) source = payload.frames_detail || null;
  if (Array.isArray(source)) {
    for (const item of source) {
      if (!item || typeof item !== "object") continue;
      const r = rowOf(item);
      if (r < 0) continue;
      if (Array.isArray(item.candidates)) {
        const e = ensure(r);
        e.candidates = item.candidates.map(normaliseCandidate);
        if (typeof item.chosen === "number" && item.chosen >= 0) e.chosen = item.chosen;
        if (item.mirrored !== undefined) e.mirrored = !!item.mirrored;
      } else if (item.centerline_xy) {
        const e = ensure(r);
        e.candidates.push(normaliseCandidate(item, e.candidates.length));
      }
    }
  } else if (source && typeof source === "object") {
    for (const [key, list] of Object.entries(source)) { const r = parseInt(key, 10); if (r >= 0 && Array.isArray(list)) ensure(r).candidates = list.map(normaliseCandidate); }
  }
  const path = payload.path;
  const setChoice = (r, index, mirrored) => { if (r === null || r === undefined || r < 0 || typeof index !== "number" || index < 0) return; const e = ensure(r); e.chosen = index; e.mirrored = !!mirrored; };
  if (Array.isArray(path)) for (const p of path) { if (Array.isArray(p)) setChoice(p[0], p[1], p[2]); else if (p && typeof p === "object") setChoice(rowOf(p), p.index, p.mirrored); }
  else if (path && typeof path === "object") for (const [key, v] of Object.entries(path)) { const r = parseInt(key, 10); if (Array.isArray(v)) setChoice(r, v[0], v[1]); else if (v && typeof v === "object") setChoice(r, v.index, v.mirrored); else if (typeof v === "number") setChoice(r, v, false); }
  for (const e of byRow.values()) {
    // The chosen index names a candidate's index field when they carry one, else its position.
    if (e.chosen !== null) { const pos = e.candidates.findIndex((c) => c.index === e.chosen); e.chosen = pos >= 0 ? pos : e.chosen < e.candidates.length ? e.chosen : null; }
    if (e.chosen === null) { const k = e.candidates.findIndex((c) => c.chosen); if (k >= 0) { e.chosen = k; e.mirrored = !!e.candidates[k].mirrored; } }
    e.candidates.forEach((c, i) => { c.chosen = i === e.chosen; });
    e.chosenIou = e.chosen !== null && e.candidates[e.chosen] ? e.candidates[e.chosen].iou : null;
  }
  detail.byRow = byRow;
  detail.current_metrics = payload.current_metrics || null;
  detail.path_length = [...byRow.values()].filter((e) => e.chosen !== null).length;
  return detail;
}

async function loadCandidateDetail(id, force = false) {
  if (!force && state.candidateDetails.has(id)) return state.candidateDetails.get(id);
  const name = state.runName;
  const payload = await api(regionUrl(`candidates/${encodeURIComponent(id)}`));
  if (state.runName !== name) throw new Error("workspace changed while loading the candidate set");
  const detail = normaliseCandidateDetail(payload, candidateSetEntry(id));
  state.candidateDetails.set(id, detail);
  return detail;
}

function shownDetail(slot) {
  const id = state.shownSets[slot];
  return id ? state.candidateDetails.get(id) || null : null;
}

// Show a set in a free slot (or the slot the other set left last); a third
// set replaces B.
async function showCandidateSet(id, slot) {
  if (candidateSlot(id) >= 0) return;
  if (slot === undefined) { slot = state.shownSets.indexOf(null); if (slot < 0) slot = 1; }
  const replaced = state.shownSets[slot];
  state.shownSets[slot] = id;
  renderCandidateSets();
  if (regions.loadingDetail.has(id)) return;
  regions.loadingDetail.add(id);
  setLoading(1);
  try {
    await loadCandidateDetail(id);
    setStatus(`${id} shown as ${CANDIDATE_SLOTS[slot]}${replaced ? ` (replacing ${replaced})` : ""}`, "ok");
  } catch (error) {
    state.shownSets[slot] = null;
    setStatus(`cannot show ${id}: ${error.message}`, "error");
  } finally {
    regions.loadingDetail.delete(id);
    setLoading(-1);
    renderCandidateSets();
    renderComparison();
    renderFrameCandidateSets();
    relayerCandidates();
  }
}

function hideCandidateSet(id) {
  const slot = candidateSlot(id);
  if (slot < 0) return;
  state.shownSets[slot] = null;
  renderCandidateSets();
  renderComparison();
  renderFrameCandidateSets();
  relayerCandidates();
}

function toggleCandidateSet(id) { return candidateSlot(id) >= 0 ? hideCandidateSet(id) : showCandidateSet(id); }

// Layer names follow the shown sets; the stage, legend and charts redraw.
function relayerCandidates() {
  CANDIDATE_LAYER_IDS.forEach((layerId, slot) => {
    const l = layer(layerId);
    const id = state.shownSets[slot];
    const entry = id ? candidateSetEntry(id) : null;
    l.name = id ? `Candidate set ${CANDIDATE_SLOTS[slot]}: ${id}${entry ? " " + shortAlgorithm(entry.algorithm) : ""}` : `Candidate set ${CANDIDATE_SLOTS[slot]}`;
    const node = document.querySelector(`.layer[data-layer="${layerId}"] > span`);
    if (node) { const key = node.querySelector(".key"); node.innerHTML = `<span class="swatch" style="background:rgb(${l.color.join(",")})"></span>${escapeHtml(l.name)} `; if (key) node.appendChild(key); }
  });
  renderLegend();
  renderLayerAvailability();
  draw();
  drawCharts();
}

function candidateSetLegendText(layerId) {
  const slot = CANDIDATE_LAYER_IDS.indexOf(layerId);
  const id = state.shownSets[slot];
  if (!id) return null;
  const entry = candidateSetEntry(id);
  return `${CANDIDATE_SLOTS[slot]}: ${id}${entry ? " " + shortAlgorithm(entry.algorithm) : ""}`;
}

// The chosen candidate of each shown set solid (head □ / tail ○), the other
// candidates of the row faint; a mirrored choice is drawn reversed.  Falls
// back to the frame payload's candidate_sets for a row the detail lacks.
function drawCandidateSetOverlays() {
  const f = state.frame;
  if (!f || !state.run) return;
  const row = f.stats && f.stats.row !== undefined ? f.stats.row : rowOfFrame(f.frame_index);
  const fromFrame = (f.pose && f.pose.candidate_sets) || [];
  state.shownSets.forEach((id, slot) => {
    if (!id) return;
    const l = layer(CANDIDATE_LAYER_IDS[slot]);
    if (!l.on || l.alpha <= 0) return;
    const color = `rgb(${l.color.join(",")})`;
    const detail = state.candidateDetails.get(id);
    const entry = detail && detail.byRow.get(row);
    if (entry) {
      entry.candidates.forEach((c, i) => { if (i !== entry.chosen && c.centerline_xy) drawCurve(c.centerline_xy, color, 1, [3, 3], 0.35 * l.alpha); });
      const chosen = entry.chosen !== null ? entry.candidates[entry.chosen] : null;
      if (chosen && chosen.centerline_xy) {
        const points = entry.mirrored ? chosen.centerline_xy.slice().reverse() : chosen.centerline_xy;
        drawCurve(points, color, 2.5, null, l.alpha);
        drawEnds(points, color);
      }
      return;
    }
    const covering = fromFrame.find((s) => String(s.id) === id);
    if (covering && covering.centerline_xy) { drawCurve(covering.centerline_xy, color, 2.5, null, l.alpha); drawEnds(covering.centerline_xy, color); }
  });
}

// The chosen candidates' IoU over the rows of each shown set, on the IoU chart.
function drawCandidateSetIou(c, xs, Y, start, end) {
  state.shownSets.forEach((id, slot) => {
    const detail = id && state.candidateDetails.get(id);
    const l = layer(CANDIDATE_LAYER_IDS[slot]);
    if (!detail || !l.on) return;
    const ys = [];
    for (let i = start; i < end; i++) { const e = detail.byRow.get(i); ys.push(e && e.chosenIou !== null && e.chosenIou !== undefined ? Y(e.chosenIou) : null); }
    polyline(c, xs, ys, `rgb(${l.color.join(",")})`, null, 1.6);
  });
}

// ---- accept and discard

// One accept at a time: the buttons go grey while the request runs (a long
// region takes seconds), and the edit guard shared with the picks and flips
// (edits.js) refuses a second edit meanwhile.
let acceptPending = false;

async function acceptCandidateSet(id) {
  if (!regionsSupported()) { setStatus(regionsWhyNot(), "error"); return; }
  const entry = candidateSetEntry(id);
  if (entry && entry.accepted) { setStatus(`${id} is already accepted`, "error"); return; }
  if (acceptPending) { setStatus(`accept of ${id}: another accept is still being applied`, "error"); return; }
  if (!claimEdit()) return;
  acceptPending = true;
  renderCandidateSets();
  renderComparison();
  const name = state.runName;
  setLoading(1);
  try {
    const payload = await post(regionUrl(`candidates/${encodeURIComponent(id)}/accept`), { use_path: true });
    if (state.runName !== name) return;
    const edit = payload.edit || {};
    hideCandidateSet(id);
    state.candidateDetails.clear();  // current_metrics of every set changed with the state
    await applyEditResponse(payload);
    await Promise.all([loadCandidateSets(), loadOutcomes()]);
    for (const shown of state.shownSets) if (shown) await loadCandidateDetail(shown, true).catch(() => {});
    renderComparison();
    setStatus(`accepted ${id}${edit.rows ? ` (${edit.rows.length} rows)` : ""} · ${edit.edit_id || ""}`, "ok");
  } catch (error) {
    setStatus(`accept ${id} failed: ${error.message}`, "error");
  } finally {
    acceptPending = false;
    releaseEdit();
    setLoading(-1);
    if (state.runName === name) { renderCandidateSets(); renderComparison(); }
  }
}

async function discardCandidateSet(id) {
  if (!regionsSupported()) { setStatus(regionsWhyNot(), "error"); return; }
  const name = state.runName;
  try {
    await api(regionUrl(`candidates/${encodeURIComponent(id)}`), { method: "DELETE" });
    if (state.runName !== name) return;
    hideCandidateSet(id);
    state.candidateDetails.delete(id);
    await loadCandidateSets();
    setStatus(`discarded ${id}`, "ok");
  } catch (error) {
    setStatus(`discard ${id} failed: ${error.message}`, "error");
  }
}

// ---- comparison of the current state with the shown sets

function sameRows(a, b) { return !!(a && b && a.rows[0] === b.rows[0] && a.rows[1] === b.rows[1]); }

function renderComparison() {
  const box = $("#comparison");
  if (!box) return;
  const details = [shownDetail(0), shownDetail(1)];
  if (!details[0] && !details[1]) { box.hidden = true; box.innerHTML = ""; return; }
  box.hidden = false;
  const both = details[0] && details[1];
  const split = both && !sameRows(details[0], details[1]);
  const columns = [];
  if (!split) columns.push({ label: "current", metrics: (details[0] || details[1]).current_metrics, cls: "current", title: "the workspace's state over the region now" });
  details.forEach((d, slot) => {
    if (!d) return;
    if (split) columns.push({ label: `current (${CANDIDATE_SLOTS[slot]} rows)`, metrics: d.current_metrics, cls: "current" });
    columns.push({ label: CANDIDATE_SLOTS[slot], metrics: d.metrics, cls: `slot-${CANDIDATE_SLOTS[slot].toLowerCase()}`, color: slotCss(slot), title: `${d.id} · ${d.algorithm} · ${paramsSummary(d.params)}` });
  });
  const head = columns.map((col) => `<th class="num ${col.cls}" ${col.color ? `style="color:${col.color}"` : ""} title="${escapeHtml(col.title || "")}">${escapeHtml(col.label)}</th>`).join("");
  const rows = COMPARE_METRICS.map(([key, label, digits, better]) => {
    const values = columns.map((col) => (col.metrics ? col.metrics[key] : null));
    const finite = values.filter((v) => typeof v === "number" && Number.isFinite(v));
    const best = better && finite.length > 1 ? (better === "up" ? Math.max(...finite) : Math.min(...finite)) : null;
    const worst = better && finite.length > 1 ? (better === "up" ? Math.min(...finite) : Math.max(...finite)) : null;
    const cells = values.map((v) => `<td class="num ${best !== null && v === best && best !== worst ? "better" : ""}">${metricText(v === undefined ? null : v, digits)}</td>`).join("");
    return `<tr><td>${label}</td>${cells}</tr>`;
  }).join("");
  const region = (d) => (d && d.frames ? `frames ${d.frames[0]}–${d.frames[1]}` : "");
  const caption = both ? (split ? `A ${region(details[0])} · B ${region(details[1])} (different regions: each has its own current column)` : `${region(details[0])} · A ${details[0].id} ${shortAlgorithm(details[0].algorithm)} · B ${details[1].id} ${shortAlgorithm(details[1].algorithm)}`) : `${region(details[0] || details[1])} · ${(details[0] || details[1]).id} ${shortAlgorithm((details[0] || details[1]).algorithm)}`;
  const accepts = details.map((d, slot) => (d ? `<button type="button" data-accept-slot="${slot}" class="primary" ${d.accepted || acceptPending ? "disabled" : ""} style="border-color:${slotCss(slot)}">Accept ${CANDIDATE_SLOTS[slot]}</button>` : "")).join("");
  box.innerHTML = `<div class="meta">${escapeHtml(caption)}</div><table class="stats comparison-table"><thead><tr><th>metric</th>${head}</tr></thead><tbody>${rows}</tbody></table><div class="row">${accepts}</div>`;
  for (const button of box.querySelectorAll("[data-accept-slot]")) button.addEventListener("click", () => { const id = state.shownSets[parseInt(button.dataset.acceptSlot, 10)]; if (id) acceptCandidateSet(id); });
}

// ---- the frame payload's candidate sets (details panel)

// Sets covering the current frame (the frame payload's pose.candidate_sets,
// completed from the list) with Show buttons and the chosen candidate's IoU
// from the loaded detail.
function renderFrameCandidateSets() {
  const section = $("#frame-candidates-section");
  const list = $("#frame-candidates");
  if (!section || !list) return;
  const f = state.frame;
  if (!f || !state.run || !isWorkspace()) { section.hidden = true; list.innerHTML = ""; return; }
  const row = f.stats && f.stats.row !== undefined ? f.stats.row : rowOfFrame(f.frame_index);
  const covering = new Map();
  for (const s of (f.pose && f.pose.candidate_sets) || []) covering.set(String(s.id), { id: String(s.id), algorithm: s.algorithm, index: s.index });
  for (const s of state.candidateSets) if (!s.accepted && s.rows[0] <= row && row <= s.rows[1] && !covering.has(s.id)) covering.set(s.id, { id: s.id, algorithm: s.algorithm, index: null });
  if (!covering.size) { section.hidden = true; list.innerHTML = ""; return; }
  section.hidden = false;
  $("#frame-candidates-head").textContent = `${covering.size} open`;
  list.innerHTML = "";
  for (const s of covering.values()) {
    const slot = candidateSlot(s.id);
    const detail = state.candidateDetails.get(s.id);
    const entry = detail && detail.byRow.get(row);
    const chosen = entry && entry.chosen !== null ? entry.candidates[entry.chosen] : null;
    const item = document.createElement("div");
    item.className = "item";
    const facts = chosen ? `chosen ${chosen.source || "#" + chosen.index}${entry.mirrored ? " ↔" : ""} · IoU ${fmt(chosen.iou)} · ${entry.candidates.length} candidates` : s.index !== null && s.index !== undefined ? `chosen #${s.index}` : "";
    item.innerHTML = `<span>${slot >= 0 ? `<span class="slot" style="background:${slotCss(slot)}">${CANDIDATE_SLOTS[slot]}</span> ` : ""}<b>${escapeHtml(s.id)}</b> ${escapeHtml(shortAlgorithm(s.algorithm || ""))}<div class="meta">${escapeHtml(facts)}</div></span><span><button type="button">${slot >= 0 ? "hide" : "show"}</button></span>`;
    item.querySelector("button").addEventListener("click", (event) => { event.stopPropagation(); toggleCandidateSet(s.id); });
    item.addEventListener("click", () => showTab("pipeline"));
    list.appendChild(item);
  }
}

// ---------------------------------------------------------------- outcomes

async function loadOutcomes() {
  const table = $("#outcomes tbody");
  if (!table) return;
  if (!serverIsApp()) { state.outcomes = []; state.outcomesError = regionsWhyNot(); renderOutcomes(); return; }
  const mine = $("#outcomes-mine").checked && isWorkspace();
  try {
    const payload = await api(`/api/outcomes${mine ? `?workspace=${encodeURIComponent(state.runName)}` : ""}`);
    const list = Array.isArray(payload) ? payload : payload.outcomes || [];
    state.outcomes = mine ? list.filter((o) => !o.workspace || o.workspace === state.runName) : list;
    state.outcomesError = null;
  } catch (error) {
    state.outcomes = [];
    state.outcomesError = regions.unsupported || error.message;
  }
  renderOutcomes();
}

function renderOutcomes() {
  const body = $("#outcomes tbody");
  const count = $("#outcomes-count");
  const select = $("#outcomes-algorithm");
  if (!body) return;
  const previous = select.value;
  const algorithms = [...new Set(state.outcomes.map((o) => o.algorithm).filter(Boolean))].sort();
  select.innerHTML = '<option value="">all algorithms</option>' + algorithms.map((id) => `<option value="${escapeHtml(id)}">${escapeHtml(shortAlgorithm(id))}</option>`).join("");
  if (algorithms.includes(previous)) select.value = previous;
  const filter = select.value;
  const shown = state.outcomes.filter((o) => !filter || o.algorithm === filter);
  body.innerHTML = "";
  if (state.outcomesError) { count.textContent = ""; body.innerHTML = `<tr><td colspan="8" class="empty">${escapeHtml(state.outcomesError)}</td></tr>`; return; }
  count.textContent = state.outcomes.length ? `${shown.length}/${state.outcomes.length}` : "";
  if (!shown.length) { body.innerHTML = `<tr><td colspan="8" class="empty">${state.outcomes.length ? "no runs of that algorithm" : "no region runs logged yet"}</td></tr>`; return; }
  const accepted = state.outcomes.filter((o) => o.accepted).length;
  count.textContent += ` · ${accepted} accepted`;
  for (const o of shown) {
    const before = o.metrics_before || {}, after = o.metrics_after || o.metrics || {};
    const tr = document.createElement("tr");
    const frames = o.frames && o.frames.length === 2 ? `${o.frames[0]}–${o.frames[1]}` : o.first !== undefined ? `rows ${o.first}–${o.last}` : "";
    const mine = !o.workspace || (isWorkspace() && o.workspace === state.runName);
    tr.className = `${o.accepted ? "accepted" : ""}${mine ? "" : " other"}`;
    tr.title = [`${o.workspace || ""} · ${o.candidate_set || ""}${o.job ? " · job " + o.job : ""}`, `anchors ${JSON.stringify(o.anchors || {})}`, `params ${JSON.stringify(o.params || {})}`, `${o.candidates || "?"} candidates, ${o.path_rows || "?"} on the path`].join("\n");
    tr.innerHTML = `<td>${escapeHtml(fmtTime(o.time))}</td><td>${escapeHtml(shortAlgorithm(o.algorithm))}</td><td>${escapeHtml(frames)}${mine ? "" : `<div class="meta">${escapeHtml(o.workspace)}</div>`}</td>` +
      `<td class="num">${metricDelta(before.median_iou, after.median_iou, 3, "up")}</td><td class="num">${metricDelta(before.frames_below_0_9, after.frames_below_0_9, 0, "down")}</td>` +
      `<td class="num">${metricDelta(before.pose_jumps_over_width, after.pose_jumps_over_width, 0, "down")} · ${metricDelta(before.length_jumps_over_3pct, after.length_jumps_over_3pct, 0, "down")}</td>` +
      `<td class="num">${metricDelta(before.orientation_flips, after.orientation_flips, 0, "down")}</td><td>${o.accepted ? `<span class="ok" title="accepted ${escapeHtml(fmtTime(o.accepted_at))}">✓</span>` : ""}</td>`;
    tr.addEventListener("click", () => {
      if (!state.run) return;
      if (o.workspace && (!isWorkspace() || o.workspace !== state.runName)) { if (workspaceEntry(o.workspace)) selectSource("workspace", o.workspace, o.frames ? o.frames[0] : undefined); else setStatus(`workspace ${o.workspace} is not in the catalog`, "error"); return; }
      const r = o.frames ? rowOfFrame(o.frames[0]) : o.first;
      if (r !== undefined && r >= 0) showRow(r, { keepView: true });
      if (o.candidate_set && candidateSetEntry(o.candidate_set)) { const node = document.querySelector(`.cand-set[data-set="${o.candidate_set}"]`); if (node) node.scrollIntoView({ block: "nearest" }); }
    });
    body.appendChild(tr);
  }
}

// ---------------------------------------------------------------- wiring

// Called by panels.onSourceChanged once a source is loaded (also after a job
// finished or an edit landed on the same source).  A different source drops
// the region, the shown sets and the cached details.
function regionsOnSourceChanged() {
  const key = currentSourceKey();
  if (key !== regions.sourceKey) {
    regions.sourceKey = key;
    state.region = null;
    state.rangeSelect = null;
    state.shownSets = [null, null];
    state.candidateDetails.clear();
    writeRegionForm();
  } else {
    // Same source reloaded: the state may have changed under the shown sets.
    state.candidateDetails.clear();
    for (const id of state.shownSets) if (id) loadCandidateDetail(id).then(() => { renderComparison(); renderFrameCandidateSets(); relayerCandidates(); }).catch(() => {});
  }
  renderRegionInfo();
  renderCandidateJobs();
  loadCandidateSets();
  loadOutcomes();
  relayerCandidates();
}

function initRegions() {
  $("#region-propose").addEventListener("click", proposeRegion);
  $("#region-around").addEventListener("click", regionAroundFrame);
  $("#region-clear").addEventListener("click", clearRegion);
  $("#region-anchors").addEventListener("click", proposeAnchors);
  for (const input of Object.values(regionInputs())) {
    input.addEventListener("change", readRegionForm);
    input.addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); input.blur(); } });
  }
  $("#region-algorithm").addEventListener("change", (e) => { state.algorithm = e.target.value || null; writeStorage("poseViewer.regionAlgorithm", state.algorithm); renderAlgorithmForm(); });
  $("#region-run").addEventListener("click", runRegion);
  $("#candidates-refresh").addEventListener("click", () => { dismissCandidateJobs(); state.candidateDetails.clear(); loadCandidateSets().then(() => Promise.all(state.shownSets.filter(Boolean).map((id) => loadCandidateDetail(id, true)))).then(() => { renderComparison(); relayerCandidates(); }).catch(() => {}); });
  $("#outcomes-algorithm").addEventListener("change", renderOutcomes);
  $("#outcomes-mine").addEventListener("change", () => loadOutcomes());
  $("#outcomes-refresh").addEventListener("click", () => loadOutcomes());
  renderRegionInfo();
  loadAlgorithms();
}
