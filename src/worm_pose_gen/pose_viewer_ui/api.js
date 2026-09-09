"use strict";

// Shared state, constants and HTTP helpers of the pose viewer.  Loaded first;
// every other script is a plain file that adds functions to the page scope.
//
// The viewer shows one *source* at a time: a workspace (read-write, served
// under /api/workspaces/<name>/...) or a run directory (read-only, the old
// /api/run and /api/frame endpoints).  ``sourceUrl`` hides the difference.

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
  { id: "hyp_forward", name: "Hypotheses: forward chain", kind: "vector", on: true, alpha: 0.9, color: [255, 211, 77] },
  { id: "hyp_backward", name: "Hypotheses: backward chain", kind: "vector", on: true, alpha: 0.9, color: [192, 128, 255] },
  { id: "hyp_independent", name: "Hypotheses: independent refit", kind: "vector", on: true, alpha: 0.9, color: [120, 200, 255] },
  { id: "prediction", name: "Chain prediction (autoregressive)", kind: "vector", on: true, alpha: 0.9, color: [255, 255, 255] },
  { id: "compare", name: "Compare run pose", kind: "vector", on: true, alpha: 1.0, color: [255, 170, 60] },
  { id: "starts", name: "Fitter starts (press Starts)", kind: "vector", on: true, alpha: 0.9, color: [120, 255, 255] },
  { id: "crop", name: "Crop window", kind: "vector", on: false, alpha: 0.8, color: [150, 150, 150] },
  // Phase 3: the chosen candidates of up to two shown candidate sets (regions.js fills the names in).
  { id: "cand_a", name: "Candidate set A", kind: "vector", on: true, alpha: 1.0, color: [0, 220, 255] },
  { id: "cand_b", name: "Candidate set B", kind: "vector", on: true, alpha: 1.0, color: [255, 110, 40] },
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
  ["prediction_distance_px", "Distance to chain prediction px"],
  ["path_energy_gap", "Path energy gap (chosen − lowest)"],
  ["hypotheses_count", "Hypotheses per frame"],
  ["tube_coverage", "Tube coverage by mask"],
  ["max_bend_widths", "Tightest bend (width / radius)"],
];
const HYP_COLORS = { forward: [255, 211, 77], backward: [192, 128, 255], independent: [120, 200, 255] };

// Pipeline stages in their natural order; the server's /api/stages payload
// is the authority, this is the fallback when it is unavailable.
const STAGE_ORDER = ["segment", "prior", "fit", "ambiguity", "propagate", "track", "export"];
const JOB_STATES = ["queued", "running", "done", "failed", "cancelled"];

const state = {
  info: null,            // /api/state payload
  runs: [],              // catalog rows of run directories
  workspaces: [],        // WorkspaceInfo (+summary) rows
  sourceKind: null,      // "run" | "workspace"
  run: null,             // payload of the selected source, in the /api/run shape
  runName: null,
  compare: null,         // payload of the compare source
  compareName: null,
  compareKind: null,
  compareRows: null,     // frame_index -> row in the compare source
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
  // Phase 1 panels.
  recordings: [],
  recording: null,       // selected RecordingInfo (by path)
  stages: null,          // [{name, params: [{name, type, default, help}]}]
  stagesError: null,
  stageValues: {},       // stage -> {param: value} as edited in the forms
  jobs: [],
  jobStates: new Map(),  // job id -> last seen state, to notice transitions
  openLogs: new Set(),   // job ids whose log viewer is expanded
  chain: null,           // {workspace, queue: [stage], current: jobId | null}
  // Phase 2 edits.
  edits: [],             // list_edits entries of the current workspace, newest first
  editsError: null,
  segments: new Map(),   // row -> /segment payload ({frames, rows, in_stretch}) of the current workspace
  // Phase 3 regions.
  algorithms: null,      // /api/algorithms entries [{id, label, scope, description, parameters}]
  algorithmsError: null,
  algorithm: null,       // selected algorithm id
  regionParams: {},      // algorithm id -> {param: value} as edited in the form
  region: null,          // {first, last, anchor_before, anchor_after, reason} in ROWS of the current source
  rangeSelect: null,     // Shift+drag in progress on a timeline chart: {anchor, current} rows
  candidateSets: [],     // list entries of the workspace's candidate sets, newest first
  candidateSetsError: null,
  candidateDetails: new Map(),  // set id -> normalised detail payload (per-row candidates and path)
  shownSets: [null, null],      // set ids in the overlay slots A and B
  outcomes: [],
  outcomesError: null,
};

let toastTimer = null;

// Errors also appear as a toast over the stage: the status line sits at the
// bottom of a sidebar tab and is easy to miss.
function showToast(text, kind) {
  const node = document.querySelector("#toast");
  if (!node) return;
  node.textContent = text;
  node.className = "toast" + (kind ? " " + kind : "");
  node.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { node.hidden = true; }, kind === "error" ? 9000 : 3500);
}

function setStatus(text, kind) {
  const node = $("#status");
  node.textContent = text;
  node.className = "status" + (kind ? " " + kind : "");
  if (kind === "error") showToast(text, "error");
}

function describeHttpError(path, response, payload) {
  const message = (payload && payload.error) || response.statusText || `HTTP ${response.status}`;
  if (response.status === 404 && message === "not found") return `${path.split("?")[0]} is not served by this server (${message})`;
  return message;
}

async function api(path, options) {
  const response = await fetch(path, options);
  let payload;
  try { payload = await response.json(); } catch (error) { payload = { error: response.statusText || "invalid response" }; }
  if (!response.ok || (payload && payload.error)) throw new Error(describeHttpError(path, response, payload));
  return payload;
}

function post(path, body) {
  return api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
}

async function fetchJson(url, controller) {
  const response = await fetch(url, controller ? { signal: controller.signal } : undefined);
  const payload = await response.json();
  if (!response.ok || payload.error) throw new Error(payload.error || response.statusText);
  return payload;
}

const isAbort = (error) => error && error.name === "AbortError";

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

function escapeHtml(text) {
  return String(text === null || text === undefined ? "" : text).replace(/[&<>"']/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]));
}

function fmtBytes(bytes) {
  if (bytes === null || bytes === undefined) return "–";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let v = bytes, u = 0;
  while (v >= 1024 && u < units.length - 1) { v /= 1024; u++; }
  return `${v.toFixed(u >= 3 ? 2 : u >= 1 ? 1 : 0)} ${units[u]}`;
}

// ISO or unix-second timestamps as a short local time.
function fmtTime(value) {
  if (value === null || value === undefined || value === "" || Number.isNaN(value)) return "–";
  const date = typeof value === "number" ? new Date(value * 1000) : new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  const today = new Date();
  const sameDay = date.toDateString() === today.toDateString();
  return sameDay ? date.toLocaleTimeString() : date.toLocaleString();
}

function fmtDuration(seconds) {
  if (seconds === null || seconds === undefined || !Number.isFinite(seconds)) return "–";
  if (seconds < 60) return `${Math.round(seconds)} s`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)} m ${Math.round(seconds % 60)} s`;
  return `${Math.floor(seconds / 3600)} h ${Math.floor((seconds % 3600) / 60)} m`;
}

// ---------------------------------------------------------------- sources
//
// Select option values and URL hashes name a source as "run:<name>" or
// "ws:<name>"; the API paths differ, the payload shapes are the same.

function sourceKey(kind, name) { return `${kind === "workspace" ? "ws" : "run"}:${name}`; }

function parseSourceKey(key) {
  if (!key) return null;
  const [prefix, ...rest] = key.split(":");
  const name = rest.join(":");
  if (!name) return null;
  return { kind: prefix === "ws" ? "workspace" : "run", name };
}

// The payload URL of a source (the /api/run shape).
function sourcePayloadUrl(kind, name) {
  return kind === "workspace" ? `/api/workspaces/${encodeURIComponent(name)}` : `/api/run?name=${encodeURIComponent(name)}`;
}

// A per-frame endpoint of a source: frame, pose or starts, with extra query.
function sourceUrl(kind, name, endpoint, query) {
  const extra = query ? `&${query}` : "";
  if (kind === "workspace") return `/api/workspaces/${encodeURIComponent(name)}/${endpoint}?${(extra || "&").slice(1)}`;
  return `/api/${endpoint}?run=${encodeURIComponent(name)}${extra}`;
}

function currentSourceKey() { return state.runName ? sourceKey(state.sourceKind, state.runName) : null; }

function isWorkspace() { return state.sourceKind === "workspace"; }

function workspaceEntry(name) { return state.workspaces.find((w) => w.name === name) || null; }

function runEntry(name) { return state.runs.find((r) => r.name === name) || null; }

// The recording stem a catalog row (run or workspace) belongs to.
function recordingStem(path) {
  if (!path) return "";
  const base = String(path).split("/").pop();
  return base.replace(/\.h5$/, "");
}
